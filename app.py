import io
import re
import json
import time
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import streamlit as st

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="RLoop-G4 | Orientation-Aware Instability Predictor",
    page_icon="🧬",
    layout="wide",
)

# ==============================================================================
# MODULE 1: LIVE PUBLIC DATABASE RETRIEVAL (ENSEMBL + UCSC GRCh38/hg38)
# ==============================================================================
def http_get_json(url, timeout=20):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RLoopG4-Streamlit/2.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gene_from_public_dbs(symbol, upstream_bp=800, downstream_bp=1200):
    """Fetches real GRCh38/hg38 sequence & UCSC tracks for any human gene symbol."""
    clean_sym = urllib.parse.quote(symbol.strip().upper())
    ens_url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{clean_sym}?content-type=application/json"
    ens_data = http_get_json(ens_url)

    chrom = str(ens_data["seq_region_name"])
    if not chrom.startswith("chr"):
        chrom = f"chr{chrom}"
    strand = int(ens_data["strand"])
    tss = int(ens_data["start"]) if strand == 1 else int(ens_data["end"])

    if strand == 1:
        win_start = tss - upstream_bp
        win_end = tss + downstream_bp
        rev_comp = 0
    else:
        win_start = tss - downstream_bp
        win_end = tss + upstream_bp
        rev_comp = 1

    # Fetch real hg38 DNA sequence from UCSC
    seq_url = (
        f"https://api.genome.ucsc.edu/getData/sequence?"
        f"genome=hg38;chrom={chrom};start={win_start};end={win_end};revComp={rev_comp}"
    )
    seq = http_get_json(seq_url)["dna"].upper()

    # Fetch real UCSC annotation tracks
    def get_track(track_name):
        t_url = (
            f"https://api.genome.ucsc.edu/getData/track?"
            f"genome=hg38;track={track_name};chrom={chrom};start={win_start};end={win_end}"
        )
        try:
            raw_items = http_get_json(t_url).get(track_name, [])
        except Exception:
            return []

        mapped = []
        for item in raw_items:
            s = int(item.get("chromStart", 0))
            e = int(item.get("chromEnd", 0))
            if e <= win_start or s >= win_end:
                continue
            s_c, e_c = max(s, win_start), min(e, win_end)
            if strand == 1:
                rel_s, rel_e = s_c - tss, e_c - tss
            else:
                rel_s, rel_e = tss - e_c, tss - s_c
            mapped.append({
                "rel_start": rel_s,
                "rel_end": rel_e,
                "width": rel_e - rel_s,
                "label": item.get("ucscLabel") or item.get("name") or "Feature",
            })
        return mapped

    cpg_islands = get_track("cpgIslandExt")
    encode_ccres = get_track("encodeCcreCombined")

    return {
        "symbol": ens_data.get("display_name", symbol.upper()),
        "ensembl_id": ens_data.get("id", "N/A"),
        "description": ens_data.get("description", ""),
        "chrom": chrom,
        "strand": strand,
        "tss": tss,
        "win_start": win_start,
        "win_end": win_end,
        "upstream_bp": upstream_bp,
        "downstream_bp": downstream_bp,
        "seq": seq,
        "cpg_islands": cpg_islands,
        "encode_ccres": encode_ccres,
    }


# ==============================================================================
# MODULE 2 & 3: ORIENTATION-AWARE BIOPHYSICAL ENGINE
# ==============================================================================
class RLoopG4Framework:
    def __init__(self, window_size=100, g4_window=25, gc_skew_threshold=0.15, g4h_threshold=1.2):
        self.window_size = window_size
        self.g4_window = g4_window
        self.gc_skew_threshold = gc_skew_threshold
        self.g4h_threshold = g4h_threshold
        self.pqs_plus = re.compile(r"(?=(G{3,}[ATGC]{1,7}G{3,}[ATGC]{1,7}G{3,}[ATGC]{1,7}G{3,}))")
        self.pqs_minus = re.compile(r"(?=(C{3,}[ATGC]{1,7}C{3,}[ATGC]{1,7}C{3,}[ATGC]{1,7}C{3,}))")

    def compute_gc_skew_profile(self, seq):
        n = len(seq)
        profile = np.zeros(n)
        hw = self.window_size // 2
        for i in range(n):
            sub = seq[max(0, i - hw) : min(n, i + hw)]
            g, c = sub.count("G"), sub.count("C")
            profile[i] = (g - c) / (g + c) if (g + c) > 0 else 0.0
        return profile

    def compute_g4hunter_profile(self, seq):
        n = len(seq)
        base_scores = np.zeros(n)
        i = 0
        while i < n:
            if seq[i] in ("G", "C"):
                char = seq[i]
                run_len = 1
                while i + run_len < n and seq[i + run_len] == char:
                    run_len += 1
                val = min(run_len, 4)
                base_scores[i : i + run_len] = val if char == "G" else -val
                i += run_len
            else:
                i += 1
        hw = self.g4_window // 2
        g4h_profile = np.zeros(n)
        for idx in range(n):
            g4h_profile[idx] = np.mean(base_scores[max(0, idx - hw) : min(n, idx + hw)])
        return g4h_profile

    def detect_pqs_motifs(self, seq):
        motifs = []
        for m in self.pqs_plus.finditer(seq):
            motifs.append({
                "start": m.start(),
                "end": m.start() + len(m.group(1)),
                "strand": "+ (Non-Template / Displaced)",
                "short_strand": "+",
                "motif": m.group(1),
            })
        for m in self.pqs_minus.finditer(seq):
            motifs.append({
                "start": m.start(),
                "end": m.start() + len(m.group(1)),
                "strand": "- (Template / Hybridized)",
                "short_strand": "-",
                "motif": m.group(1),
            })
        return motifs

    def analyze(self, gene_data, conflict="Head-On"):
        seq = gene_data["seq"]
        upstream_bp = gene_data["upstream_bp"]

        gc_skew = self.compute_gc_skew_profile(seq)
        g4h = self.compute_g4hunter_profile(seq)

        rloop_pot = np.clip(gc_skew, 0, None)
        displaced_g4 = np.clip(g4h, 0, None)
        template_g4 = np.clip(-g4h, 0, None)

        trc_weight = 2.2 if conflict == "Head-On" else 1.0
        rfi_profile = trc_weight * (
            (rloop_pot * displaced_g4 * 4.5)
            + (rloop_pot * 0.4)
            + (template_g4 * 0.15)
        )

        motifs = self.detect_pqs_motifs(seq)
        motif_rows = []
        for m in motifs:
            local_skew = float(np.mean(gc_skew[m["start"] : m["end"]]))
            local_g4h = float(np.mean(g4h[m["start"] : m["end"]]))
            local_rfi = float(np.max(rfi_profile[m["start"] : m["end"]]))
            coforms = (local_skew >= self.gc_skew_threshold) and (m["short_strand"] == "+")

            motif_rows.append({
                "Gene": gene_data["symbol"],
                "Conflict": conflict,
                "TSS_Rel_Start": m["start"] - upstream_bp,
                "TSS_Rel_End": m["end"] - upstream_bp,
                "G4_Strand": m["strand"],
                "Local_GC_Skew": round(local_skew, 3),
                "Local_G4Hunter": round(local_g4h, 3),
                "G_Loop_CoFormation": "YES (Active G-Loop)" if coforms else "NO (Suppressed)",
                "Peak_RFI_Score": round(local_rfi, 3),
                "Motif_Sequence": m["motif"],
            })

        peak_rfi = float(np.max(rfi_profile))
        mean_rfi = float(np.mean(rfi_profile))
        gc_content = round(((seq.count("G") + seq.count("C")) / len(seq)) * 100, 1)

        if peak_rfi >= 2.0:
            risk = "CRITICAL"
        elif peak_rfi >= 0.8:
            risk = "MODERATE"
        else:
            risk = "LOW"

        summary = {
            "Gene": gene_data["symbol"],
            "Locus_hg38": f"{gene_data['chrom']}:{gene_data['win_start']}-{gene_data['win_end']}",
            "Strand": "+" if gene_data["strand"] == 1 else "-",
            "Conflict": conflict,
            "GC_%": gc_content,
            "CpG_Islands": len(gene_data["cpg_islands"]),
            "Max_GC_Skew": round(float(np.max(gc_skew)), 3),
            "Max_Displaced_G4H": round(float(np.max(displaced_g4)), 3),
            "Active_GLoop_G4s": sum(1 for r in motif_rows if "YES" in r["G_Loop_CoFormation"]),
            "Suppressed_G4s": sum(1 for r in motif_rows if "NO" in r["G_Loop_CoFormation"]),
            "Mean_RFI": round(mean_rfi, 3),
            "Peak_RFI": round(peak_rfi, 3),
            "Risk_Category": risk,
        }

        tracks = {
            "gc_skew": gc_skew,
            "g4h": g4h,
            "rfi": rfi_profile,
        }
        return summary, pd.DataFrame(motif_rows), tracks


# ==============================================================================
# PLOTTING HELPERS
# ==============================================================================
def build_browser_figure(gene_data, tracks, conflict, gc_thresh, g4h_thresh):
    x = np.arange(len(tracks["gc_skew"])) - gene_data["upstream_bp"]
    strand_sym = "+" if gene_data["strand"] == 1 else "-"

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.1, 0.7, 1.2]}
    )
    fig.suptitle(
        f"RLoop-G4 Genomic Track: {gene_data['symbol']} ({gene_data['chrom']}:{gene_data['win_start']}-{gene_data['win_end']}, "
        f"Strand {strand_sym} | {conflict} TRC)",
        fontsize=12, fontweight="bold"
    )

    for ax in axes:
        ax.axvline(0, color="black", ls="-.", lw=1.0, alpha=0.6)

    # Panel 1: GC Skew
    ax1 = axes[0]
    ax1.plot(x, tracks["gc_skew"], color="#1f77b4", lw=1.7, label="GC Skew (G-C)/(G+C)")
    ax1.axhline(gc_thresh, color="red", ls="--", lw=1, label=f"R-Loop Threshold ({gc_thresh})")
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.fill_between(x, tracks["gc_skew"], gc_thresh, where=(tracks["gc_skew"] >= gc_thresh), color="#1f77b4", alpha=0.25)
    ax1.set_ylabel("GC Skew")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.25)

    # Panel 2: G4Hunter
    ax2 = axes[1]
    ax2.plot(x, tracks["g4h"], color="#2ca02c", lw=1.7, label="G4Hunter Score")
    ax2.axhline(g4h_thresh, color="darkgreen", ls="--", lw=1, label="Non-Template (+) G4")
    ax2.axhline(-g4h_thresh, color="purple", ls="--", lw=1, label="Template (-) G4")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.fill_between(x, tracks["g4h"], 0, where=(tracks["g4h"] > 0), color="#2ca02c", alpha=0.25)
    ax2.fill_between(x, tracks["g4h"], 0, where=(tracks["g4h"] < 0), color="#9467bd", alpha=0.25)
    ax2.set_ylabel("G4Hunter")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.25)

    # Panel 3: UCSC Annotations
    ax3 = axes[2]
    for iv in gene_data["cpg_islands"]:
        ax3.broken_barh([(iv["rel_start"], iv["width"])], (0.7, 0.55), facecolors="#17becf", edgecolors="black", alpha=0.85)
    for iv in gene_data["encode_ccres"]:
        ax3.broken_barh([(iv["rel_start"], iv["width"])], (0.0, 0.55), facecolors="#ff7f0e", edgecolors="black", alpha=0.85)
    ax3.set_yticks([0.27, 0.97])
    ax3.set_yticklabels(["ENCODE cCRE", "CpG Island"])
    ax3.set_ylim(-0.2, 1.5)
    ax3.set_ylabel("UCSC Tracks")
    ax3.grid(alpha=0.25)
    ax3.legend(
        handles=[
            Patch(facecolor="#17becf", edgecolor="black", label="UCSC CpG Island"),
            Patch(facecolor="#ff7f0e", edgecolor="black", label="ENCODE cCRE"),
        ],
        loc="upper right", fontsize=8
    )

    # Panel 4: RFI Score
    ax4 = axes[3]
    ax4.plot(x, tracks["rfi"], color="#d62728", lw=2.0, label="Replication Fork Instability (RFI)")
    ax4.axhline(2.0, color="darkred", ls=":", lw=1.2, label="Critical Collapse Threshold (2.0)")
    ax4.fill_between(x, tracks["rfi"], 0, color="#d62728", alpha=0.25)
    ax4.set_ylabel("RFI Score")
    ax4.set_xlabel("Position Relative to TSS = 0 bp (5' -> 3' Non-Template Strand)")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(alpha=0.25)

    plt.tight_layout()
    return fig


def build_comparison_figure(summary_df):
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(summary_df) * 0.65)))
    colors = ["#d62728" if s >= 2.0 else "#ff7f0e" if s >= 0.8 else "#2ca02c" for s in summary_df["Peak_RFI"]]
    labels = [f"{row['Gene']} ({row['Conflict']})" for _, row in summary_df.iterrows()]

    bars = ax.barh(labels, summary_df["Peak_RFI"], color=colors, edgecolor="black", height=0.55)
    ax.axvline(2.0, color="darkred", ls="--", lw=1.2, label="Critical Threshold (2.0)")
    ax.axvline(0.8, color="darkorange", ls=":", lw=1.2, label="Moderate Threshold (0.8)")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.08, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontweight="bold")

    ax.set_xlabel("Peak Replication Fork Instability (RFI) Score")
    ax.set_title("Cross-Gene Instability Comparison (Live GRCh38/hg38)", fontweight="bold")
    ax.set_xlim(0, max(summary_df["Peak_RFI"].max() * 1.2, 2.5))
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf


# ==============================================================================
# STREAMLIT UI LAYOUT
# ==============================================================================
st.title("🧬 RLoop-G4: Orientation-Aware Instability Framework")
st.caption("Live GRCh38/hg38 R-Loop–G-Quadruplex Co-Formation & Replication Fork Instability Predictor (Ensembl + UCSC REST APIs)")

with st.sidebar:
    st.header("Analysis Settings")
    mode = st.radio("Analysis Mode", ["Single Gene Deep-Dive", "Multi-Gene Batch Comparison"])

    if mode == "Single Gene Deep-Dive":
        gene_input = st.text_input("Human Gene Symbol (HGNC)", value="MYC", help="Examples: MYC, BRCA1, BCL2, KRAS, TP53, HBB")
        conflict_type = st.selectbox("TRC Orientation", ["Head-On", "Co-Directional"])
    else:
        batch_input = st.text_area(
            "Gene Symbols (comma-separated)",
            value="MYC, BRCA1, BCL2, KRAS, HBB",
            help="Enter human gene symbols separated by commas."
        )
        conflict_type = st.selectbox("TRC Orientation for Batch", ["Head-On", "Co-Directional"])

    st.divider()
    st.subheader("Biophysical Parameters")
    upstream_bp = st.slider("Upstream of TSS (bp)", 200, 1500, 800, step=100)
    downstream_bp = st.slider("Downstream of TSS (bp)", 400, 2000, 1200, step=100)
    gc_window = st.slider("GC-Skew Sliding Window (bp)", 50, 200, 100, step=25)
    g4_window = st.slider("G4Hunter Window (bp)", 15, 50, 25, step=5)
    gc_thresh = st.slider("R-Loop GC-Skew Threshold", 0.05, 0.40, 0.15, step=0.05)
    g4h_thresh = st.slider("G4Hunter Threshold", 0.8, 2.0, 1.2, step=0.1)

    run_btn = st.button("Run RLoop-G4 Analysis", type="primary", use_container_width=True)

model = RLoopG4Framework(
    window_size=gc_window,
    g4_window=g4_window,
    gc_skew_threshold=gc_thresh,
    g4h_threshold=g4h_thresh,
)

# ==============================================================================
# EXECUTION: SINGLE GENE MODE
# ==============================================================================
if mode == "Single Gene Deep-Dive":
    if run_btn or "single_loaded" not in st.session_state:
        st.session_state["single_loaded"] = True
        try:
            with st.spinner(f"Fetching {gene_input.upper()} (GRCh38/hg38) from Ensembl & UCSC..."):
                gene_data = fetch_gene_from_public_dbs(gene_input, upstream_bp, downstream_bp)
                summary, motifs_df, tracks = model.analyze(gene_data, conflict=conflict_type)

            # Top KPI Row
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Peak RFI Score", f"{summary['Peak_RFI']:.2f}")
            c2.metric("Risk Category", summary["Risk_Category"])
            c3.metric("Active G-Loop G4s", summary["Active_GLoop_G4s"])
            c4.metric("GC Content", f"{summary['GC_%']}%")
            c5.metric("UCSC CpG Islands", summary["CpG_Islands"])

            st.info(
                f"**{gene_data['symbol']}** (`{gene_data['ensembl_id']}`) | "
                f"**Coordinates:** `{summary['Locus_hg38']}` (Strand `{summary['Strand']}`) | "
                f"{gene_data['description']}"
            )

            tab1, tab2, tab3 = st.tabs([
                "📈 Multi-Track Genomic Browser",
                "🧬 Detected G4 Motifs",
                "📄 Sequence & Export"
            ])

            with tab1:
                fig = build_browser_figure(gene_data, tracks, conflict_type, gc_thresh, g4h_thresh)
                st.pyplot(fig)
                st.download_button(
                    label="Download Genomic Track (PNG)",
                    data=fig_to_png_bytes(fig),
                    file_name=f"{gene_data['symbol']}_{conflict_type}_tracks.png",
                    mime="image/png",
                )
                plt.close(fig)

            with tab2:
                if not motifs_df.empty:
                    st.dataframe(motifs_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        label="Download G4 Motifs Table (CSV)",
                        data=motifs_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{gene_data['symbol']}_g4_motifs.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No canonical Putative G-Quadruplex Sequences (PQS) detected in this window.")

            with tab3:
                st.subheader("Locus Summary")
                st.dataframe(pd.DataFrame([summary]), use_container_width=True, hide_index=True)
                fasta_text = f">{gene_data['symbol']}|{summary['Locus_hg38']}|strand={summary['Strand']}|conflict={conflict_type}\n{gene_data['seq']}\n"
                st.download_button(
                    label="Download hg38 FASTA Sequence",
                    data=fasta_text.encode("utf-8"),
                    file_name=f"{gene_data['symbol']}_hg38.fasta",
                    mime="text/plain",
                )

        except Exception as e:
            st.error(f"Could not analyze gene '{gene_input}'. Verify it is a valid human HGNC symbol. Details: {e}")

# ==============================================================================
# EXECUTION: MULTI-GENE BATCH MODE
# ==============================================================================
else:
    if run_btn:
        symbols = [s.strip().upper() for s in batch_input.split(",") if s.strip()]
        if not symbols:
            st.warning("Please enter at least one gene symbol.")
        else:
            summaries, all_motifs = [], []
            progress = st.progress(0, text="Querying Ensembl & UCSC...")

            for idx, sym in enumerate(symbols):
                try:
                    gene_data = fetch_gene_from_public_dbs(sym, upstream_bp, downstream_bp)
                    summary, motifs_df, _ = model.analyze(gene_data, conflict=conflict_type)
                    summaries.append(summary)
                    if not motifs_df.empty:
                        all_motifs.append(motifs_df)
                    time.sleep(0.15)
                except Exception as e:
                    st.warning(f"Skipped '{sym}': {e}")
                progress.progress((idx + 1) / len(symbols), text=f"Processed {sym} ({idx + 1}/{len(symbols)})")

            progress.empty()

            if summaries:
                summary_df = pd.DataFrame(summaries)
                tab1, tab2 = st.tabs(["📊 Cross-Gene Comparison", "🧬 Combined G4 Motifs"])

                with tab1:
                    fig = build_comparison_figure(summary_df)
                    st.pyplot(fig)
                    st.download_button(
                        label="Download Comparison Chart (PNG)",
                        data=fig_to_png_bytes(fig),
                        file_name="rloop_g4_batch_comparison.png",
                        mime="image/png",
                    )
                    plt.close(fig)

                    st.subheader("Batch Summary Table")
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        label="Download Summary Table (CSV)",
                        data=summary_df.to_csv(index=False).encode("utf-8"),
                        file_name="rloop_g4_batch_summary.csv",
                        mime="text/csv",
                    )

                with tab2:
                    if all_motifs:
                        combined_motifs = pd.concat(all_motifs, ignore_index=True)
                        st.dataframe(combined_motifs, use_container_width=True, hide_index=True)
                        st.download_button(
                            label="Download All Detected Motifs (CSV)",
                            data=combined_motifs.to_csv(index=False).encode("utf-8"),
                            file_name="rloop_g4_batch_motifs.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("No canonical G4 motifs found across the selected genes.")
    else:
        st.info("Enter comma-separated human gene symbols in the sidebar and click **Run RLoop-G4 Analysis**.")
