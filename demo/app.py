#!/usr/bin/env python3
"""KeloidBench interactive demonstration application."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEMO_DIR = Path(__file__).resolve().parent
ASSET_DIR = DEMO_DIR / "assets"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from demo_utils import (  # noqa: E402
    CURATED_EXAMPLES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    NON_TRANSFERABLE_ACCESSIONS,
    PRIMARY_ENDPOINT,
    PROGRAM_LABELS,
    SOURCE_RECORDS,
    TRANSFERABLE_ACCESSIONS,
    build_evidence_packet,
    expression_qc,
    generate_grounded_summary,
    load_feature_views,
    load_loso_evidence,
    load_product_bundle,
    load_profile_manifest,
    normalize_expression_table,
    primary_example_ids,
    program_score_table,
    raw_expression_for_sample,
    score_sample,
)
from product_features import build_product_feature_views  # noqa: E402


st.set_page_config(
    page_title="KeloidBench · Program-grounded transcriptomic triage",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: transparent; border: 0; border-left: 3px solid #4e8d7b;
      padding: .25rem .65rem; border-radius: 0;}
    .kb-hero {min-height:430px; padding:2rem; border-radius:1rem; color:white; display:flex;
      align-items:flex-end; background-position:center; background-size:cover; margin-bottom:.35rem;
      box-shadow:0 18px 45px rgba(16,29,49,.18);}
    .kb-hero-copy {max-width:590px; padding:1.2rem 1.3rem; border-radius:.8rem;
      background:rgba(10,25,42,.82); backdrop-filter:blur(7px);}
    .kb-hero h1 {margin: 0 0 .3rem 0; font-size: 2.8rem; letter-spacing:-.04em;}
    .kb-hero p {margin: 0; color: #e7f3f2; font-size: 1.06rem;}
    .kb-credit {color:#68717f; font-size:.72rem; line-height:1.35; margin:.2rem 0 .9rem;}
    .kb-credit a {color:#456c71;}
    .kb-media {padding:.65rem; border-radius:.85rem; background:#f0f6f5; height:100%;}
    .kb-media img {display:block; width:100%; height:300px; object-fit:cover; border-radius:.6rem;}
    .kb-media.anatomy img {object-fit:contain; background:white;}
    .kb-media h4 {margin:.7rem .2rem .25rem; color:#183d3a;}
    .kb-media p {margin:.2rem; color:#536171; font-size:.88rem;}
    .kb-eyebrow {font-size:.76rem; letter-spacing:.12em; text-transform:uppercase; color:#b9e3d8; font-weight:700;}
    .kb-chip {display:inline-block; padding:.18rem .55rem; margin:.15rem .25rem .15rem 0;
      border-radius:1rem; background:#eaf3f1; color:#20584f; font-size:.82rem;}
    .kb-warn {border-left: 4px solid #d18324; padding: .7rem 1rem; background: #fff7e8; border-radius: .4rem;}
    .kb-good {border-left: 4px solid #36866f; padding: .7rem 1rem; background: #eef8f4; border-radius: .4rem;}
    .kb-result {padding:1rem 1.15rem; border:1px solid #dce5e7; border-radius:.8rem; background:#fbfcfd;}
    .kb-result strong {font-size:1.15rem;}
    .kb-story {min-height:7.8rem; padding:.8rem .85rem; border-radius:.7rem;
      background:linear-gradient(145deg,#edf6f3,#f7faf9); border-top:3px solid #4e8d7b;}
    .kb-story b {color:#183d3a;}
    .kb-step {font-size:.84rem; color:#536171; padding:.55rem .7rem; border-radius:.5rem; background:#f5f7f9; text-align:center;}
    .kb-muted {color:#5e6978; font-size:.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_bundle():
    return load_product_bundle(PRIMARY_ENDPOINT)


@st.cache_data
def cached_manifest():
    return load_profile_manifest()


@st.cache_data
def cached_features():
    return load_feature_views()


@st.cache_data
def cached_raw_sample(sample_id: str):
    return raw_expression_for_sample(sample_id, cached_manifest())


@st.cache_data
def cached_loso():
    return load_loso_evidence()


@st.cache_data
def asset_data_url(filename: str) -> str:
    """Embed a versioned local image so styled visual cards work in cloud builds."""
    path = ASSET_DIR / filename
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


@st.cache_data
def cached_primary_sample_map():
    """PCA projection of real program features used by the public demo."""
    metadata = cached_manifest().set_index("sample_id")
    fused = cached_features()["fused_multiview"].copy()
    eligible = metadata[
        metadata[PRIMARY_ENDPOINT].isin(["keloid", "non_keloid"])
        & metadata["accession"].isin(TRANSFERABLE_ACCESSIONS | NON_TRANSFERABLE_ACCESSIONS)
    ]
    ids = [sample for sample in eligible.index if sample in fused.index]
    matrix = fused.loc[ids].astype(float)
    coordinates = PCA(n_components=2, random_state=13).fit_transform(
        StandardScaler().fit_transform(matrix)
    )
    return pd.DataFrame(
        {
            "sample_id": ids,
            "PC1": coordinates[:, 0],
            "PC2": coordinates[:, 1],
            "reference_label": eligible.loc[ids, PRIMARY_ENDPOINT].to_numpy(),
            "accession": eligible.loc[ids, "accession"].astype(str).to_numpy(),
            "transfer_group": np.where(
                eligible.loc[ids, "accession"].isin(NON_TRANSFERABLE_ACCESSIONS),
                "non-transferable platform",
                "transferable subset",
            ),
        }
    )


@st.cache_data
def cached_program_references():
    metadata = cached_manifest().set_index("sample_id")
    fused = cached_features()["fused_multiview"].copy()
    eligible = metadata[
        metadata[PRIMARY_ENDPOINT].isin(["keloid", "non_keloid"])
        & metadata["accession"].isin(TRANSFERABLE_ACCESSIONS)
    ]
    ids = [sample for sample in eligible.index if sample in fused.index]
    table = fused.loc[ids].copy()
    table["reference_label"] = eligible.loc[ids, PRIMARY_ENDPOINT].to_numpy()
    return table.groupby("reference_label").median(numeric_only=True)


def read_uploaded_csv(uploaded, transpose: bool) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(uploaded.getvalue()), index_col=0)
    return normalize_expression_table(raw, transpose=transpose)


def display_status(decision: str) -> None:
    if decision == "abstain":
        st.warning("ABSTAIN · confidence below the operational threshold")
    elif decision == "keloid":
        st.error("KELOID-LIKE · research triage output")
    else:
        st.success("UNAFFECTED-LIKE · research triage output")


def activate_example(slug: str) -> None:
    """Switch to a curated profile before Streamlit reruns the script."""
    st.session_state["source_mode"] = "Public example"
    st.session_state["profile_selector"] = CURATED_EXAMPLES[slug]["sample_id"]
    st.session_state["active_example_slug"] = slug
    st.query_params["example"] = slug


def example_slug_for_sample(sample_id: str) -> str | None:
    for slug, item in CURATED_EXAMPLES.items():
        if item["sample_id"] == sample_id:
            return slug
    return None


def sync_profile_route() -> None:
    slug = example_slug_for_sample(str(st.session_state.get("profile_selector", "")))
    st.session_state["active_example_slug"] = slug
    if slug:
        st.query_params["example"] = slug
    elif "example" in st.query_params:
        del st.query_params["example"]


hero_image = asset_data_url("keloidbench-hero-generated.png")
st.markdown(
    f"""
    <div class="kb-hero" role="img" aria-label="Illustrated keloid skin cross-section, fibroblasts, and program-level data"
      style="background-image:linear-gradient(90deg,rgba(7,20,36,.22),rgba(7,20,36,.02)),url('{hero_image}')">
      <div class="kb-hero-copy">
        <div class="kb-eyebrow">Interactive research demonstration</div>
        <h1>KeloidBench</h1>
        <p>From expression profile to biological programs, confidence-aware prediction, and an auditable explanation.</p>
      </div>
    </div>
    <div class="kb-credit">Original KeloidBench concept illustration generated for this project. It is explanatory artwork—not a clinical specimen or model input.</div>
    """,
    unsafe_allow_html=True,
)

product_manifest, bundle = cached_bundle()
manifest = cached_manifest()
feature_views = cached_features()

requested_slug = str(
    st.query_params.get("example", os.environ.get("KELOIDBENCH_DEMO_EXAMPLE", "confident_keloid"))
)
if requested_slug not in CURATED_EXAMPLES:
    requested_slug = "confident_keloid"
if "source_mode" not in st.session_state:
    st.session_state["source_mode"] = "Public example"
if "profile_selector" not in st.session_state:
    st.session_state["profile_selector"] = CURATED_EXAMPLES[requested_slug]["sample_id"]
    st.session_state["active_example_slug"] = requested_slug

headline_1, headline_2, headline_3, headline_4 = st.columns(4)
headline_1.metric("Transferable studies", "7", help="Studies in the documented transferable subset")
headline_2.metric("Study-held-out macro F1", "0.918", help="Equal-study donor-primary mean")
headline_3.metric("Held-out accuracy", "77/92", help="Pooled sample accuracy; not the headline metric")
headline_4.metric("Selective coverage", "80.4%", help="At the fixed 0.55 confidence rule")
st.caption(
    "Headline metrics describe the later nested LOSO ensemble. The live scorer below is a separately versioned Stage-B artifact."
)

st.markdown("### What KeloidBench is modeling")
st.caption("Clinical appearance provides context; the model itself operates on de-identified public transcriptomic profiles, not photographs.")
clinical_image = asset_data_url("keloid-ear-clinical-cc-by-4.jpg")
anatomy_image = asset_data_url("skin-anatomy-nci-public-domain.jpg")
clinical_column, anatomy_column = st.columns(2)
with clinical_column:
    st.markdown(
        f"""
        <div class="kb-media">
          <img src="{clinical_image}" alt="Clinical photograph of an ear-lobe keloid">
          <h4>Clinical phenotype</h4>
          <p>A visible example of excessive scar growth. The photograph is included only to ground the biological problem.</p>
        </div>
        <div class="kb-credit">Photo: Bobjgalindo, <a href="https://commons.wikimedia.org/wiki/File:A_keloid_scar_on_ear_lobe.jpg" target="_blank">Wikimedia Commons</a>, <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">CC BY 4.0</a>. Displayed without image edits.</div>
        """,
        unsafe_allow_html=True,
    )
with anatomy_column:
    st.markdown(
        f"""
        <div class="kb-media anatomy">
          <img src="{anatomy_image}" alt="National Cancer Institute illustration of human skin layers">
          <h4>Tissue and fibroblast programs</h4>
          <p>KeloidBench summarizes expression into interpretable wound-healing, matrix, inflammation, and mechanotransduction programs.</p>
        </div>
        <div class="kb-credit">Illustration: Don Bliss / National Cancer Institute, <a href="https://commons.wikimedia.org/wiki/File:Anatomy_The_Skin_-_NCI_Visuals_Online.jpg" target="_blank">Wikimedia Commons</a>, U.S. government <a href="https://creativecommons.org/publicdomain/mark/1.0/" target="_blank">public domain</a>.</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Choose a demo story")
st.caption("Each example is public and selected to demonstrate a different system behavior.")
example_columns = st.columns(4)
for column, (slug, item) in zip(example_columns, CURATED_EXAMPLES.items(), strict=True):
    with column:
        st.markdown(
            f'<div class="kb-story"><b>{item["icon"]} {item["title"]}</b>'
            f'<p class="kb-muted">{item["description"]}</p></div>',
            unsafe_allow_html=True,
        )
        st.button(
            f"Open {item['short_title']}",
            key=f"open_{slug}",
            on_click=activate_example,
            args=(slug,),
            width="stretch",
            type=(
                "primary"
                if slug == st.session_state.get("active_example_slug", requested_slug)
                else "secondary"
            ),
        )

with st.sidebar:
    st.header("Run an analysis")
    st.caption("Choose a public example or score your own expression matrix.")
    source_mode = st.radio(
        "Input source",
        ["Public example", "Upload CSV"],
        key="source_mode",
        horizontal=True,
    )
    confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    uploaded = None
    if source_mode == "Public example":
        examples = primary_example_ids(manifest)
        if st.session_state.get("profile_selector") not in examples:
            st.session_state["profile_selector"] = CURATED_EXAMPLES["confident_keloid"]["sample_id"]
        manifest_by_id = manifest.set_index("sample_id")

        def profile_label(value: str) -> str:
            slug = example_slug_for_sample(value)
            accession_value = str(manifest_by_id.loc[value, "accession"])
            if slug:
                return f"★ {CURATED_EXAMPLES[slug]['short_title']} · {accession_value}"
            return f"{value} · {accession_value}"

        sample_id = st.selectbox(
            "Selected public profile",
            examples,
            key="profile_selector",
            format_func=profile_label,
            help="Curated profiles are marked with a star; all eligible public profiles remain available.",
            on_change=sync_profile_route,
        )
        row = manifest.set_index("sample_id").loc[sample_id]
        accession = str(row["accession"])
        known_label = str(row[PRIMARY_ENDPOINT])
        st.caption(f"Accession: {accession} · reference label: {known_label}")
        expression = cached_raw_sample(sample_id)
        full_feature_row = feature_views["fused_multiview"].loc[[sample_id]]
        live_feature_row = feature_views[bundle["meta"]["feature_set"]].loc[[sample_id]]
        score = score_sample(
            live_feature_row,
            bundle,
            is_expression=False,
            confidence_threshold=confidence_threshold,
        )
    else:
        uploaded = st.file_uploader("Expression CSV", type=["csv"])
        transpose = st.toggle("Genes are rows", value=False, help="Turn on for genes × samples files.")
        known_label = "unknown"
        accession = st.text_input("Accession or cohort name", value="uploaded")
        if uploaded is None:
            st.info("Upload a numeric expression matrix to begin, or choose a public example above.")
            st.stop()
        expression = read_uploaded_csv(uploaded, transpose)
        sample_id = st.selectbox("Sample to inspect", expression.index.astype(str).tolist())
        expression = expression.loc[[sample_id]]
        views = build_product_feature_views(expression)
        full_feature_row = views["fused_multiview"]
        score = score_sample(
            expression,
            bundle,
            is_expression=True,
            confidence_threshold=confidence_threshold,
        )

    with st.expander("Decision settings", expanded=False):
        confidence_threshold = st.slider(
            "Abstention confidence threshold",
            min_value=0.50,
            max_value=0.90,
            value=DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.01,
            help="A profile is withheld when max(P(keloid), 1−P(keloid)) is below this value.",
        )
        st.caption("The workshop result uses the fixed 0.55 rule.")
    # Reapply an adjusted threshold without rerunning the deterministic model.
    score["confidence_threshold"] = float(confidence_threshold)
    score["decision"] = (
        score["pred_label"] if score["confidence"] >= confidence_threshold else "abstain"
    )

    with st.expander("Model and artifact details", expanded=False):
        st.code(
            f"KeloidBench / {score['artifact_stage']}\n"
            f"{score['model_name']} · {score['feature_set']}",
            language=None,
        )
    st.warning("Research prototype—not a clinical diagnostic.", icon="⚠️")

qc = expression_qc(expression, accession=accession)
programs = program_score_table(full_feature_row)
packet = build_evidence_packet(
    score,
    qc,
    programs,
    accession=accession,
    sample_label=str(sample_id),
)

active_slug = example_slug_for_sample(str(sample_id))
st.divider()
overview_left, overview_right = st.columns([1.35, 1])
with overview_left:
    st.markdown("### Result at a glance")
    display_status(score["decision"])
    if active_slug:
        active = CURATED_EXAMPLES[active_slug]
        st.markdown(
            f"**Why this example matters:** {active['description']}  \n"
            f"*Presenter cue: {active['presenter_note']}*"
        )
    else:
        st.caption(f"Profile {sample_id} from {accession}")
with overview_right:
    result_1, result_2, result_3 = st.columns(3)
    result_1.metric("P(keloid)", f"{score['prob_keloid']:.3f}")
    result_2.metric("Confidence", f"{score['confidence']:.3f}")
    result_3.metric("Transfer check", qc["transfer_status"].replace("-", " ").title())
    if score["decision"] == "abstain":
        st.caption("No class is returned because confidence does not meet the selected rule.")
    elif qc["transfer_status"] == "non-transferable":
        st.caption("The prediction is shown, but the platform warning should take precedence.")
    else:
        st.caption("Continue through the guided tabs to inspect how this result was produced.")

step_columns = st.columns(5)
for column, label in zip(
    step_columns,
    ["1  Validate input", "2  Inspect biology", "3  Review decision", "4  Read explanation", "5  Audit evidence"],
    strict=True,
):
    column.markdown(f'<div class="kb-step">{label}</div>', unsafe_allow_html=True)

tab_qc, tab_programs, tab_prediction, tab_explanation, tab_loso = st.tabs(
    [
        "1 · Start here",
        "2 · Biology",
        "3 · Decision",
        "4 · Copilot",
        "5 · Evidence",
    ]
)

with tab_qc:
    st.subheader("Is this profile suitable to score?")
    st.caption("KeloidBench checks gene coverage and known platform behavior before interpreting a prediction.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expression genes", f"{qc['n_genes']:,}")
    c2.metric("Pinned genes", f"{qc['pinned_genes_present']}/{qc['pinned_genes_expected']}")
    c3.metric("Usable programs", f"{qc['usable_programs']}/{qc['total_programs']}")
    c4.metric("Transferability", qc["transfer_status"].replace("-", " ").title())
    status_class = "kb-good" if qc["transfer_status"] == "represented" else "kb-warn"
    st.markdown(
        f'<div class="{status_class}"><b>{qc["transfer_status"].upper()}</b><br>{qc["transfer_note"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Dataset provenance")
    provenance_left, provenance_right = st.columns([1, 1.25])
    with provenance_left:
        source = SOURCE_RECORDS.get(accession)
        sample_metadata = manifest.set_index("sample_id").loc[sample_id] if sample_id in set(manifest["sample_id"]) else None
        if source:
            st.markdown(f"**{source['study_title']}**")
            st.write(source["study_design"])
            st.caption(source["paper_citation"])
            link_left, link_right = st.columns(2)
            link_left.link_button("Open original paper ↗", source["paper_url"], width="stretch")
            link_right.link_button(f"Open {accession} on GEO ↗", source["geo_url"], width="stretch")
        else:
            st.info("No curated paper record is attached to this uploaded cohort.")
        if sample_metadata is not None:
            provenance_rows = pd.DataFrame(
                {
                    "Field": ["Profile", "Reference label", "Modality", "Platform", "Donor/group"],
                    "Value": [
                        str(sample_id),
                        known_label,
                        str(sample_metadata.get("modality", "unknown")),
                        str(sample_metadata.get("platform_id", "unknown")),
                        str(sample_metadata.get("patient_id", "unknown")),
                    ],
                }
            )
            st.dataframe(provenance_rows, width="stretch", hide_index=True)
    with provenance_right:
        sample_map = cached_primary_sample_map().copy()
        sample_map["Selected"] = np.where(sample_map["sample_id"] == sample_id, "selected profile", "public profile")
        base_map = (
            alt.Chart(sample_map)
            .mark_circle(size=72, opacity=0.62)
            .encode(
                x=alt.X("PC1:Q", title="Program-space PC1"),
                y=alt.Y("PC2:Q", title="Program-space PC2"),
                color=alt.Color(
                    "reference_label:N",
                    title="Reference",
                    scale=alt.Scale(
                        domain=["keloid", "non_keloid"],
                        range=["#b35c44", "#3d806e"],
                    ),
                ),
                shape=alt.Shape("transfer_group:N", title="Cohort status"),
                tooltip=["sample_id:N", "accession:N", "reference_label:N", "transfer_group:N"],
            )
        )
        selected_map = (
            alt.Chart(sample_map[sample_map["sample_id"] == sample_id])
            .mark_point(size=330, filled=False, stroke="#111827", strokeWidth=3)
            .encode(x="PC1:Q", y="PC2:Q")
        )
        st.altair_chart((base_map + selected_map).properties(height=330), width="stretch")
        st.caption("Real fused program scores projected into two dimensions; the selected profile is ringed.")
    st.markdown("#### Program coverage")
    coverage_display = qc["coverage_table"][[
        "display_name", "present_genes", "expected_genes", "coverage", "minimum_required", "usable", "missing"
    ]].rename(
        columns={
            "display_name": "Program",
            "present_genes": "Present",
            "expected_genes": "Expected",
            "coverage": "Coverage",
            "minimum_required": "Minimum",
            "usable": "Usable",
            "missing": "Missing genes",
        }
    )
    with st.expander("Show the complete coverage table", expanded=False):
        st.dataframe(
            coverage_display.style.format({"Coverage": "{:.0%}"}),
            width="stretch",
            hide_index=True,
        )

with tab_programs:
    st.subheader("Which biological programs are active?")
    st.caption(
        "Scores are within-sample percentile-rank contrasts: mean positive-gene rank minus mean negative-gene rank."
    )
    reference_programs = cached_program_references()
    selected_program_names = (
        programs.assign(magnitude=programs["score"].abs())
        .nlargest(8, "magnitude")["program"]
        .tolist()
    )
    fingerprint = pd.DataFrame(
        [
            full_feature_row.iloc[0].reindex(selected_program_names),
            reference_programs.loc["keloid"].reindex(selected_program_names),
            reference_programs.loc["non_keloid"].reindex(selected_program_names),
        ],
        index=["Selected profile", "Keloid median", "Unaffected median"],
    )
    fingerprint.index.name = "Profile"
    fingerprint_long = fingerprint.reset_index().melt(
        id_vars="Profile", var_name="program", value_name="score"
    )
    fingerprint_long["Program"] = fingerprint_long["program"].map(PROGRAM_LABELS)
    fingerprint_long["Score label"] = fingerprint_long["score"].map(lambda value: f"{value:+.2f}")
    score_limit = max(0.1, float(fingerprint_long["score"].abs().max()))
    fingerprint_chart = (
        alt.Chart(fingerprint_long)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X("Program:N", sort=None, axis=alt.Axis(labelAngle=-32), title=None),
            y=alt.Y(
                "Profile:N",
                sort=["Selected profile", "Keloid median", "Unaffected median"],
                title=None,
            ),
            color=alt.Color(
                "score:Q",
                title="Program score",
                scale=alt.Scale(domain=[-score_limit, score_limit], range=["#3d6d8f", "#f5f5f1", "#b35c44"]),
            ),
            tooltip=["Profile:N", "Program:N", alt.Tooltip("score:Q", format="+.3f")],
        )
    )
    fingerprint_text = (
        alt.Chart(fingerprint_long)
        .mark_text(fontSize=11)
        .encode(
            x=alt.X("Program:N", sort=None),
            y=alt.Y("Profile:N", sort=["Selected profile", "Keloid median", "Unaffected median"]),
            text="Score label:N",
            color=alt.condition("abs(datum.score) > 0.22", alt.value("white"), alt.value("#263238")),
        )
    )
    st.altair_chart((fingerprint_chart + fingerprint_text).properties(height=235), width="stretch")
    st.caption("Actual selected-profile scores compared with medians from the seven transferable studies.")
    strongest = programs.assign(magnitude=programs["score"].abs()).nlargest(3, "magnitude")
    insight_columns = st.columns(3)
    for column, (_, program) in zip(insight_columns, strongest.iterrows(), strict=True):
        column.metric(program["display_name"], f"{program['score']:+.3f}", program["direction"])
    st.caption("These are the three largest absolute program scores for the selected profile.")
    show = programs[["display_name", "score", "direction"]].rename(
        columns={"display_name": "Program", "score": "Score", "direction": "Direction"}
    )
    with st.expander("Show all 16 program scores", expanded=False):
        chart = programs.set_index("display_name")[["score"]].sort_values("score")
        st.bar_chart(chart, horizontal=True, color="#2f7667", height=480)
        st.dataframe(show.style.format({"Score": "{:+.3f}"}), width="stretch", hide_index=True)
        export_scores = programs[["program", "display_name", "score", "direction"]].copy()
        export_scores.insert(0, "sample_id", str(sample_id))
        export_scores.insert(1, "accession", str(accession))
        st.download_button(
            "Download this program-score profile",
            data=export_scores.to_csv(index=False),
            file_name=f"keloidbench_{sample_id}_program_scores.csv",
            mime="text/csv",
        )
    with st.expander("Inspect the versioned program definitions"):
        selected_program = st.selectbox("Program", programs["program"].tolist())
        from gene_modules import ALL_RANK_PROGRAM_SETS

        spec = ALL_RANK_PROGRAM_SETS[selected_program]
        left, right = st.columns(2)
        left.markdown("**Positive genes**")
        left.write(", ".join(spec.get("positive", [])))
        right.markdown("**Negative/reference genes**")
        right.write(", ".join(spec.get("negative", [])) or "None")

with tab_prediction:
    st.subheader("What decision does the model make?")
    display_status(score["decision"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(keloid)", f"{score['prob_keloid']:.3f}")
    c2.metric("Model confidence", f"{score['confidence']:.3f}")
    c3.metric("Class threshold", f"{score['decision_threshold']:.3f}")
    c4.metric("Abstention threshold", f"{score['confidence_threshold']:.2f}")
    probability_frame = pd.DataFrame(
        {
            "class": ["Keloid", "Unaffected"],
            "probability": [score["prob_keloid"], 1.0 - score["prob_keloid"]],
        }
    )
    probability_left, probability_right = st.columns([1, 1.3])
    with probability_left:
        probability_arc = (
            alt.Chart(probability_frame)
            .mark_arc(innerRadius=72, outerRadius=112, cornerRadius=5)
            .encode(
                theta=alt.Theta("probability:Q", stack=True),
                color=alt.Color(
                    "class:N",
                    scale=alt.Scale(domain=["Keloid", "Unaffected"], range=["#b35c44", "#3d806e"]),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                tooltip=["class:N", alt.Tooltip("probability:Q", format=".1%")],
            )
        )
        center_label = (
            alt.Chart(pd.DataFrame({"label": [f"{score['prob_keloid']:.1%}\nP(keloid)"]}))
            .mark_text(size=20, fontWeight="bold", lineBreak="\n")
            .encode(text="label:N")
        )
        st.altair_chart((probability_arc + center_label).properties(height=285), width="stretch")
    with probability_right:
        st.markdown("**How the two rules work**")
        st.markdown(
            f"1. The class boundary is **{score['decision_threshold']:.3f}**. "
            f"The live score is **{score['prob_keloid']:.3f}**.\n\n"
            f"2. The abstention rule requires confidence ≥ **{score['confidence_threshold']:.2f}**. "
            f"Observed confidence is **{score['confidence']:.3f}**."
        )
        if score["decision"] == "abstain":
            st.warning("The class score exists, but KeloidBench withholds it from the final decision.")
        elif qc["transfer_status"] == "non-transferable":
            st.warning("A class is returned, but the transfer warning marks this result as unreliable.")
        else:
            st.success(f"Final research-triage decision: {score['decision'].replace('_', ' ')}")
    st.markdown("#### Transparent feature contributions")
    contributions = pd.DataFrame(score["contributions"])
    if contributions.empty:
        st.info("This artifact does not expose linear feature contributions.")
    else:
        support_column, oppose_column = st.columns(2)
        strongest_support = contributions.sort_values("contribution", ascending=False).iloc[0]
        strongest_oppose = contributions.sort_values("contribution", ascending=True).iloc[0]
        support_column.markdown(
            f"**Strongest push toward keloid**  \n{strongest_support['display_name']} "
            f"({strongest_support['contribution']:+.3f})"
        )
        oppose_column.markdown(
            f"**Strongest push toward unaffected**  \n{strongest_oppose['display_name']} "
            f"({strongest_oppose['contribution']:+.3f})"
        )
        contribution_chart = contributions.set_index("display_name")[["contribution"]].sort_values("contribution")
        st.bar_chart(contribution_chart, horizontal=True, color="#b86846", height=300)
        st.caption("Positive values push the model toward keloid; negative values push it toward unaffected skin.")
    if known_label != "unknown":
        st.caption(f"Public reference label: **{known_label}**. It is displayed for demonstration only and is not used by the scorer.")

with tab_explanation:
    st.subheader("How does the copilot explain the result?")
    st.caption(
        "The language model receives this structured evidence after scoring. It cannot alter the probability, label, or abstention decision."
    )
    narrative_key = f"{sample_id}:{score['confidence_threshold']:.2f}:{score['prob_keloid']:.6f}"
    if st.session_state.get("narrative_key") != narrative_key or st.button("Generate grounded summary", type="primary"):
        try:
            narrative, narrative_mode = generate_grounded_summary(packet)
        except Exception as exc:  # noqa: BLE001
            narrative = f"Narrative endpoint failed: {exc}"
            narrative_mode = "error"
        st.session_state["narrative"] = narrative
        st.session_state["narrative_mode"] = narrative_mode
        st.session_state["narrative_key"] = narrative_key
    mode = st.session_state.get("narrative_mode", "deterministic_fallback")
    st.info(st.session_state.get("narrative", "Select Generate grounded summary."))
    if mode == "grounded_llm":
        st.success("Generated by the configured grounded language-model endpoint.")
    elif mode == "deterministic_fallback":
        st.caption(
            "Deterministic fallback active. Set KELOIDBENCH_LLM_API_URL and KELOIDBENCH_LLM_MODEL to enable the constrained generative layer."
        )
    with st.expander("View the exact evidence packet"):
        st.json(packet)
    st.download_button(
        "Download evidence JSON",
        data=json.dumps(packet, indent=2),
        file_name=f"keloidbench_{sample_id}_evidence.json",
        mime="application/json",
    )

with tab_loso:
    st.subheader("How does KeloidBench behave on unseen studies?")
    predictions, selected_folds, transferable_report, selective_report = cached_loso()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transferable studies", "7")
    c2.metric("Mean donor-primary macro F1", f"{transferable_report['mean_macro_f1']:.3f}")
    c3.metric("Pooled held-out accuracy", "77/92 · 0.837")
    fixed = selective_report["fixed_confidence_baselines"]["0.55"]
    c4.metric("Selective F1 @ 0.55", f"{fixed['mean_selective_macro_f1']:.3f}")
    st.markdown(
        '<div class="kb-warn"><b>Known non-transferable platform:</b> GSE173900 is shown explicitly. '
        'Including it reduces the eight-study score and exposes a severe failure fold.</div>',
        unsafe_allow_html=True,
    )
    fold_options = sorted(predictions["accession"].unique().tolist())
    default_fold = accession if accession in fold_options else "E-MTAB-4945"
    fold = st.selectbox(
        "Held-out accession",
        fold_options,
        index=fold_options.index(default_fold),
        key=f"fold_selector_{accession}",
    )
    fold_preds = predictions[predictions["accession"] == fold].copy()
    fold_row = selected_folds[selected_folds["accession"] == fold].iloc[0]
    f1, cov, acc = st.columns(3)
    f1.metric("Sample macro F1", f"{fold_row['macro_f1']:.3f}")
    cov.metric("Coverage @ conf 0.55", f"{(fold_preds['confidence'] >= 0.55).mean():.0%}")
    acc.metric("Sample accuracy", f"{fold_row['accuracy']:.3f}")
    fold_plot = fold_preds.sort_values("prob_keloid").reset_index(drop=True)
    fold_plot["Profile order"] = np.arange(1, len(fold_plot) + 1)
    abstention_band = (
        alt.Chart(pd.DataFrame({"lower": [0.45], "upper": [0.55]}))
        .mark_rect(color="#d99b45", opacity=0.13)
        .encode(y="lower:Q", y2="upper:Q")
    )
    class_rule = (
        alt.Chart(pd.DataFrame({"threshold": [float(fold_row["threshold"])]}))
        .mark_rule(color="#30343b", strokeDash=[6, 4])
        .encode(y="threshold:Q")
    )
    probability_points = (
        alt.Chart(fold_plot)
        .mark_circle(size=115, opacity=0.88)
        .encode(
            x=alt.X("Profile order:Q", axis=alt.Axis(tickMinStep=1), title="Held-out profiles (sorted)"),
            y=alt.Y("prob_keloid:Q", scale=alt.Scale(domain=[0, 1]), title="P(keloid)"),
            color=alt.Color(
                "true_label:N",
                title="Reference",
                scale=alt.Scale(domain=["keloid", "non_keloid"], range=["#b35c44", "#3d806e"]),
            ),
            shape=alt.Shape("selective_decision:N", title="Decision"),
            tooltip=[
                "sample_id:N",
                "true_label:N",
                "pred_label:N",
                "selective_decision:N",
                alt.Tooltip("prob_keloid:Q", format=".3f"),
                alt.Tooltip("confidence:Q", format=".3f"),
            ],
        )
    )
    st.altair_chart(
        (abstention_band + class_rule + probability_points).properties(height=350),
        width="stretch",
    )
    st.caption("Orange band: fixed confidence <0.55 abstention region. Dashed line: fold class threshold.")
    table = fold_preds[[
        "sample_id", "true_label", "prob_keloid", "pred_label", "confidence", "selective_decision", "correct"
    ]].rename(
        columns={
            "sample_id": "Sample",
            "true_label": "Reference",
            "prob_keloid": "P(keloid)",
            "pred_label": "Prediction",
            "confidence": "Confidence",
            "selective_decision": "At conf ≥0.55",
            "correct": "Correct",
        }
    )
    st.dataframe(
        table.style.format({"P(keloid)": "{:.3f}", "Confidence": "{:.3f}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "LOSO figures are stored outer-fold predictions from the later ensemble evaluation; they are not recomputed by the older live Stage-B artifact."
    )

st.divider()
st.markdown(
    "<span class='kb-chip'>Research use only</span>"
    "<span class='kb-chip'>Versioned programs</span>"
    "<span class='kb-chip'>Selective abstention</span>"
    "<span class='kb-chip'>Visible failure modes</span>",
    unsafe_allow_html=True,
)
