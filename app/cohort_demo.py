"""Governed cohort service, public demo (research Phase G6).

Run locally: ``streamlit run app/cohort_demo.py`` (needs ``pip install -e ".[cohort,demo]"``).
The logic lives in ``cdm.demo``; this file is the page. Counts only, no images, concept IDs
and ISIC source labels only (no SNOMED text). Deployed on Streamlit Community Cloud:
https://cdm-cohort-demo.streamlit.app/
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cdm.cohort import Refused, load
from cdm.demo import (
    ATTRIBUTION,
    DemoSession,
    concepts_in,
    custom_definition,
    diagnosis_options,
    fetch_tables,
)
from cdm.demo import library as phenotype_library

st.set_page_config(page_title="Governed cohort counts", page_icon=":bar_chart:", layout="wide")


@st.cache_data(show_spinner="Fetching public ISIC metadata and building the OMOP tables...")
def tables() -> dict[str, pd.DataFrame]:
    return fetch_tables()


if "session" not in st.session_state:
    st.session_state.session = DemoSession(tables())
session: DemoSession = st.session_state.session

st.title("Governed cohort counts on OMOP")
st.caption(
    "A portfolio demo on public research images, not a clinical tool. " + ATTRIBUTION + " "
    "Concepts are shown as OMOP concept IDs; follow the Athena link for a name."
)
left, right = st.columns([1, 1])

with left:
    st.subheader("1. Pick a cohort")
    mode = st.radio("Cohort", ["From the phenotype library", "Build your own"], horizontal=True)
    if mode == "From the phenotype library":
        paths = {p.parent.name: p for p in phenotype_library()}
        name = st.selectbox("Phenotype", sorted(paths))
        path = paths[name]
    else:
        options = diagnosis_options()
        labels = st.multiselect(
            "Diagnosis (ISIC label, with its OMOP concept ID)",
            sorted(options),
            default=["Melanoma, NOS", "Melanoma Invasive", "Melanoma in situ"],
            format_func=lambda x: f"{x} ({options[x]})",
        )
        low, high = st.slider("Age at diagnosis", 0, 100, (0, 100))
        histo = st.checkbox("Only people with a histopathology procedure (concept 43531286)")
        breakdown = st.multiselect("Break down by", ["gender_concept_id", "age_band"])
        if not labels:
            st.stop()
        definition = custom_definition(
            labels, low if low > 0 else None, high if high < 100 else None, histo, breakdown
        )
        path = session.write(definition)
    pheno = load(path)
    st.code(path.read_text(encoding="utf-8"), language="yaml")
    st.write(f"Definition hash `{pheno.sha256[:12]}`, status: **{session.status(path)}**")
    with st.expander("Concepts in this definition"):
        st.dataframe(
            concepts_in(pheno.definition),
            hide_index=True,
            column_config={"name (Athena)": st.column_config.LinkColumn()},
        )

with right:
    st.subheader("2. Approve, then run")
    st.write(
        "A definition runs only after a reviewer approves its exact hash. Here you are the "
        "reviewer, in a scratch approvals file that lasts for this session only."
    )
    run, approve = st.columns(2)
    if approve.button("Approve this exact definition"):
        session.approve(path)
        st.rerun()
    if run.button("Run"):
        try:
            result = session.run(path)
        except Refused as refusal:
            st.error(f"Refused: {refusal}")
        else:
            st.write("**Attrition** (people left after each step)")
            st.dataframe(pd.DataFrame(result.attrition), hide_index=True)
            if result.breakdown:
                st.write("**Breakdown**")
                st.dataframe(pd.DataFrame(result.breakdown), hide_index=True)
            st.info(result.note)
    st.subheader("3. Audit log")
    st.write(
        "Every refusal, decision and run, each entry chained to the one before by hash. In "
        "this demo each visitor has their own log, so the check against earlier results "
        "covers this session only."
    )
    st.dataframe(session.audit(), hide_index=True)
