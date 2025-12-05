#!/usr/bin/env python3
"""
Streamlit dashboard to explore Meta-AC outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from agents import BayesianAgent

STATS_PATH = "meta_ac_stats.csv"
PREDICTIONS_PATH = "meta_ac_predictions.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    stats_df = pd.read_csv(STATS_PATH)
    preds_df = pd.read_csv(PREDICTIONS_PATH)
    if "paper_id" not in preds_df.columns:
        preds_df = preds_df.copy()
        preds_df["paper_id"] = stats_df.loc[: len(preds_df) - 1, "paper_id"].values
    merged = stats_df.merge(preds_df, on="paper_id", how="inner")

    bayes = BayesianAgent()
    merged["reliability"] = merged.apply(bayes.get_reliability_score, axis=1)
    merged["raw_score_norm"] = merged["raw_avg"] / 10.0
    if "sentiment_score" not in merged.columns:
        merged["sentiment_score"] = merged["final_prob"]
    merged["sentiment_score"] = np.clip(merged["sentiment_score"].astype(float), 0.0, 1.0)
    return merged


def render_radar(row: pd.Series) -> None:
    categories = ["Raw Score", "Sentiment", "Reliability"]
    values = [
        row["raw_score_norm"],
        row["sentiment_score"],
        row["reliability"],
    ]
    radar_df = pd.DataFrame({"Metric": categories, "Value": values})
    fig = px.line_polar(
        radar_df,
        r="Value",
        theta="Metric",
        line_close=True,
        range_r=[0, 1],
        title="Meta-AC Signal Balance",
    )
    fig.update_traces(fill="toself")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Meta-AC Dashboard", layout="wide")
    st.title("Meta-AC Paper Explorer")

    df = load_data()
    title_to_row = {row["title"]: row for _, row in df.iterrows()}
    selection = st.sidebar.selectbox(
        "Select a paper", options=list(title_to_row.keys())
    )
    row = title_to_row[selection]

    st.header(selection)
    st.metric("Meta-AC Probability", f"{row['final_prob']:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Raw Metrics**")
        st.write(f"- Raw Avg Rating: {row['raw_avg']}")
        st.write(f"- Variance: {row['rating_variance']}")
        st.write(f"- Reliability: {row['reliability']:.2f}")
    with col2:
        render_radar(row)


if __name__ == "__main__":
    main()
