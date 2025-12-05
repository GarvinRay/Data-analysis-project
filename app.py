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
from meta_ac.config import PREDICTIONS_PATH, STATS_PATH


@st.cache_data
def load_data() -> pd.DataFrame:
    stats_path = STATS_PATH if STATS_PATH.exists() else Path(STATS_PATH.name)
    preds_path = PREDICTIONS_PATH if PREDICTIONS_PATH.exists() else Path(PREDICTIONS_PATH.name)

    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    stats_df = pd.read_csv(stats_path)
    preds_df = pd.read_csv(preds_path)

    # Normalize column names
    stats_df = stats_df.rename(columns={"avg_rating": "raw_avg", "rating_variance": "variance"})
    preds_df = preds_df.rename(columns={"probability": "final_prob"})

    # Attach title if missing
    if "title" not in preds_df.columns and "paper_id" in stats_df.columns:
        preds_df = preds_df.merge(stats_df[["paper_id", "decision"]], on="paper_id", how="left")
        preds_df["title"] = preds_df["paper_id"]

    if "paper_id" not in preds_df.columns and "paper_id" in stats_df.columns:
        preds_df = preds_df.copy()
        preds_df["paper_id"] = stats_df.loc[: len(preds_df) - 1, "paper_id"].values

    merged = stats_df.merge(preds_df, on="paper_id", how="inner")

    bayes = BayesianAgent()
    merged["reliability"] = merged.apply(bayes.get_reliability_score, axis=1)
    merged["raw_score_norm"] = merged["raw_avg"] / 10.0
    if "sentiment_score" not in merged.columns:
        merged["sentiment_score"] = merged.get("final_prob", 0.0)
    merged["sentiment_score"] = np.clip(merged["sentiment_score"].astype(float), 0.0, 1.0)

    # Backfill title if missing
    if "title" not in merged.columns:
        merged["title"] = merged["paper_id"]
    if "rating_variance" not in merged.columns and "variance" in merged.columns:
        merged["rating_variance"] = merged["variance"]
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

    try:
        df = load_data()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return
    if "title" not in df.columns:
        # Fallback: use paper_id as title if missing
        df["title"] = df["paper_id"]

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
        st.write(f"- Variance: {row.get('rating_variance', row.get('variance', 'N/A'))}")
        st.write(f"- Reliability: {row['reliability']:.2f}")
    with col2:
        render_radar(row)


if __name__ == "__main__":
    main()
