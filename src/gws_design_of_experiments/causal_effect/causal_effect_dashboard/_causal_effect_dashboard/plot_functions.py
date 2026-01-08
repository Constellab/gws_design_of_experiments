import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from gws_design_of_experiments.causal_effect.causal_effect_dashboard._causal_effect_dashboard.causal_effect_state import (
    CausalEffectState,
)
from gws_design_of_experiments.causal_effect.causal_effect_task import CausalEffect


def display_barplot(
    df_filtered: pd.DataFrame, thresholds: bool, treatment_order: list | None = None
):
    if thresholds:
        # Calculate min and max values for slider ranges
        all_values = df_filtered[CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME]
        negative_values = all_values[all_values < 0]
        positive_values = all_values[all_values >= 0]

        # Add sliders for threshold selection
        col1, col2 = st.columns(2)
        with col1:
            if len(negative_values) > 0:
                min_negative = negative_values.min()
                max_negative = negative_values.max()
                step_negative = (min_negative - max_negative) / 100
                negative_threshold = st.slider(
                    "Negative effects threshold",
                    min_value=min_negative,
                    max_value=max_negative,
                    value=(min_negative, max_negative),
                    step=step_negative,
                    key="negative_threshold",
                )
            else:
                negative_threshold = (0.0, 0.0)
                st.write("No negative values found")

        with col2:
            if len(positive_values) > 0:
                min_positive = positive_values.min()
                max_positive = positive_values.max()
                step_positive = (max_positive - min_positive) / 100
                positive_threshold = st.slider(
                    "Positive effects threshold",
                    min_value=min_positive,
                    max_value=max_positive,
                    value=(min_positive, max_positive),
                    step=step_positive,
                    key="positive_threshold",
                )
            else:
                positive_threshold = (0.0, 0.0)
                st.write("No positive values found")

        # Filter data based on thresholds
        df_filtered = df_filtered[
            (
                (df_filtered[CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME] >= negative_threshold[0])
                & (df_filtered[CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME] <= negative_threshold[1])
            )
            | (
                (df_filtered[CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME] >= positive_threshold[0])
                & (df_filtered[CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME] <= positive_threshold[1])
            )
        ]

    # Apply display names to treatment order
    if treatment_order:
        treatment_order_display = [
            CausalEffectState.get_display_name(name) for name in treatment_order
        ]
    else:
        # We ranked the values manually
        # Calculate mean absolute effect for each treatment to determine order
        treatment_order = (
            df_filtered.groupby(CausalEffect.TREATMENT_NAME)[
                CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME
            ]
            .apply(lambda x: np.mean(np.abs(x)))
            .sort_values(ascending=False)
            .index.tolist()
        )
        treatment_order_display = [
            CausalEffectState.get_display_name(name) for name in treatment_order
        ]

    # Create a copy of the dataframe with display names
    df_display = df_filtered.copy()
    df_display[CausalEffect.TREATMENT_NAME + "_Display"] = df_display[
        CausalEffect.TREATMENT_NAME
    ].apply(CausalEffectState.get_display_name)
    df_display[f"{CausalEffect.TARGET_NAME}_Combo_Display"] = df_display[
        f"{CausalEffect.TARGET_NAME}_Combo"
    ].apply(
        lambda x: CausalEffectState.get_display_name(x.split(" [")[0])
        + " ["
        + CausalEffectState.get_display_name(x.split(" [")[1].rstrip("]"))
        + "]"
    )

    fig = px.bar(
        df_display,
        x=CausalEffect.TREATMENT_NAME + "_Display",
        y=CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME,
        color=f"{CausalEffect.TARGET_NAME}_Combo_Display",
        barmode="group",
        title="CATE by treatment and target",
        labels={
            CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME: "Effect (log scale)",
            CausalEffect.TREATMENT_NAME + "_Display": "Treatment",
        },
        category_orders={CausalEffect.TREATMENT_NAME + "_Display": treatment_order_display},
    )

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
