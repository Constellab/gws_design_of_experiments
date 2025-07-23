import streamlit as st
import pandas as pd
import os
import plotly.express as px
import seaborn as sns
import numpy as np
from gws_design_of_experiments.causal_effect.causal_effect_task import CausalEffect
from gws_design_of_experiments.causal_effect.causal_effect_dashboard._causal_effect_dashboard.causal_effect_state import CausalEffectState


def render_multiple_effect_page():

    # --- Tabs d'affichage ---
    tab_clustermap, tab_barplot= st.tabs(["🧩 Clustermap", "📊 Barplot"])

    # --- HEATMAP ---
    with tab_clustermap:
        # TODO : Add a warning if target is only one -> can't do the plot
        pivot = CausalEffectState.get_df_filtered().pivot(index=f"{CausalEffect.TARGET_NAME}_Combo", columns=CausalEffect.TREATMENT_NAME, values=CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME).fillna(0)
        fig = sns.clustermap(pivot, cmap="coolwarm", center=0, figsize=(14, 12))
        st.pyplot(fig.figure)


    # --- BARPLOT ---
    with tab_barplot:
        fig = px.bar(
            CausalEffectState.get_df_filtered(),
            x=CausalEffect.TREATMENT_NAME,
            y=CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME,
            color=f"{CausalEffect.TARGET_NAME}_Combo",
            barmode="group",
            title="CATE by treatment and target",
            labels={CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME: "Effect (log scale)"},
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

