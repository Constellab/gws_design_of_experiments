import streamlit as st
import pandas as pd
import os
import plotly.express as px
import seaborn as sns
import numpy as np
from gws_design_of_experiments.causal_effect.causal_effect_task import CausalEffect
from gws_design_of_experiments.causal_effect.causal_effect_dashboard._causal_effect_dashboard.causal_effect_state import CausalEffectState



def render_single_effect_page():

    # --- BARPLOT ---

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
