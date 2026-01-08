import seaborn as sns
import streamlit as st
from gws_design_of_experiments.causal_effect.causal_effect_dashboard._causal_effect_dashboard.causal_effect_state import (
    CausalEffectState,
)
from gws_design_of_experiments.causal_effect.causal_effect_dashboard._causal_effect_dashboard.plot_functions import (
    display_barplot,
)
from gws_design_of_experiments.causal_effect.causal_effect_task import CausalEffect


def render_multiple_effect_page():
    # --- Tabs d'affichage ---
    tab_clustermap, tab_barplot = st.tabs(["🧩 Clustermap", "📊 Barplot"])

    df_filtered = CausalEffectState.get_df_filtered()

    # --- HEATMAP ---
    with tab_clustermap:
        # Create display version of the data
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

        pivot = df_display.pivot(
            index=f"{CausalEffect.TARGET_NAME}_Combo_Display",
            columns=CausalEffect.TREATMENT_NAME + "_Display",
            values=CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME,
        ).fillna(0)
        fig = sns.clustermap(pivot, cmap="coolwarm", center=0, figsize=(14, 8))
        st.pyplot(fig.figure)

        # Get the column order from the clustermap (display names)
        clustermap_column_order_display = fig.data2d.columns.tolist()
        # Convert back to original names for the barplot function
        clustermap_column_order = []
        for display_name in clustermap_column_order_display:
            # Find original name that corresponds to this display name
            for original, display in CausalEffectState.get_settings().items():
                if display == display_name:
                    clustermap_column_order.append(original)
                    break
            else:
                clustermap_column_order.append(display_name)  # fallback

    # --- BARPLOT ---
    with tab_barplot:
        display_barplot(df_filtered, thresholds=True, treatment_order=clustermap_column_order)
