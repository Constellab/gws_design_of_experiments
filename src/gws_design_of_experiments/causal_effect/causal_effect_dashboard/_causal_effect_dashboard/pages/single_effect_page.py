
from gws_design_of_experiments.causal_effect.causal_effect_dashboard._causal_effect_dashboard.causal_effect_state import \
    CausalEffectState
from gws_design_of_experiments.causal_effect.causal_effect_dashboard._causal_effect_dashboard.plot_functions import \
    display_barplot


def render_single_effect_page():

    # --- BARPLOT ---
    # Get the filtered data and sort treatments by absolute effect values
    df_filtered = CausalEffectState.get_df_filtered()
    display_barplot(df_filtered, thresholds=True)
