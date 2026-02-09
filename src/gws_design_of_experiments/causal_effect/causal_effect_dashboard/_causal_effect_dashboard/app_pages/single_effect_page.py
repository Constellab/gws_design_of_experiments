from ..causal_effect_state import CausalEffectState
from ..plot_functions import display_barplot


def render_single_effect_page():
    # --- BARPLOT ---
    # Get the filtered data and sort treatments by absolute effect values
    df_filtered = CausalEffectState.get_df_filtered()
    display_barplot(df_filtered, thresholds=True)
