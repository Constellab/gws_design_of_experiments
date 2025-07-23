import os
import pandas as pd
import numpy as np
from gws_design_of_experiments.causal_effect.causal_effect_task import CausalEffect

import streamlit as st
from pages import single_effect_page, multiple_effect_page, settings_page
from gws_design_of_experiments.causal_effect.causal_effect_dashboard._causal_effect_dashboard.causal_effect_state import CausalEffectState
from gws_core.streamlit import StreamlitRouter

sources: list
params: dict

# Uncomment if you want to hide the Streamlit sidebar toggle and always show the sidebar
# from gws_core.streamlit import StreamlitHelper
# StreamlitHelper.hide_sidebar_toggle()


# Pages

def _render_single_effect_page():
    single_effect_page.render_single_effect_page()

def _render_multiple_effect_page():
    multiple_effect_page.render_multiple_effect_page()

def _render_settings_page():
    settings_page.render_settings_page()


# --- Loading data ---
@st.cache_data
def load_data(folder_path: str) -> pd.DataFrame:
    all_results = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file == "causal_effects.csv":
                path = os.path.join(root, file)
                combinaison = os.path.basename(root)
                df = pd.read_csv(path)
                df["Combinaison"] = combinaison
                df[f"{CausalEffect.TARGET_NAME}_Combo"] = df[CausalEffect.TARGET_NAME] + " [" + combinaison + "]"
                all_results.append(df)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        raise Exception("No data found in the specified folder.")


def show_sidebar():
    with st.sidebar:
        # 1. Combinaisons
        combinaisons = sorted(df_all["Combinaison"].unique())
        # Map combinations to display names
        display_combinaisons = [CausalEffectState.get_display_name(combo) for combo in combinaisons]
        selected_display = st.selectbox("Combinations of targets", options=display_combinaisons, index=0, key="combinations_selectbox_display")
        # Find the original combination name
        selected_combo = combinaisons[display_combinaisons.index(selected_display)] if selected_display in display_combinaisons else combinaisons[0]
        st.session_state["combinations_selectbox"] = selected_combo

    # 2. Traitements - auto-désélection de ceux à 0
    df_temp = df_all[df_all["Combinaison"]== CausalEffectState.get_combinations()]
    traitement_stats = df_temp.groupby(CausalEffect.TREATMENT_NAME)[CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME].apply(lambda x: (x != 0).any())
    traitements_valides = sorted(traitement_stats[traitement_stats].index.tolist())
    traitements = sorted(df_all[CausalEffect.TREATMENT_NAME].unique())

    # Filtrage selon combinaison + traitement
    df_filtre = df_all[
        (df_all["Combinaison"]== CausalEffectState.get_combinations()) &
        (df_all[CausalEffect.TREATMENT_NAME].isin(traitements_valides))
    ]

    # 3. Targets
    cibles_combo = sorted(df_filtre[f"{CausalEffect.TARGET_NAME}_Combo"].unique())

    df_filtre = df_filtre[df_filtre[f"{CausalEffect.TARGET_NAME}_Combo"].isin(cibles_combo)]
    df_filtre[CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME] = np.sign(df_filtre[CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME]) * np.log10(1 + np.abs(df_filtre[CausalEffect.AVERAGE_CAUSAL_EFFECT_NAME]))
    if df_filtre.empty:
        st.warning("No data to display.")
        return
    CausalEffectState.set_df_filtered(df_filtre)


if sources:
    router = StreamlitRouter.load_from_session()
    folder_results = sources[0].path

    CausalEffectState.set_input_folder(folder_results)
    CausalEffectState.load_settings()  # Load settings at startup

    df_all = load_data(folder_results)
    show_sidebar()

    pages = {}

    if "_" in CausalEffectState.get_combinations():
        _multiple_effect_page = st.Page(_render_multiple_effect_page, title='Multiple effect', url_path='multiple-effect-page', icon='📊')
        pages['Multiple effect'] = [_multiple_effect_page]
    else:
        _single_effect_page = st.Page(_render_single_effect_page, title='Single effect', url_path='single-effect-page', icon='📈')
        pages['Single effect'] = [_single_effect_page]


    _settings_page = st.Page(_render_settings_page, title='Settings', url_path='settings', icon='⚙️')
    pages['Settings'] = [_settings_page]

    pg = st.navigation(pages)
    pg.run()


