import streamlit as st
import os
import json
from gws_design_of_experiments.causal_effect.causal_effect_task import CausalEffect
from gws_design_of_experiments.causal_effect.causal_effect_dashboard._causal_effect_dashboard.causal_effect_state import CausalEffectState
from gws_core.streamlit import StreamlitContainers



def render_settings_page():
    st.write("Change the display of your variables in the dashboard by defining the settings below.")

    input_folder = CausalEffectState.get_input_folder()
    settings_path = os.path.join(input_folder, "settings.json")
    # Load settings
    with open(settings_path, 'r') as f:
        settings = json.load(f)

    # Create input fields for each setting
    updated_settings = {}
    settings_changed = False

    # Create header
    cols = st.columns([2, 2, 2])
    with cols[0]:
        st.write("**Original name**")
    with cols[1]:
        st.write("**Name to display**")

    # Create rows for each setting
    for key, value in settings.items():
        cols = st.columns([2, 2, 2])
        with cols[0]:
            st.text(key)
        with cols[1]:
            new_value = st.text_input(
                label="",
                value=str(value),
                key=f"setting_{key}",
                label_visibility="collapsed"
            )
            updated_settings[key] = new_value

        # Check if value changed
        if str(value) != new_value:
            settings_changed = True

    # Save updated settings if any changes were made
    if settings_changed:
        with open(settings_path, 'w') as f:
            json.dump(updated_settings, f, indent=2)
        st.rerun()
