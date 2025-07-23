from typing import List
import pandas as pd
import streamlit as st
import json
import os

class CausalEffectState:
    """Class to manage the state of the app.
    """

    COMBINATIONS_KEY = "combinations_selectbox"
    DF_FILTERED_KEY = "df_filtered"
    INPUT_FOLDER_KEY = "input_folder"
    SETTINGS_KEY = "settings"


    ###### Getters and Setters for the state ######

    @classmethod
    def get_combinations(cls) -> str:
        return st.session_state.get(cls.COMBINATIONS_KEY, None)

    @classmethod
    def get_df_filtered(cls) -> pd.DataFrame:
        return st.session_state.get(cls.DF_FILTERED_KEY, None)

    @classmethod
    def set_df_filtered(cls, df: pd.DataFrame):
        st.session_state[cls.DF_FILTERED_KEY] = df

    @classmethod
    def get_input_folder(cls) -> str:
        return st.session_state.get(cls.INPUT_FOLDER_KEY, None)

    @classmethod
    def set_input_folder(cls, folder: str):
        st.session_state[cls.INPUT_FOLDER_KEY] = folder

    @classmethod
    def get_settings(cls) -> dict:
        return st.session_state.get(cls.SETTINGS_KEY, {})

    @classmethod
    def set_settings(cls, settings: dict):
        st.session_state[cls.SETTINGS_KEY] = settings

    @classmethod
    def load_settings(cls):
        """Load settings from settings.json file"""
        input_folder = cls.get_input_folder()
        if input_folder:
            settings_path = os.path.join(input_folder, "settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    cls.set_settings(settings)
                    return settings
        return {}

    @classmethod
    def get_display_name(cls, original_name: str) -> str:
        """Get display name for an original name from settings"""
        settings = cls.get_settings()
        return settings.get(original_name, original_name)
