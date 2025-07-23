from typing import List
import pandas as pd
import streamlit as st

class CausalEffectState:
    """Class to manage the state of the app.
    """

    COMBINATIONS_KEY = "combinations_selectbox"
    DF_FILTERED_KEY = "df_filtered"


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
