import streamlit as st
import pandas as pd
import os
import plotly.express as px
import seaborn as sns
import numpy as np
from gws_design_of_experiments.causal_effect.causal_effect_task import CausalEffect
from gws_design_of_experiments.causal_effect.causal_effect_dashboard._causal_effect_dashboard.causal_effect_state import CausalEffectState



def render_settings_page():

    st.write("### Settings")
