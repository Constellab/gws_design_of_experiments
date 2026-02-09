import streamlit as st
from gws_streamlit_main import StreamlitMainState

# Initialize GWS - MUST be at the top
StreamlitMainState.initialize()

from gws_design_of_experiments.optimization.optimization_dashboard._optimization_dashboard.app_pages.optimization_page import (
    render_first_page,
)

sources = StreamlitMainState.get_sources()
params = StreamlitMainState.get_params()

# Uncomment if you want to hide the Streamlit sidebar toggle and always show the sidebar
# from gws_streamlit_main import StreamlitHelper
# StreamlitHelper.hide_sidebar_toggle()

if sources:
    folder_results = sources[0].path


def _render_first_page():
    render_first_page(folder_results)


_first_page = st.Page(_render_first_page, title="First page", url_path="first-page", icon="📦")
pg = st.navigation([_first_page])

pg.run()
