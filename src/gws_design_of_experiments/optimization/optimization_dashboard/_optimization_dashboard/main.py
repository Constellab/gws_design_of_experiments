

import streamlit as st
from pages import optimization_page

sources: list
params: dict

# Uncomment if you want to hide the Streamlit sidebar toggle and always show the sidebar
# from gws_core.streamlit import StreamlitHelper
# StreamlitHelper.hide_sidebar_toggle()

if sources:
    folder_results = sources[0].path

def _render_first_page():
    optimization_page.render_first_page(folder_results)


_first_page = st.Page(_render_first_page, title='First page', url_path='first-page', icon='📦')
pg = st.navigation([_first_page])

pg.run()
