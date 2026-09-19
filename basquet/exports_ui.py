"""Botones de exportación (PDF y Excel) que se repiten en cada pestaña."""

from typing import Dict

import pandas as pd
import streamlit as st

from .excel_export import render_excel_button
from .pdf_export import render_pdf_button


def render_export_buttons(tablas: Dict[str, pd.DataFrame], key: str) -> None:
    col1, col2 = st.columns(2)
    with col1:
        render_pdf_button(tablas, key=key)
    with col2:
        render_excel_button(tablas, key=key)
