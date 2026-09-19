"""Pestaña 'Definiciones': glosario de todas las métricas usadas en la app."""

from typing import Dict

import pandas as pd
import streamlit as st

from ..exports_ui import render_export_buttons
from ..metric_definitions import DEFINICIONES


def render_definiciones(tablas: Dict[str, pd.DataFrame]) -> None:
    render_export_buttons(tablas, key='definiciones')

    st.write("")
    st.caption(
        "Qué significa cada columna y cada métrica que aparece en el resto de las pestañas, "
        "agrupadas por dónde se usan."
    )

    for seccion, items in DEFINICIONES.items():
        st.subheader(seccion)
        for nombre, texto in items:
            st.markdown(f"- **{nombre}**: {texto}")
        st.write("")
