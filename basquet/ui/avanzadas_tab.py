"""Pestaña 'Avanzadas': posesiones, eficiencia ofensiva/defensiva, eFG%, TS%, etc.

Replica los cálculos de la hoja "PartidoUnicoAvanzadas" de la planilla del
club, a partir de las estadísticas oficiales por jugador de cada equipo.
"""

from typing import Dict

import altair as alt
import pandas as pd
import streamlit as st

from ..advanced_stats import calcular_avanzadas_equipo, calcular_avanzadas_jugador, totales_raw_equipo
from ..colors import _parse_color, _text_color_for_bg
from ..pdf_export import render_pdf_button
from ..utils import _first_of

METRICAS_PORCENTAJE = {
    '% Rebotes Defensivos', '% Rebotes Ofensivos', '% Rebotes Totales',
    '% Asistencias', '% Pérdidas', '% Robos', '% Bloqueos',
    '3p/FG', 'eFG%', 'TS%', 'FT%',
}
METRICAS_CHART = ['Eficiencia Ofensiva', 'Eficiencia Defensiva', 'Net Rating', 'eFG%', 'TS%', '% Rebotes Totales', '% Asistencias', '% Pérdidas']
ORDEN_METRICAS = [
    'Posesiones', 'Eficiencia Ofensiva', 'Eficiencia Defensiva', 'Net Rating',
    '% Rebotes Defensivos', '% Rebotes Ofensivos', '% Rebotes Totales',
    '% Asistencias', '% Pérdidas', '% Robos', '% Bloqueos', '3p/FG', 'eFG%', 'TS%', 'FT%',
]


def _fmt(nombre: str, valor: float) -> str:
    if nombre in METRICAS_PORCENTAJE:
        return f"{valor * 100:.1f}%"
    return f"{valor:.2f}"


def render_avanzadas(tablas: Dict[str, pd.DataFrame]) -> None:
    render_pdf_button(tablas, key='avanzadas')

    part_df = tablas.get('partido', pd.DataFrame())
    est_loc_df = tablas.get('estadisticas_equipolocal', pd.DataFrame())
    est_vis_df = tablas.get('estadisticas_equipovisitante', pd.DataFrame())

    row = part_df.iloc[0] if not part_df.empty else {}
    local_name = str(_first_of(row, ['local', 'equipo_local', 'nombre_local'], ''))
    visitante_name = str(_first_of(row, ['visitante', 'equipo_visitante', 'nombre_visitante'], ''))
    if (not local_name.strip()) and not est_loc_df.empty:
        local_name = str(_first_of(est_loc_df.iloc[0], ['equipo', 'nombre_equipo'], 'Local'))
    if (not visitante_name.strip()) and not est_vis_df.empty:
        visitante_name = str(_first_of(est_vis_df.iloc[0], ['equipo', 'nombre_equipo'], 'Visitante'))
    local_name = local_name or 'Local'
    visitante_name = visitante_name or 'Visitante'

    color_local = _parse_color(_first_of(row, ['color_local', 'local_color', 'colorLocal', 'colorlocal'], '#1f77b4'), '#1f77b4')
    color_visitante = _parse_color(_first_of(row, ['color_visitante', 'visitante_color', 'colorVisitante', 'colorvisitante'], '#ff7f0e'), '#ff7f0e')
    tc_local = _text_color_for_bg(color_local)
    tc_visitante = _text_color_for_bg(color_visitante)

    st.markdown(f"""
    <div style='display:flex; gap:12px; margin-bottom:8px;'>
        <div style='flex:1; background:{color_local}; color:{tc_local}; padding:10px 14px; border-radius:8px; text-align:center; font-weight:700;'>🏀 LOCAL - {local_name}</div>
        <div style='flex:1; background:{color_visitante}; color:{tc_visitante}; padding:10px 14px; border-radius:8px; text-align:center; font-weight:700;'>🏀 VISITANTE - {visitante_name}</div>
    </div>
    """, unsafe_allow_html=True)

    if est_loc_df.empty and est_vis_df.empty:
        st.info("No hay estadísticas oficiales por jugador para calcular las métricas avanzadas.")
        return

    tot_local = totales_raw_equipo(est_loc_df)
    tot_visit = totales_raw_equipo(est_vis_df)
    av_local = calcular_avanzadas_equipo(tot_local, tot_visit)
    av_visit = calcular_avanzadas_equipo(tot_visit, tot_local)

    st.write("")
    st.caption(
        "Posesiones = Tiros de campo intentados + 0.44 × Tiros libres intentados − Rebotes ofensivos + Pérdidas. "
        "Eficiencia = puntos por posesión (no está multiplicada por 100)."
    )

    color_scale_equipo = alt.Scale(domain=[local_name, visitante_name], range=[color_local, color_visitante])

    def bar_chart_metric(nombre_metrica: str):
        dfc = pd.DataFrame([
            {'Equipo': local_name, 'Valor': av_local.get(nombre_metrica, 0.0)},
            {'Equipo': visitante_name, 'Valor': av_visit.get(nombre_metrica, 0.0)},
        ])
        es_pct = nombre_metrica in METRICAS_PORCENTAJE
        bars = alt.Chart(dfc).mark_bar(stroke='#000000', strokeWidth=1).encode(
            x=alt.X('Equipo:N', title=None, sort=[local_name, visitante_name], axis=alt.Axis(labelAngle=315)),
            y=alt.Y('Valor:Q', title=None, axis=alt.Axis(format='%') if es_pct else alt.Axis()),
            color=alt.Color('Equipo:N', scale=color_scale_equipo, legend=None),
            tooltip=[alt.Tooltip('Equipo:N'), alt.Tooltip('Valor:Q', format='.1%' if es_pct else '.2f')],
        )
        fmt = '.1%' if es_pct else '.2f'
        text = alt.Chart(dfc).mark_text(dy=-6, fontSize=13, fontWeight='bold').encode(
            x=alt.X('Equipo:N', sort=[local_name, visitante_name]),
            y=alt.Y('Valor:Q'),
            text=alt.Text('Valor:Q', format=fmt),
        )
        return (bars + text).properties(height=220, title=nombre_metrica)

    filas_charts = [METRICAS_CHART[:4], METRICAS_CHART[4:]]
    for fila in filas_charts:
        cols = st.columns(len(fila))
        for c, metrica in zip(cols, fila):
            with c:
                st.altair_chart(bar_chart_metric(metrica), use_container_width=True)

    st.write("")
    st.subheader('Todas las métricas de equipo')
    tabla_equipo = pd.DataFrame({
        'Métrica': ORDEN_METRICAS,
        local_name: [_fmt(m, av_local.get(m, 0.0)) for m in ORDEN_METRICAS],
        visitante_name: [_fmt(m, av_visit.get(m, 0.0)) for m in ORDEN_METRICAS],
    })
    st.dataframe(tabla_equipo, use_container_width=True, hide_index=True)

    st.write("")
    st.subheader('Métricas por jugador (3p/FG%, eFG%, TS%, FT%)')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**LOCAL - {local_name}**")
        jug_local = calcular_avanzadas_jugador(est_loc_df)
        if not jug_local.empty:
            jug_local = jug_local.sort_values('TS%', ascending=False).copy()
            for c in ['3p/FG%', 'eFG%', 'TS%', 'FT%']:
                jug_local[c] = jug_local[c].apply(lambda v: f"{v * 100:.0f}%")
            st.dataframe(jug_local, use_container_width=True, hide_index=True)
        else:
            st.info('Sin datos.')
    with col2:
        st.markdown(f"**VISITANTE - {visitante_name}**")
        jug_visit = calcular_avanzadas_jugador(est_vis_df)
        if not jug_visit.empty:
            jug_visit = jug_visit.sort_values('TS%', ascending=False).copy()
            for c in ['3p/FG%', 'eFG%', 'TS%', 'FT%']:
                jug_visit[c] = jug_visit[c].apply(lambda v: f"{v * 100:.0f}%")
            st.dataframe(jug_visit, use_container_width=True, hide_index=True)
        else:
            st.info('Sin datos.')
