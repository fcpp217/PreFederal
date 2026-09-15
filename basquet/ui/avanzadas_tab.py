"""Pestaña 'Avanzadas': posesiones, eficiencia ofensiva/defensiva, eFG%, TS%, etc.

Replica los cálculos de la hoja "PartidoUnicoAvanzadas" de la planilla del
club, a partir de jugadoresAgregado (mismo origen que las pestañas
Jugadores y Quintetos), con los mismos 3 filtros: número de período,
situación del marcador y momento del período. Se usa jugadoresAgregado en
vez de la planilla oficial final porque ahí los intentos de tiro se pueden
calcular siempre como convertidos + fallados (garantizado), y porque así
se pueden aplicar los filtros.
"""

from typing import Dict

import altair as alt
import pandas as pd
import streamlit as st

from ..advanced_stats import calcular_avanzadas_equipo, calcular_avanzadas_jugador, conteos_desde_jugadores_agregado, totales_raw_equipo
from ..colors import _parse_color, _text_color_for_bg
from ..pdf_export import render_pdf_button
from ..utils import _first_col, _first_of, _stay_estadistica

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
    jg = tablas.get('jugadoresAgregado', pd.DataFrame())

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

    if jg.empty:
        st.info("No hay datos agregados por jugador para calcular las métricas avanzadas.")
        return

    period_col = _first_col(jg, ['numero_periodo', 'periodo', 'Periodo'])
    situ_col = _first_col(jg, ['SituacionMarcador', 'Situacion marcador', 'situacion_marcador', 'situacionMarcador', 'situacion', 'Situacion'])
    u2m_col = _first_col(jg, ['ultimos_dos_minutos', 'ultimos2min', 'ultimos_dos', 'u2m'])

    with st.form(key='avanzadas_filters'):
        fcols = st.columns(3)
        with fcols[0]:
            if period_col and period_col in jg.columns:
                per_opts = ['TODOS'] + sorted(pd.to_numeric(jg[period_col], errors='coerce').dropna().astype(int).unique().tolist())
                sel_periodo = st.selectbox('Número de periodo', per_opts, index=0, key='av_sel_per')
            else:
                sel_periodo = 'TODOS'
        with fcols[1]:
            if situ_col and situ_col in jg.columns:
                situ_vals = jg[situ_col].astype(str).fillna('').unique().tolist()
                situ_opts = ['TODOS'] + sorted([s for s in situ_vals if s != ''])
                sel_situ = st.selectbox('Situacion marcador', situ_opts, index=0, key='av_sel_situ')
            else:
                sel_situ = 'TODOS'
        with fcols[2]:
            if u2m_col and u2m_col in jg.columns:
                u2_vals = jg[u2m_col].astype(str).fillna('').unique().tolist()
                u2_opts = ['TODOS'] + sorted([u for u in u2_vals if u != ''])
                sel_u2m = st.selectbox('Momento del periodo', u2_opts, index=0, key='av_sel_u2m')
            else:
                sel_u2m = 'TODOS'
        submitted = st.form_submit_button('Aplicar filtros')
        if submitted:
            _stay_estadistica()

    jg_f = jg.copy()
    if sel_periodo != 'TODOS' and period_col:
        try:
            jg_f = jg_f[pd.to_numeric(jg_f[period_col], errors='coerce') == int(sel_periodo)]
        except Exception:
            pass
    if sel_situ != 'TODOS' and situ_col:
        jg_f = jg_f[jg_f[situ_col].astype(str) == str(sel_situ)]
    if sel_u2m != 'TODOS' and u2m_col:
        jg_f = jg_f[jg_f[u2m_col].astype(str) == str(sel_u2m)]

    conteos_local = conteos_desde_jugadores_agregado(jg_f, 'LOCAL')
    conteos_visit = conteos_desde_jugadores_agregado(jg_f, 'VISITANTE')
    if conteos_local.empty and conteos_visit.empty:
        st.info("No hay jugadas para este filtro.")
        return

    tot_local = totales_raw_equipo(conteos_local)
    tot_visit = totales_raw_equipo(conteos_visit)
    av_local = calcular_avanzadas_equipo(tot_local, tot_visit)
    av_visit = calcular_avanzadas_equipo(tot_visit, tot_local)

    st.write("")
    st.caption(
        "Posesiones = Tiros de campo intentados + 0.44 × Tiros libres intentados − Rebotes ofensivos + Pérdidas. "
        "Eficiencia = puntos por posesión (no está multiplicada por 100). % Bloqueos no está disponible con "
        "este origen de datos y siempre muestra 0%."
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
        jug_local = calcular_avanzadas_jugador(conteos_local)
        if not jug_local.empty:
            jug_local = jug_local.sort_values('TS%', ascending=False).copy()
            for c in ['3p/FG%', 'eFG%', 'TS%', 'FT%']:
                jug_local[c] = jug_local[c].apply(lambda v: f"{v * 100:.0f}%")
            st.dataframe(jug_local, use_container_width=True, hide_index=True)
        else:
            st.info('Sin datos.')
    with col2:
        st.markdown(f"**VISITANTE - {visitante_name}**")
        jug_visit = calcular_avanzadas_jugador(conteos_visit)
        if not jug_visit.empty:
            jug_visit = jug_visit.sort_values('TS%', ascending=False).copy()
            for c in ['3p/FG%', 'eFG%', 'TS%', 'FT%']:
                jug_visit[c] = jug_visit[c].apply(lambda v: f"{v * 100:.0f}%")
            st.dataframe(jug_visit, use_container_width=True, hide_index=True)
        else:
            st.info('Sin datos.')
