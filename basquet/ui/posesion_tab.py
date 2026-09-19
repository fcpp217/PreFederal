"""Pestaña 'Posesión': estadísticas de tiro según el tiempo de posesión previo.

Se agrupan los tiros de campo (2P/3P, convertidos o no) en tres momentos del
ataque -0 a 8s, 9 a 16s y 17 a 24s desde que el equipo recuperó la pelota-,
más una categoría aparte para los tiros que llegan tras un rebote ofensivo
propio (no compiten contra un reloj de 24s nuevo). Los tiros libres no se
consideran: no reflejan un uso real del reloj de posesión. Sigue la misma
lógica de filtros que las pestañas de Jugadores y Quintetos (período,
situación del marcador y momento del período).
"""

from typing import Dict

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ..colors import _parse_color, _text_color_for_bg
from ..data_processing import BUCKET_A_REVISAR, BUCKETS_POSESION
from ..exports_ui import render_export_buttons
from ..metric_definitions import render_definiciones_markdown
from ..possession_stats import preparar_tiros, resumen_por_bucket, resumen_por_bucket_y_tipo
from ..utils import _first_col, _first_of, _stay_estadistica

BUCKETS_VALIDOS = [b for b in BUCKETS_POSESION if b != BUCKET_A_REVISAR]


def render_posesion(tablas: Dict[str, pd.DataFrame]) -> None:
    render_export_buttons(tablas, key='posesion')

    pbp_df = tablas.get('pbp', pd.DataFrame())
    part_df = tablas.get('partido', pd.DataFrame())
    est_loc_df = tablas.get('estadisticas_equipolocal', pd.DataFrame())
    est_vis_df = tablas.get('estadisticas_equipovisitante', pd.DataFrame())

    if pbp_df.empty or 'bucket_posesion' not in pbp_df.columns:
        st.info("No hay datos de jugadas para calcular el tiempo de posesión.")
        return

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

    period_col = _first_col(pbp_df, ['numero_periodo', 'periodo', 'Periodo'])
    situ_col = _first_col(pbp_df, ['SituacionMarcador', 'Situacion marcador', 'situacion_marcador', 'situacionMarcador', 'situacion', 'Situacion'])
    u2m_col = _first_col(pbp_df, ['ultimos_dos_minutos', 'ultimos2min', 'ultimos_dos', 'u2m'])

    with st.form(key='posesion_filters'):
        fcols = st.columns(3)
        with fcols[0]:
            if period_col and period_col in pbp_df.columns:
                per_opts = ['TODOS'] + sorted(pd.to_numeric(pbp_df[period_col], errors='coerce').dropna().astype(int).unique().tolist())
                sel_periodo = st.selectbox('Número de periodo', per_opts, index=0, key='pos_sel_per')
            else:
                sel_periodo = 'TODOS'
        with fcols[1]:
            if situ_col and situ_col in pbp_df.columns:
                situ_vals = pbp_df[situ_col].astype(str).fillna('').unique().tolist()
                situ_opts = ['TODOS'] + sorted([s for s in situ_vals if s != ''])
                sel_situ = st.selectbox('Situacion marcador', situ_opts, index=0, key='pos_sel_situ')
            else:
                sel_situ = 'TODOS'
        with fcols[2]:
            if u2m_col and u2m_col in pbp_df.columns:
                u2_vals = pbp_df[u2m_col].astype(str).fillna('').unique().tolist()
                u2_opts = ['TODOS'] + sorted([u for u in u2_vals if u != ''])
                sel_u2m = st.selectbox('Momento del periodo', u2_opts, index=0, key='pos_sel_u2m')
            else:
                sel_u2m = 'TODOS'
        submitted = st.form_submit_button('Aplicar filtros')
        if submitted:
            _stay_estadistica()

    d = preparar_tiros(pbp_df)
    if not d.empty:
        if sel_periodo != 'TODOS' and period_col:
            try:
                d = d[pd.to_numeric(d[period_col], errors='coerce') == int(sel_periodo)]
            except Exception:
                pass
        if sel_situ != 'TODOS' and situ_col:
            d = d[d[situ_col].astype(str) == str(sel_situ)]
        if sel_u2m != 'TODOS' and u2m_col:
            d = d[d[u2m_col].astype(str) == str(sel_u2m)]

    if d.empty:
        st.info("No hay tiros de campo con tiempo de posesión calculado para este filtro.")
        return

    color_scale_equipo = alt.Scale(domain=[local_name, visitante_name], range=[color_local, color_visitante])
    d = d.copy()
    d['Equipo'] = np.where(d['Condicion'] == 'LOCAL', local_name, np.where(d['Condicion'] == 'VISITANTE', visitante_name, 'Otro'))
    d = d[d['Equipo'] != 'Otro']

    # Las posesiones "a revisar" (>24s, básicamente imposibles en básquet) se
    # excluyen de los gráficos y del resumen para no distorsionar los
    # porcentajes; quedan disponibles aparte para poder auditar la carga.
    revisar = d[d['bucket_posesion'] == BUCKET_A_REVISAR]
    d_validos = d[d['bucket_posesion'] != BUCKET_A_REVISAR]

    if not revisar.empty:
        st.warning(
            f"⚠️ Se detectaron {len(revisar)} tiro(s) con más de 24s de posesión previa, algo imposible en "
            "básquet (el reloj de posesión nunca supera los 24s). Probablemente falte un evento en la carga "
            "del partido (un rebote, una recuperación o una pérdida). Se excluyeron de los gráficos y del "
            "resumen; el detalle está más abajo para poder revisarlos."
        )

    if d_validos.empty:
        st.info("No hay tiros de campo válidos (fuera de los casos a revisar) para este filtro.")
        return

    resumen_equipo = resumen_por_bucket(d_validos, ['Equipo', 'bucket_posesion'])

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        chart_pct = (
            alt.Chart(resumen_equipo)
            .mark_bar()
            .encode(
                x=alt.X('bucket_posesion:N', title='Tiempo de posesión antes del tiro', sort=BUCKETS_VALIDOS),
                xOffset=alt.XOffset('Equipo:N', sort=[local_name, visitante_name]),
                y=alt.Y('%Efectividad:Q', title='Efectividad (%)'),
                color=alt.Color('Equipo:N', scale=color_scale_equipo, legend=alt.Legend(orient='top', title=None)),
                tooltip=[
                    alt.Tooltip('Equipo:N'),
                    alt.Tooltip('bucket_posesion:N', title='Momento'),
                    alt.Tooltip('Intentos:Q'),
                    alt.Tooltip('Convertidos:Q'),
                    alt.Tooltip('%Efectividad:Q', title='Efectividad (%)', format='.1f'),
                ],
            )
            .properties(height=320, title=alt.TitleParams(text='% de efectividad según tiempo de posesión', anchor='middle'))
        )
        st.altair_chart(chart_pct, use_container_width=True)
    with col2:
        chart_vol = (
            alt.Chart(resumen_equipo)
            .mark_bar()
            .encode(
                x=alt.X('bucket_posesion:N', title='Tiempo de posesión antes del tiro', sort=BUCKETS_VALIDOS),
                xOffset=alt.XOffset('Equipo:N', sort=[local_name, visitante_name]),
                y=alt.Y('Intentos:Q', title='Tiros de campo intentados'),
                color=alt.Color('Equipo:N', scale=color_scale_equipo, legend=alt.Legend(orient='top', title=None)),
                tooltip=[
                    alt.Tooltip('Equipo:N'),
                    alt.Tooltip('bucket_posesion:N', title='Momento'),
                    alt.Tooltip('Intentos:Q'),
                    alt.Tooltip('Puntos:Q'),
                ],
            )
            .properties(height=320, title=alt.TitleParams(text='Volumen de tiros según tiempo de posesión', anchor='middle'))
        )
        st.altair_chart(chart_vol, use_container_width=True)

    st.write("")
    st.subheader('Resumen por equipo (2P y 3P por separado)')
    orden_bucket = {v: i for i, v in enumerate(BUCKETS_VALIDOS)}
    resumen_equipo_tipo = resumen_por_bucket_y_tipo(d_validos, ['Equipo', 'bucket_posesion'])
    resumen_equipo_tipo = resumen_equipo_tipo.sort_values(
        ['Equipo', 'bucket_posesion'],
        key=lambda s: s.map(orden_bucket) if s.name == 'bucket_posesion' else s,
    )
    resumen_equipo_tipo = resumen_equipo_tipo.rename(columns={'bucket_posesion': 'Momento'})
    st.dataframe(resumen_equipo_tipo, use_container_width=True, hide_index=True)

    with st.expander('Detalle por jugador'):
        resumen_jugador = resumen_por_bucket_y_tipo(d_validos, ['Equipo', 'nombre', 'bucket_posesion'])
        resumen_jugador = resumen_jugador.rename(columns={'nombre': 'Nombre', 'bucket_posesion': 'Momento'})
        st.dataframe(
            resumen_jugador.sort_values(['Equipo', 'Nombre', 'Momento']),
            use_container_width=True,
            hide_index=True,
        )

    if not revisar.empty:
        with st.expander(f'⚠️ Posesiones a revisar ({len(revisar)})'):
            cols_show = [c for c in ['autoincremental_id', 'Equipo', 'nombre', 'numero_periodo', 'tiempo_segundos', 'accion_tipo', 'tiempo_posesion'] if c in revisar.columns]
            st.dataframe(
                revisar[cols_show].rename(columns={
                    'autoincremental_id': 'ID jugada', 'nombre': 'Jugador', 'numero_periodo': 'Periodo',
                    'tiempo_segundos': 'Tiempo partido (s)', 'accion_tipo': 'Tiro', 'tiempo_posesion': 'Posesión (s)',
                }),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander('ℹ️ Qué significa cada métrica'):
        st.markdown(render_definiciones_markdown(['Posesión (tiempo de posesión antes del tiro)']))
