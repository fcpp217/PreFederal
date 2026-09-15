"""Pestaña 'Posesión': estadísticas de tiro según el tiempo de posesión previo.

Se agrupan los tiros de campo (2P/3P, convertidos o no) en tres momentos del
ataque -0 a 8s, 9 a 16s y 17 a 24s desde que el equipo recuperó la pelota-,
más un cuarto grupo residual ("+24s") para las posesiones más largas que no
se quiere ocultar. El objetivo es ver si un equipo tira mejor en transición,
a mitad de posesión o cuando el ataque se estanca.
"""

from typing import Dict

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ..colors import _parse_color, _text_color_for_bg
from ..data_processing import ACCIONES_TIRO_DE_CAMPO, BUCKETS_POSESION
from ..pdf_export import render_pdf_button
from ..utils import _first_of

CONVERTIDAS = {'CANASTA-2P', 'CANASTA-3P'}
PUNTOS_POR_ACCION = {'CANASTA-2P': 2, 'CANASTA-3P': 3}


def _preparar_tiros(pbp_df: pd.DataFrame) -> pd.DataFrame:
    if pbp_df.empty or 'accion_tipo' not in pbp_df.columns:
        return pd.DataFrame()
    d = pbp_df[pbp_df['accion_tipo'].isin(ACCIONES_TIRO_DE_CAMPO)].copy()
    if d.empty:
        return d
    d = d[d.get('bucket_posesion').notna()]
    d['Tipo'] = np.where(d['accion_tipo'].isin(['CANASTA-2P', 'TIRO2-FALLADO']), '2P', '3P')
    d['Convertido'] = d['accion_tipo'].isin(CONVERTIDAS)
    d['Puntos'] = d['accion_tipo'].map(PUNTOS_POR_ACCION).fillna(0)
    d['Condicion'] = d.get('Condicion', '').astype(str).str.upper()
    return d


def _resumen_por_bucket(d: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame(columns=group_cols + ['Intentos', 'Convertidos', '%Efectividad', 'Puntos'])
    agg = d.groupby(group_cols, as_index=False).agg(
        Intentos=('accion_tipo', 'count'),
        Convertidos=('Convertido', 'sum'),
        Puntos=('Puntos', 'sum'),
    )
    agg['%Efectividad'] = np.where(agg['Intentos'] > 0, (agg['Convertidos'] / agg['Intentos'] * 100.0).round(1), 0.0)
    return agg


def render_posesion(tablas: Dict[str, pd.DataFrame]) -> None:
    render_pdf_button(tablas, key='posesion')

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

    try:
        periodos = sorted(pd.to_numeric(pbp_df['numero_periodo'], errors='coerce').dropna().astype(int).unique().tolist())
    except Exception:
        periodos = []
    sel_periodo = st.selectbox('Seleccionar periodo', ['TODOS'] + periodos, index=0, key='pos_sel_per')

    d = _preparar_tiros(pbp_df)
    if sel_periodo != 'TODOS' and not d.empty:
        try:
            d = d[pd.to_numeric(d['numero_periodo'], errors='coerce') == int(sel_periodo)]
        except Exception:
            pass

    if d.empty:
        st.info("No hay tiros de campo con tiempo de posesión calculado para este filtro.")
        return

    color_scale_equipo = alt.Scale(domain=[local_name, visitante_name], range=[color_local, color_visitante])
    d = d.copy()
    d['Equipo'] = np.where(d['Condicion'] == 'LOCAL', local_name, np.where(d['Condicion'] == 'VISITANTE', visitante_name, 'Otro'))
    d = d[d['Equipo'] != 'Otro']

    resumen_equipo = _resumen_por_bucket(d, ['Equipo', 'bucket_posesion'])

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        chart_pct = (
            alt.Chart(resumen_equipo)
            .mark_bar()
            .encode(
                x=alt.X('bucket_posesion:N', title='Tiempo de posesión antes del tiro', sort=BUCKETS_POSESION),
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
                x=alt.X('bucket_posesion:N', title='Tiempo de posesión antes del tiro', sort=BUCKETS_POSESION),
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

    resumen_tipo = _resumen_por_bucket(d, ['bucket_posesion', 'Tipo'])
    chart_tipo = (
        alt.Chart(resumen_tipo)
        .mark_bar()
        .encode(
            x=alt.X('bucket_posesion:N', title='Tiempo de posesión antes del tiro', sort=BUCKETS_POSESION),
            y=alt.Y('Intentos:Q', title='Tiros intentados', stack='normalize', axis=alt.Axis(format='%')),
            color=alt.Color('Tipo:N', scale=alt.Scale(domain=['2P', '3P'], range=['#1e88e5', '#43a047']), legend=alt.Legend(orient='top', title='Tipo de tiro')),
            order=alt.Order('Tipo:N'),
            tooltip=[alt.Tooltip('bucket_posesion:N', title='Momento'), alt.Tooltip('Tipo:N'), alt.Tooltip('Intentos:Q')],
        )
        .properties(height=220, title=alt.TitleParams(text='Selección de tiro (2P vs 3P) según el momento de la posesión', anchor='middle'))
    )
    st.altair_chart(chart_tipo, use_container_width=True)

    st.write("")
    st.subheader('Resumen por equipo')
    tabla = resumen_equipo.sort_values(['Equipo', 'bucket_posesion'], key=lambda s: s.map({v: i for i, v in enumerate(BUCKETS_POSESION)}) if s.name == 'bucket_posesion' else s)
    tabla = tabla.rename(columns={'bucket_posesion': 'Momento'})
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    with st.expander('Detalle por jugador'):
        resumen_jugador = _resumen_por_bucket(d, ['Equipo', 'nombre', 'bucket_posesion'])
        resumen_jugador = resumen_jugador.rename(columns={'nombre': 'Nombre', 'bucket_posesion': 'Momento'})
        st.dataframe(
            resumen_jugador.sort_values(['Equipo', 'Nombre', 'Momento']),
            use_container_width=True,
            hide_index=True,
        )
