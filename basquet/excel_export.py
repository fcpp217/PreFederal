"""Exportación a Excel (.xlsx), una hoja por cada área de la app.

Reutiliza las mismas funciones de cálculo que ya alimentan la UI y el PDF
(basquet/tables.py, basquet/possession_stats.py, basquet/advanced_stats.py y
algunos helpers de basquet/pdf_export.py), para que los números sean
siempre los mismos sin importar qué exportación se mire.
"""

import io
from typing import Any, Dict

import pandas as pd
import streamlit as st

from .advanced_stats import calcular_avanzadas_equipo, calcular_avanzadas_jugador, conteos_desde_jugadores_agregado, totales_raw_equipo
from .data_processing import BUCKET_A_REVISAR
from .metric_definitions import definiciones_dataframe
from .pdf_export import _ORDEN_METRICAS_AVANZADAS, _METRICAS_PORCENTAJE_PDF, _resumen_basico, _totales_equipo, _top_quintetos
from .possession_stats import preparar_tiros, resumen_por_bucket_y_tipo
from .tables import _build_table


def _hoja_resumen(writer: "pd.ExcelWriter", tablas: Dict[str, pd.DataFrame], info: Dict[str, Any]) -> None:
    sheet = 'Resumen'
    periodos = sorted(set(info['per_local']) | set(info['per_visit']))
    header = ['Equipo'] + [f'P{p}' for p in periodos] + ['Total']
    fila_local = [info['local_name']] + [info['per_local'].get(p, '') for p in periodos] + [info['tanteo_local']]
    fila_visit = [info['visitante_name']] + [info['per_visit'].get(p, '') for p in periodos] + [info['tanteo_visitante']]
    df_marcador = pd.DataFrame([fila_local, fila_visit], columns=header)
    df_marcador.to_excel(writer, sheet_name=sheet, index=False, startrow=0)

    tot_local = _totales_equipo(tablas.get('estadisticas_equipolocal', pd.DataFrame()))
    tot_visit = _totales_equipo(tablas.get('estadisticas_equipovisitante', pd.DataFrame()))
    metricas = ['Puntos', '%1P', '%2P', '%3P', 'Reb. Def.', 'Reb. Of.', 'Reb. Tot.', 'Asistencias', 'Pérdidas', 'Recuperaciones']
    df_comp = pd.DataFrame({
        'Métrica': metricas,
        info['local_name']: [round(tot_local.get(m, 0.0), 1) for m in metricas],
        info['visitante_name']: [round(tot_visit.get(m, 0.0), 1) for m in metricas],
    })
    df_comp.to_excel(writer, sheet_name=sheet, index=False, startrow=len(df_marcador) + 3)


def _hoja_jugadores(writer: "pd.ExcelWriter", df_equipo: pd.DataFrame, nombre_hoja: str) -> None:
    df_show = _build_table(df_equipo)
    if df_show is None or df_show.empty:
        df_show = pd.DataFrame({'Aviso': ['Sin datos disponibles.']})
    df_show.to_excel(writer, sheet_name=nombre_hoja, index=False)


def _hoja_quintetos(writer: "pd.ExcelWriter", qg: pd.DataFrame, condicion: str, nombre_hoja: str) -> None:
    df_top = _top_quintetos(qg, condicion, top_n=50)
    if df_top.empty:
        df_top = pd.DataFrame({'Aviso': ['Sin datos disponibles.']})
    df_top.to_excel(writer, sheet_name=nombre_hoja, index=False)


def _hoja_posesion(writer: "pd.ExcelWriter", tablas: Dict[str, pd.DataFrame], info: Dict[str, Any]) -> None:
    sheet = 'Posesion'
    pbp_df = tablas.get('pbp', pd.DataFrame())
    d = preparar_tiros(pbp_df)
    if d.empty:
        pd.DataFrame({'Aviso': ['Sin tiros de campo con posesión calculada.']}).to_excel(writer, sheet_name=sheet, index=False)
        return
    d = d.copy()
    d['Equipo'] = d['Condicion'].map({'LOCAL': info['local_name'], 'VISITANTE': info['visitante_name']}).fillna('Otro')
    d = d[d['Equipo'] != 'Otro']
    revisar = d[d['bucket_posesion'] == BUCKET_A_REVISAR]
    d_validos = d[d['bucket_posesion'] != BUCKET_A_REVISAR]

    resumen = resumen_por_bucket_y_tipo(d_validos, ['Equipo', 'bucket_posesion']).rename(columns={'bucket_posesion': 'Momento'})
    resumen.to_excel(writer, sheet_name=sheet, index=False, startrow=0)

    if not revisar.empty:
        startrow = len(resumen) + 3
        cols_show = [c for c in ['autoincremental_id', 'Equipo', 'nombre', 'numero_periodo', 'tiempo_segundos', 'accion_tipo', 'tiempo_posesion'] if c in revisar.columns]
        aviso = pd.DataFrame({'Aviso': [f'{len(revisar)} tiro(s) con posesión >24s excluidos por dato a revisar:']})
        aviso.to_excel(writer, sheet_name=sheet, index=False, startrow=startrow, header=False)
        revisar[cols_show].rename(columns={
            'autoincremental_id': 'ID jugada', 'nombre': 'Jugador', 'numero_periodo': 'Periodo',
            'tiempo_segundos': 'Tiempo partido (s)', 'accion_tipo': 'Tiro', 'tiempo_posesion': 'Posesión (s)',
        }).to_excel(writer, sheet_name=sheet, index=False, startrow=startrow + 2)


def _fmt_valor_avanzada(nombre: str, valor: float) -> float:
    return round(valor * 100, 1) if nombre in _METRICAS_PORCENTAJE_PDF else round(valor, 2)


def _hoja_avanzadas(writer: "pd.ExcelWriter", tablas: Dict[str, pd.DataFrame], info: Dict[str, Any]) -> None:
    jg = tablas.get('jugadoresAgregado', pd.DataFrame())
    conteos_local = conteos_desde_jugadores_agregado(jg, 'LOCAL')
    conteos_visit = conteos_desde_jugadores_agregado(jg, 'VISITANTE')
    tot_local = totales_raw_equipo(conteos_local)
    tot_visit = totales_raw_equipo(conteos_visit)
    av_local = calcular_avanzadas_equipo(tot_local, tot_visit)
    av_visit = calcular_avanzadas_equipo(tot_visit, tot_local)

    df_equipo = pd.DataFrame({
        'Métrica': _ORDEN_METRICAS_AVANZADAS,
        info['local_name']: [_fmt_valor_avanzada(m, av_local.get(m, 0.0)) for m in _ORDEN_METRICAS_AVANZADAS],
        info['visitante_name']: [_fmt_valor_avanzada(m, av_visit.get(m, 0.0)) for m in _ORDEN_METRICAS_AVANZADAS],
    })
    df_equipo.to_excel(writer, sheet_name='Avanzadas_Equipo', index=False)

    for conteos, nombre_hoja in ((conteos_local, 'Avanzadas_Jug_Local'), (conteos_visit, 'Avanzadas_Jug_Visit')):
        jug = calcular_avanzadas_jugador(conteos)
        if jug.empty:
            jug = pd.DataFrame({'Aviso': ['Sin datos disponibles.']})
        else:
            jug = jug.copy()
            for c in ['3p/FG%', 'eFG%', 'TS%', 'FT%']:
                jug[c] = (jug[c] * 100).round(1)
            jug = jug.sort_values('TS%', ascending=False)
        jug.to_excel(writer, sheet_name=nombre_hoja, index=False)


def _hoja_definiciones(writer: "pd.ExcelWriter") -> None:
    definiciones_dataframe().to_excel(writer, sheet_name='Definiciones', index=False)


def _construir_excel(tablas: Dict[str, pd.DataFrame]) -> bytes:
    info = _resumen_basico(tablas)
    qg = tablas.get('quintetosAgregado', pd.DataFrame())

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        _hoja_resumen(writer, tablas, info)
        _hoja_jugadores(writer, tablas.get('estadisticas_equipolocal', pd.DataFrame()), 'Jugadores_Local')
        _hoja_jugadores(writer, tablas.get('estadisticas_equipovisitante', pd.DataFrame()), 'Jugadores_Visitante')
        _hoja_quintetos(writer, qg, 'LOCAL', 'Quintetos_Local')
        _hoja_quintetos(writer, qg, 'VISITANTE', 'Quintetos_Visitante')
        _hoja_posesion(writer, tablas, info)
        _hoja_avanzadas(writer, tablas, info)
        _hoja_definiciones(writer)
    buf.seek(0)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _construir_excel_cacheado(partido_id: str, _tablas: Dict[str, pd.DataFrame]) -> bytes:
    # Igual que en pdf_export: el cache se indexa por partido_id, no por
    # `_tablas` (el guión bajo le dice a Streamlit que no lo hashee).
    return _construir_excel(_tablas)


def render_excel_button(tablas: Dict[str, pd.DataFrame], key: str) -> None:
    """Botón de descarga de un Excel con una hoja por cada área de la app."""
    part_df = tablas.get('partido', pd.DataFrame())
    partido_id = str(part_df.iloc[0].get('_id')) if (not part_df.empty and '_id' in part_df.columns) else 'partido'
    try:
        excel_bytes = _construir_excel_cacheado(partido_id, tablas)
        st.download_button(
            "📊 Descargar Excel",
            data=excel_bytes,
            file_name=f"estadisticas_{partido_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"download_excel_{key}",
        )
    except Exception as e:
        st.error(f"No se pudo generar el Excel: {e}")
