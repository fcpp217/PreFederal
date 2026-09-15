"""Generación de un informe en PDF, construido en el servidor.

El enfoque anterior capturaba la pantalla con html2canvas/jsPDF inyectados
en el DOM del documento padre de Streamlit (fuera del iframe del
componente). Ese approach es inherentemente frágil: depende de atributos
internos (`data-testid`) que cambian entre versiones de Streamlit, de
esperas fijas (`setTimeout`) para que el DOM termine de renderizar, de que
el navegador soporte ciertas funciones de `canvas`/CSS, y de que no haya
restricciones de CORS al cargar las librerías. Cualquiera de esos motivos
alcanza para que la descarga falle o produzca un PDF vacío o cortado.

Este módulo arma el PDF directamente a partir de los DataFrames ya
calculados (los mismos que alimentan la UI), por lo que el resultado es
determinístico y no depende del navegador, del layout ni del tema activo.
"""

import io
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .colors import _parse_color, _text_color_for_bg
from .tables import _build_table
from .utils import _first_of


def _resumen_basico(tablas: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Recalcula lo esencial del partido (nombres, colores, tanteo) para el PDF."""
    part_df = tablas.get('partido', pd.DataFrame())
    pbp_df = tablas.get('pbp', pd.DataFrame())
    est_loc_df = tablas.get('estadisticas_equipolocal', pd.DataFrame())
    est_vis_df = tablas.get('estadisticas_equipovisitante', pd.DataFrame())
    row = part_df.iloc[0] if not part_df.empty else {}

    local_name = str(_first_of(row, [
        'local', 'equipo_local', 'nombre_local', 'localnombre', 'nombreLocal', 'equipoLocal', 'club_local', 'clubLocal'
    ], ''))
    visitante_name = str(_first_of(row, [
        'visitante', 'equipo_visitante', 'nombre_visitante', 'visitantenombre', 'nombreVisitante', 'equipoVisitante', 'club_visitante', 'clubVisitante'
    ], ''))
    if (not local_name or not local_name.strip()) and not est_loc_df.empty:
        local_name = str(_first_of(est_loc_df.iloc[0], ['equipo', 'nombre_equipo'], 'Local'))
    if (not visitante_name or not visitante_name.strip()) and not est_vis_df.empty:
        visitante_name = str(_first_of(est_vis_df.iloc[0], ['equipo', 'nombre_equipo'], 'Visitante'))
    local_name = local_name or 'Local'
    visitante_name = visitante_name or 'Visitante'

    color_local = _parse_color(_first_of(row, ['color_local', 'local_color', 'colorLocal', 'colorlocal'], '#1f77b4'), '#1f77b4')
    color_visitante = _parse_color(_first_of(row, ['color_visitante', 'visitante_color', 'colorVisitante', 'colorvisitante'], '#ff7f0e'), '#ff7f0e')

    tanteo_local = _first_of(row, ['tanteo_local', 'puntos_local', 'marcador_local', 'score_local', 'tanteoLocal', 'marcadorLocal'], None)
    tanteo_visitante = _first_of(row, ['tanteo_visitante', 'puntos_visitante', 'marcador_visitante', 'score_visitante', 'tanteoVisitante', 'marcadorVisitante'], None)

    per_local: Dict[int, int] = {}
    per_visit: Dict[int, int] = {}
    dfp = pd.DataFrame()
    if not pbp_df.empty:
        dfp = pbp_df.copy()
        if 'autoincremental_id' in dfp.columns:
            dfp['autoincremental_id_num'] = pd.to_numeric(dfp['autoincremental_id'], errors='coerce').fillna(0)
            dfp = dfp.sort_values(by=['_id', 'numero_periodo', 'autoincremental_id_num'])
        elif 'tiempo_segundos' in dfp.columns:
            dfp = dfp.sort_values(by=['_id', 'numero_periodo', 'tiempo_segundos'])
        dfp = dfp.reset_index(drop=True)

        if (tanteo_local is None or str(tanteo_local).strip() == '') and not dfp.empty:
            tanteo_local = dfp.iloc[-1].get('puntosLocal', tanteo_local)
        if (tanteo_visitante is None or str(tanteo_visitante).strip() == '') and not dfp.empty:
            tanteo_visitante = dfp.iloc[-1].get('puntosVisitante', tanteo_visitante)

        if 'numero_periodo' in dfp.columns:
            for per in sorted(pd.to_numeric(dfp['numero_periodo'], errors='coerce').dropna().astype(int).unique().tolist()):
                sub = dfp[pd.to_numeric(dfp['numero_periodo'], errors='coerce') == per]
                if not sub.empty:
                    last = sub.iloc[-1]
                    per_local[per] = int(float(pd.to_numeric(last.get('puntosLocal', 0), errors='coerce') or 0))
                    per_visit[per] = int(float(pd.to_numeric(last.get('puntosVisitante', 0), errors='coerce') or 0))

    def _to_int(v: Any) -> Optional[int]:
        try:
            if v is None or str(v).strip() == '':
                return None
            return int(float(v))
        except Exception:
            return None

    return {
        'local_name': local_name,
        'visitante_name': visitante_name,
        'color_local': color_local,
        'color_visitante': color_visitante,
        'tanteo_local': _to_int(tanteo_local),
        'tanteo_visitante': _to_int(tanteo_visitante),
        'per_local': per_local,
        'per_visit': per_visit,
        'pbp_df': dfp,
    }


def _grafico_evolucion(dfp: pd.DataFrame, info: Dict[str, Any]) -> Optional[io.BytesIO]:
    """Gráfico de evolución del marcador (equivalente simplificado al de la pestaña Resumen)."""
    if dfp is None or dfp.empty or 'puntosLocal' not in dfp.columns:
        return None
    try:
        if 'x_period' not in dfp.columns:
            orden = np.arange(len(dfp))
            tiempo_num = pd.to_numeric(dfp.get('tiempo_segundos', np.nan), errors='coerce').fillna(orden)
            periodo_num = pd.to_numeric(dfp.get('numero_periodo', 1), errors='coerce').fillna(1)
            x_period = np.where(
                periodo_num <= 4,
                (600 - tiempo_num) + (periodo_num - 1) * 600,
                2400 + (300 - tiempo_num) + (periodo_num - 5) * 300,
            )
        else:
            x_period = pd.to_numeric(dfp['x_period'], errors='coerce')

        y_local = pd.to_numeric(dfp['puntosLocal'], errors='coerce').fillna(0)
        y_visit = pd.to_numeric(dfp.get('puntosVisitante', 0), errors='coerce').fillna(0)

        fig, ax = plt.subplots(figsize=(9.5, 3.3), dpi=150)
        ax.plot(x_period, y_local, color=info['color_local'], linewidth=1.8, label=info['local_name'])
        ax.plot(x_period, y_visit, color=info['color_visitante'], linewidth=1.8, label=info['visitante_name'])
        for per_num in sorted(pd.to_numeric(dfp.get('numero_periodo', pd.Series(dtype=float)), errors='coerce').dropna().unique().tolist())[:-1]:
            ax.axvline(x=per_num * 600, color='#999999', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Tiempo de partido (s)')
        ax.set_ylabel('Puntos acumulados')
        ax.set_title('Evolución del marcador')
        ax.legend(loc='upper left', fontsize=8, frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None


def _tabla_estilo(header: list, rows: list, header_color: str) -> Table:
    data = [header] + rows
    t = Table(data, repeatRows=1)
    header_text_color = rl_colors.HexColor(_text_color_for_bg(header_color))
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor(header_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), header_text_color),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.4, rl_colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#f2f2f2')]),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return t


def _boxscore_table(df_equipo: pd.DataFrame, header_color: str) -> Optional[Table]:
    df_show = _build_table(df_equipo)
    if df_show is None or df_show.empty:
        return None
    header = [str(c).replace('\n', ' ') for c in df_show.columns]
    rows = df_show.astype(str).values.tolist()
    return _tabla_estilo(header, rows, header_color)


def _top_quintetos(qg: pd.DataFrame, condicion: str, top_n: int = 5) -> pd.DataFrame:
    if qg is None or qg.empty or 'Condicion' not in qg.columns or 'quinteto' not in qg.columns:
        return pd.DataFrame()
    sub = qg[qg['Condicion'].astype(str).str.upper() == condicion].copy()
    if sub.empty:
        return pd.DataFrame()
    sub['quinteto_str'] = sub['quinteto'].apply(lambda q: ', '.join(q) if isinstance(q, (list, tuple)) else str(q))
    for col in ['tiempo_jugado', 'puntos_favor', 'puntos_contra']:
        if col not in sub.columns:
            sub[col] = 0
    agg = sub.groupby('quinteto_str', as_index=False)[['tiempo_jugado', 'puntos_favor', 'puntos_contra']].sum()
    agg = agg.sort_values('tiempo_jugado', ascending=False).head(top_n)
    agg['Tiempo'] = agg['tiempo_jugado'].apply(lambda s: f"{int(s // 60)}:{int(s % 60):02d}")
    agg['+/-'] = (agg['puntos_favor'] - agg['puntos_contra']).astype(int)
    agg['puntos_favor'] = agg['puntos_favor'].astype(int)
    agg['puntos_contra'] = agg['puntos_contra'].astype(int)
    return agg.rename(columns={
        'quinteto_str': 'Quinteto', 'puntos_favor': 'PF', 'puntos_contra': 'PC',
    })[['Quinteto', 'Tiempo', 'PF', 'PC', '+/-']]


def _construir_pdf(tablas: Dict[str, pd.DataFrame]) -> bytes:
    info = _resumen_basico(tablas)
    est_loc_df = tablas.get('estadisticas_equipolocal', pd.DataFrame())
    est_vis_df = tablas.get('estadisticas_equipovisitante', pd.DataFrame())
    qg = tablas.get('quintetosAgregado', pd.DataFrame())

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=landscape(A4),
        leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm, bottomMargin=12 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TituloInforme', parent=styles['Title'], fontSize=18, spaceAfter=4)
    h2 = ParagraphStyle('Subtitulo', parent=styles['Heading2'], spaceBefore=10, spaceAfter=4)
    normal = styles['Normal']

    story = []
    tl = info['tanteo_local'] if info['tanteo_local'] is not None else '-'
    tv = info['tanteo_visitante'] if info['tanteo_visitante'] is not None else '-'
    story.append(Paragraph("Estadísticas del partido", title_style))
    story.append(Paragraph(f"{info['local_name']} {tl}  -  {tv} {info['visitante_name']}", h2))

    periodos = sorted(set(info['per_local']) | set(info['per_visit']))
    if periodos:
        header = ['Equipo'] + [f"P{p}" for p in periodos]
        fila_local = [info['local_name']] + [str(info['per_local'].get(p, '-')) for p in periodos]
        fila_visit = [info['visitante_name']] + [str(info['per_visit'].get(p, '-')) for p in periodos]
        story.append(_tabla_estilo(header, [fila_local, fila_visit], '#e0e0e0'))

    chart_buf = _grafico_evolucion(info['pbp_df'], info)
    if chart_buf is not None:
        story.append(Spacer(1, 8))
        story.append(Image(chart_buf, width=250 * mm, height=87 * mm))

    for titulo, df_equipo, color in (
        (f"Estadísticas por jugador - LOCAL ({info['local_name']})", est_loc_df, info['color_local']),
        (f"Estadísticas por jugador - VISITANTE ({info['visitante_name']})", est_vis_df, info['color_visitante']),
    ):
        story.append(Spacer(1, 10))
        story.append(Paragraph(titulo, h2))
        tabla = _boxscore_table(df_equipo, color)
        story.append(tabla if tabla is not None else Paragraph("Sin datos disponibles.", normal))

    top_loc = _top_quintetos(qg, 'LOCAL')
    top_vis = _top_quintetos(qg, 'VISITANTE')
    if not top_loc.empty or not top_vis.empty:
        story.append(PageBreak())
        story.append(Paragraph("Quintetos con más minutos en cancha", title_style))
        for titulo, df_top, color in (
            (f"LOCAL ({info['local_name']})", top_loc, info['color_local']),
            (f"VISITANTE ({info['visitante_name']})", top_vis, info['color_visitante']),
        ):
            if df_top.empty:
                continue
            story.append(Paragraph(titulo, h2))
            story.append(_tabla_estilo(df_top.columns.tolist(), df_top.astype(str).values.tolist(), color))
            story.append(Spacer(1, 10))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _construir_pdf_cacheado(partido_id: str, _tablas: Dict[str, pd.DataFrame]) -> bytes:
    # `_tablas` (con guión bajo) le indica a Streamlit que no intente hashear
    # el diccionario de DataFrames: el cache se indexa solo por partido_id,
    # que ya identifica unívocamente a `tablas` dentro de la sesión.
    return _construir_pdf(_tablas)


def render_pdf_button(tablas: Dict[str, pd.DataFrame], key: str) -> None:
    """Botón de descarga de un informe PDF generado en el servidor.

    A diferencia del botón anterior (que intentaba fotografiar el DOM del
    navegador), este arma el PDF a partir de los datos ya procesados, por
    lo que el resultado no depende del estado visual de la página ni de
    scripts de terceros cargados en el navegador del usuario.
    """
    part_df = tablas.get('partido', pd.DataFrame())
    partido_id = str(part_df.iloc[0].get('_id')) if (not part_df.empty and '_id' in part_df.columns) else 'partido'
    try:
        pdf_bytes = _construir_pdf_cacheado(partido_id, tablas)
        st.download_button(
            "📄 Descargar PDF",
            data=pdf_bytes,
            file_name=f"estadisticas_{partido_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"download_pdf_{key}",
        )
    except Exception as e:
        st.error(f"No se pudo generar el PDF: {e}")
