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

from .advanced_stats import calcular_avanzadas_equipo, totales_raw_equipo
from .colors import _parse_color, _text_color_for_bg
from .possession_stats import preparar_tiros, resumen_por_bucket
from .data_processing import BUCKET_A_REVISAR, BUCKETS_POSESION
from .tables import _build_table
from .utils import _first_col, _first_of

PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)
PAGE_MARGIN = 14 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * PAGE_MARGIN
_CELL_STYLE = ParagraphStyle('Celda', fontName='Helvetica', fontSize=6.5, leading=8, alignment=1)
_CELL_STYLE_LEFT = ParagraphStyle('CeldaIzq', parent=_CELL_STYLE, alignment=0)
_HEADER_STYLE_BASE = ParagraphStyle('Encabezado', fontName='Helvetica-Bold', fontSize=6.8, leading=8.2, alignment=1)
# Columnas que suelen tener texto largo (nombres de jugador, quintetos):
# reciben más ancho relativo que el resto para no forzar un wrap excesivo.
_COLUMNAS_ANCHAS = {'Nombre', 'Quinteto', 'Equipo', 'Jugador'}


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


def _color_visible(hex_color: str) -> str:
    """Evita líneas invisibles cuando el color de equipo es blanco (fondo de página)."""
    return '#cfd8dc' if str(hex_color).strip().lower() in ('#fff', '#ffffff', 'white') else hex_color


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
        ax.plot(x_period, y_local, color=_color_visible(info['color_local']), linewidth=1.8, label=info['local_name'])
        ax.plot(x_period, y_visit, color=_color_visible(info['color_visitante']), linewidth=1.8, label=info['visitante_name'])
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


def _col_widths(header: list, total_width: float = CONTENT_WIDTH) -> list:
    """Reparte el ancho disponible entre columnas sin que la tabla nunca supere
    el ancho de la página (la causa de que antes se vieran tablas cortadas)."""
    pesos = [2.2 if str(h).strip() in _COLUMNAS_ANCHAS else 1.0 for h in header]
    total_peso = sum(pesos) or 1.0
    return [total_width * p / total_peso for p in pesos]


def _tabla_estilo(header: list, rows: list, header_color: str, total_width: float = CONTENT_WIDTH) -> Table:
    header_style = ParagraphStyle('EncabezadoColor', parent=_HEADER_STYLE_BASE, textColor=rl_colors.HexColor(_text_color_for_bg(header_color)))
    data = [[Paragraph(str(h).replace('\n', '<br/>'), header_style) for h in header]]
    for row in rows:
        data.append([
            Paragraph(str(v), _CELL_STYLE_LEFT if header[i] in _COLUMNAS_ANCHAS else _CELL_STYLE)
            for i, v in enumerate(row)
        ])
    t = Table(data, colWidths=_col_widths(header, total_width), repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor(header_color)),
        ('GRID', (0, 0), (-1, -1), 0.4, rl_colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#f2f2f2')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    return t


def _boxscore_table(df_equipo: pd.DataFrame, header_color: str) -> Optional[Table]:
    df_show = _build_table(df_equipo)
    if df_show is None or df_show.empty:
        return None
    header = [str(c).replace('\n', ' ') for c in df_show.columns]
    rows = df_show.astype(str).values.tolist()
    return _tabla_estilo(header, rows, header_color)


def _totales_equipo(df: pd.DataFrame) -> Dict[str, float]:
    """Replica los totales de la sección 'Comparativas por variable' del Resumen."""
    if df is None or df.empty:
        return {}

    def total(col: Optional[str]) -> float:
        if not col or col not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[col], errors='coerce').fillna(0).sum())

    puntos_col = _first_col(df, ['puntos', 'pts'])
    can1, tir1 = _first_col(df, ['canasta1p', 'canastas1p', 'conv1p', 'convierte1p']), _first_col(df, ['tiro1p', 'tiros1p', 'int1p'])
    can2, tir2 = _first_col(df, ['canasta2p', 'canastas2p', 'conv2p']), _first_col(df, ['tiro2p', 'tiros2p', 'int2p'])
    can3, tir3 = _first_col(df, ['canasta3p', 'canastas3p', 'conv3p']), _first_col(df, ['tiro3p', 'tiros3p', 'int3p'])
    rd_col, ro_col = _first_col(df, ['rebotedefensivo']), _first_col(df, ['reboteofensivo'])
    ast_col = _first_col(df, ['asistencias'])
    per_col = _first_col(df, ['perdidas'])
    rec_col = _first_col(df, ['recuperaciones'])

    made1, att1 = total(can1), total(tir1)
    made2, att2 = total(can2), total(tir2)
    made3, att3 = total(can3), total(tir3)
    pct = lambda m, a: (m / a * 100.0) if a > 0 else 0.0
    rd, ro = total(rd_col), total(ro_col)
    return {
        'Puntos': total(puntos_col),
        '%1P': pct(made1, att1), '%2P': pct(made2, att2), '%3P': pct(made3, att3),
        'Reb. Def.': rd, 'Reb. Of.': ro, 'Reb. Tot.': rd + ro,
        'Asistencias': total(ast_col), 'Pérdidas': total(per_col), 'Recuperaciones': total(rec_col),
    }


def _grafico_comparativas(tot_local: Dict[str, float], tot_visit: Dict[str, float], info: Dict[str, Any]) -> Optional[io.BytesIO]:
    """Gráfico de barras Local vs Visitante por variable, igual al de la pestaña Resumen."""
    if not tot_local and not tot_visit:
        return None
    metricas = ['Puntos', '%1P', '%2P', '%3P', 'Reb. Def.', 'Reb. Of.', 'Reb. Tot.', 'Asistencias', 'Pérdidas', 'Recuperaciones']
    try:
        fig, axes = plt.subplots(2, 5, figsize=(11.6, 4.6), dpi=150)
        for ax, m in zip(axes.flat, metricas):
            vl, vv = tot_local.get(m, 0.0), tot_visit.get(m, 0.0)
            valores = [vl, vv]
            etiquetas = [info['local_name'][:12], info['visitante_name'][:12]]
            colores = [info['color_local'], info['color_visitante']]
            barras = ax.bar(etiquetas, valores, color=colores, edgecolor='black', linewidth=0.5)
            ax.set_title(m, fontsize=8.5)
            ax.tick_params(axis='x', labelsize=6.5, rotation=20)
            ax.tick_params(axis='y', labelsize=6.5)
            tope = max(valores + [1.0])
            ax.set_ylim(0, tope * 1.25)
            for barra, v in zip(barras, valores):
                etiqueta = f"{v:.0f}%" if m.startswith('%') else f"{v:.0f}"
                ax.text(barra.get_x() + barra.get_width() / 2, v + tope * 0.03, etiqueta, ha='center', fontsize=6.5, fontweight='bold')
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
        fig.suptitle('Comparativas por variable (Local vs Visitante)', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None


def _tabla_posesion(pbp_df: pd.DataFrame, info: Dict[str, Any]):
    """Tabla resumen de la pestaña Posesión (intentos/%/puntos por momento y equipo)."""
    d = preparar_tiros(pbp_df)
    if d.empty:
        return None, 0
    d = d.copy()
    d['Equipo'] = np.where(
        d['Condicion'] == 'LOCAL', info['local_name'],
        np.where(d['Condicion'] == 'VISITANTE', info['visitante_name'], 'Otro'),
    )
    d = d[d['Equipo'] != 'Otro']
    n_revisar = int((d['bucket_posesion'] == BUCKET_A_REVISAR).sum())
    d_validos = d[d['bucket_posesion'] != BUCKET_A_REVISAR]
    if d_validos.empty:
        return None, n_revisar

    resumen = resumen_por_bucket(d_validos, ['Equipo', 'bucket_posesion'])
    orden = {b: i for i, b in enumerate(BUCKETS_POSESION)}
    resumen = resumen.sort_values(
        ['Equipo', 'bucket_posesion'],
        key=lambda s: s.map(orden) if s.name == 'bucket_posesion' else s,
    )
    resumen = resumen.rename(columns={'bucket_posesion': 'Momento'})
    tabla = _tabla_estilo(resumen.columns.tolist(), resumen.astype(str).values.tolist(), '#455a64')
    return tabla, n_revisar


_METRICAS_PORCENTAJE_PDF = {
    '% Rebotes Defensivos', '% Rebotes Ofensivos', '% Rebotes Totales',
    '% Asistencias', '% Pérdidas', '% Robos', '% Bloqueos',
    '3p/FG', 'eFG%', 'TS%', 'FT%',
}
_ORDEN_METRICAS_AVANZADAS = [
    'Posesiones', 'Eficiencia Ofensiva', 'Eficiencia Defensiva', 'Net Rating',
    '% Rebotes Defensivos', '% Rebotes Ofensivos', '% Rebotes Totales',
    '% Asistencias', '% Pérdidas', '% Robos', '% Bloqueos', '3p/FG', 'eFG%', 'TS%', 'FT%',
]


def _tabla_avanzadas(est_loc_df: pd.DataFrame, est_vis_df: pd.DataFrame, info: Dict[str, Any]) -> Optional[Table]:
    if (est_loc_df is None or est_loc_df.empty) and (est_vis_df is None or est_vis_df.empty):
        return None
    tot_local = totales_raw_equipo(est_loc_df)
    tot_visit = totales_raw_equipo(est_vis_df)
    av_local = calcular_avanzadas_equipo(tot_local, tot_visit)
    av_visit = calcular_avanzadas_equipo(tot_visit, tot_local)

    def fmt(nombre, valor):
        if nombre in _METRICAS_PORCENTAJE_PDF:
            return f"{valor * 100:.1f}%"
        return f"{valor:.2f}"

    header = ['Métrica', info['local_name'], info['visitante_name']]
    rows = [[m, fmt(m, av_local.get(m, 0.0)), fmt(m, av_visit.get(m, 0.0))] for m in _ORDEN_METRICAS_AVANZADAS]
    return _tabla_estilo(header, rows, '#455a64')


def _top_quintetos(qg: pd.DataFrame, condicion: str, top_n: int = 8) -> pd.DataFrame:
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

    tot_local = _totales_equipo(est_loc_df)
    tot_visit = _totales_equipo(est_vis_df)
    comparativas_buf = _grafico_comparativas(tot_local, tot_visit, info)
    if comparativas_buf is not None:
        story.append(PageBreak())
        story.append(Image(comparativas_buf, width=250 * mm, height=99 * mm))

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

    tabla_posesion, n_revisar = _tabla_posesion(tablas.get('pbp', pd.DataFrame()), info)
    if tabla_posesion is not None:
        story.append(PageBreak())
        story.append(Paragraph("Tiros de campo según tiempo de posesión previo", title_style))
        story.append(Paragraph(
            "No incluye tiros libres. \"Reb. Of.\" son tiros tras un rebote ofensivo propio "
            "(no compiten contra un reloj de 24s nuevo).",
            normal,
        ))
        story.append(Spacer(1, 6))
        story.append(tabla_posesion)
        if n_revisar:
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                f"⚠ Se excluyeron {n_revisar} tiro(s) con más de 24s de posesión previa (dato imposible en "
                "básquet, probablemente un evento faltante en la carga del partido).",
                normal,
            ))

    tabla_av = _tabla_avanzadas(est_loc_df, est_vis_df, info)
    if tabla_av is not None:
        story.append(PageBreak())
        story.append(Paragraph("Estadísticas avanzadas", title_style))
        story.append(Paragraph(
            "Posesiones = Tiros de campo intentados + 0.44 × Tiros libres intentados − Rebotes ofensivos + "
            "Pérdidas. Eficiencia = puntos por posesión (no está multiplicada por 100).",
            normal,
        ))
        story.append(Spacer(1, 6))
        story.append(tabla_av)

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
