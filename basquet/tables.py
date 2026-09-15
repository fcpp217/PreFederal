"""Construcción de la tabla de estadísticas por jugador para su despliegue en UI."""

import re

import pandas as pd
import streamlit as st

from .utils import _first_col, _num

def _build_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = df.columns.tolist()
    # identificar bloque desde rebotedefensivo (case-insensitive)
    start_idx = None
    for i, c in enumerate(cols):
        if str(c).lower().startswith('rebotedefensivo'):
            start_idx = i
            break
    tail_cols = cols[start_idx:] if start_idx is not None else []
    # excluir tapones
    tail_cols = [c for c in tail_cols if str(c).lower() not in (
        'taponescometidos', 'taponesrecibidos',
        'tiro1fallado','tiro2fallado','tiro3fallado',
        'tiro1p','tiro2p','tiro3p',
        'dorsal',
        'tiempo_jugado'
    )]

    # columnas base
    nombre_col = _first_col(df, ['nombre', 'jugador', 'nombre_jugador'])
    dorsal_col = None  # ocultar dorsal en tabla
    puntos_col = _first_col(df, ['puntos', 'pts'])

    can1_col = _first_col(df, ['canasta1p', 'canastas1p', 'conv1p', 'convierte1p'])
    tir1_col = _first_col(df, ['tiro1p', 'tiros1p', 'int1p'])
    can2_col = _first_col(df, ['canasta2p', 'canastas2p', 'conv2p'])
    tir2_col = _first_col(df, ['tiro2p', 'tiros2p', 'int2p'])
    can3_col = _first_col(df, ['canasta3p', 'canastas3p', 'conv3p'])
    tir3_col = _first_col(df, ['tiro3p', 'tiros3p', 'int3p'])
    tiempo_col = _first_col(df, ['tiempo_jugado', 'tiempoJugado', 'Tiempo Jugado', 'tiempo'])

    rows_out = []
    for _, r in df.iterrows():
        made1 = _num(df, r, can1_col)
        # Attempts already include made + missed; do not add made again
        att1 = _num(df, r, tir1_col)
        made2 = _num(df, r, can2_col)
        att2 = _num(df, r, tir2_col)
        made3 = _num(df, r, can3_col)
        att3 = _num(df, r, tir3_col)

        pct = lambda m, a: (m / a * 100.0) if a > 0 else 0.0

        row_out = {
            'nombre': r.get(nombre_col, ''),
            'puntos': int(round(_num(df, r, puntos_col))),
            '1P': f"{int(made1)}/{int(att1)}",
            '%1P': f"{pct(made1, att1):.0f}%",
            '2P': f"{int(made2)}/{int(att2)}",
            '%2P': f"{pct(made2, att2):.0f}%",
            '3P': f"{int(made3)}/{int(att3)}",
            '%3P': f"{pct(made3, att3):.0f}%",
        }
        # tiempo jugado en segundos -> mm:ss
        if tiempo_col is not None and tiempo_col in df.columns:
            # Asegurar segundos numéricos
            val = r.get(tiempo_col)
            seg = pd.to_numeric(val, errors='coerce')
            if pd.isna(seg):
                try:
                    seg = float(str(val).strip())
                except Exception:
                    seg = 0
            mins = int(seg // 60)
            secs = int(seg % 60)
            row_out['tiempo_jugado'] = f"{mins}:{secs:02d}"
        # asegurar presencia de métricas clave aunque no estén en tail
        for extra in ['rebotedefensivo','reboteofensivo','rebotetotal','asistencias','perdidas','recuperaciones','+-']:
            if extra in df.columns:
                row_out[extra] = r.get(extra)
        # anexar métricas desde rebotedefensivo en adelante
        for c in tail_cols:
            row_out[c] = r.get(c)
        rows_out.append(row_out)

    # ordenar columnas: base primero, luego tail
    # Orden base y extras solicitados (Tiempo Jugado va luego de Nombre)
    base_order = ['nombre', 'tiempo_jugado', 'puntos', '1P', '%1P', '2P', '%2P', '3P', '%3P']
    # Reordenar cola: def, of, total, asistencias, perdidas, recuperaciones, '+-' y luego el resto
    preferred_tail = ['rebotedefensivo', 'reboteofensivo', 'rebotetotal', 'asistencias', 'perdidas', 'recuperaciones', '+-']
    rest_tail = [c for c in tail_cols if c not in preferred_tail]
    final_cols = base_order + preferred_tail + [c for c in rest_tail if c not in base_order]
    out_df = pd.DataFrame(rows_out)
    # mantener solo columnas existentes
    final_cols = [c for c in final_cols if c in out_df.columns]
    out_df = out_df[final_cols]

    # Renombrar columnas a nombres amigables, usando saltos de línea
    rename_map = {
        'nombre': 'Nombre',
        'puntos': 'Puntos',
        '1P': '1P\n(conv/att)',
        '%1P': '%1P',
        '2P': '2P\n(conv/att)',
        '%2P': '%2P',
        '3P': '3P\n(conv/att)',
        '%3P': '%3P',
        'tiempo_jugado': 'Tiempo\nJugado',
    }
    # métricas desde rebotes con nombres más legibles
    def prettify_tail(c: str) -> str:
        cl = str(c).lower()
        mapping = {
            'rebotedefensivo': 'Rebote\nDef.',
            'reboteofensivo': 'Rebote\nOf.',
            'rebotetotal': 'Rebotes\nTotales',
            'asistencias': 'Asist.',
            'perdidas': 'Pérdidas',
            'recuperaciones': 'Recup.',
            'faltacomendida': 'Falta\nCometida',
            'faltarecibida': 'Falta\nRecibida',
        }
        for k, v in mapping.items():
            if cl.startswith(k):
                return v
        # Title Case básico con saltos para palabras largas
        base = re.sub(r'[_]+', ' ', c).title()
        return base
    for c in tail_cols:
        rename_map[c] = prettify_tail(c)
    out_df = out_df.rename(columns=rename_map)
    # Convertir numéricos a enteros en texto (excepto % columnas y Tiempo Jugado ya formateado)
    for c in out_df.columns:
        if c == 'Nombre' or c.startswith('%') or c in ('Tiempo\nJugado',):
            continue
        try:
            out_df[c] = out_df[c].apply(lambda v: str(int(round(float(v)))) if pd.notna(v) and str(v).strip() != '' else '')
        except Exception:
            out_df[c] = out_df[c].astype(str)
    return out_df

def build_column_config(df_show: pd.DataFrame):
    cfg = {}
    for col in df_show.columns:
        if col == 'Nombre':
            cfg[col] = st.column_config.TextColumn(width=220)
        elif col == 'Quinteto':
            cfg[col] = st.column_config.TextColumn(width=480)
        elif col == 'Titular':
            # Sin título y bien estrecha para mostrar solo la estrella
            cfg[col] = st.column_config.TextColumn(label="", width=40)
        elif col == 'Marca':
            # Columna de símbolos (quintetos): sin título y angosta
            cfg[col] = st.column_config.TextColumn(label="", width=40)
        else:
            cfg[col] = st.column_config.TextColumn(width=80)
    return cfg
