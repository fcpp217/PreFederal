import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import requests
import streamlit as st
import re
import altair as alt

# Variable global para debug de nombres corregidos
df_nombres_corregidos = pd.DataFrame()

# ------------------------------
# Config UI
# ------------------------------
st.set_page_config(page_title="Estadística", layout="wide")

# ------------------------------
# Utilidades (adaptadas de sqlite_loader/download_to_sqlite.py)
# ------------------------------
PARTIDO_URL = "https://appaficioncabb.indalweb.net/envivonavegador/partido.ashx"
ESTADISTICAS_URL = "https://appaficioncabb.indalweb.net/envivonavegador/estadisticas.ashx"


# ------------------------------
# Helpers UI
# ------------------------------
def _parse_color(c: Any, fallback: str) -> str:
    try:
        s = str(c).strip()
        if not s:
            return fallback
        # Mapeo de nombres en español a HEX
        name = s.lower().replace('ó','o').replace('á','a').replace('é','e').replace('í','i').replace('ú','u').strip()
        es_map = {
            'negro': '#000000',
            'blanco': '#ffffff',
            'rojo': '#e53935',
            'azul': '#1e88e5',
            'verde': '#43a047',
            'amarillo': '#fdd835',
            'naranja': '#fb8c00',
            'violeta': '#8e24aa',
            'morado': '#6a1b9a',
            'gris': '#9e9e9e',
            'gris claro': '#cfd8dc',
            'gris oscuro': '#616161',
            'celeste': '#03a9f4',
            'cian': '#00bcd4',
            'turquesa': '#26a69a',
            'bordo': '#7b1fa2',
            'granate': '#800000',
            'marron': '#6d4c41',
            'cafe': '#6d4c41',
            'rosa': '#ec407a',
            'magenta': '#d81b60',
            'lima': '#c0ca33',
            'oliva': '#827717',
            'dorado': '#b8860b',
            'plateado': '#b0bec5',
        }
        if name in es_map:
            return es_map[name]
        # Aceptar hex sin '#'
        if re.fullmatch(r"[0-9A-Fa-f]{6}", s):
            return f"#{s}"
        # Aceptar hex con '#'
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", s):
            return s
        # Cualquier otro formato no-hex: forzar fallback para asegurar contraste visible
        return fallback
    except Exception:
        return fallback


def _text_color_for_bg(hex_color: str) -> str:
    # Contraste WCAG aproximado para elegir blanco/negro
    try:
        c = hex_color.lstrip('#')
        if len(c) != 6:
            return "#ffffff"
        r = int(c[0:2], 16) / 255.0
        g = int(c[2:4], 16) / 255.0
        b = int(c[4:6], 16) / 255.0
        # luminancia relativa
        def lin(u: float) -> float:
            return u / 12.92 if u <= 0.03928 else ((u + 0.055) / 1.055) ** 2.4
        L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
        return "#000000" if (L > 0.55) else "#ffffff"
    except Exception:
        return "#ffffff"


def _clamp(v: int) -> int:
    return max(0, min(255, v))


def _adjust_color(hex_color: str, factor: float) -> str:
    """Aclarar u oscurecer un color HEX. factor>1 aclara, <1 oscurece."""
    try:
        c = hex_color.lstrip('#')
        if len(c) != 6:
            return hex_color
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)
        r = _clamp(int(r * factor))
        g = _clamp(int(g * factor))
        b = _clamp(int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color


def _first_of(row: Any, keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in row:
            v = row.get(k)
            # Considerar vacío como faltante
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            if isinstance(v, str) and v.strip() == "":
                continue
            return v
    return default


def to_seconds(tiempo_str: Any) -> Optional[float]:
    try:
        if tiempo_str is None:
            return None
        s = str(tiempo_str).strip()
        parts = s.split(":")
        if len(parts) == 3:
            minutos = int(parts[1])
            segundos = float(parts[2])
        elif len(parts) == 2:
            minutos = int(parts[0])
            segundos = float(parts[1])
        else:
            return None
        return minutos * 60 + segundos
    except Exception:
        return None


def puntos_canasta(accion: Any) -> int:
    if accion == "CANASTA-1P":
        return 1
    if accion == "CANASTA-2P":
        return 2
    if accion == "CANASTA-3P":
        return 3
    return 0

# ------------------------------
# Helpers para construir tabla de Estadística
# ------------------------------
def _stay_estadistica():
    try:
        st.session_state['force_estadistica'] = True
    except Exception:
        pass
def _first_col(df: pd.DataFrame, opts: List[str]) -> Optional[str]:
    for o in opts:
        if o in df.columns:
            return o
    return None

def _num(df: pd.DataFrame, row: Any, colname: Optional[str]) -> float:
    if not colname:
        return 0.0
    try:
        return float(pd.to_numeric(row.get(colname), errors='coerce') or 0)
    except Exception:
        return 0.0

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


def agregar_equipo_condicion(df_jugadas: pd.DataFrame, df_partidos: pd.DataFrame) -> pd.DataFrame:
    if df_jugadas.empty:
        return df_jugadas
    cols_needed = ["_id", "idlocal", "idvisitante", "local", "visitante"]
    cols_needed = [c for c in cols_needed if c in df_partidos.columns]
    df_merge = df_jugadas.merge(df_partidos[cols_needed], on="_id", how="left")
    for col in ["local", "visitante"]:
        if col in df_merge.columns:
            df_merge[col] = df_merge[col].astype("string").fillna("").str.strip()
    for col in ["equipo_id", "idlocal", "idvisitante"]:
        if col in df_merge.columns:
            df_merge[col] = df_merge[col].astype("string").fillna("").str.strip()
    df_merge["equipo"] = np.where(
        df_merge["equipo_id"].str.upper() == "-1",
        "SIN EQUIPO",
        np.where(
            df_merge["equipo_id"] == df_merge.get("idlocal", ""),
            df_merge.get("local", ""),
            np.where(
                df_merge["equipo_id"] == df_merge.get("idvisitante", ""),
                df_merge.get("visitante", ""),
                ""
            )
        )
    )
    df_merge["Condicion"] = np.where(
        df_merge["equipo_id"].str.upper() == "-1",
        "NEUTRAL",
        np.where(
            df_merge["equipo_id"] == df_merge.get("idlocal", ""),
            "LOCAL",
            np.where(
                df_merge["equipo_id"] == df_merge.get("idvisitante", ""),
                "VISITANTE",
                ""
            )
        )
    )
    df_jugadas["equipo"] = df_merge["equipo"]
    df_jugadas["Condicion"] = df_merge["Condicion"]
    return df_jugadas


def fetch_partido(partido_id: str) -> Dict[str, Any]:
    resp = requests.post(PARTIDO_URL, data={'id_partido': str(partido_id)}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_estadisticas(partido_id: str) -> Dict[str, Any]:
    resp = requests.post(ESTADISTICAS_URL, data={'id_partido': str(partido_id)}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ------------------------------
# Transformaciones in-memory (sin SQLite)
# ------------------------------

def procesar_pbp_y_agregados(
    partido_df: pd.DataFrame,
    jugada_df: pd.DataFrame,
    estadisticas_local_df: pd.DataFrame,
    estadisticas_visitante_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Preparación jugadas
    df_jug = jugada_df.copy()
    if df_jug.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Enriquecer con equipo/condición
    df_par = partido_df[[c for c in ["_id", "idlocal", "idvisitante", "local", "visitante"] if c in partido_df.columns]]
    df_jug = agregar_equipo_condicion(df_jug, df_par)

    # Nombres de jugador por merge con estadisticas de equipos
    df_loc = estadisticas_local_df.copy()
    df_vis = estadisticas_visitante_df.copy()
    for col in ["_id", "equipo_id", "dorsal"]:
        if col in df_jug.columns:
            df_jug[col] = df_jug[col].astype(str).fillna("").str.strip()
    for df_stats in [df_loc, df_vis]:
        for col in ["_id", "idequipo", "dorsal", "nombre"]:
            if col in df_stats.columns:
                df_stats[col] = df_stats[col].astype(str).fillna("").str.strip()
    left_cols = ["_id", "equipo_id", "dorsal"]
    right_cols = ["_id", "idequipo", "dorsal", "nombre"]
    if all(c in df_jug.columns for c in left_cols) and all(c in df_loc.columns for c in right_cols):
        df_loc_merge = df_jug.merge(
            df_loc[right_cols],
            left_on=["_id", "equipo_id", "dorsal"],
            right_on=["_id", "idequipo", "dorsal"],
            how="left",
        ).rename(columns={"nombre": "nombre_local"})
    else:
        df_loc_merge = df_jug.copy()
        df_loc_merge["nombre_local"] = None
    if all(c in df_jug.columns for c in left_cols) and all(c in df_vis.columns for c in right_cols):
        df_vis_merge = df_jug.merge(
            df_vis[right_cols],
            left_on=["_id", "equipo_id", "dorsal"],
            right_on=["_id", "idequipo", "dorsal"],
            how="left",
        ).rename(columns={"nombre": "nombre_visitante"})
    else:
        df_vis_merge = df_jug.copy()
        df_vis_merge["nombre_visitante"] = None

    df_jug["nombre_local"] = df_loc_merge.get("nombre_local")
    df_jug["nombre_visitante"] = df_vis_merge.get("nombre_visitante")
    # Tomar SIEMPRE el nombre que venga (aunque sea 'NOMBRE') y luego concatenar con el dorsal.
    # Construir como Series (evitar ndarray de np.where que no tiene fillna/str)
    nombre_local = df_jug["nombre_local"].astype(str) if "nombre_local" in df_jug.columns else pd.Series([""]*len(df_jug), index=df_jug.index)
    nombre_visit = df_jug["nombre_visitante"].astype(str) if "nombre_visitante" in df_jug.columns else pd.Series([""]*len(df_jug), index=df_jug.index)
    base_nombre = nombre_local.str.strip()
    mask_empty = (base_nombre == "") | base_nombre.isna() | (base_nombre.str.lower() == "nan")
    base_nombre = base_nombre.where(~mask_empty, nombre_visit.str.strip())
    # Si equipo_id == -1 => SIN JUGADOR
    if "equipo_id" in df_jug.columns:
        eqid = df_jug["equipo_id"]
        sin_mask = (eqid == -1) | (eqid.astype(str).str.strip() == "-1")
        base_nombre = base_nombre.where(~sin_mask, "SIN JUGADOR")
    df_jug["nombre"] = base_nombre.fillna("").str.strip()
    if "dorsal" in df_jug.columns:
        dorsal_str = df_jug["dorsal"].astype(str).fillna("").str.strip().str.zfill(2)
        df_jug["nombre"] = (dorsal_str + "-" + df_jug["nombre"]).str.strip("-")
    drop_cols = [c for c in ["nombre_local", "nombre_visitante"] if c in df_jug.columns]
    if drop_cols:
        df_jug.drop(columns=drop_cols, inplace=True)

    # Orden y acumulados
    if "autoincremental_id" in df_jug.columns:
        df_jug["autoincremental_id_num"] = pd.to_numeric(df_jug["autoincremental_id"], errors='coerce').fillna(0)
        df = df_jug.sort_values(by=["_id", "autoincremental_id_num"]).reset_index(drop=True)
    else:
        tmp = df_jug.copy()
        if "tiempo_partido" in tmp.columns:
            tmp["tiempo_segundos"] = tmp["tiempo_partido"].apply(to_seconds)
        sort_cols = [c for c in ["_id", "numero_periodo"] if c in tmp.columns]
        if "tiempo_segundos" in tmp.columns:
            df = tmp.sort_values(by=sort_cols + ["tiempo_segundos"], ascending=[True, True, False]).reset_index(drop=True)
        else:
            df = tmp.sort_values(by=sort_cols).reset_index(drop=True)

    for col in ["puntosLocal", "puntosVisitante", "DifPuntos"]:
        df[col] = 0
    if "accion_tipo" in df.columns and "Condicion" in df.columns:
        for pid, grupo in df.groupby("_id"):
            puntos_local = 0
            puntos_visitante = 0
            for i in grupo.index:
                accion = df.loc[i, "accion_tipo"]
                condicion = df.loc[i, "Condicion"]
                pts = puntos_canasta(accion)
                if condicion == "LOCAL":
                    puntos_local += pts
                elif condicion == "VISITANTE":
                    puntos_visitante += pts
                df.at[i, "puntosLocal"] = puntos_local
                df.at[i, "puntosVisitante"] = puntos_visitante
                df.at[i, "DifPuntos"] = puntos_local - puntos_visitante

    def clasificar(row):
        dif = row.get("DifPuntos", 0)
        cond = row.get("Condicion", "")
        if cond == "LOCAL":
            if dif < -5:
                return "Perdiendo 5+"
            elif dif > 5:
                return "Ganando 5+"
        elif cond == "VISITANTE":
            if dif > 5:
                return "Perdiendo 5+"
            elif dif < -5:
                return "Ganando 5+"
        return "+-5"

    df["SituacionMarcador"] = df.apply(clasificar, axis=1)

    if "tiempo_partido" in df.columns:
        df["tiempo_segundos"] = df["tiempo_partido"].apply(to_seconds)
        df["ultimos_dos_minutos"] = df["tiempo_segundos"].apply(lambda x: "Últimos 2 min" if (x is not None and x <= 120) else "Primeros 8 min")

    # Quintetos + agregados
    if all(c in df.columns for c in ["_id", "numero_periodo", "accion_tipo", "nombre", "Condicion", "tiempo_segundos", "puntosLocal", "puntosVisitante"]):
        if "autoincremental_id_num" not in df.columns and "autoincremental_id" in df.columns:
            df["autoincremental_id_num"] = pd.to_numeric(df["autoincremental_id"], errors='coerce').fillna(0)
        order_cols = ["_id", "numero_periodo"] + (["autoincremental_id_num"] if "autoincremental_id_num" in df.columns else ["tiempo_segundos"])
        df = df.sort_values(by=order_cols, ascending=[True, True, True]).reset_index(drop=True)
        quintetos_local: List[List[str]] = []
        quintetos_visitante: List[List[str]] = []
        partido_actual = None
        periodo_actual = None
        q_local: List[str] = []
        q_visit: List[str] = []
        token_display: Dict[str, str] = {}
        for _, row in df.iterrows():
            pid = row["_id"]
            per_raw = row.get("numero_periodo")
            try:
                per = int(per_raw) if pd.notna(per_raw) else -1
            except Exception:
                per = -1
            accion = row["accion_tipo"]
            accion_s = "" if pd.isna(accion) else str(accion)
            jugador_nombre = row.get("nombre")
            jugador_nombre_s = "" if (jugador_nombre is None or (hasattr(pd, 'isna') and pd.isna(jugador_nombre))) else str(jugador_nombre)
            cond_raw = row.get("Condicion")
            cond = "" if (cond_raw is None or (isinstance(cond_raw, float) and pd.isna(cond_raw)) or (hasattr(pd, 'isna') and pd.isna(cond_raw))) else str(cond_raw)
            dorsal_val = row.get("dorsal")
            eq_val = row.get("equipo_id")
            dorsal_str = str(dorsal_val).strip() if dorsal_val is not None else ""
            eq_str = str(eq_val).strip() if eq_val is not None else ""
            token = (eq_str + "|" + dorsal_str) if (eq_str or dorsal_str) else (str(jugador_nombre) or "")
            name_clean = jugador_nombre_s.strip()
            if name_clean and re.match(r"^\d{1,2}-", name_clean):
                m1 = re.match(r"^(\d{1,2})-\1-(.*)$", name_clean)
                if m1:
                    display = f"{m1.group(1)}-{m1.group(2)}".strip("-")
                else:
                    m2 = re.match(r"^(\d{1,2})-\1$", name_clean)
                    display = m2.group(1) if m2 else name_clean
            else:
                if dorsal_str:
                    base = dorsal_str.zfill(2)
                    display = f"{base}-{name_clean}" if name_clean else base
                else:
                    display = name_clean if name_clean else ""
            if token:
                token_display[token] = display
            if pid != partido_actual or per != periodo_actual:
                q_local = []
                q_visit = []
                partido_actual = pid
                periodo_actual = per
            if accion_s == 'CAMBIO-JUGADOR-ENTRA':
                if token:
                    if cond == 'LOCAL' and token not in q_local:
                        q_local.append(token)
                    elif cond == 'VISITANTE' and token not in q_visit:
                        q_visit.append(token)
            elif accion_s == 'CAMBIO-JUGADOR-SALE':
                if token:
                    if cond == 'LOCAL' and token in q_local:
                        q_local.remove(token)
                    elif cond == 'VISITANTE' and token in q_visit:
                        q_visit.remove(token)
            else:
                if token and isinstance(jugador_nombre_s, str) and jugador_nombre_s.strip():
                    if cond == 'LOCAL' and token not in q_local and len(q_local) < 5:
                        q_local.append(token)
                    elif cond == 'VISITANTE' and token not in q_visit and len(q_visit) < 5:
                        q_visit.append(token)
            quintetos_local.append([token_display.get(t, t) for t in q_local])
            quintetos_visitante.append([token_display.get(t, t) for t in q_visit])
        df['quinteto_local'] = quintetos_local
        df['quinteto_visitante'] = quintetos_visitante

    # Agregados: jugadoresAgregado
    df_tmp = df.copy()
    for col in ["tiempo_segundos", "puntosLocal", "puntosVisitante"]:
        if col in df_tmp.columns:
            df_tmp[col] = pd.to_numeric(df_tmp[col], errors='coerce').fillna(0)

    resultados: List[Dict[str, Any]] = []
    for pid, df_partido in df_tmp.groupby('_id'):
        jugadores_en_cancha: set = set()
        fila_anterior = None
        condicion_jugador: Dict[Tuple[str, str], str] = {}
        tiempo_jugado: Dict[Tuple[Any, Any, Any, Any, Any, Any, Any], float] = {}
        puntos_favor: Dict[Tuple[Any, Any, Any, Any, Any, Any, Any], float] = {}
        puntos_contra: Dict[Tuple[Any, Any, Any, Any, Any, Any, Any], float] = {}

        # Determinar nombres de equipos local/visitante para construir claves de jugadores desde quintetos
        equipo_nombre_local = None
        equipo_nombre_visitante = None
        try:
            series_local = df_partido[df_partido.get('Condicion', '').astype(str).str.upper() == 'LOCAL']
            if not series_local.empty:
                equipo_nombre_local = str(series_local.iloc[0].get('equipo', '')).strip() or None
            series_vis = df_partido[df_partido.get('Condicion', '').astype(str).str.upper() == 'VISITANTE']
            if not series_vis.empty:
                equipo_nombre_visitante = str(series_vis.iloc[0].get('equipo', '')).strip() or None
        except Exception:
            pass

        for _, fila in df_partido.iterrows():
            jugador_key = (fila.get('equipo', ''), fila.get('nombre', ''))
            if jugador_key not in condicion_jugador:
                condicion_jugador[jugador_key] = fila.get('Condicion', '')

            accion = fila.get('accion_tipo')
            tiempo_actual = fila.get('tiempo_segundos', 0)
            numero_periodo = fila.get('numero_periodo')
            condicion_fija = condicion_jugador[jugador_key]

            # Si cambia el período o la fila anterior fue FIN DE PERIODO/FIN DE PARTIDO, reiniciar la cancha
            if fila_anterior is not None:
                try:
                    per_ant = fila_anterior.get('numero_periodo')
                except Exception:
                    per_ant = None
                if (fila_anterior.get('accion_tipo') in ('FINAL-PERIODO','FINAL-PARTIDO')) or (per_ant != numero_periodo):
                    jugadores_en_cancha.clear()

            if (
                fila_anterior is not None and
                fila_anterior.get('accion_tipo') != 'FINAL-PERIODO' and
                fila_anterior.get('numero_periodo') == numero_periodo
            ):
                # Si no hay jugadores en cancha, intentar poblar desde los quintetos de la fila anterior
                if not jugadores_en_cancha:
                    ql = fila_anterior.get('quinteto_local')
                    qv = fila_anterior.get('quinteto_visitante')
                    if isinstance(ql, list):
                        ql = tuple(sorted(ql))
                    if isinstance(qv, list):
                        qv = tuple(sorted(qv))
                    if ql is not None and len(ql) == 5 and equipo_nombre_local:
                        for nombre_disp in ql:
                            clave_j = (equipo_nombre_local, nombre_disp)
                            jugadores_en_cancha.add(clave_j)
                            if clave_j not in condicion_jugador:
                                condicion_jugador[clave_j] = 'LOCAL'
                    if qv is not None and len(qv) == 5 and equipo_nombre_visitante:
                        for nombre_disp in qv:
                            clave_j = (equipo_nombre_visitante, nombre_disp)
                            jugadores_en_cancha.add(clave_j)
                            if clave_j not in condicion_jugador:
                                condicion_jugador[clave_j] = 'VISITANTE'

                delta_tiempo = abs(float(fila_anterior.get('tiempo_segundos', 0)) - float(tiempo_actual))
                delta_local = float(fila.get('puntosLocal', 0)) - float(fila_anterior.get('puntosLocal', 0))
                delta_visitante = float(fila.get('puntosVisitante', 0)) - float(fila_anterior.get('puntosVisitante', 0))

                # Tomar el estado del marcador al INICIO del intervalo (fila_anterior)
                dif_ant = float(fila_anterior.get('DifPuntos', 0)) if 'DifPuntos' in fila_anterior else 0.0
                ult2min_ant = fila_anterior.get('ultimos_dos_minutos')

                # Determinar jugadores en cancha para el intervalo: exigir quintetos previos completos (5v5)
                set_interval = set()
                ql_prev = fila_anterior.get('quinteto_local')
                qv_prev = fila_anterior.get('quinteto_visitante')
                if isinstance(ql_prev, list):
                    ql_prev = tuple(sorted(ql_prev))
                if isinstance(qv_prev, list):
                    qv_prev = tuple(sorted(qv_prev))
                if not (ql_prev is not None and len(ql_prev) == 5 and equipo_nombre_local and qv_prev is not None and len(qv_prev) == 5 and equipo_nombre_visitante):
                    # Si no hay 5v5 definidos, no acumulamos tiempo para evitar sobreconteo
                    fila_anterior = fila
                    continue
                for nombre_disp in ql_prev:
                    clave_j = (equipo_nombre_local, nombre_disp)
                    set_interval.add(clave_j)
                    if clave_j not in condicion_jugador:
                        condicion_jugador[clave_j] = 'LOCAL'
                for nombre_disp in qv_prev:
                    clave_j = (equipo_nombre_visitante, nombre_disp)
                    set_interval.add(clave_j)
                    if clave_j not in condicion_jugador:
                        condicion_jugador[clave_j] = 'VISITANTE'

                for jug in list(set_interval):
                    cond_jug = condicion_jugador.get(jug, None)
                    if cond_jug is None:
                        continue
                    # Calcular SituacionMarcador desde la perspectiva del jugador
                    if cond_jug == 'LOCAL':
                        if dif_ant < -5:
                            situacion_para_jug = 'Perdiendo 5+'
                        elif dif_ant > 5:
                            situacion_para_jug = 'Ganando 5+'
                        else:
                            situacion_para_jug = '+-5'
                    elif cond_jug == 'VISITANTE':
                        if dif_ant > 5:
                            situacion_para_jug = 'Perdiendo 5+'
                        elif dif_ant < -5:
                            situacion_para_jug = 'Ganando 5+'
                        else:
                            situacion_para_jug = '+-5'
                    else:
                        situacion_para_jug = '+-5'

                    clave = (
                        pid, jug[0], jug[1],
                        cond_jug, situacion_para_jug,
                        ult2min_ant, numero_periodo
                    )
                    tiempo_jugado[clave] = tiempo_jugado.get(clave, 0) + delta_tiempo
                    if cond_jug == 'LOCAL':
                        puntos_favor[clave] = puntos_favor.get(clave, 0) + delta_local
                        puntos_contra[clave] = puntos_contra.get(clave, 0) + delta_visitante
                    elif cond_jug == 'VISITANTE':
                        puntos_favor[clave] = puntos_favor.get(clave, 0) + delta_visitante
                        puntos_contra[clave] = puntos_contra.get(clave, 0) + delta_local

            # Actualizar cancha
            if accion == 'CAMBIO-JUGADOR-ENTRA':
                jugadores_en_cancha.add(jugador_key)
            elif accion == 'CAMBIO-JUGADOR-SALE':
                jugadores_en_cancha.discard(jugador_key)

            fila_anterior = fila

        # Construcción simple de resultados: respetar nombre PBP; excluir SIN EQUIPO y NEUTRAL
        for clave in tiempo_jugado:
            partido_id2, equipo, nombre, condicion, situacion, ult2min, periodo = clave
            if condicion != 'NEUTRAL' and equipo != 'SIN EQUIPO':
                resultados.append({
                    '_id': partido_id2,
                    'equipo': equipo,
                    'nombre': nombre,
                    'numero_periodo': periodo,
                    'Condicion': condicion,
                    'SituacionMarcador': situacion,
                    'ultimos_dos_minutos': ult2min,
                    'tiempo_jugado': tiempo_jugado.get(clave, 0),
                    'puntos_favor': puntos_favor.get(clave, 0),
                    'puntos_contra': puntos_contra.get(clave, 0),
                    'diferencia': puntos_favor.get(clave, 0) - puntos_contra.get(clave, 0),
                })

    # Alinear nombres en quintetos al nuevo formato 'dorsal-nombre'
    try:
        # Construir mapas desde las tablas de estadísticas de equipo
        map_local = {}
        map_vis = {}
        if not estadisticas_local_df.empty and all(c in estadisticas_local_df.columns for c in ["dorsal","nombre"]):
            d = estadisticas_local_df.copy()
            d["dorsal"] = d["dorsal"].astype(str).str.strip().str.zfill(2)
            d["nombre"] = d["nombre"].astype(str).str.strip()
            map_local = {row["nombre"]: f"{row['dorsal']}-{row['nombre']}" for _, row in d.iterrows()}
        if not estadisticas_visitante_df.empty and all(c in estadisticas_visitante_df.columns for c in ["dorsal","nombre"]):
            d = estadisticas_visitante_df.copy()
            d["dorsal"] = d["dorsal"].astype(str).str.strip().str.zfill(2)
            d["nombre"] = d["nombre"].astype(str).str.strip()
            map_vis = {row["nombre"]: f"{row['dorsal']}-{row['nombre']}" for _, row in d.iterrows()}
        # Aplicar sobre listas de quintetos si existen
        def _map_quinteto(lst, mapper):
            if isinstance(lst, list):
                return [mapper.get(x, x) for x in lst]
            return lst
        if 'quinteto_local' in df.columns:
            df['quinteto_local'] = df['quinteto_local'].apply(lambda x: _map_quinteto(x, map_local))
        if 'quinteto_visitante' in df.columns:
            df['quinteto_visitante'] = df['quinteto_visitante'].apply(lambda x: _map_quinteto(x, map_vis))
    except Exception:
        pass

    # Crear DataFrame principal
    df_TiempoJugadores = pd.DataFrame(resultados)
    # Alinear tipos de claves en df_TiempoJugadores
    if not df_TiempoJugadores.empty:
        for col in ['_id','equipo','nombre','Condicion']:
            if col in df_TiempoJugadores.columns:
                df_TiempoJugadores[col] = df_TiempoJugadores[col].astype(str).fillna('').str.strip()
        if 'numero_periodo' in df_TiempoJugadores.columns:
            df_TiempoJugadores['numero_periodo'] = pd.to_numeric(df_TiempoJugadores['numero_periodo'], errors='coerce').fillna(0).astype(int)
    acciones_interes = [
        'TIRO2-FALLADO', 'REBOTE-DEFENSIVO', 'CANASTA-2P', 'PERDIDA',
        'RECUPERACION', 'TIRO3-FALLADO', 'ASISTENCIA', 'CANASTA-3P',
        'FALTA-COMETIDA', 'FALTA-RECIBIDA', 'REBOTE-OFENSIVO',
        'TIRO1-FALLADO', 'CANASTA-1P'
    ]
    # Construir acciones desde el DF final (df), para que 'nombre' coincida exactamente con tiempos
    df_acciones = df[df.get('accion_tipo', pd.Series(dtype=str)).isin(acciones_interes)] if 'accion_tipo' in df.columns else pd.DataFrame(columns=['accion_tipo'])
    if not df_acciones.empty:
        # Filtrar para excluir filas con equipo "SIN EQUIPO"
        df_acciones = df_acciones[df_acciones['equipo'].astype(str) != "SIN EQUIPO"]
        
        # Agrupar acciones por las MISMAS claves completas que tiempos
        stable_keys = ['_id','equipo','nombre','Condicion','numero_periodo','SituacionMarcador','ultimos_dos_minutos']
        # Asegurar que existan las columnas clave
        for k in stable_keys:
            if k not in df_acciones.columns:
                df_acciones[k] = ''
        df_acc_ind = (
            df_acciones
            .groupby(stable_keys)['accion_tipo']
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        # Alinear tipos de claves en df_acc_ind
        for col in ['_id','equipo','nombre','Condicion']:
            if col in df_acc_ind.columns:
                df_acc_ind[col] = df_acc_ind[col].astype(str).fillna('').str.strip()
        if 'numero_periodo' in df_acc_ind.columns:
            df_acc_ind['numero_periodo'] = pd.to_numeric(df_acc_ind['numero_periodo'], errors='coerce').fillna(0).astype(int)
        # Merge por claves estables; las columnas de situación/momento quedan del lado izquierdo
        df_JugadoresFinal = df_TiempoJugadores.merge(
            df_acc_ind,
            on=stable_keys,
            how='left'
        )
    else:
        df_JugadoresFinal = df_TiempoJugadores.copy()
    df_JugadoresFinal = df_JugadoresFinal.fillna(0)

    # Agregados: quintetos
    resultados_q: List[Dict[str, Any]] = []
    df_tmp2 = df_tmp.copy()
    if "autoincremental_id" in df_tmp2.columns:
        df_tmp2["autoincremental_id_num"] = pd.to_numeric(df_tmp2["autoincremental_id"], errors='coerce').fillna(0)
        base_df = df_tmp2.sort_values(by=['_id','numero_periodo','autoincremental_id_num']).reset_index(drop=True)
    else:
        if "tiempo_segundos" not in df_tmp2.columns and "tiempo_partido" in df_tmp2.columns:
            df_tmp2["tiempo_segundos"] = df_tmp2["tiempo_partido"].apply(to_seconds)
        if "tiempo_segundos" in df_tmp2.columns:
            base_df = df_tmp2.sort_values(by=['_id','numero_periodo','tiempo_segundos']).reset_index(drop=True)
        else:
            base_df = df_tmp2.copy()

    for partido_id3, df_partido in base_df.groupby('_id'):
        fila_anterior = None
        tiempo_quinteto: Dict[Tuple[Any,...], float] = {}
        puntos_favor_q: Dict[Tuple[Any,...], float] = {}
        puntos_contra_q: Dict[Tuple[Any,...], float] = {}
        estadisticas: Dict[str, Dict[Tuple[Any,...], int]] = {}
        for accion in acciones_interes:
            for tipo in ['Favor','Contra']:
                estadisticas[f'{accion}_{tipo}'] = {}

        for _, fila in df_partido.iterrows():
            accion = fila.get('accion_tipo')
            tiempo_actual = float(fila.get('tiempo_segundos', 0))
            numero_periodo = fila.get('numero_periodo')
            # Determinar estado del marcador y momento al INICIO del intervalo
            # (desde la fila anterior), como hicimos para jugadores
            # Si no hay fila anterior, estos valores se completarán cuando exista un intervalo válido
            situacion_local_prev = None
            situacion_visit_prev = None
            ult2min_ant = None
            dif_ant = None
            if fila_anterior is not None:
                try:
                    dif_ant = float(fila_anterior.get('DifPuntos', 0))
                except Exception:
                    dif_ant = 0.0
                ult2min_ant = fila_anterior.get('ultimos_dos_minutos')
                # Mapeo de situación desde la perspectiva de cada equipo
                if dif_ant < -5:
                    situacion_local_prev = 'Perdiendo 5+'
                    situacion_visit_prev = 'Ganando 5+'
                elif dif_ant > 5:
                    situacion_local_prev = 'Ganando 5+'
                    situacion_visit_prev = 'Perdiendo 5+'
                else:
                    situacion_local_prev = '+-5'
                    situacion_visit_prev = '+-5'

            # Usar los quintetos de la FILA ANTERIOR para representar quiénes
            # jugaron el intervalo entre fila_anterior y fila (consistencia con jugadores)
            quinteto_local_prev = None
            quinteto_visitante_prev = None
            if fila_anterior is not None:
                quinteto_local_prev = fila_anterior.get('quinteto_local')
                quinteto_visitante_prev = fila_anterior.get('quinteto_visitante')
                if isinstance(quinteto_local_prev, list):
                    quinteto_local_prev = tuple(sorted(quinteto_local_prev))
                if isinstance(quinteto_visitante_prev, list):
                    quinteto_visitante_prev = tuple(sorted(quinteto_visitante_prev))
                if quinteto_local_prev is not None and len(quinteto_local_prev) != 5:
                    quinteto_local_prev = None
                if quinteto_visitante_prev is not None and len(quinteto_visitante_prev) != 5:
                    quinteto_visitante_prev = None

            if (
                fila_anterior is not None and
                fila_anterior.get('accion_tipo') != 'FINAL-PERIODO' and
                fila_anterior.get('numero_periodo') == numero_periodo
            ):
                delta_tiempo = abs(float(fila_anterior.get('tiempo_segundos', 0)) - tiempo_actual)
                delta_local = float(fila.get('puntosLocal', 0)) - float(fila_anterior.get('puntosLocal', 0))
                delta_visitante = float(fila.get('puntosVisitante', 0)) - float(fila_anterior.get('puntosVisitante', 0))
                # Requerir que ambos quintetos previos sean 5v5 definidos para evitar sobreconteo
                if quinteto_local_prev is not None and quinteto_visitante_prev is not None:
                    if quinteto_local_prev is not None:
                        clave_local = (partido_id3, quinteto_local_prev, 'LOCAL', situacion_local_prev, ult2min_ant, numero_periodo)
                        tiempo_quinteto[clave_local] = tiempo_quinteto.get(clave_local, 0) + delta_tiempo
                        puntos_favor_q[clave_local] = puntos_favor_q.get(clave_local, 0) + delta_local
                        puntos_contra_q[clave_local] = puntos_contra_q.get(clave_local, 0) + delta_visitante
                    if quinteto_visitante_prev is not None:
                        clave_vis = (partido_id3, quinteto_visitante_prev, 'VISITANTE', situacion_visit_prev, ult2min_ant, numero_periodo)
                        tiempo_quinteto[clave_vis] = tiempo_quinteto.get(clave_vis, 0) + delta_tiempo
                        puntos_favor_q[clave_vis] = puntos_favor_q.get(clave_vis, 0) + delta_visitante
                        puntos_contra_q[clave_vis] = puntos_contra_q.get(clave_vis, 0) + delta_local

            # Contabilizar acciones en función del quinteto vigente al INICIO del instante
            # (usar quintetos y situación de la fila anterior para consistencia)
            if accion in acciones_interes and fila_anterior is not None:
                condicion = str(fila.get('Condicion', '')).upper()
                clave_local = (partido_id3, quinteto_local_prev, 'LOCAL', situacion_local_prev, ult2min_ant, numero_periodo) if quinteto_local_prev is not None else None
                clave_vis = (partido_id3, quinteto_visitante_prev, 'VISITANTE', situacion_visit_prev, ult2min_ant, numero_periodo) if quinteto_visitante_prev is not None else None

                if condicion == 'LOCAL':
                    if clave_local is not None:
                        estadisticas[f'{accion}_Favor'][clave_local] = estadisticas[f'{accion}_Favor'].get(clave_local, 0) + 1
                    if clave_vis is not None:
                        estadisticas[f'{accion}_Contra'][clave_vis] = estadisticas[f'{accion}_Contra'].get(clave_vis, 0) + 1
                elif condicion == 'VISITANTE':
                    if clave_vis is not None:
                        estadisticas[f'{accion}_Favor'][clave_vis] = estadisticas[f'{accion}_Favor'].get(clave_vis, 0) + 1
                    if clave_local is not None:
                        estadisticas[f'{accion}_Contra'][clave_local] = estadisticas[f'{accion}_Contra'].get(clave_local, 0) + 1

            fila_anterior = fila

        for clave in tiempo_quinteto:
            partido_id4, quinteto, condicion, situacion, ult2min, periodo = clave
            fila_res = {
                '_id': partido_id4,
                'quinteto': quinteto,
                'Condicion': condicion,
                'SituacionMarcador': situacion,
                'ultimos_dos_minutos': ult2min,
                'numero_periodo': periodo,
                'tiempo_jugado': tiempo_quinteto.get(clave, 0),
                'puntos_favor': puntos_favor_q.get(clave, 0),
                'puntos_contra': puntos_contra_q.get(clave, 0),
            }
            for accion in acciones_interes:
                fila_res[f'{accion}_Favor'] = estadisticas[f'{accion}_Favor'].get(clave, 0)
                fila_res[f'{accion}_Contra'] = estadisticas[f'{accion}_Contra'].get(clave, 0)
            resultados_q.append(fila_res)

    df_TiempoQuintetos = pd.DataFrame(resultados_q)

    return df, df_JugadoresFinal, df_TiempoQuintetos


@st.cache_data(show_spinner=False)
def descargar_y_transformar(partido_id: str) -> Dict[str, pd.DataFrame]:
    # Descargas
    part_payload = fetch_partido(partido_id)
    est_payload = fetch_estadisticas(partido_id)

    partido = part_payload.get('partido') or {}
    envivo = part_payload.get('envivo') or {}
    if not partido or not partido.get('idlocal') or not partido.get('local'):
        raise ValueError(
            f"Partido no válido o incompleto | part_payload_keys={list(part_payload.keys()) if isinstance(part_payload, dict) else type(part_payload)} | partido_keys={list(partido.keys()) if isinstance(partido, dict) else type(partido)}"
        )

    partido['_id'] = str(partido_id)
    partido_df = pd.DataFrame([partido])

    historialacciones = envivo.get('historialacciones') or []
    jugada_df = pd.DataFrame(historialacciones)
    if not jugada_df.empty:
        jugada_df['_id'] = str(partido_id)

    estadisticas = est_payload.get('estadisticas') or {}
    estadisticas['_id'] = str(partido_id)

    base_cols = {k: v for k, v in estadisticas.items() if k not in ('estadisticasequipolocal', 'estadisticasequipovisitante')}
    estadistica_df = pd.DataFrame([base_cols]) if base_cols else pd.DataFrame()

    local = estadisticas.get('estadisticasequipolocal') or []
    visitante = estadisticas.get('estadisticasequipovisitante') or []
    if not isinstance(local, list):
        local = []
    if not isinstance(visitante, list):
        visitante = []
    local_df = pd.DataFrame(local)
    visitante_df = pd.DataFrame(visitante)
    if not local_df.empty:
        local_df['_id'] = str(partido_id)
    if not visitante_df.empty:
        visitante_df['_id'] = str(partido_id)

    pbp_df, jugadores_df, quintetos_df = procesar_pbp_y_agregados(
        partido_df, jugada_df, local_df, visitante_df
    )

    # Asegurar columnas derivadas visibles en pbp
    if isinstance(pbp_df, pd.DataFrame) and not pbp_df.empty:
        # tiempo_segundos si falta y hay tiempo_partido
        if 'tiempo_segundos' not in pbp_df.columns and 'tiempo_partido' in pbp_df.columns:
            pbp_df = pbp_df.copy()
            pbp_df['tiempo_segundos'] = pbp_df['tiempo_partido'].apply(to_seconds)
        # x_period = (600 - tiempo_segundos) + (numero_periodo - 1) * 600
        if 'numero_periodo' in pbp_df.columns:
            tiempo_num = pd.to_numeric(pbp_df.get('tiempo_segundos', np.nan), errors='coerce').fillna(0)
            periodo_num = pd.to_numeric(pbp_df.get('numero_periodo', 1), errors='coerce').fillna(1)
            # Q1-Q4 (10 min) y OT (5 min)
            pbp_df['x_period'] = np.where(
                periodo_num <= 4,
                (600 - tiempo_num) + (periodo_num - 1) * 600,
                2400 + (300 - tiempo_num) + (periodo_num - 5) * 300
            )

    return {
        'partido': partido_df,
        'jugada': jugada_df,
        'estadistica': estadistica_df,
        'estadisticas_equipolocal': local_df,
        'estadisticas_equipovisitante': visitante_df,
        'pbp': pbp_df,
        'jugadoresAgregado': jugadores_df,
        # Mantener clave legacy y agregar la correcta
        'quintetosAgregado': quintetos_df,
    }


# ------------------------------
# UI
# ------------------------------
# Estilos ligeros (mantener tema claro) y centrado de títulos
st.markdown(
    """
    <style>
    h1, h2, h3, h4, h5, h6 {
        text-align: center;
    }
    /* Reducir ancho del input de texto */
    .stTextInput > div > div > input {
        max-width: 160px;
        width: 160px;
        text-align: center;
    }
    /* Hacer que el botón tenga el mismo ancho que el input y quede debajo */
    .stButton > button {
        max-width: 160px;
        width: 160px;
    }
    /* Centrar el contenedor de entrada */
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Estadísticas Basquet")

# Entrada en el cuerpo principal, centrada y angosta
st.subheader("Entrada")
wrap = st.container()
colL, colMid, colR = wrap.columns([2, 1, 2])
with colMid:
    partido_id_input = st.text_input("ID de partido", value="", placeholder="Ej: 123456", max_chars=6)
    ejecutar = st.button("Buscar partido", type="primary")

if (ejecutar or ('tablas' in st.session_state)):
    if ejecutar:
        if not partido_id_input.strip().isdigit():
            st.error("Ingrese un ID numérico válido.")
        else:
            try:
                # Resetear filtros de las 3 pestañas al buscar un nuevo partido
                for k in ['res_sel_per','estad_sel_per','estad_sel_situ','estad_sel_u2m','q_sel_per','q_sel_situ','q_sel_u2m']:
                    try:
                        if k in st.session_state:
                            del st.session_state[k]
                    except Exception:
                        pass
                with st.spinner("Descargando y procesando datos..."):
                    tablas = descargar_y_transformar(partido_id_input.strip())
                # Persistir en sesión para evitar re-descarga al cambiar filtros
                st.session_state['tablas'] = tablas
                # Forzar reinicio visual de filtros a 'TODOS'
                st.session_state['res_sel_per'] = 'TODOS'
                st.session_state['estad_sel_per'] = 'TODOS'
                st.session_state['estad_sel_situ'] = 'TODOS'
                st.session_state['estad_sel_u2m'] = 'TODOS'
                st.session_state['q_sel_per'] = 'TODOS'
                st.session_state['q_sel_situ'] = 'TODOS'
                st.session_state['q_sel_u2m'] = 'TODOS'
            except Exception as e:
                # Mostrar error y detalle para diagnóstico
                st.error("ID de Partido no encontrado")
                tablas = None
    else:
        tablas = st.session_state.get('tablas')

    if tablas is not None:
        # Mantener orden fijo pero mostrando primero 'Resumen' al abrir la app
        # Pestañas: Resumen, Estadisticas por jugador, Estadistica por Quintetos
        nombres = ["Resumen", "Estadisticas por jugador", "Estadistica por Quintetos"]
        # Mostrar pestañas
        tabs = st.tabs(nombres)
        # Referencias por nombre
        t_resumen = tabs[0]
        t_estadistica = tabs[1]
        t_quintetos = tabs[2]

        # Pestaña Resumen
        with t_resumen:
            part_df = tablas.get('partido', pd.DataFrame())
            pbp_df = tablas.get('pbp', pd.DataFrame())
            est_loc_df = tablas.get('estadisticas_equipolocal', pd.DataFrame())
            est_vis_df = tablas.get('estadisticas_equipovisitante', pd.DataFrame())
            # Derivar valores base aún si 'partido' está vacío
            row = part_df.iloc[0] if not part_df.empty else {}
            # Nombres (variantes)
            local_name = str(_first_of(row, [
                'local', 'equipo_local', 'nombre_local', 'localnombre', 'nombreLocal', 'equipoLocal', 'club_local', 'clubLocal'
            ], ''))
            visitante_name = str(_first_of(row, [
                'visitante', 'equipo_visitante', 'nombre_visitante', 'visitantenombre', 'nombreVisitante', 'equipoVisitante', 'club_visitante', 'clubVisitante'
            ], ''))
            if (not local_name or local_name.strip() == '') and not est_loc_df.empty:
                local_name = str(_first_of(est_loc_df.iloc[0], ['equipo', 'nombre_equipo'], 'Local'))
            if (not visitante_name or visitante_name.strip() == '') and not est_vis_df.empty:
                visitante_name = str(_first_of(est_vis_df.iloc[0], ['equipo', 'nombre_equipo'], 'Visitante'))
            if not local_name:
                local_name = 'Local'
            if not visitante_name:
                visitante_name = 'Visitante'

            # Colores
            color_local_raw = _first_of(row, ['color_local', 'local_color', 'colorLocal', 'colorlocal'], '#1f77b4')
            color_visitante_raw = _first_of(row, ['color_visitante', 'visitante_color', 'colorVisitante', 'colorvisitante'], '#ff7f0e')
            color_local = _parse_color(color_local_raw, '#1f77b4')
            color_visitante = _parse_color(color_visitante_raw, '#ff7f0e')
            tc_local = _text_color_for_bg(color_local)
            tc_visitante = _text_color_for_bg(color_visitante)

            # Tanteos con fallback a pbp
            tanteo_local = _first_of(row, ['tanteo_local', 'puntos_local', 'marcador_local', 'score_local', 'tanteoLocal', 'marcadorLocal'], None)
            tanteo_visitante = _first_of(row, ['tanteo_visitante', 'puntos_visitante', 'marcador_visitante', 'score_visitante', 'tanteoVisitante', 'marcadorVisitante'], None)
            if (tanteo_local is None or (hasattr(pd, 'isna') and pd.isna(tanteo_local)) or tanteo_local == '') or \
               (tanteo_visitante is None or (hasattr(pd, 'isna') and pd.isna(tanteo_visitante)) or tanteo_visitante == ''):
                if not pbp_df.empty:
                    dfp = pbp_df.copy()
                    if 'autoincremental_id' in dfp.columns and 'autoincremental_id_num' not in dfp.columns:
                        dfp['autoincremental_id_num'] = pd.to_numeric(dfp['autoincremental_id'], errors='coerce').fillna(0)
                    if 'autoincremental_id_num' in dfp.columns:
                        dfp = dfp.sort_values(by=['_id','numero_periodo','autoincremental_id_num'])
                    elif 'tiempo_segundos' in dfp.columns:
                        dfp = dfp.sort_values(by=['_id','numero_periodo','tiempo_segundos'])
                    last = dfp.iloc[-1]
                    tanteo_local = last.get('puntosLocal', tanteo_local)
                    tanteo_visitante = last.get('puntosVisitante', tanteo_visitante)

            # Resumen por periodo para tarjetas (si hay pbp)
            local_line = ''
            visit_line = ''
            if not pbp_df.empty:
                try:
                    dfpp = pbp_df.copy()
                    if 'autoincremental_id' in dfpp.columns and 'autoincremental_id_num' not in dfpp.columns:
                        dfpp['autoincremental_id_num'] = pd.to_numeric(dfpp['autoincremental_id'], errors='coerce').fillna(0)
                    if 'autoincremental_id_num' in dfpp.columns:
                        dfpp = dfpp.sort_values(by=['_id','numero_periodo','autoincremental_id_num'])
                    else:
                        if 'tiempo_segundos' not in dfpp.columns and 'tiempo_partido' in dfpp.columns:
                            dfpp['tiempo_segundos'] = dfpp['tiempo_partido'].apply(to_seconds)
                        if 'tiempo_segundos' in dfpp.columns:
                            dfpp = dfpp.sort_values(by=['_id','numero_periodo','tiempo_segundos'])
                    # Asegurar numéricos
                    dfpp['puntosLocal_num'] = pd.to_numeric(dfpp.get('puntosLocal', 0), errors='coerce').fillna(0)
                    dfpp['puntosVisitante_num'] = pd.to_numeric(dfpp.get('puntosVisitante', 0), errors='coerce').fillna(0)
                    # Último por periodo
                    per_local = {}
                    per_visit = {}
                    if 'numero_periodo' in dfpp.columns:
                        for per in sorted(pd.to_numeric(dfpp['numero_periodo'], errors='coerce').dropna().astype(int).unique().tolist()):
                            sub = dfpp[pd.to_numeric(dfpp['numero_periodo'], errors='coerce') == per]
                            if not sub.empty:
                                last = sub.iloc[-1]
                                per_local[per] = int(float(last.get('puntosLocal_num', 0)))
                                per_visit[per] = int(float(last.get('puntosVisitante_num', 0)))
                    if per_local:
                        local_line = ' | '.join([f"{per_local.get(p, 0)}" for p in sorted(per_local.keys())])
                    if per_visit:
                        visit_line = ' | '.join([f"{per_visit.get(p, 0)}" for p in sorted(per_visit.keys())])
                except Exception:
                    pass

            local_line_html = f"<div style='font-size:13px; opacity:0.9; margin-top:6px; text-align:right'>{local_line}</div>" if local_line else ''
            visit_line_html = f"<div style='font-size:13px; opacity:0.9; margin-top:6px; text-align:right'>{visit_line}</div>" if visit_line else ''

            # Render con columnas confiables
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""
                    <div style='background:{color_local}; color:{tc_local}; padding:16px; border-radius:16px; box-shadow:0 2px 8px rgba(0,0,0,0.08)'>
                        <div style='font-size:13px; font-weight:600; opacity:0.9; text-transform:uppercase; letter-spacing:.04em'>Local</div>
                        <div style='display:flex; justify-content:space-between; align-items:center; gap:8px'>
                            <div style='font-size:28px; font-weight:700; line-height:1.1'>{local_name}</div>
                            <div style='font-size:44px; font-weight:800'>{int(tanteo_local) if (tanteo_local is not None and str(tanteo_local).strip() != '' and not pd.isna(tanteo_local)) else '-'}</div>
                        </div>
                        {local_line_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div style='background:{color_visitante}; color:{tc_visitante}; padding:16px; border-radius:16px; box-shadow:0 2px 8px rgba(0,0,0,0.08)'>
                        <div style='font-size:13px; font-weight:600; opacity:0.9; text-transform:uppercase; letter-spacing:.04em'>Visitante</div>
                        <div style='display:flex; justify-content:space-between; align-items:center; gap:8px'>
                            <div style='font-size:28px; font-weight:700; line-height:1.1'>{visitante_name}</div>
                            <div style='font-size:44px; font-weight:800'>{int(tanteo_visitante) if (tanteo_visitante is not None and str(tanteo_visitante).strip() != '' and not pd.isna(tanteo_visitante)) else '-'}</div>
                        </div>
                        {visit_line_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Gráfico de evolución de puntos desde PBP
            if not pbp_df.empty:
                    # Selector de periodo (aplica a todo lo siguiente en esta pestaña)
                    try:
                        periodos = sorted(pd.to_numeric(pbp_df['numero_periodo'], errors='coerce').dropna().astype(int).unique().tolist())
                    except Exception:
                        periodos = []
                    opciones_periodo = ['TODOS'] + periodos
                    sel_periodo = st.selectbox('Seleccionar periodo', opciones_periodo, index=0, key='res_sel_per')
                    # Espaciado
                    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                    chart_height = 600
                    point_size = 120

                    dfp = pbp_df.copy()
                    # Filtrar por periodo si corresponde
                    if sel_periodo != 'TODOS':
                        try:
                            sel_val = int(sel_periodo)
                            dfp = dfp[pd.to_numeric(dfp['numero_periodo'], errors='coerce') == sel_val].copy()
                        except Exception:
                            pass
                    # Orden temporal por autoincremental_id si existe, sino por tiempo
                    if 'autoincremental_id' in dfp.columns and 'autoincremental_id_num' not in dfp.columns:
                        dfp['autoincremental_id_num'] = pd.to_numeric(dfp['autoincremental_id'], errors='coerce').fillna(0)
                    if 'autoincremental_id_num' in dfp.columns:
                        dfp = dfp.sort_values(by=['_id','numero_periodo','autoincremental_id_num'])
                    elif 'tiempo_segundos' in dfp.columns:
                        dfp = dfp.sort_values(by=['_id','numero_periodo','tiempo_segundos'])
                    dfp = dfp.reset_index(drop=True)
                    dfp['orden'] = np.arange(len(dfp))

                    # Eje X: usar x_period de pbp si existe; sino calcular
                    if 'x_period' not in dfp.columns:
                        # x_period: Q1-Q4 (10 min) y OT (5 min). Si no hay tiempo, usar 'orden'.
                        tiempo_num = pd.to_numeric(dfp.get('tiempo_segundos', np.nan), errors='coerce')
                        periodo_num = pd.to_numeric(dfp.get('numero_periodo', 1), errors='coerce').fillna(1)
                        tiempo_num = tiempo_num.fillna(dfp['orden'])
                        dfp['x_period'] = np.where(
                            periodo_num <= 4,
                            (600 - tiempo_num) + (periodo_num - 1) * 600,
                            2400 + (300 - tiempo_num) + (periodo_num - 5) * 300
                        )

                    # Asegurar x_period numérico exacto del pbp
                    dfp['x_period'] = pd.to_numeric(dfp['x_period'], errors='coerce')
                    dfp = dfp[~dfp['x_period'].isna()].copy()
                    # Selección interactiva sobre eje X
                    brush = alt.selection_interval(encodings=['x'], name='Seleccion')
                    # Series explícitas por equipo para evitar valores no presentes en pbp
                    dfp['puntosLocal_num'] = pd.to_numeric(dfp.get('puntosLocal', 0), errors='coerce').fillna(0)
                    dfp['puntosVisitante_num'] = pd.to_numeric(dfp.get('puntosVisitante', 0), errors='coerce').fillna(0)

                    # Ajustar color de línea si el color de equipo es blanco para que se vea
                    def _is_white(c: str) -> bool:
                        s = str(c).strip().lower()
                        return s in ('#fff', '#ffffff', 'white')
                    local_line_color = '#cfd8dc' if _is_white(color_local) else color_local
                    visit_line_color = '#cfd8dc' if _is_white(color_visitante) else color_visitante

                    line_local = (
                        alt.Chart(dfp)
                        .mark_line(point=False, color=local_line_color)
                        .encode(
                            x=alt.X('x_period:Q', title='Tiempo', sort=None),
                            y=alt.Y('puntosLocal_num:Q', title='Puntos acumulados'),
                            order=alt.Order('orden:Q'),
                            tooltip=[
                                alt.Tooltip('numero_periodo:Q', title='Periodo'),
                                alt.Tooltip('tiempo_segundos:Q', title='Tiempo (s)'),
                                alt.Tooltip('puntosLocal_num:Q', title=local_name)
                            ]
                        )
                        .properties(height=chart_height, title=alt.TitleParams(text='Puntos local vs visitante', anchor='middle'))
                        .transform_filter(brush)
                    )
                    line_visit = (
                        alt.Chart(dfp)
                        .mark_line(point=False, color=visit_line_color)
                        .encode(
                            x=alt.X('x_period:Q', title='Tiempo', sort=None),
                            y=alt.Y('puntosVisitante_num:Q', title='Puntos acumulados'),
                            order=alt.Order('orden:Q'),
                            tooltip=[
                                alt.Tooltip('numero_periodo:Q', title='Periodo'),
                                alt.Tooltip('tiempo_segundos:Q', title='Tiempo (s)'),
                                alt.Tooltip('puntosVisitante_num:Q', title=visitante_name)
                            ]
                        )
                        .properties(height=chart_height)
                        .transform_filter(brush)
                    )
                    line = line_local + line_visit

                    # Reglas verticales al cambio de periodo
                    period_changes = []
                    prev = None
                    for _, rowp in dfp.iterrows():
                        cur = rowp.get('numero_periodo')
                        if prev is not None and cur != prev:
                            period_changes.append({'x_period': rowp.get('x_period'), 'numero_periodo': cur})
                        prev = cur
                    chart = line
                    if period_changes:
                        rules_df = pd.DataFrame(period_changes)
                        # Reglas más visibles
                        rules = alt.Chart(rules_df).mark_rule(color='#333333', strokeDash=[8,4], strokeWidth=3).encode(
                            x=alt.X('x_period:Q'),
                            tooltip=[alt.Tooltip('numero_periodo:N', title='Inicio periodo')]
                        ).transform_filter(brush)
                        # Etiquetas del número de periodo (debajo y corridas)
                        try:
                            rules_df['label'] = 'P ' + rules_df['numero_periodo'].astype(str)
                            # Posicionar texto al 10% del eje Y
                            y_min = float(min(dfp['puntosLocal_num'].min(), dfp['puntosVisitante_num'].min())) if len(dfp) else 0.0
                            y_max = float(max(dfp['puntosLocal_num'].max(), dfp['puntosVisitante_num'].max())) if len(dfp) else 1.0
                            y_pos = y_min + 0.1 * (y_max - y_min)
                            rules_df['y_pos'] = y_pos
                            labels = alt.Chart(rules_df).mark_text(align='left', baseline='bottom', dx=6, color='#333333', fontSize=16).encode(
                                x=alt.X('x_period:Q'), y=alt.Y('y_pos:Q'), text='label:N'
                            ).transform_filter(brush)
                            chart = chart + rules + labels
                        except Exception:
                            chart = chart + rules

                    # Puntos de canastas con tonos por 1P/2P/3P y color por equipo
                    if 'accion_tipo' in dfp.columns:
                        ev = dfp[dfp['accion_tipo'].isin(['CANASTA-1P','CANASTA-2P','CANASTA-3P'])].copy()
                        if not ev.empty:
                            ev['Equipo'] = np.where(ev.get('Condicion','').astype(str).str.upper()=='LOCAL', local_name, visitante_name)
                            ev['y_points'] = np.where(ev.get('Condicion','').astype(str).str.upper()=='LOCAL',
                                                      pd.to_numeric(ev.get('puntosLocal', 0), errors='coerce').fillna(0),
                                                      pd.to_numeric(ev.get('puntosVisitante', 0), errors='coerce').fillna(0))
                            ev['orden'] = pd.to_numeric(ev['orden'], errors='coerce').fillna(0)

                            # Leyenda externa removida; crearemos una leyenda interna personalizada
                            points = alt.Chart(ev).mark_point(filled=True, size=point_size).encode(
                                x=alt.X('x_period:Q', title='Tiempo', sort=None),
                                y=alt.Y('y_points:Q'),
                                color=alt.Color('accion_tipo:N', scale=alt.Scale(
                                    domain=['CANASTA-1P','CANASTA-2P','CANASTA-3P'],
                                    range=['#fdd835','#1e88e5','#43a047']
                                ), legend=None),
                                opacity=alt.value(1.0),
                                order=alt.Order('orden:Q'),
                                tooltip=[
                                    alt.Tooltip('Equipo:N'),
                                    alt.Tooltip('accion_tipo:N', title='Acción'),
                                    alt.Tooltip('numero_periodo:Q', title='Periodo'),
                                    alt.Tooltip('tiempo_segundos:Q', title='Tiempo (s)'),
                                    alt.Tooltip('x_period:Q', title='x_period'),
                                    alt.Tooltip('y_points:Q', title='Puntos')
                                ]
                            ).transform_filter(brush)
                            chart = chart + points

                            # Leyenda interna (esquina superior izquierda dentro del gráfico)
                            try:
                                x_min = float(dfp['x_period'].min()) if len(dfp) else 0.0
                                x_max = float(dfp['x_period'].max()) if len(dfp) else 1.0
                                y_min_v = float(min(dfp['puntosLocal_num'].min(), dfp['puntosVisitante_num'].min())) if len(dfp) else 0.0
                                y_max_v = float(max(dfp['puntosLocal_num'].max(), dfp['puntosVisitante_num'].max())) if len(dfp) else 1.0
                                x_pad = (x_max - x_min) * 0.02
                                y_range = (y_max_v - y_min_v) if (y_max_v - y_min_v) != 0 else 1.0
                                base_x = x_min + x_pad
                                base_y = y_max_v - 0.06 * y_range
                                step = 0.07 * y_range
                                legend_df = pd.DataFrame({
                                    'accion_tipo': ['CANASTA-1P','CANASTA-2P','CANASTA-3P'],
                                    'lx': [base_x, base_x, base_x],
                                    'ly': [base_y, base_y - step, base_y - 2*step],
                                    'label': ['1 Punto','2 Puntos','3 Puntos']
                                })
                                leg_points = alt.Chart(legend_df).mark_point(filled=True, size=point_size*0.6).encode(
                                    x=alt.X('lx:Q'), y=alt.Y('ly:Q'),
                                    color=alt.Color('accion_tipo:N', scale=alt.Scale(
                                        domain=['CANASTA-1P','CANASTA-2P','CANASTA-3P'],
                                        range=['#fdd835','#1e88e5','#43a047']
                                    ), legend=None)
                                ).transform_filter(brush)
                                leg_text = alt.Chart(legend_df).mark_text(align='left', dx=10, dy=4, color='#333333', fontSize=14).encode(
                                    x=alt.X('lx:Q'), y=alt.Y('ly:Q'), text='label:N'
                                ).transform_filter(brush)
                                chart = chart + leg_points + leg_text
                            except Exception:
                                pass

                    # Sin leyenda externa; ya colocamos una interna personalizada
                    chart = chart.add_params(brush)
                    # Guardar para renderizar en columnas
                    chart_full = chart

                    # Segundo gráfico: Diferencia de puntos (sin puntos de canasta)
                    try:
                        dfp['DifPuntos_num'] = dfp['puntosLocal_num'] - dfp['puntosVisitante_num']
                        # Dominio simétrico alrededor de 0
                        dmin = float(dfp['DifPuntos_num'].min()) if not dfp.empty else -1.0
                        dmax = float(dfp['DifPuntos_num'].max()) if not dfp.empty else 1.0
                        dabs = max(abs(dmin), abs(dmax)) if not dfp.empty else 1.0
                        y_domain = [-dabs, dabs]

                        diff_line = (
                            alt.Chart(dfp)
                            .mark_line(point=False, color='#43a047')
                            .encode(
                                x=alt.X('x_period:Q', title='Tiempo', sort=None),
                                y=alt.Y('DifPuntos_num:Q', title='Diferencia de puntos', scale=alt.Scale(domain=y_domain))
                            )
                            .properties(height=chart_height, title=alt.TitleParams(text='Diferencia de puntos (local - visitante)', anchor='middle'))
                            .transform_filter(brush)
                        )
                        # Línea horizontal en 0 (más oscura y punteada)
                        zero_rule = (
                            alt.Chart(pd.DataFrame({'y':[0]}))
                            .mark_rule(color='#333333', strokeDash=[6,3], strokeWidth=2)
                            .encode(y='y:Q')
                        )

                        # Reglas y etiquetas de cambio de periodo igual que en el primer gráfico
                        diff_chart = diff_line + zero_rule
                        if period_changes:
                            rules_df2 = pd.DataFrame(period_changes)
                            rules2 = alt.Chart(rules_df2).mark_rule(color='#333333', strokeDash=[8,4], strokeWidth=3).encode(
                                x=alt.X('x_period:Q'),
                                tooltip=[alt.Tooltip('numero_periodo:N', title='Inicio periodo')]
                            ).transform_filter(brush)
                            try:
                                rules_df2['label'] = 'P ' + rules_df2['numero_periodo'].astype(str)
                                # Posicionar al 10% del eje Y simétrico
                                y_pos2 = y_domain[0] + 0.1 * (y_domain[1] - y_domain[0])
                                rules_df2['y_pos'] = y_pos2
                                labels2 = alt.Chart(rules_df2).mark_text(align='left', baseline='bottom', dx=6, color='#333333', fontSize=14).encode(
                                    x=alt.X('x_period:Q'), y=alt.Y('y_pos:Q'), text='label:N'
                                ).transform_filter(brush)
                                diff_chart = diff_chart + rules2 + labels2
                            except Exception:
                                diff_chart = diff_chart + rules2

                        # Marcas de máximo y mínimo de diferencia con etiquetas (dentro del brush actual)
                        try:
                            # Máximo dentro del rango seleccionado: última aparición por x_period
                            if dmax >= 0:
                                max_layer = (
                                    alt.Chart(dfp)
                                    .transform_filter(brush)
                                    .transform_joinaggregate(maxDiff='max(DifPuntos_num)')
                                    .transform_filter('datum.DifPuntos_num == datum.maxDiff')
                                    .transform_window(rn='row_number()', sort=[alt.SortField('x_period', order='descending')])
                                    .transform_filter('datum.rn == 1')
                                )
                                max_point = max_layer.mark_point(filled=True, size=140, color='#000000').encode(
                                    x='x_period:Q', y='DifPuntos_num:Q'
                                )
                                max_label = (
                                    max_layer
                                    .transform_calculate(label="'Max dif Local = ' + toString(datum.DifPuntos_num)")
                                    .mark_text(color='#000000', fontSize=16, dy=-12)
                                    .encode(x='x_period:Q', y='DifPuntos_num:Q', text='label:N')
                                )
                                diff_chart = diff_chart + max_point + max_label
                            # Mínimo dentro del rango seleccionado: última aparición por x_period
                            if dmin <= 0:
                                min_layer = (
                                    alt.Chart(dfp)
                                    .transform_filter(brush)
                                    .transform_joinaggregate(minDiff='min(DifPuntos_num)')
                                    .transform_filter('datum.DifPuntos_num == datum.minDiff')
                                    .transform_window(rn='row_number()', sort=[alt.SortField('x_period', order='descending')])
                                    .transform_filter('datum.rn == 1')
                                )
                                min_point = min_layer.mark_point(filled=True, size=140, color='#000000').encode(
                                    x='x_period:Q', y='DifPuntos_num:Q'
                                )
                                min_label = (
                                    min_layer
                                    .transform_calculate(label="'Max dif Visitante = ' + toString(datum.DifPuntos_num)")
                                    .mark_text(color='#000000', fontSize=16, dy=18)
                                    .encode(x='x_period:Q', y='DifPuntos_num:Q', text='label:N')
                                )
                                diff_chart = diff_chart + min_point + min_label
                        except Exception as e:
                            st.warning(f"No se pudieron calcular las marcas de máximo/mínimo: {e}")

                        # Marcar tiempos muertos solicitados con puntos del color del equipo
                        try:
                            tm = dfp[dfp.get('accion_tipo', '').astype(str) == 'TIEMPO-MUERTO-SOLICITADO'].copy()
                            if not tm.empty:
                                # Asegurar columnas necesarias
                                if 'DifPuntos_num' not in tm.columns:
                                    tm['DifPuntos_num'] = pd.to_numeric(tm.get('puntosLocal', 0), errors='coerce').fillna(0) - pd.to_numeric(tm.get('puntosVisitante', 0), errors='coerce').fillna(0)
                                tm['EquipoTM'] = np.where(tm.get('Condicion', '').astype(str).str.upper()=='LOCAL', local_name, visitante_name)
                                tm_color = alt.Scale(domain=[local_name, visitante_name], range=[color_local, color_visitante])
                                tm_points = (
                                    alt.Chart(tm)
                                    .mark_point(filled=True, size=180, stroke='#000', strokeWidth=0.5)
                                    .encode(
                                        x=alt.X('x_period:Q'),
                                        y=alt.Y('DifPuntos_num:Q'),
                                        color=alt.Color('EquipoTM:N', scale=tm_color, legend=alt.Legend(orient='top-left', title='Tiempo muerto')),
                                        tooltip=[
                                            alt.Tooltip('EquipoTM:N', title='Equipo'),
                                            alt.Tooltip('numero_periodo:Q', title='Periodo'),
                                            alt.Tooltip('tiempo_segundos:Q', title='Tiempo (s)'),
                                            alt.Tooltip('DifPuntos_num:Q', title='Dif puntos')
                                        ]
                                    )
                                    .transform_filter(brush)
                                )
                                diff_chart = diff_chart + tm_points
                        except Exception:
                            pass

                        diff_chart = diff_chart.add_params(brush)
                        # Mostrar ambos gráficos en la misma línea (layout original)
                        diff_chart_full = diff_chart
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.altair_chart(chart_full, use_container_width=True)
                        with col_b:
                            st.altair_chart(diff_chart_full, use_container_width=True)
                    except Exception:
                        pass

                # (Se elimina sección de estadística desde PBP a pedido del usuario)

            # Tercer gráfico: Proporción de tiempo por estado del marcador (Local ganando / Empate / Visitante ganando)
            try:
                if not pbp_df.empty:
                    # Usar el mismo dfp (filtrado y ordenado) si existe; en su defecto, construirlo
                    if 'dfp' not in locals():
                        dfp = pbp_df.copy()
                        # Quitar columnas legacy con underscore si existen
                        dfp = dfp.drop(columns=['puntos_local', 'puntos_visitante'], errors='ignore')
                        if 'autoincremental_id' in dfp.columns and 'autoincremental_id_num' not in dfp.columns:
                            dfp['autoincremental_id_num'] = pd.to_numeric(dfp['autoincremental_id'], errors='coerce').fillna(0)
                        if 'autoincremental_id_num' in dfp.columns:
                            dfp = dfp.sort_values(by=['_id','numero_periodo','autoincremental_id_num'])
                        elif 'tiempo_segundos' in dfp.columns:
                            dfp = dfp.sort_values(by=['_id','numero_periodo','tiempo_segundos'])
                        dfp = dfp.reset_index(drop=True)
                        dfp['orden'] = np.arange(len(dfp))
                        if 'x_period' not in dfp.columns:
                            tiempo_num = pd.to_numeric(dfp.get('tiempo_segundos', np.nan), errors='coerce')
                            periodo_num = pd.to_numeric(dfp.get('numero_periodo', 1), errors='coerce').fillna(1)
                            tiempo_num = tiempo_num.fillna(dfp['orden'])
                            dfp['x_period'] = (600 - tiempo_num) + (periodo_num - 1) * 600
                        # Asegurar conversión on-the-fly cuando se use

                    # Aplicar filtro de periodo si corresponde (para mantener coherencia con los gráficos anteriores)
                    if 'sel_periodo' in locals() and sel_periodo != 'TODOS':
                        try:
                            sel_val = int(sel_periodo)
                            dfp = dfp[pd.to_numeric(dfp['numero_periodo'], errors='coerce') == sel_val].copy()
                        except Exception:
                            pass

                    # Asegurar tipos y ordenar SIEMPRE por autoincremental_id_num cuando exista
                    work = dfp.copy()
                    work['numero_periodo'] = pd.to_numeric(work.get('numero_periodo', 1), errors='coerce').fillna(1).astype(int)
                    work['x_period'] = pd.to_numeric(work.get('x_period', 0), errors='coerce').fillna(0)
                    if 'autoincremental_id_num' not in work.columns and 'autoincremental_id' in work.columns:
                        work['autoincremental_id_num'] = pd.to_numeric(work['autoincremental_id'], errors='coerce').fillna(0)
                    if 'autoincremental_id_num' in work.columns:
                        work = work.sort_values(by=['numero_periodo', 'autoincremental_id_num']).reset_index(drop=True)
                    else:
                        work = work.sort_values(by=['numero_periodo', 'x_period']).reset_index(drop=True)

                    t_local = 0.0
                    t_empate = 0.0
                    t_visit = 0.0
                    # Acumular por periodo, incluyendo desde el inicio del periodo hasta la primera jugada (empate)
                    # y desde la última jugada hasta el final del periodo usando el signo de la última diferencia
                    for per, df_per in work.groupby('numero_periodo'):
                        df_per = df_per.sort_values(by=['x_period']).reset_index(drop=True)
                        # Inicio y fin absolutos del periodo en x_period
                        try:
                            per_int = int(per)
                        except Exception:
                            per_int = 1
                        x_start = (per_int - 1) * 600.0
                        x_end = per_int * 600.0
                        # Si hay al menos un evento, sumar tramo inicial como empate hasta la primera jugada
                        if len(df_per) > 0:
                            first = df_per.iloc[0]
                            first_x = float(first.get('x_period', x_start))
                            if first_x > x_start:
                                t_empate += (first_x - x_start)
                            # Entre jugadas
                            prev = None
                            for _, r in df_per.iterrows():
                                if prev is not None:
                                    if str(prev.get('accion_tipo', '')).upper() != 'FINAL-PERIODO':
                                        dt = float(r.get('x_period', 0)) - float(prev.get('x_period', 0))
                                        if dt < 0:
                                            dt = abs(float(r.get('tiempo_segundos', 0)) - float(prev.get('tiempo_segundos', 0)))
                                        try:
                                            pl_prev = float(prev.get('puntosLocal', 0))
                                            pv_prev = float(prev.get('puntosVisitante', 0))
                                        except Exception:
                                            pl_prev = float(pd.to_numeric(prev.get('puntosLocal', 0), errors='coerce'))
                                            pv_prev = float(pd.to_numeric(prev.get('puntosVisitante', 0), errors='coerce'))
                                        dif_prev = pl_prev - pv_prev
                                        if dif_prev > 0:
                                            t_local += dt
                                        elif dif_prev < 0:
                                            t_visit += dt
                                        else:
                                            t_empate += dt
                                prev = r
                            # Tramo final hasta el fin del periodo
                            last = df_per.iloc[-1]
                            last_x = float(last.get('x_period', x_end))
                            if x_end > last_x:
                                dif_last = float(last.get('DifPuntos_num', 0))
                                dt_end = x_end - last_x
                                if dif_last > 0:
                                    t_local += dt_end
                                elif dif_last < 0:
                                    t_visit += dt_end
                                else:
                                    t_empate += dt_end

                    total_t = t_local + t_empate + t_visit
                    if total_t > 0:
                        # Datos para barra apilada (etiquetas con nombre de equipo en MAYÚSCULAS)
                        estados = [f'{local_name.upper()} GANANDO', 'EMPATE', f'{visitante_name.upper()} GANANDO']  # orden fijo solicitado
                        segundos = [t_local, t_empate, t_visit]
                        # Ajustar color de Empate si coincide con algún color de equipo (gris) para no confundir
                        empate_base = '#9e9e9e'
                        empate_color = empate_base
                        if str(color_local).lower() == empate_base or str(color_visitante).lower() == empate_base:
                            # Aclarar un poco el gris de empate para distinguir
                            empate_color = _adjust_color(empate_base, 1.2)
                        colores = [color_local, empate_color, color_visitante]
                        fracciones = [s / total_t for s in segundos]
                        porcentajes = [f * 100.0 for f in fracciones]
                        # Color de texto por segmento para asegurar contraste (etiquetas legibles)
                        text_colors = [
                            _text_color_for_bg(color_local),
                            _text_color_for_bg(empate_color),
                            _text_color_for_bg(color_visitante)
                        ]
                        # Trazo para equipos con color blanco para que se vean los límites
                        local_is_white = str(color_local).lower() in ('#ffffff', '#fff', 'white')
                        visit_is_white = str(color_visitante).lower() in ('#ffffff', '#fff', 'white')
                        strokes = [
                            '#000000' if local_is_white else None,
                            None,
                            '#000000' if visit_is_white else None
                        ]
                        start_fracs = [0.0, fracciones[0], fracciones[0] + fracciones[1]]
                        mid_fracs = [start_fracs[i] + fracciones[i] / 2.0 for i in range(3)]
                        # Color de etiqueta: negro por defecto, blanco si el color del equipo es negro
                        def is_black(c: str) -> bool:
                            c = str(c).strip().lower()
                            return c in ('#000000', '#000', 'black')
                        label_colors = [
                            '#ffffff' if is_black(color_local) else '#000000',  # Local
                            '#000000',                                         # Empate
                            '#ffffff' if is_black(color_visitante) else '#000000'  # Visitante
                        ]
                        df_bar = pd.DataFrame({
                            'Estado': estados,
                            'EstadoOrden': [0, 1, 2],  # asegurar orden en la pila (izq->der): Local, Empate, Visitante
                            'Segundos': segundos,
                            'Fraccion': fracciones,
                            'Porcentaje': porcentajes,
                            'StartFrac': start_fracs,
                            'MidFrac': mid_fracs,
                            'Stroke': strokes,
                            'TextColor': text_colors,
                            'LabelColor': label_colors,
                            'y': [' ', ' ', ' ']  # única fila para que sea una barra única
                        })

                        # Separador visual entre filas
                        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                        bar = (
                            alt.Chart(df_bar)
                            .mark_bar()
                            .encode(
                                y=alt.Y('y:N', axis=None, scale=alt.Scale(paddingInner=0, paddingOuter=0)),
                                x=alt.X('Segundos:Q', stack='normalize', axis=None),
                                color=alt.Color(
                                    'Estado:N',
                                    scale=alt.Scale(domain=estados, range=colores),
                                    legend=alt.Legend(orient='top', direction='horizontal', title=None, labelLimit=1000)
                                ),
                                order=alt.Order('EstadoOrden:Q'),
                                stroke=alt.Color('Stroke:N', scale=None, legend=None),
                                tooltip=[
                                    alt.Tooltip('Estado:N'),
                                    alt.Tooltip('Porcentaje:Q', format='.0f', title='Porcentaje (%)')
                                ]
                            )
                            .properties(height=180, title=alt.TitleParams(text='Tiempo por estado del marcador', anchor='middle'))
                        )
                        # Etiquetas de porcentaje centradas dentro de cada segmento
                        labels = (
                            alt.Chart(df_bar)
                            .mark_text(fontSize=18, fontWeight=700, align='center', stroke='#dddddd', strokeWidth=0.35)
                            .encode(
                                y=alt.Y('y:N'),
                                x=alt.X('MidFrac:Q', axis=None, scale=alt.Scale(domain=[0,1])),
                                color=alt.Color('LabelColor:N', scale=None, legend=None),
                                order=alt.Order('EstadoOrden:Q'),
                                detail='Estado:N',
                                text=alt.Text('Fraccion:Q', format='.0%')
                            )
                        )
                        # Mostrar la barra en una nueva fila de columnas, ocupando la primera columna
                        row2_col1, row2_col2 = st.columns(2)
                        with row2_col1:
                            st.altair_chart(bar + labels, use_container_width=True)
                        with row2_col2:
                            # Resumen textual: mejor racha pura (X-0) y mayor sequía para cada equipo, con rango temporal
                            try:
                                timeline = dfp[['x_period','puntosLocal','puntosVisitante','numero_periodo','tiempo_segundos']].copy()
                                if 'autoincremental_id_num' not in dfp.columns and 'autoincremental_id' in dfp.columns:
                                    dfp['autoincremental_id_num'] = pd.to_numeric(dfp['autoincremental_id'], errors='coerce').fillna(0)
                                if 'autoincremental_id_num' in dfp.columns:
                                    timeline = timeline.join(dfp['autoincremental_id_num'])
                                    timeline = timeline.sort_values(by=['numero_periodo','autoincremental_id_num']).reset_index(drop=True)
                                else:
                                    timeline = timeline.sort_values(by=['numero_periodo','x_period']).reset_index(drop=True)
                                # Convertir a numérico para cálculos
                                timeline['puntosLocal'] = pd.to_numeric(timeline['puntosLocal'], errors='coerce').fillna(0)
                                timeline['puntosVisitante'] = pd.to_numeric(timeline['puntosVisitante'], errors='coerce').fillna(0)
                                if len(timeline) >= 2:
                                    # Longest drought per team: track start/end rows
                                    first_x = float(timeline.iloc[0]['x_period'])
                                    last_x = float(timeline.iloc[-1]['x_period'])
                                    last_change_L = first_x
                                    last_change_V = first_x
                                    drought_start_L = timeline.iloc[0]
                                    drought_start_V = timeline.iloc[0]
                                    drought_best_L = (0.0, timeline.iloc[0], timeline.iloc[0])
                                    drought_best_V = (0.0, timeline.iloc[0], timeline.iloc[0])
                                    prev_PL = float(timeline.iloc[0]['puntosLocal'])
                                    prev_PV = float(timeline.iloc[0]['puntosVisitante'])
                                    for i in range(1, len(timeline)):
                                        rowi = timeline.iloc[i]
                                        x = float(rowi['x_period'])
                                        PL = float(rowi['puntosLocal'])
                                        PV = float(rowi['puntosVisitante'])
                                        if PL != prev_PL:
                                            dur = x - last_change_L
                                            if dur > drought_best_L[0]:
                                                drought_best_L = (dur, drought_start_L, rowi)
                                            last_change_L = x
                                            drought_start_L = rowi
                                        if PV != prev_PV:
                                            dur = x - last_change_V
                                            if dur > drought_best_V[0]:
                                                drought_best_V = (dur, drought_start_V, rowi)
                                            last_change_V = x
                                            drought_start_V = rowi
                                        prev_PL, prev_PV = PL, PV
                                    # considerar sequía hasta el final del juego
                                    end_dummy = timeline.iloc[-1]
                                    dur = last_x - last_change_L
                                    if dur > drought_best_L[0]:
                                        drought_best_L = (dur, drought_start_L, end_dummy)
                                    dur = last_x - last_change_V
                                    if dur > drought_best_V[0]:
                                        drought_best_V = (dur, drought_start_V, end_dummy)

                                    # Best pure scoring run per team: X-0 (oponente no anota en ese tramo)
                                    def best_pure_run(favor_is_local: bool):
                                        prev = timeline.iloc[0]
                                        prev_PL = float(prev['puntosLocal'])
                                        prev_PV = float(prev['puntosVisitante'])
                                        run_pts = 0.0
                                        run_start = None
                                        best_pts = 0.0
                                        best_start = None
                                        best_end = None
                                        for i in range(1, len(timeline)):
                                            cur = timeline.iloc[i]
                                            PL = float(cur['puntosLocal'])
                                            PV = float(cur['puntosVisitante'])
                                            dL = PL - prev_PL
                                            dV = PV - prev_PV
                                            # Determinar si anota el equipo a analizar y el otro NO
                                            if favor_is_local:
                                                scores = (dL > 0 and dV == 0)
                                                rival_scores = (dV > 0)
                                                inc = dL
                                            else:
                                                scores = (dV > 0 and dL == 0)
                                                rival_scores = (dL > 0)
                                                inc = dV
                                            if scores:
                                                if run_start is None:
                                                    run_start = timeline.iloc[i-1]
                                                    run_pts = 0.0
                                                run_pts += inc
                                            if rival_scores or (dL > 0 and dV > 0):
                                                # Cierra la racha pura si estaba abierta
                                                if run_start is not None and run_pts > best_pts:
                                                    best_pts, best_start, best_end = run_pts, run_start, timeline.iloc[i]
                                                run_start = None
                                                run_pts = 0.0
                                            prev_PL, prev_PV = PL, PV
                                        # cerrar al final
                                        if run_start is not None and run_pts > best_pts:
                                            best_pts, best_start, best_end = run_pts, run_start, timeline.iloc[-1]
                                        return int(best_pts), best_start, best_end

                                    best_L_pts, sL, eL = best_pure_run(True)
                                    best_V_pts, sV, eV = best_pure_run(False)

                                    def fmt_when(srow, erow):
                                        try:
                                            p1 = int(srow['numero_periodo'])
                                            t1 = float(srow['tiempo_segundos']) if pd.notna(srow['tiempo_segundos']) else 0.0
                                            p2 = int(erow['numero_periodo'])
                                            t2 = float(erow['tiempo_segundos']) if pd.notna(erow['tiempo_segundos']) else 0.0
                                            def mmss(x):
                                                m = int(x // 60)
                                                s = int(x % 60)
                                                return f"{m}:{s:02d}"
                                            return f"P{p1} {mmss(t1)} → P{p2} {mmss(t2)}"
                                        except Exception:
                                            return ""

                                    def fmt_drought_tuple(dt):
                                        sec = dt[0]
                                        srow = dt[1]
                                        erow = dt[2]
                                        m = int(sec // 60)
                                        s = int(sec % 60)
                                        return f"{m}m {s:02d}s ({fmt_when(srow, erow)})"

                                    # Tarjetas con color de equipo
                                    card_style = "border-radius:10px;padding:12px 14px;margin:6px 0;box-shadow:0 1px 5px rgba(0,0,0,.08)"
                                    # Calcular puntos titulares/suplentes por equipo cruzando PBP con planillas (respetando filtro de periodo ya aplicado a dfp)
                                    try:
                                        pbp_pts = dfp.copy()
                                        # Columnas candidatas
                                        dorsal_candidates = ['dorsal','numero','nro','número','numero_camiseta','n_camisa']
                                        pbp_dorsal_col = _first_col(pbp_pts, dorsal_candidates)
                                        if pbp_dorsal_col is None or pbp_dorsal_col not in pbp_pts.columns:
                                            raise ValueError('PBP sin columna de dorsal reconocible')
                                        # Normalizar Condicion
                                        if 'Condicion' in pbp_pts.columns:
                                            pbp_pts['Condicion'] = pbp_pts['Condicion'].astype(str).str.upper().fillna('')
                                        else:
                                            pbp_pts['Condicion'] = ''
                                        # Mapear puntos por tipo de canasta
                                        at = pbp_pts.get('accion_tipo', '').astype(str).str.upper()
                                        pts_event = np.where(at == 'CANASTA-1P', 1,
                                                      np.where(at == 'CANASTA-2P', 2,
                                                      np.where(at == 'CANASTA-3P', 3, 0)))
                                        pbp_pts['__puntos'] = pts_event
                                        # Agregar por Condicion y dorsal
                                        jg_grp = (
                                            pbp_pts.groupby(['Condicion', pbp_dorsal_col], as_index=False)['__puntos']
                                            .sum()
                                            .rename(columns={pbp_dorsal_col: '__dorsal'})
                                        )
                                    except Exception:
                                        jg_grp = pd.DataFrame(columns=['Condicion','__dorsal','__puntos'])

                                    # Obtener planillas por equipo y detectar titularidad y dorsal
                                    def titular_sets(planilla_df: pd.DataFrame):
                                        if planilla_df is None or planilla_df.empty:
                                            return set(), set()
                                        dcol = _first_col(planilla_df, ['dorsal','numero','nro','número','numero_camiseta','n_camisa'])
                                        tcol = _first_col(planilla_df, ['quintetotitular','quinteto_titular','QuintetoTitular','quintetoTitular','titular','es_titular'])
                                        if not dcol or dcol not in planilla_df.columns or not tcol or tcol not in planilla_df.columns:
                                            return set(), set()
                                        dfp2 = planilla_df[[dcol, tcol]].copy()
                                        def is_true(v):
                                            try:
                                                if isinstance(v, (int, float)):
                                                    return float(v) != 0.0
                                                s = str(v).strip().lower()
                                                return s in ('si','sí','true','t','1','x','s')
                                            except Exception:
                                                return False
                                        dfp2['__titular'] = dfp2[tcol].apply(is_true)
                                        tit = set(pd.to_numeric(dfp2.loc[dfp2['__titular'], dcol], errors='coerce').dropna().astype(int).tolist())
                                        supl = set(pd.to_numeric(dfp2.loc[~dfp2['__titular'], dcol], errors='coerce').dropna().astype(int).tolist())
                                        return tit, supl

                                    tit_loc, supl_loc = titular_sets(est_loc_df)
                                    tit_vis, supl_vis = titular_sets(est_vis_df)

                                    # Función para sumar puntos por conjunto de dorsales
                                    def sum_pts(cond: str, dorsales: set) -> float:
                                        if jg_grp is None or jg_grp.empty or not dorsales:
                                            return 0.0
                                        tmp = jg_grp[jg_grp['Condicion'].astype(str).str.upper()==cond.upper()].copy()
                                        if tmp.empty:
                                            return 0.0
                                        tmp['__d'] = pd.to_numeric(tmp['__dorsal'], errors='coerce').astype('Int64')
                                        s = tmp[tmp['__d'].isin(list(dorsales))]['__puntos'].sum()
                                        try:
                                            return float(s)
                                        except Exception:
                                            return 0.0

                                    # Calcular puntos por equipo
                                    pts_tit_loc = sum_pts('LOCAL', tit_loc)
                                    pts_sup_loc = sum_pts('LOCAL', supl_loc)
                                    pts_tit_vis = sum_pts('VISITANTE', tit_vis)
                                    pts_sup_vis = sum_pts('VISITANTE', supl_vis)

                                    # Puntos en la zona (def: apariciones de CANASTA-2P * 2) respetando filtros de periodo aplicados a dfp
                                    try:
                                        df_zone = dfp.copy()
                                        if 'accion_tipo' in df_zone.columns:
                                            at2 = df_zone['accion_tipo'].astype(str).str.upper()
                                            df_zone['__is_c2p'] = (at2 == 'CANASTA-2P')
                                        else:
                                            df_zone['__is_c2p'] = False
                                        if 'Condicion' in df_zone.columns:
                                            df_zone['Condicion'] = df_zone['Condicion'].astype(str).str.upper().fillna('')
                                        # Detectar columna de zona
                                        zona_candidates = ['zona', 'zona_tiro', 'tiro_zona', 'zona_accion', 'zona_tiro_codigo', 'zonaCodigo', 'zona_codigo']
                                        zona_col = None
                                        for zc in zona_candidates:
                                            if zc in df_zone.columns:
                                                zona_col = zc
                                                break
                                        if zona_col is not None:
                                            zvals = df_zone[zona_col].astype(str).fillna('')
                                            df_zone['__is_z1'] = zvals.str.upper().str.startswith('Z1-')
                                        else:
                                            df_zone['__is_z1'] = False
                                        mask_loc = (df_zone['Condicion']=='LOCAL') & df_zone['__is_c2p'] & df_zone['__is_z1']
                                        mask_vis = (df_zone['Condicion']=='VISITANTE') & df_zone['__is_c2p'] & df_zone['__is_z1']
                                        cnt_loc = int(df_zone[mask_loc].shape[0])
                                        cnt_vis = int(df_zone[mask_vis].shape[0])
                                        pzona_loc = 2 * cnt_loc
                                        pzona_vis = 2 * cnt_vis
                                    except Exception:
                                        pzona_loc = 0
                                        pzona_vis = 0

                                    # Tarjetas (posición original), con contenido reordenado
                                    st.markdown(
                                        f"""
                                        <div style='{card_style};background:{color_local};color:{tc_local};'>
                                            <div style='font-weight:700;margin-bottom:6px;'>{local_name}</div>
                                            <div>Puntos titulares: <strong>{int(pts_tit_loc)}</strong></div>
                                            <div>Puntos suplentes: <strong>{int(pts_sup_loc)}</strong></div>
                                            <div>Puntos en la zona (Z1): <strong>{int(pzona_loc)}</strong></div>
                                            <div>Mejor racha: <strong>{best_L_pts}-0</strong> ({fmt_when(sL, eL) if sL is not None else ''})</div>
                                            <div>Mayor sequía: <strong>{fmt_drought_tuple(drought_best_L)}</strong></div>
                                        </div>
                                        <div style='{card_style};background:{color_visitante};color:{tc_visitante};'>
                                            <div style='font-weight:700;margin-bottom:6px;'>{visitante_name}</div>
                                            <div>Puntos titulares: <strong>{int(pts_tit_vis)}</strong></div>
                                            <div>Puntos suplentes: <strong>{int(pts_sup_vis)}</strong></div>
                                            <div>Puntos en la zona (Z1): <strong>{int(pzona_vis)}</strong></div>
                                            <div>Mejor racha: <strong>{best_V_pts}-0</strong> ({fmt_when(sV, eV) if sV is not None else ''})</div>
                                            <div>Mayor sequía: <strong>{fmt_drought_tuple(drought_best_V)}</strong></div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.info('Sin suficientes eventos para resumen de rachas')
                            except Exception as e:
                                st.warning(f"No se pudo calcular el resumen de rachas: {e}")
                    else:
                        st.info('No se pudo calcular el tiempo por estado del marcador (total de tiempo = 0).')
            except Exception as e:
                st.warning(f"No se pudo renderizar la barra de estados del marcador: {e}")

            # (Se eliminó la fila 3 comparativa en Resumen a pedido del usuario)

            # Comparativa de totales (desde "estadisticas por jugador")
            try:
                jg_tot = tablas.get('jugadoresAgregado', pd.DataFrame()).copy()
                if isinstance(jg_tot, pd.DataFrame) and not jg_tot.empty:
                    # Respetar filtro de periodo de la pestaña (si existe columna y selección)
                    if 'sel_periodo' in locals() and sel_periodo != 'TODOS':
                        if 'numero_periodo' in jg_tot.columns:
                            try:
                                sel_val = int(sel_periodo)
                                jg_tot = jg_tot[pd.to_numeric(jg_tot['numero_periodo'], errors='coerce') == sel_val].copy()
                            except Exception:
                                pass
                    # Normalizar Condicion
                    if 'Condicion' in jg_tot.columns:
                        jg_tot['Condicion'] = jg_tot['Condicion'].astype(str).str.upper().fillna('')
                    # Mapeo estricto como en la tabla de jugadores
                    strict_map = [
                        ('ASISTENCIA', 'asistencias'),
                        ('CANASTA-1P', 'canasta1p'),
                        ('CANASTA-2P', 'canasta2p'),
                        ('CANASTA-3P', 'canasta3p'),
                        ('FALTA-COMETIDA', 'faltascometidas'),
                        ('FALTA-RECIBIDA', 'faltasrecibidas'),
                        ('PERDIDA', 'perdidas'),
                        ('REBOTE-DEFENSIVO', 'rebotedefensivo'),
                        ('REBOTE-OFENSIVO', 'reboteofensivo'),
                        ('RECUPERACION', 'recuperaciones'),
                        ('TIRO1-FALLADO', 'tiro1fallado'),
                        ('TIRO2-FALLADO', 'tiro2fallado'),
                        ('TIRO3-FALLADO', 'tiro3fallado'),
                    ]
                    for src, dst in strict_map:
                        if src in jg_tot.columns:
                            jg_tot[dst] = pd.to_numeric(jg_tot[src], errors='coerce').fillna(0)
                        elif dst not in jg_tot.columns:
                            jg_tot[dst] = 0
                    # Derivadas
                    jg_tot['rebotetotal'] = jg_tot['rebotedefensivo'] + jg_tot['reboteofensivo']
                    jg_tot['puntos'] = jg_tot['canasta1p'] + 2*jg_tot['canasta2p'] + 3*jg_tot['canasta3p']
                    # Plus-minus
                    pm_cands = ['diferencia', 'plusminus', 'plus_minus', '+-']
                    pm_series = None
                    for pmc in pm_cands:
                        if pmc in jg_tot.columns:
                            s = pd.to_numeric(jg_tot[pmc], errors='coerce').fillna(0)
                            pm_series = s if pm_series is None else (pm_series + s)
                    if pm_series is None:
                        jg_tot['pm'] = 0
                    else:
                        jg_tot['pm'] = pm_series

                    # Sumar por condicion
                    vars_keep = {
                        'puntos': 'Puntos',
                        'rebotedefensivo': 'Rebote Def.',
                        'reboteofensivo': 'Rebote Of.',
                        'rebotetotal': 'Rebotes Totales',
                        'asistencias': 'Asistencias',
                        'perdidas': 'Pérdidas',
                        'recuperaciones': 'Recuperaciones',
                        'pm': '+/-',
                    }
                    agg_tot = (
                        jg_tot.groupby('Condicion')[list(vars_keep.keys())]
                        .sum(numeric_only=True)
                        .reset_index()
                    )
                    # Asegurar presencia de ambas filas LOCAL y VISITANTE con 0s si faltan
                    needed = ['LOCAL', 'VISITANTE']
                    for cond in needed:
                        if cond not in agg_tot['Condicion'].astype(str).tolist():
                            row0 = {**{k: 0.0 for k in vars_keep.keys()}, 'Condicion': cond}
                            agg_tot = pd.concat([agg_tot, pd.DataFrame([row0])], ignore_index=True)
                    # Preparar tidy para Altair
                    def equipo_name(cond):
                        return local_name if str(cond).upper()=='LOCAL' else (visitante_name if str(cond).upper()=='VISITANTE' else str(cond))
                    rows = []
                    for _, r in agg_tot.iterrows():
                        eq = equipo_name(r['Condicion'])
                        for k, label in vars_keep.items():
                            rows.append({'Equipo': eq, 'Variable': label, 'Valor': float(r.get(k, 0))})
                    df_vars = pd.DataFrame(rows)
                    if not df_vars.empty:
                        # Colores por equipo y helper de chart
                        color_scale = alt.Scale(domain=[local_name, visitante_name], range=[color_local, color_visitante])

                        # Construir agregados por equipo para porcentajes
                        def ensure_cols(df, cols):
                            for c in cols:
                                if c not in df.columns:
                                    df[c] = 0
                            return df
                        jg_cnt = jg_tot.copy()
                        jg_cnt = ensure_cols(jg_cnt, ['CANASTA-1P','TIRO1-FALLADO','CANASTA-2P','TIRO2-FALLADO','CANASTA-3P','TIRO3-FALLADO'])
                        grp_cnt = (
                            jg_cnt.groupby('Condicion')[['CANASTA-1P','TIRO1-FALLADO','CANASTA-2P','TIRO2-FALLADO','CANASTA-3P','TIRO3-FALLADO']]
                            .sum(numeric_only=True)
                            .reset_index()
                        )
                        # Asegurar filas para ambos equipos
                        for cond in ['LOCAL','VISITANTE']:
                            if cond not in grp_cnt['Condicion'].astype(str).tolist():
                                grp_cnt = pd.concat([
                                    grp_cnt,
                                    pd.DataFrame([{'Condicion': cond, 'CANASTA-1P':0,'TIRO1-FALLADO':0,'CANASTA-2P':0,'TIRO2-FALLADO':0,'CANASTA-3P':0,'TIRO3-FALLADO':0}])
                                ], ignore_index=True)

                        def equipo_name(cond):
                            return local_name if str(cond).upper()=='LOCAL' else (visitante_name if str(cond).upper()=='VISITANTE' else str(cond))

                        def bar_chart_for(variable_label, valores_dict, y_title='Total', is_percent=False):
                            data = [{'Equipo': k, 'Valor': valores_dict.get(k, 0)} for k in [local_name, visitante_name]]
                            dfc = pd.DataFrame(data)
                            enc_y = alt.Y('Valor:Q', title=y_title, scale=alt.Scale(domain=[0,100])) if is_percent else alt.Y('Valor:Q', title=y_title)
                            # Barras
                            bars = (
                                alt.Chart(dfc)
                                .mark_bar(stroke='#000000', strokeWidth=1)
                                .encode(
                                    x=alt.X('Equipo:N', title=None, sort=[local_name, visitante_name], axis=alt.Axis(labelAngle=315)),
                                    y=enc_y,
                                    color=alt.Color('Equipo:N', scale=color_scale, legend=None),
                                    tooltip=[alt.Tooltip('Equipo:N'), alt.Tooltip('Valor:Q', format='.0f' if not is_percent else '.0f')]
                                )
                            )
                            # Etiquetas
                            if is_percent:
                                text = (
                                    alt.Chart(dfc)
                                    .transform_calculate(label="toString(round(datum.Valor)) + '%'")
                                    .mark_text(dy=-6, fontSize=16, fontWeight='bold')
                                    .encode(
                                        x=alt.X('Equipo:N', sort=[local_name, visitante_name], axis=alt.Axis(labelAngle=315)),
                                        y=enc_y,
                                        text='label:N',
                                        color=alt.value('#000000')
                                    )
                                )
                            else:
                                text = (
                                    alt.Chart(dfc)
                                    .mark_text(dy=-6, fontSize=16, fontWeight='bold')
                                    .encode(
                                        x=alt.X('Equipo:N', sort=[local_name, visitante_name], axis=alt.Axis(labelAngle=315)),
                                        y=enc_y,
                                        text=alt.Text('Valor:Q', format='.0f'),
                                        color=alt.value('#000000')
                                    )
                                )
                            return (bars + text).properties(height=340, title=variable_label)

                        # Calcular valores por equipo
                        def get_total(cond, col):
                            try:
                                return float(agg_tot.loc[agg_tot['Condicion'].astype(str)==cond, col].sum())
                            except Exception:
                                return 0.0
                        # Puntos y rebotes/asist/perd/recup
                        vals_puntos = {equipo_name('LOCAL'): get_total('LOCAL','puntos'), equipo_name('VISITANTE'): get_total('VISITANTE','puntos')}
                        vals_rd = {equipo_name('LOCAL'): get_total('LOCAL','rebotedefensivo'), equipo_name('VISITANTE'): get_total('VISITANTE','rebotedefensivo')}
                        vals_ro = {equipo_name('LOCAL'): get_total('LOCAL','reboteofensivo'), equipo_name('VISITANTE'): get_total('VISITANTE','reboteofensivo')}
                        vals_rt = {equipo_name('LOCAL'): get_total('LOCAL','rebotetotal'), equipo_name('VISITANTE'): get_total('VISITANTE','rebotetotal')}
                        vals_ast = {equipo_name('LOCAL'): get_total('LOCAL','asistencias'), equipo_name('VISITANTE'): get_total('VISITANTE','asistencias')}
                        vals_per = {equipo_name('LOCAL'): get_total('LOCAL','perdidas'), equipo_name('VISITANTE'): get_total('VISITANTE','perdidas')}
                        vals_rec = {equipo_name('LOCAL'): get_total('LOCAL','recuperaciones'), equipo_name('VISITANTE'): get_total('VISITANTE','recuperaciones')}

                        # Porcentajes
                        def pct(conv, fall):
                            a = conv + fall
                            return (conv / a * 100.0) if a > 0 else 0.0
                        def get_pct(cond, conv_col, miss_col):
                            row = grp_cnt[grp_cnt['Condicion'].astype(str)==cond]
                            c = float(row[conv_col].sum()) if not row.empty else 0.0
                            m = float(row[miss_col].sum()) if not row.empty else 0.0
                            return pct(c, m)
                        vals_1p = {
                            equipo_name('LOCAL'): get_pct('LOCAL','CANASTA-1P','TIRO1-FALLADO'),
                            equipo_name('VISITANTE'): get_pct('VISITANTE','CANASTA-1P','TIRO1-FALLADO')
                        }
                        vals_2p = {
                            equipo_name('LOCAL'): get_pct('LOCAL','CANASTA-2P','TIRO2-FALLADO'),
                            equipo_name('VISITANTE'): get_pct('VISITANTE','CANASTA-2P','TIRO2-FALLADO')
                        }
                        vals_3p = {
                            equipo_name('LOCAL'): get_pct('LOCAL','CANASTA-3P','TIRO3-FALLADO'),
                            equipo_name('VISITANTE'): get_pct('VISITANTE','CANASTA-3P','TIRO3-FALLADO')
                        }

                        # Render en grilla 5 por fila
                        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                        st.subheader('Comparativas por variable (Local vs Visitante)')
                        row1 = st.columns(5)
                        with row1[0]:
                            st.altair_chart(bar_chart_for('Puntos', vals_puntos, 'Total'), use_container_width=True)
                        with row1[1]:
                            st.altair_chart(bar_chart_for('%1P', vals_1p, '%', is_percent=True), use_container_width=True)
                        with row1[2]:
                            st.altair_chart(bar_chart_for('%2P', vals_2p, '%', is_percent=True), use_container_width=True)
                        with row1[3]:
                            st.altair_chart(bar_chart_for('%3P', vals_3p, '%', is_percent=True), use_container_width=True)
                        with row1[4]:
                            st.altair_chart(bar_chart_for('Reb. Def.', vals_rd, 'Total'), use_container_width=True)

                        row2 = st.columns(5)
                        with row2[0]:
                            st.altair_chart(bar_chart_for('Reb. Of.', vals_ro, 'Total'), use_container_width=True)
                        with row2[1]:
                            st.altair_chart(bar_chart_for('Reb. Tot.', vals_rt, 'Total'), use_container_width=True)
                        with row2[2]:
                            st.altair_chart(bar_chart_for('Asistencias', vals_ast, 'Total'), use_container_width=True)
                        with row2[3]:
                            st.altair_chart(bar_chart_for('Pérdidas', vals_per, 'Total'), use_container_width=True)
                        with row2[4]:
                            st.altair_chart(bar_chart_for('Recuperaciones', vals_rec, 'Total'), use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo construir la comparativa de totales: {e}")

            # Pestaña Estadisticas por jugador (desde jugadoresAgregado por jugador, con Totales y derivadas)
            with t_estadistica:
                part_df = tablas.get('partido', pd.DataFrame())
                local_title = str(part_df.iloc[0].get('local')) if not part_df.empty else 'Local'
                visitante_title = str(part_df.iloc[0].get('visitante')) if not part_df.empty else 'Visitante'
                jg = tablas.get('jugadoresAgregado', pd.DataFrame()).copy()
                # Planillas por equipo para detectar titulares
                plan_loc = tablas.get('estadisticas_equipolocal', pd.DataFrame()).copy()
                plan_vis = tablas.get('estadisticas_equipovisitante', pd.DataFrame()).copy()
                def titular_sets(planilla_df: pd.DataFrame):
                    if planilla_df is None or planilla_df.empty:
                        return set()
                    dcol = _first_col(planilla_df, ['dorsal','numero','nro','número','numero_camiseta','n_camisa'])
                    tcol = _first_col(planilla_df, ['quintetotitular','quinteto_titular','QuintetoTitular','quintetoTitular','titular','es_titular'])
                    if not dcol or dcol not in planilla_df.columns or not tcol or tcol not in planilla_df.columns:
                        return set()
                    dfp2 = planilla_df[[dcol, tcol]].copy()
                    def is_true(v):
                        try:
                            if isinstance(v, (int, float)):
                                return float(v) != 0.0
                            s = str(v).strip().lower()
                            return s in ('si','sí','true','t','1','x','s')
                        except Exception:
                            return False
                    dfp2['__titular'] = dfp2[tcol].apply(is_true)
                    tit = set(pd.to_numeric(dfp2.loc[dfp2['__titular'], dcol], errors='coerce').dropna().astype(int).tolist())
                    return tit
                tit_set_loc = titular_sets(plan_loc)
                tit_set_vis = titular_sets(plan_vis)
                if not jg.empty:
                    # Ya no necesitamos arreglar nombres "NOMBRE" aquí, ya están formateados correctamente
                    # en la generación de jugadoresAgregado
                    # Normalizar y detectar columnas
                    if 'Condicion' in jg.columns:
                        jg['Condicion'] = jg['Condicion'].astype(str).str.upper().fillna('')
                    if 'equipo' in jg.columns:
                        jg['equipo'] = jg['equipo'].astype(str).str.strip()

                    period_col = _first_col(jg, ['numero_periodo', 'periodo', 'Periodo'])
                    situ_col = _first_col(jg, ['SituacionMarcador', 'Situacion marcador', 'situacion_marcador', 'situacionMarcador', 'situacion', 'Situacion'])
                    u2m_col = _first_col(jg, ['ultimos_dos_minutos', 'ultimos2min', 'ultimos_dos', 'u2m'])

                    # Filtros en formulario para evitar cambio de pestaña al aplicar
                    st.markdown("### Filtros")
                    with st.form(key='estadisticas_filters'):
                        cols_f = st.columns(3)
                        with cols_f[0]:
                            if period_col and period_col in jg.columns:
                                per_opts = ['TODOS'] + sorted(pd.to_numeric(jg[period_col], errors='coerce').dropna().astype(int).unique().tolist())
                                sel_per = st.selectbox('Periodo', per_opts, index=0, key='estad_sel_per')
                            else:
                                sel_per = 'TODOS'
                        with cols_f[1]:
                            if situ_col and situ_col in jg.columns:
                                situ_vals = jg[situ_col].astype(str).fillna('').unique().tolist()
                                situ_opts = ['TODOS'] + sorted([s for s in situ_vals if s != ''])
                                sel_situ = st.selectbox('Situacion', situ_opts, index=0, key='estad_sel_situ')
                            else:
                                sel_situ = 'TODOS'
                        with cols_f[2]:
                            if u2m_col and u2m_col in jg.columns:
                                u2m_vals = jg[u2m_col].astype(str).fillna('').unique().tolist()
                                u2m_opts = ['TODOS'] + sorted([s for s in u2m_vals if s != ''])
                                sel_u2m = st.selectbox('Últimos 2 min', u2m_opts, index=0, key='estad_sel_u2m')
                            else:
                                sel_u2m = 'TODOS'
                        apply_filters = st.form_submit_button('Aplicar filtros')

                    jg_f = jg.copy()
                    if apply_filters:
                        if sel_per != 'TODOS' and period_col and period_col in jg_f.columns:
                            try:
                                jg_f = jg_f[pd.to_numeric(jg_f[period_col], errors='coerce') == int(sel_per)]
                            except Exception:
                                pass
                        if sel_situ != 'TODOS' and situ_col and situ_col in jg_f.columns:
                            jg_f = jg_f[jg_f[situ_col].astype(str) == str(sel_situ)]
                        if sel_u2m != 'TODOS' and u2m_col and u2m_col in jg_f.columns:
                            jg_f = jg_f[jg_f[u2m_col].astype(str) == str(sel_u2m)]

                    # Top-5 jugadores por minutos en últimos 2 min (Q4+) usando quintetosAgregado
                    def compute_top5_u2m_names(cond: str, u2m_enabled: bool) -> set:
                        if not u2m_enabled:
                            return set()
                        qg_all = tablas.get('quintetosAgregado', pd.DataFrame()).copy()
                        if qg_all is None or qg_all.empty:
                            return set()
                        qg_all['Condicion'] = qg_all.get('Condicion','').astype(str).str.upper().fillna('')
                        base = qg_all[qg_all['Condicion']==cond.upper()].copy()
                        if base.empty:
                            return set()
                        # Normalizar columnas
                        per_col = _first_col(base, ['numero_periodo','periodo','Periodo']) or 'numero_periodo'
                        u2m_col = _first_col(base, ['ultimos_dos_minutos','ultimos2min','ultimos_dos','u2m','Ultimos 2 min','Últimos dos min']) or 'ultimos_dos_minutos'
                        base['__per'] = pd.to_numeric(base.get(per_col, 0), errors='coerce').fillna(0)
                        def _norm_txt(x: Any) -> str:
                            s = str(x)
                            s = s.replace('Ú','U').replace('ú','u').replace('ó','o').replace('í','i').replace('á','a').replace('é','e')
                            return s.strip().lower()
                        base['__u2m'] = base.get(u2m_col, '').apply(_norm_txt).isin(['ultimos dos min','u2m','true','1','si','sí','s'])
                        # Parse tiempo_jugado en formato mm:ss a segundos
                        def parse_secs(v: Any) -> float:
                            try:
                                s = str(v)
                                if ':' in s:
                                    mm, ss = s.split(':', 1)
                                    return float(int(mm) * 60 + int(ss))
                                return float(pd.to_numeric(v, errors='coerce'))
                            except Exception:
                                return 0.0
                        base['__t'] = base.get('tiempo_jugado', 0).apply(parse_secs)
                        filt = base[(base['__per']>=4) & (base['__u2m'])]
                        if filt.empty:
                            return set()
                        # Acumular minutos por jugador a partir del quinteto
                        acc = {}
                        for _, r in filt.iterrows():
                            t = float(r['__t'])
                            q = r.get('quinteto')
                            members = []
                            if isinstance(q, (list, tuple)):
                                members = [str(x).strip() for x in q]
                            else:
                                members = [s.strip() for s in re.split(r"\s*/\s*", str(q)) if s.strip()]
                            for m in members:
                                acc[m] = acc.get(m, 0.0) + t
                        # Top-5 por tiempo
                        top5 = sorted(acc.items(), key=lambda x: x[1], reverse=True)[:5]
                        return set([str(k) for k,_ in top5])

                    # Usar el filtro de Quintetos para decidir si marcamos 🔥
                    def _norm_txt2(x: Any) -> str:
                        s = str(x)
                        return s.replace('Ú','U').replace('ú','u').replace('ó','o').replace('í','i').replace('á','a').replace('é','e').strip().lower()
                    u2m_enabled_flag = False
                    try:
                        u2m_enabled_flag = (_norm_txt2(st.session_state.get('q_sel_u2m', '')) == 'ultimos dos min') or (_norm_txt2(locals().get('sel_u2m_q', '')) == 'ultimos dos min')
                    except Exception:
                        u2m_enabled_flag = False
                    top5_u2m_loc = compute_top5_u2m_names('LOCAL', u2m_enabled_flag)
                    top5_u2m_vis = compute_top5_u2m_names('VISITANTE', u2m_enabled_flag)

                    def make_table(df_src: pd.DataFrame, condicion: str) -> pd.DataFrame:
                        df_t = df_src[df_src.get('Condicion', '').astype(str).str.upper() == condicion.upper()].copy()
                        if df_t.empty:
                            return pd.DataFrame()
                        name_col = _first_col(df_t, ['nombre', 'jugador', 'nombre_jugador']) or 'nombre'
                        # Formatear el nombre que se mostrará en la tabla como "DD-Nombre" si hay dorsal (sin espacios)
                        dorsal_candidates = ['dorsal','numero','nro','número','numero_camiseta','n_camisa']
                        dcol = _first_col(df_t, dorsal_candidates)
                        def two_digit(v: Any) -> str:
                            try:
                                iv = int(pd.to_numeric(v, errors='coerce'))
                                return f"{iv:02d}"
                            except Exception:
                                return ''
                        def fmt_name_row(r: pd.Series) -> str:
                            nm = str(r.get(name_col, '')).strip()
                            if dcol and dcol in r.index:
                                dd = two_digit(r.get(dcol))
                                return f"{dd}-{nm}" if dd else nm
                            return nm
                        df_t[name_col] = df_t.apply(fmt_name_row, axis=1)

                        # Armar set de nombres titulares desde la planilla con el mismo formato "DD-Nombre" (sin espacios)
                        plan_df = plan_loc if condicion.upper()=='LOCAL' else plan_vis
                        tit_name_set = set()
                        try:
                            if plan_df is not None and not plan_df.empty:
                                plan_dcol = _first_col(plan_df, dorsal_candidates)
                                name_cands = ['Nombre','nombre','Jugador','jugador','nombre_jugador','NombreJugador']
                                plan_name_col = _first_col(plan_df, name_cands)
                                tit_cands = ['quintetotitular','quinteto_titular','QuintetoTitular','quintetoTitular','titular','es_titular']
                                plan_tit_col = _first_col(plan_df, tit_cands)
                                if plan_dcol and plan_name_col and plan_tit_col and \
                                   plan_dcol in plan_df.columns and plan_name_col in plan_df.columns and plan_tit_col in plan_df.columns:
                                    tmp = plan_df[[plan_dcol, plan_name_col, plan_tit_col]].copy()
                                    def truthy(v: Any) -> bool:
                                        try:
                                            if isinstance(v, (int, float)):
                                                return float(v) != 0.0
                                            s = str(v).strip().lower()
                                            return s in ('si','sí','true','t','1','x','s')
                                        except Exception:
                                            return False
                                    tmp['__tit'] = tmp[plan_tit_col].apply(truthy)
                                    tmp['__dd'] = tmp[plan_dcol].apply(two_digit)
                                    tmp['__nn'] = tmp[plan_name_col].astype(str).str.strip()
                                    tmp['__disp'] = np.where(tmp['__dd']!='', tmp['__dd'] + '-' + tmp['__nn'], tmp['__nn'])
                                    tit_name_set = set(tmp.loc[tmp['__tit']==True, '__disp'].astype(str).str.lower().tolist())
                        except Exception:
                            tit_name_set = set()

                        # Mapeo estricto de la tabla provista (izquierda -> derecha)
                        strict_map = [
                            ('ASISTENCIA', 'asistencias'),
                            ('CANASTA-1P', 'canasta1p'),
                            ('CANASTA-2P', 'canasta2p'),
                            ('CANASTA-3P', 'canasta3p'),
                            ('FALTA-COMETIDA', 'faltascometidas'),
                            ('FALTA-RECIBIDA', 'faltasrecibidas'),
                            ('PERDIDA', 'perdidas'),
                            ('REBOTE-DEFENSIVO', 'rebotedefensivo'),
                            ('REBOTE-OFENSIVO', 'reboteofensivo'),
                            ('RECUPERACION', 'recuperaciones'),
                            ('TIRO1-FALLADO', 'tiro1fallado'),
                            ('TIRO2-FALLADO', 'tiro2fallado'),
                            ('TIRO3-FALLADO', 'tiro3fallado'),
                            ('tiempo_jugado', 'tiempo_jugado'),
                        ]
                        # Construir columnas destino sumando desde la fuente estricta
                        for src, dst in strict_map:
                            if src in df_t.columns:
                                df_t[dst] = pd.to_numeric(df_t[src], errors='coerce').fillna(0)
                            else:
                                # si no existe en la fuente, crear en 0
                                df_t[dst] = 0

                        cols_sum = [dst for _, dst in strict_map]
                        # Agregar columna de plus-minus desde candidatos
                        pm_cands = ['diferencia', 'plusminus', 'plus_minus', '+-']
                        pm_series = None
                        for pmc in pm_cands:
                            if pmc in df_t.columns:
                                s = pd.to_numeric(df_t[pmc], errors='coerce').fillna(0)
                                pm_series = s if pm_series is None else (pm_series + s)
                        if pm_series is None:
                            df_t['+-'] = 0
                        else:
                            df_t['+-'] = pm_series
                        cols_sum = cols_sum + ['+-']
                        # Agrupar por jugador
                        agg = df_t.groupby(name_col, as_index=False)[cols_sum].sum()
                        # Columna Titular (estrella y 🔥 si aplica) por nombre (comparación exacta con el formato "DD-Nombre")
                        try:
                            def _norm_name(s: Any) -> str:
                                s = re.sub(r"\s+", " ", str(s).strip())
                                return s.lower()
                            tit_name_norm = set((_norm_name(x) for x in tit_name_set)) if tit_name_set else set()
                            # set de fuego (top-5 u2m) según condición
                            top_u2m = top5_u2m_loc if condicion.upper()=='LOCAL' else top5_u2m_vis
                            top_u2m_norm = set((_norm_name(x) for x in top_u2m)) if top_u2m else set()
                            def is_tit(nm: Any) -> str:
                                s = str(nm).strip()
                                if s == 'Totales':
                                    return ''
                                by_name = (_norm_name(s) in tit_name_norm) if tit_name_norm else False
                                has_fire = (_norm_name(s) in top_u2m_norm) if top_u2m_norm else False
                                return ('⭐' if by_name else '') + ('🔥' if has_fire else '')
                            agg['Titular'] = agg[name_col].map(is_tit)
                        except Exception:
                            agg['Titular'] = ''
                        # Derivadas simples
                        agg['rebotetotal'] = agg['rebotedefensivo'] + agg['reboteofensivo']
                        agg['puntos'] = agg['canasta1p'] + 2*agg['canasta2p'] + 3*agg['canasta3p']

                        # Intentos: sumar directamente desde columnas fuente por jugador, sin crear columnas intermedias
                        idx = agg[name_col]
                        def sum_by(col: str):
                            if col in df_t.columns:
                                return df_t.groupby(name_col)[col].sum().reindex(idx).fillna(0).values
                            else:
                                return np.zeros(len(idx))
                        conv1 = sum_by('CANASTA-1P')
                        fall1 = sum_by('TIRO1-FALLADO')
                        conv2 = sum_by('CANASTA-2P')
                        fall2 = sum_by('TIRO2-FALLADO')
                        conv3 = sum_by('CANASTA-3P')
                        fall3 = sum_by('TIRO3-FALLADO')

                        agg['tiro1p'] = conv1 + fall1
                        agg['tiro2p'] = conv2 + fall2
                        agg['tiro3p'] = conv3 + fall3
                        # Fila Totales
                        totals = agg.select_dtypes(include=[np.number]).sum(numeric_only=True)
                        total_row = {col: '' for col in agg.columns}
                        total_row[name_col] = 'Totales'
                        for c in totals.index:
                            total_row[c] = totals[c]
                        # Sin total para '+-'
                        if '+-' in total_row:
                            total_row['+-'] = ''
                        agg = pd.concat([agg, pd.DataFrame([total_row])], ignore_index=True)
                        # Asegurar que 'Titular' quede vacío en la fila Totales
                        if 'Titular' in agg.columns:
                            agg.loc[agg[name_col]=='Totales','Titular'] = ''
                        # Pasar por _build_table para formato (conv/att y %)
                        out = _build_table(agg)
                        # Reordenar para que 'Titular' sea la 2da columna (sin título)
                        name_display = None
                        for cand in ['Nombre', 'nombre', 'Jugador', name_col]:
                            if cand in out.columns:
                                name_display = cand
                                break
                        if name_display and 'Titular' in out.columns:
                            cols = out.columns.tolist()
                            # remover y reinsertar
                            cols.remove('Titular')
                            insert_idx = cols.index(name_display) + 1 if name_display in cols else 1
                            cols.insert(insert_idx, 'Titular')
                            out = out[cols]
                        return out

                    tbl_loc = make_table(jg_f, 'LOCAL')
                    tbl_vis = make_table(jg_f, 'VISITANTE')

                    # Usar mismos colores que en Resumen
                    row = part_df.iloc[0] if not part_df.empty else {}
                    color_local_raw = _first_of(row, ['color_local', 'local_color', 'colorLocal', 'colorlocal'], '#1f77b4')
                    color_visitante_raw = _first_of(row, ['color_visitante', 'visitante_color', 'colorVisitante', 'colorvisitante'], '#ff7f0e')
                    color_local = _parse_color(color_local_raw, '#1f77b4')
                    color_visitante = _parse_color(color_visitante_raw, '#ff7f0e')
                    tc_local = _text_color_for_bg(color_local)
                    tc_visitante = _text_color_for_bg(color_visitante)

                    # Títulos con cajas coloreadas
                    st.markdown(f"""
                        <div style='text-align:center;background:{color_local};color:{tc_local};padding:10px;border-radius:6px;margin:8px 0;'>
                            <strong>LOCAL - {local_title}</strong>
                        </div>
                    """, unsafe_allow_html=True)
                    # Colorear estrella con color del equipo
                    def style_star(df: pd.DataFrame, color_hex: str):
                        if isinstance(df, pd.DataFrame) and 'Titular' in df.columns:
                            return df.style.applymap(lambda v: f'color: {color_hex}' if str(v) == '⭐' else '', subset=['Titular'])
                        return df
                    # (Eliminado post-proceso de fueguito en jugadores)
                    styled_loc = style_star(tbl_loc, color_local)
                    st.dataframe(styled_loc, use_container_width=True, hide_index=True, column_config=build_column_config(tbl_loc))
                    # Leyenda solo para estrella
                    st.caption("⭐ Quinteto titular")
                    # (Timeline LOCAL movido a pestaña Quintetos)

                    st.markdown(f"""
                        <div style='text-align:center;background:{color_visitante};color:{tc_visitante};padding:10px;border-radius:6px;margin:16px 0 8px;'>
                            <strong>VISITANTE - {visitante_title}</strong>
                        </div>
                    """, unsafe_allow_html=True)
                    styled_vis = style_star(tbl_vis, color_visitante)
                    st.dataframe(styled_vis, use_container_width=True, hide_index=True, column_config=build_column_config(tbl_vis))
                    st.caption("⭐ Quinteto titular")
                    # (Timeline VISITANTE movido a pestaña Quintetos)
                else:
                    st.info('No hay jugadoresAgregado para generar estadística')

            # Pestaña Quintetos (agregado de quintetos)
            with t_quintetos:
                part_df = tablas.get('partido', pd.DataFrame())
                local_title = str(part_df.iloc[0].get('local')) if not part_df.empty else 'Local'
                visitante_title = str(part_df.iloc[0].get('visitante')) if not part_df.empty else 'Visitante'
                qg = tablas.get('quintetosAgregado', pd.DataFrame()).copy()
                if not qg.empty:
                    # Normalizar
                    if 'Condicion' in qg.columns:
                        qg['Condicion'] = qg['Condicion'].astype(str).str.upper().fillna('')
                    # Formateo de quinteto a string estandar para agrupar y mostrar
                    if 'quinteto' in qg.columns:
                        def fmt_quinteto_group(v):
                            if isinstance(v, (list, tuple)):
                                return ' / '.join([str(x) for x in v])
                            return str(v)
                        qg['quinteto'] = qg['quinteto'].apply(fmt_quinteto_group)

                    period_col = _first_col(qg, ['numero_periodo', 'periodo', 'Periodo'])
                    situ_col = _first_col(qg, ['SituacionMarcador', 'Situacion marcador', 'situacion_marcador', 'situacionMarcador', 'situacion', 'Situacion'])
                    u2m_col = _first_col(qg, ['ultimos_dos_minutos', 'ultimos2min', 'ultimos_dos', 'u2m'])

                    with st.form(key='quintetos_filters'):
                        fcols = st.columns(3)
                        with fcols[0]:
                            if period_col and period_col in qg.columns:
                                per_opts = ['TODOS'] + sorted(pd.to_numeric(qg[period_col], errors='coerce').dropna().astype(int).unique().tolist())
                                sel_per_q = st.selectbox('Número de periodo', per_opts, index=0, key='q_sel_per')
                            else:
                                sel_per_q = 'TODOS'
                        with fcols[1]:
                            if situ_col and situ_col in qg.columns:
                                situ_vals = qg[situ_col].astype(str).fillna('').unique().tolist()
                                situ_opts = ['TODOS'] + sorted([s for s in situ_vals if s != ''])
                                sel_situ_q = st.selectbox('Situacion marcador', situ_opts, index=0, key='q_sel_situ')
                            else:
                                sel_situ_q = 'TODOS'
                        with fcols[2]:
                            if u2m_col and u2m_col in qg.columns:
                                u2_vals = qg[u2m_col].astype(str).fillna('').unique().tolist()
                                u2_opts = ['TODOS'] + sorted([u for u in u2_vals if u != ''])
                                sel_u2m_q = st.selectbox('Momento del periodo', u2_opts, index=0, key='q_sel_u2m')
                            else:
                                sel_u2m_q = 'TODOS'
                        submitted_q = st.form_submit_button('Aplicar filtros')
                        if submitted_q:
                            _stay_estadistica()

                    # Aplicar filtros
                    qg_f = qg.copy()
                    if sel_per_q != 'TODOS' and period_col and period_col in qg_f.columns:
                        try:
                            qg_f = qg_f[pd.to_numeric(qg_f[period_col], errors='coerce') == int(sel_per_q)]
                        except Exception:
                            pass
                    if sel_situ_q != 'TODOS' and situ_col and situ_col in qg_f.columns:
                        qg_f = qg_f[qg_f[situ_col].astype(str) == str(sel_situ_q)]
                    if sel_u2m_q != 'TODOS' and u2m_col and u2m_col in qg_f.columns:
                        qg_f = qg_f[qg_f[u2m_col].astype(str) == str(sel_u2m_q)]

                    # Construcción de tablas por Condicion, agrupando por quinteto
                    def make_quintetos_table(
                        df_src: pd.DataFrame,
                        condicion: str,
                        mark_init: Optional[set]=None,
                        mark_pm_max: Optional[set]=None,
                        mark_pm_min: Optional[set]=None,
                    ) -> pd.DataFrame:
                        df_t = df_src[df_src.get('Condicion', '').astype(str).str.upper() == condicion.upper()].copy()
                        if df_t.empty:
                            return pd.DataFrame()
                        name_col = 'quinteto'
                        # Seleccionar métricas disponibles
                        base_cols = []
                        for c in ['tiempo_jugado', 'puntos_favor', 'puntos_contra', 'diferencia']:
                            if c in df_t.columns:
                                df_t[c] = pd.to_numeric(df_t[c], errors='coerce').fillna(0)
                                base_cols.append(c)
                        # Variables de Favor/Contra
                        fav_cols = [c for c in df_t.columns if c.endswith('_Favor')]
                        con_cols = [c for c in df_t.columns if c.endswith('_Contra')]
                        sum_cols = base_cols + fav_cols + con_cols
                        if not sum_cols:
                            return pd.DataFrame()
                        keep_cols = [name_col] + sum_cols
                        agg = (
                            df_t
                            .groupby([name_col], as_index=False)[keep_cols]
                            .sum(numeric_only=True)
                        )
                        # Agregar columna de marca (símbolos) antes de formatear nombres
                        init_sym = '⭐'    # quinteto inicial (igual que en tabla de jugadores)
                        pm_up_sym = '🟢↑' # mayor +/-
                        pm_dn_sym = '🔴↓' # menor +/-
                        def mark_symbol(qs: str) -> str:
                            sym = ''
                            q = str(qs)
                            if mark_init and q in mark_init:
                                sym += init_sym
                            if mark_pm_max and q in mark_pm_max:
                                sym += pm_up_sym
                            if mark_pm_min and q in mark_pm_min:
                                sym += pm_dn_sym
                            return sym
                        try:
                            agg['Marca'] = agg[name_col].apply(mark_symbol)
                        except Exception:
                            agg['Marca'] = ''
                        # Ordenar por tiempo_jugado desc si existe antes de agregar Totales
                        if 'tiempo_jugado' in agg.columns:
                            agg = agg.sort_values(by=['tiempo_jugado'], ascending=False).reset_index(drop=True)
                        # Eficiencias de 1P/2P/3P a Favor y en Contra
                        def get_col(df, *cands):
                            for c in cands:
                                if c in df.columns:
                                    return df[c]
                            return pd.Series([0]*len(df))
                        # Favor
                        c1F = pd.to_numeric(get_col(agg, 'CANASTA-1P_Favor', 'CANASTA_1P_Favor'), errors='coerce').fillna(0)
                        m1F = pd.to_numeric(get_col(agg, 'TIRO1-FALLADO_Favor', 'TIRO1_FALLADO_Favor'), errors='coerce').fillna(0)
                        a1F = c1F + m1F
                        c2F = pd.to_numeric(get_col(agg, 'CANASTA-2P_Favor', 'CANASTA_2P_Favor'), errors='coerce').fillna(0)
                        m2F = pd.to_numeric(get_col(agg, 'TIRO2-FALLADO_Favor', 'TIRO2_FALLADO_Favor'), errors='coerce').fillna(0)
                        a2F = c2F + m2F
                        c3F = pd.to_numeric(get_col(agg, 'CANASTA-3P_Favor', 'CANASTA_3P_Favor'), errors='coerce').fillna(0)
                        m3F = pd.to_numeric(get_col(agg, 'TIRO3-FALLADO_Favor', 'TIRO3_FALLADO_Favor'), errors='coerce').fillna(0)
                        a3F = c3F + m3F
                        # Contra
                        c1C = pd.to_numeric(get_col(agg, 'CANASTA-1P_Contra', 'CANASTA_1P_Contra'), errors='coerce').fillna(0)
                        m1C = pd.to_numeric(get_col(agg, 'TIRO1-FALLADO_Contra', 'TIRO1_FALLADO_Contra'), errors='coerce').fillna(0)
                        a1C = c1C + m1C
                        c2C = pd.to_numeric(get_col(agg, 'CANASTA-2P_Contra', 'CANASTA_2P_Contra'), errors='coerce').fillna(0)
                        m2C = pd.to_numeric(get_col(agg, 'TIRO2-FALLADO_Contra', 'TIRO2_FALLADO_Contra'), errors='coerce').fillna(0)
                        a2C = c2C + m2C
                        c3C = pd.to_numeric(get_col(agg, 'CANASTA-3P_Contra', 'CANASTA_3P_Contra'), errors='coerce').fillna(0)
                        m3C = pd.to_numeric(get_col(agg, 'TIRO3-FALLADO_Contra', 'TIRO3_FALLADO_Contra'), errors='coerce').fillna(0)
                        a3C = c3C + m3C

                        def pct(conv, att):
                            with np.errstate(divide='ignore', invalid='ignore'):
                                p = np.where(att > 0, (conv / att) * 100.0, 0.0)
                            return pd.Series(p).round(0).astype(int)

                        agg['1P Favor'] = (c1F.astype(int).astype(str) + '/' + a1F.astype(int).astype(str))
                        agg['%1P Favor'] = pct(c1F, a1F).astype(str) + '%'
                        agg['1P Contra'] = (c1C.astype(int).astype(str) + '/' + a1C.astype(int).astype(str))
                        agg['%1P Contra'] = pct(c1C, a1C).astype(str) + '%'
                        agg['2P Favor'] = (c2F.astype(int).astype(str) + '/' + a2F.astype(int).astype(str))
                        agg['%2P Favor'] = pct(c2F, a2F).astype(str) + '%'
                        agg['2P Contra'] = (c2C.astype(int).astype(str) + '/' + a2C.astype(int).astype(str))
                        agg['%2P Contra'] = pct(c2C, a2C).astype(str) + '%'
                        agg['3P Favor'] = (c3F.astype(int).astype(str) + '/' + a3F.astype(int).astype(str))
                        agg['%3P Favor'] = pct(c3F, a3F).astype(str) + '%'
                        agg['3P Contra'] = (c3C.astype(int).astype(str) + '/' + a3C.astype(int).astype(str))
                        agg['%3P Contra'] = pct(c3C, a3C).astype(str) + '%'
                        # Diferencias de % (Favor - Contra)
                        # No agregar columnas de diferencias de % en la tabla
                        # Derivar plus/minus de puntos
                        if 'puntos_favor' in agg.columns and 'puntos_contra' in agg.columns:
                            try:
                                agg['+-'] = pd.to_numeric(agg['puntos_favor'], errors='coerce').fillna(0) - pd.to_numeric(agg['puntos_contra'], errors='coerce').fillna(0)
                            except Exception:
                                agg['+-'] = 0
                        # Totales
                        totals = agg.select_dtypes(include=[np.number]).sum(numeric_only=True)
                        total_row = {col: '' for col in agg.columns}
                        total_row[name_col] = 'Totales'
                        for c in totals.index:
                            total_row[c] = totals[c]
                        agg = pd.concat([agg, pd.DataFrame([total_row])], ignore_index=True)
                        # Formatos
                        if 'tiempo_jugado' in agg.columns:
                            def fmt_t(s):
                                try:
                                    s = float(s)
                                except Exception:
                                    return str(s)
                                return f"{int(s//60)}:{int(s%60):02d}" if s == s and s != '' else ''
                            agg['tiempo_jugado'] = agg['tiempo_jugado'].apply(fmt_t)
                            agg = agg.rename(columns={'tiempo_jugado': 'Tiempo Jugado'})
                        # Acortar quinteto a apellidos
                        def short_quinteto(qs: str) -> str:
                            try:
                                s = str(qs)
                                if s == 'Totales':
                                    return s
                                parts = [p.strip() for p in re.split(r"\s*/\s*|\s*-\s*", s) if p.strip()]
                                apes = []
                                for p in parts:
                                    # Tomar después de guión en caso de '00-Nombre Apellido'
                                    if '-' in p:
                                        p = p.split('-', 1)[-1].strip()
                                    # Si viene 'APELLIDO, NOMBRE'
                                    if ',' in p:
                                        apes.append(p.split(',', 1)[0].strip())
                                    else:
                                        toks = [t for t in p.split(' ') if t]
                                        apes.append(toks[-1] if toks else p)
                                return ' - '.join(apes)
                            except Exception:
                                return str(qs)
                        agg[name_col] = agg[name_col].apply(short_quinteto)
                        # Reordenar para que 'Marca' sea segunda columna (sin título)
                        if 'Marca' in agg.columns:
                            cols = agg.columns.tolist()
                            if name_col in cols:
                                cols.remove('Marca')
                                insert_idx = 1
                                cols.insert(insert_idx, 'Marca')
                                agg = agg[cols]
                        # Orden de columnas: como jugadores: Quinteto, Tiempo, Puntos a favor/en contra, Diferencia, eficiencias, luego pares Favor/Contra
                        preferred = [
                            name_col, 'Marca', 'Tiempo Jugado', 'puntos_favor', 'puntos_contra', 'diferencia', '+-',
                            '1P Favor', '%1P Favor', '1P Contra', '%1P Contra',
                            '2P Favor', '%2P Favor', '2P Contra', '%2P Contra',
                            '3P Favor', '%3P Favor', '3P Contra', '%3P Contra',
                        ]
                        base_priority = [
                            'REBOTE-DEFENSIVO', 'REBOTE-OFENSIVO', 'ASISTENCIA', 'PERDIDA', 'RECUPERACION',
                            'FALTA-COMETIDA', 'FALTA-RECIBIDA',
                            'TIRO1-FALLADO', 'CANASTA-1P',
                            'TIRO2-FALLADO', 'CANASTA-2P',
                            'TIRO3-FALLADO', 'CANASTA-3P'
                        ]
                        pair_cols = []
                        for base in base_priority:
                            fcol = f'{base}_Favor'
                            ccol = f'{base}_Contra'
                            if fcol in agg.columns:
                                pair_cols.append(fcol)
                            if ccol in agg.columns:
                                pair_cols.append(ccol)
                        ordered = [c for c in preferred if c in agg.columns] + [c for c in pair_cols if c not in preferred]
                        ordered += [c for c in agg.columns if c not in ordered]
                        agg = agg[ordered]
                        # Renombrar columnas a etiquetas amigables
                        def nice_label(col: str) -> str:
                            mapping_base = {
                                'puntos_favor': 'Puntos a favor',
                                'puntos_contra': 'Puntos en contra',
                                'diferencia': 'Diferencia de puntos',
                                'REBOTE-DEFENSIVO': 'Rebote Def.',
                                'REBOTE-OFENSIVO': 'Rebote Of.',
                                'ASISTENCIA': 'Asist.',
                                'PERDIDA': 'Pérdidas',
                                'RECUPERACION': 'Recup.',
                                'FALTA-COMETIDA': 'Falta Cometida',
                                'FALTA-RECIBIDA': 'Falta Recibida',
                                'TIRO1-FALLADO': 'Libres Fallados',
                                'CANASTA-1P': 'Libres Convertidos',
                                'TIRO2-FALLADO': 'Dobles Fallados',
                                'CANASTA-2P': 'Dobles Convertidos',
                                'TIRO3-FALLADO': 'Triples Fallados',
                                'CANASTA-3P': 'Triples Convertidos',
                            }
                            if col in (name_col, 'Tiempo Jugado'):
                                return 'Quinteto' if col == name_col else col
                            if col in ('1P Favor','%1P Favor','1P Contra','%1P Contra','%1P Dif',
                                       '2P Favor','%2P Favor','2P Contra','%2P Contra','%2P Dif',
                                       '3P Favor','%3P Favor','3P Contra','%3P Contra','%3P Dif'):
                                return col
                            if col in ('puntos_favor','puntos_contra','diferencia','+-'):
                                return mapping_base.get(col, col)
                            # Favor/Contra metrics
                            if col.endswith('_Favor') or col.endswith('_Contra'):
                                base = col.replace('_Favor','').replace('_Contra','')
                                side = 'Favor' if col.endswith('_Favor') else 'Contra'
                                base_label = mapping_base.get(base, base.title())
                                return f"{base_label} {side}"
                            return col
                        agg = agg.rename(columns={c: nice_label(c) for c in agg.columns})
                        return agg

                    # Calcular marcas de quinteto inicial y más minutos en últimos 2 min (Q4+)
                    def compute_marks(df: pd.DataFrame, condicion: str):
                        base = df[df.get('Condicion', '').astype(str).str.upper() == condicion.upper()].copy()
                        if base.empty:
                            return set(), set()
                        # Asegurar columnas (detectar nombres alternativos)
                        per_col = _first_col(base, ['numero_periodo','periodo','Periodo']) or 'numero_periodo'
                        u2m_col = _first_col(base, ['ultimos_dos_minutos','ultimos2min','ultimos_dos','u2m','Ultimos 2 min','Últimos dos min']) or 'ultimos_dos_minutos'
                        per = pd.to_numeric(base.get(per_col, 0), errors='coerce').fillna(0)
                        base['__per'] = per
                        # ultimos_dos_minutos puede venir como texto 'Últimos dos min'. Hacer comparación robusta sin acentos ni mayúsculas
                        def _norm_txt(x: Any) -> str:
                            s = str(x)
                            s = s.replace('Ú','U').replace('ú','u').replace('ó','o').replace('í','i').replace('á','a').replace('é','e')
                            return s.strip().lower()
                        base['__u2m'] = base.get(u2m_col, '').apply(_norm_txt).isin(['ultimos dos min','u2m','true','1','si','sí','s'])
                        # Parse tiempo_jugado a segundos (mm:ss o numérico)
                        def _to_secs(v: Any) -> float:
                            try:
                                s = str(v)
                                if ':' in s:
                                    mm, ss = s.split(':', 1)
                                    return float(int(mm) * 60 + int(ss))
                                return float(pd.to_numeric(v, errors='coerce'))
                            except Exception:
                                return 0.0
                        base['__t'] = base.get('tiempo_jugado', 0).apply(_to_secs)
                        # Función para extraer dorsales de un quinteto (como conjunto de dos dígitos)
                        def two_digit(v):
                            try:
                                iv = int(pd.to_numeric(v, errors='coerce'))
                                return f"{iv:02d}"
                            except Exception:
                                return None
                        def quinteto_dorsales(qv):
                            s = ''
                            if isinstance(qv, (list, tuple)):
                                parts = [str(x) for x in qv]
                                s = ' '.join(parts)
                            else:
                                s = str(qv)
                            # Buscar números de 1 o 2 dígitos y también prefijos 'DD-'
                            nums = set()
                            for m in re.findall(r"\b(\d{1,2})\b", s):
                                td = two_digit(m)
                                if td:
                                    nums.add(td)
                            for m in re.findall(r"\b(\d{2})-", s):
                                td = two_digit(m)
                                if td:
                                    nums.add(td)
                            return nums
                        # Obtener dorsales titulares desde planillas
                        plan_df = tablas.get('estadisticas_equipolocal' if condicion.upper()=='LOCAL' else 'estadisticas_equipovisitante', pd.DataFrame())
                        starters = set()
                        if not plan_df.empty:
                            dcol = None
                            for c in ['dorsal','numero','nro','número','numero_camiseta','n_camisa']:
                                if c in plan_df.columns:
                                    dcol = c
                                    break
                            tcol = None
                            for c in ['quintetotitular','quinteto_titular','QuintetoTitular','quintetoTitular','titular','es_titular']:
                                if c in plan_df.columns:
                                    tcol = c
                                    break
                            if dcol and tcol:
                                tmp = plan_df[[dcol, tcol]].copy()
                                tmp['__tit'] = tmp[tcol].astype(str).str.lower().isin(['true','1','si','sí','t','x'])
                                starters = set([two_digit(v) for v in tmp.loc[tmp['__tit'], dcol].tolist() if two_digit(v) is not None])
                        init_set = set()
                        if starters and len(starters) >= 5:
                            # Encontrar fila cuyo quinteto contenga todos los dorsales titulares
                            for _, r in base.iterrows():
                                qs = r.get('quinteto')
                                nums = quinteto_dorsales(qs)
                                if starters.issubset(nums):
                                    init_set.add(str(qs))
                        # Últimos 2 min: periodo >=4 y flag u2m -> más tiempo jugado
                        u2 = base[(base['__per'] >= 4) & (base['__u2m'])]
                        u2m_set = set()
                        if not u2.empty:
                            idx2 = u2.groupby('quinteto')['__t'].sum().sort_values(ascending=False).head(1).index
                            u2m_set = set([str(x) for x in idx2.tolist()])
                        return init_set, u2m_set

                    init_loc, u2m_loc = compute_marks(qg, 'LOCAL')
                    init_vis, u2m_vis = compute_marks(qg, 'VISITANTE')

                    # Marcar mayor y menor +/- por equipo
                    def mark_pm(df: pd.DataFrame, condicion: str) -> tuple[set, set]:
                        b = df[df.get('Condicion','').astype(str).str.upper()==condicion.upper()].copy()
                        if b.empty:
                            return set(), set()
                        # Si no existe 'diferencia', derivar como puntos_favor - puntos_contra
                        if 'diferencia' not in b.columns:
                            try:
                                b['diferencia'] = pd.to_numeric(b.get('puntos_favor',0), errors='coerce').fillna(0) - pd.to_numeric(b.get('puntos_contra',0), errors='coerce').fillna(0)
                            except Exception:
                                b['diferencia'] = 0
                        g = b.groupby('quinteto')['diferencia'].sum()
                        if g.empty:
                            return set(), set()
                        srt = g.sort_values(ascending=False)
                        top = srt.head(1).index
                        bottom = srt.tail(1).index
                        return set([str(x) for x in top.tolist()]), set([str(x) for x in bottom.tolist()])

                    maxpm_loc, minpm_loc = mark_pm(qg, 'LOCAL')
                    maxpm_vis, minpm_vis = mark_pm(qg, 'VISITANTE')

                    # Pasar conjuntos combinados a la tabla
                    tbl_loc_q = make_quintetos_table(qg_f, 'LOCAL', init_loc, maxpm_loc, minpm_loc)
                    tbl_vis_q = make_quintetos_table(qg_f, 'VISITANTE', init_vis, maxpm_vis, minpm_vis)
                    # Post-proceso: marcar 🔥 en quintetos usando SOLO la tabla mostrada y filtro de esta pestaña
                    def _norm_txt_q(x: Any) -> str:
                        s = str(x)
                        return s.replace('Ú','U').replace('ú','u').replace('ó','o').replace('í','i').replace('á','a').replace('é','e').strip().lower()
                    u2m_sel_q = _norm_txt_q(st.session_state.get('q_sel_u2m','TODOS'))
                    def _parse_secs_q(v: Any) -> float:
                        try:
                            s = str(v)
                            if ':' in s:
                                mm, ss = s.split(':', 1)
                                return float(int(mm)*60 + int(ss))
                            return float(pd.to_numeric(v, errors='coerce'))
                        except Exception:
                            return 0.0
                    def add_fire_quinteto(df_in: pd.DataFrame) -> pd.DataFrame:
                        if not isinstance(df_in, pd.DataFrame) or df_in.empty:
                            return df_in
                        if u2m_sel_q != 'ultimos dos min':
                            return df_in
                        if 'Tiempo Jugado' not in df_in.columns:
                            return df_in
                        work = df_in.copy()
                        work['__sec'] = work['Tiempo Jugado'].apply(_parse_secs_q)
                        # excluir Totales
                        name_col_cur = 'Quinteto' if 'Quinteto' in work.columns else None
                        if name_col_cur is None:
                            return df_in
                        mask = work[name_col_cur].astype(str).str.strip().ne('Totales')
                        if not mask.any():
                            return df_in
                        idx_max = work[mask]['__sec'].idxmax()
                        if 'Marca' not in work.columns:
                            work['Marca'] = ''
                        work.at[idx_max, 'Marca'] = str(work.at[idx_max, 'Marca']) + '🔥'
                        return work
                    tbl_loc_q = add_fire_quinteto(tbl_loc_q)
                    tbl_vis_q = add_fire_quinteto(tbl_vis_q)

                    # Usar mismos colores que en Resumen
                    row = part_df.iloc[0] if not part_df.empty else {}
                    color_local_raw = _first_of(row, ['color_local', 'local_color', 'colorLocal', 'colorlocal'], '#1f77b4')
                    color_visitante_raw = _first_of(row, ['color_visitante', 'visitante_color', 'colorVisitante', 'colorvisitante'], '#ff7f0e')
                    color_local = _parse_color(color_local_raw, '#1f77b4')
                    color_visitante = _parse_color(color_visitante_raw, '#ff7f0e')
                    tc_local = _text_color_for_bg(color_local)
                    tc_visitante = _text_color_for_bg(color_visitante)

                    # Títulos y tablas
                    st.markdown(f"""
                        <div style='text-align:center;background:{color_local};color:{tc_local};padding:10px;border-radius:6px;margin:8px 0;'>
                            <strong>LOCAL - {local_title}</strong>
                        </div>
                    """, unsafe_allow_html=True)
                    if not tbl_loc_q.empty:
                        st.dataframe(tbl_loc_q, use_container_width=True, hide_index=True, column_config=build_column_config(tbl_loc_q))
                        st.caption("⭐ Quinteto inicial   ·   🟢↑ Mayor +/-   ·   🔴↓ Menor +/-")
                        # Timeline de presencia (LOCAL)
                        try:
                            pbp_df_full = tablas.get('pbp', pd.DataFrame()).copy()
                            if not pbp_df_full.empty and 'quinteto_local' in pbp_df_full.columns:
                                dfp = pbp_df_full.copy()
                                # Orden temporal consistente
                                if 'autoincremental_id' in dfp.columns and 'autoincremental_id_num' not in dfp.columns:
                                    dfp['autoincremental_id_num'] = pd.to_numeric(dfp['autoincremental_id'], errors='coerce').fillna(0)
                                if 'autoincremental_id_num' in dfp.columns:
                                    dfp = dfp.sort_values(by=['_id','numero_periodo','autoincremental_id_num'])
                                elif 'tiempo_segundos' in dfp.columns:
                                    dfp = dfp.sort_values(by=['_id','numero_periodo','tiempo_segundos'])
                                dfp = dfp.reset_index(drop=True)
                                # x_period con soporte de prórrogas (OT)
                                if 'x_period' not in dfp.columns:
                                    tnum = pd.to_numeric(dfp.get('tiempo_segundos', np.nan), errors='coerce').fillna(0)
                                    pnum = pd.to_numeric(dfp.get('numero_periodo', 1), errors='coerce').fillna(1)
                                    dfp['x_period'] = np.where(
                                        pnum <= 4,
                                        (600 - tnum) + (pnum - 1) * 600,
                                        2400 + (300 - tnum) + (pnum - 4) * 300
                                    )
                                dfp['x_period'] = pd.to_numeric(dfp['x_period'], errors='coerce')
                                dfp = dfp[~dfp['x_period'].isna()].copy()
                                # Y-order desde planilla local
                                est_loc_df2 = tablas.get('estadisticas_equipolocal', pd.DataFrame()).copy()
                                y_order_loc: List[str] = []
                                if not est_loc_df2.empty:
                                    dcol = _first_col(est_loc_df2, ['dorsal','numero','nro','número','numero_camiseta','n_camisa'])
                                    ncol = _first_col(est_loc_df2, ['nombre','jugador','nombre_jugador'])
                                    if dcol and ncol:
                                        tmp = est_loc_df2[[dcol, ncol]].copy()
                                        def _fmt_label_loc(r):
                                            try:
                                                dd = int(pd.to_numeric(r.get(dcol, ''), errors='coerce'))
                                                dd2 = f"{dd:02d}"
                                            except Exception:
                                                dd2 = str(r.get(dcol, '')).strip()
                                            nm = str(r.get(ncol, '')).strip()
                                            return (dd2 + '-' + nm).strip('-') if dd2 or nm else nm
                                        y_order_loc = [_fmt_label_loc(r) for _, r in tmp.iterrows()]
                                        seen = set()
                                        y_order_loc = [x for x in y_order_loc if not (x in seen or seen.add(x))]
                                # Normalizar quinteto_local en dfp
                                def _to_list(v):
                                    if isinstance(v, list):
                                        return [str(x) for x in v]
                                    if isinstance(v, str):
                                        try:
                                            parsed = json.loads(v)
                                            if isinstance(parsed, list):
                                                return [str(x) for x in parsed]
                                        except Exception:
                                            pass
                                        parts = [p.strip() for p in re.split(r"\s*/\s*", v) if p.strip()]
                                        return parts if parts else [v]
                                    return []
                                dfp['quinteto_local'] = dfp['quinteto_local'].apply(_to_list)
                                # Intervalos [x1,x2] por jugador del quinteto previo
                                intervals = []
                                for i in range(1, len(dfp)):
                                    r_prev = dfp.iloc[i-1]
                                    r_cur = dfp.iloc[i]
                                    try:
                                        per_prev = int(pd.to_numeric(r_prev.get('numero_periodo'), errors='coerce'))
                                        per_cur = int(pd.to_numeric(r_cur.get('numero_periodo'), errors='coerce'))
                                    except Exception:
                                        per_prev, per_cur = None, None
                                    if per_prev is None or per_cur is None or per_prev != per_cur:
                                        continue
                                    x1 = float(pd.to_numeric(r_prev.get('x_period'), errors='coerce'))
                                    x2 = float(pd.to_numeric(r_cur.get('x_period'), errors='coerce'))
                                    if x2 < x1:
                                        x1, x2 = x2, x1
                                    for p in (r_prev.get('quinteto_local') or []):
                                        intervals.append({'player': str(p), 'x1': x1, 'x2': x2})
                                intervals_df = pd.DataFrame(intervals)
                                # Eventos ENTRA/SALE (LOCAL) -> círculos verde/rojo
                                ev = pd.DataFrame()
                                if 'accion_tipo' in dfp.columns and 'Condicion' in dfp.columns:
                                    ev = dfp[dfp['accion_tipo'].isin(['CAMBIO-JUGADOR-ENTRA','CAMBIO-JUGADOR-SALE']) & (dfp['Condicion'].astype(str).str.upper()=='LOCAL')].copy()
                                    if not ev.empty:
                                        ev['player'] = ev.get('nombre', '').astype(str)
                                if not intervals_df.empty:
                                    intervals_df['player'] = intervals_df['player'].astype(str)
                                    # Alto suficiente: ~28px por jugador
                                    players_count = len(y_order_loc) if y_order_loc else intervals_df['player'].nunique()
                                    chart_height = max(380, 28 * players_count)
                                    # Halo para mejorar contraste + línea principal
                                    halo_loc = (
                                        alt.Chart(intervals_df)
                                        .mark_rule(color='#000000', strokeWidth=9, opacity=0.15)
                                        .encode(
                                            x=alt.X('x1:Q', title='Tiempo (x_period)'),
                                            x2='x2:Q',
                                            y=alt.Y('player:N', sort=y_order_loc if y_order_loc else None, title='Jugador (Local)', axis=alt.Axis(labelFontSize=13))
                                        )
                                    )
                                    base = (
                                        alt.Chart(intervals_df)
                                        .mark_rule(color=color_local, strokeWidth=6)
                                        .encode(
                                            x=alt.X('x1:Q', title='Tiempo (x_period)'),
                                            x2='x2:Q',
                                            y=alt.Y('player:N', sort=y_order_loc if y_order_loc else None, title='Jugador (Local)', axis=alt.Axis(labelFontSize=13))
                                        )
                                        .properties(height=chart_height, title=alt.TitleParams(text=f'Jugadores en cancha - {local_title}', anchor='middle'))
                                    )
                                    # Reglas verticales y etiquetas de periodo
                                    period_changes = []
                                    prev = None
                                    for _, rp in dfp.iterrows():
                                        cur = rp.get('numero_periodo')
                                        if prev is not None and cur != prev:
                                            period_changes.append({'x_period': rp.get('x_period'), 'numero_periodo': cur})
                                        prev = cur
                                    chart_loc = halo_loc + base
                                    if period_changes:
                                        rules_df = pd.DataFrame(period_changes)
                                        rules = alt.Chart(rules_df).mark_rule(color='#333333', strokeDash=[8,4], strokeWidth=3).encode(
                                            x=alt.X('x_period:Q'),
                                            tooltip=[alt.Tooltip('numero_periodo:N', title='Inicio periodo')]
                                        )
                                        # Etiqueta en la parte superior con texto "P <n>"
                                        top_name = y_order_loc[0] if y_order_loc else intervals_df['player'].unique().tolist()[0]
                                        labels_df = rules_df.copy()
                                        labels_df['label'] = 'P ' + labels_df['numero_periodo'].astype(str)
                                        labels_df['y_label'] = top_name
                                        labels = alt.Chart(labels_df).mark_text(align='left', baseline='bottom', dx=6, dy=-8, color='#333333', fontSize=14).encode(
                                            x=alt.X('x_period:Q'), y=alt.Y('y_label:N'), text='label:N'
                                        )
                                        chart_loc = chart_loc + rules + labels
                                    # Marcas ENTRA (verde) / SALE (rojo) como círculos
                                    if not ev.empty:
                                        enter = ev[ev['accion_tipo']=='CAMBIO-JUGADOR-ENTRA'].copy()
                                        sale = ev[ev['accion_tipo']=='CAMBIO-JUGADOR-SALE'].copy()
                                        if not enter.empty:
                                            enter_layer = (
                                                alt.Chart(enter)
                                                .mark_point(filled=True, size=140, color='#2e7d32', shape='circle')
                                                .encode(x=alt.X('x_period:Q'), y=alt.Y('player:N', sort=y_order_loc if y_order_loc else None))
                                            )
                                            chart_loc = chart_loc + enter_layer
                                        if not sale.empty:
                                            sale_layer = (
                                                alt.Chart(sale)
                                                .mark_point(filled=True, size=140, color='#c62828', shape='circle')
                                                .encode(x=alt.X('x_period:Q'), y=alt.Y('player:N', sort=y_order_loc if y_order_loc else None))
                                            )
                                            chart_loc = chart_loc + sale_layer
                                    # Marcas de tiros (LOCAL)
                                    if 'accion_tipo' in dfp.columns and 'Condicion' in dfp.columns:
                                        shots = dfp[(dfp['Condicion'].astype(str).str.upper()=='LOCAL') & (dfp['accion_tipo'].isin([
                                            'CANASTA-1P','TIRO1-FALLADO','CANASTA-2P','TIRO2-FALLADO','CANASTA-3P','TIRO3-FALLADO'
                                        ]))].copy()
                                        if not shots.empty:
                                            shots['player'] = shots.get('nombre','').astype(str)
                                            # Separación adicional: ajustar levemente el tiempo (x) según tipo y si fue convertido/fallado
                                            try:
                                                at = shots['accion_tipo'].astype(str)
                                                tipo_num = at.str.extract(r'(\d)P', expand=False).fillna('0').astype(int)
                                                is_made = at.str.startswith('CANASTA-')
                                                base_off = np.where(is_made, 0.25, -0.25)
                                                tipo_off = np.where(tipo_num == 1, 0.00, np.where(tipo_num == 2, 0.08, 0.16))
                                                xper = pd.to_numeric(shots.get('x_period', 0), errors='coerce').fillna(0)
                                                shots['__x_adj'] = xper + base_off + tipo_off
                                            except Exception:
                                                shots['__x_adj'] = shots.get('x_period', 0)
                                            # Definir capas por tipo usando glifos de texto (✔ para convertidos, ✚ para fallados) con offset para no taparse
                                            def shot_layer_text(df_in, color_hex, glyph, dx_px):
                                                if df_in.empty:
                                                    return None
                                                return (
                                                    alt.Chart(df_in)
                                                    .mark_text(fontWeight='bold', fontSize=26, color=color_hex, stroke='black', strokeWidth=0.6, dx=dx_px)
                                                    .encode(
                                                        x=alt.X('__x_adj:Q'),
                                                        y=alt.Y('player:N', sort=y_order_loc if y_order_loc else None),
                                                        text=alt.value(glyph)
                                                    )
                                                )
                                            grn = '#2e7d32'
                                            red = '#c62828'
                                            # Offsets: convertidos +8px, fallados -8px (para mismo tiempo)
                                            l_m1 = shot_layer_text(shots[shots['accion_tipo']=='CANASTA-1P'], grn, '1', 8)
                                            l_x1 = shot_layer_text(shots[shots['accion_tipo']=='TIRO1-FALLADO'], red, '1', -8)
                                            l_m2 = shot_layer_text(shots[shots['accion_tipo']=='CANASTA-2P'], grn, '2', 8)
                                            l_x2 = shot_layer_text(shots[shots['accion_tipo']=='TIRO2-FALLADO'], red, '2', -8)
                                            l_m3 = shot_layer_text(shots[shots['accion_tipo']=='CANASTA-3P'], grn, '3', 8)
                                            l_x3 = shot_layer_text(shots[shots['accion_tipo']=='TIRO3-FALLADO'], red, '3', -8)
                                            for lyr in [l_m1,l_x1,l_m2,l_x2,l_m3,l_x3]:
                                                if lyr is not None:
                                                    chart_loc = chart_loc + lyr
                                    st.altair_chart(chart_loc, use_container_width=True)
                                    st.caption("1 verde: 1P convertido · 1 rojo: 1P fallado · 2 verde: 2P convertido · 2 rojo: 2P fallado · 3 verde: 3P convertido · 3 rojo: 3P fallado · ● Verde: entra · ● Rojo: sale")
                                else:
                                    st.info('No se pudieron derivar intervalos de presencia del PBP (LOCAL).')
                        except Exception as e:
                            st.warning(f'No se pudo generar el timeline LOCAL: {e}')
                    else:
                        st.info('Sin datos de quintetos LOCAL para los filtros seleccionados')

                    st.markdown(f"""
                        <div style='text-align:center;background:{color_visitante};color:{tc_visitante};padding:10px;border-radius:6px;margin:16px 0 8px;'>
                            <strong>VISITANTE - {visitante_title}</strong>
                        </div>
                    """, unsafe_allow_html=True)
                    if not tbl_vis_q.empty:
                        st.dataframe(tbl_vis_q, use_container_width=True, hide_index=True, column_config=build_column_config(tbl_vis_q))
                        st.caption("⭐ Quinteto inicial   ·   🟢↑ Mayor +/-   ·   🔴↓ Menor +/-")
                        # Timeline de presencia (VISITANTE)
                        try:
                            pbp_df_full = tablas.get('pbp', pd.DataFrame()).copy()
                            if not pbp_df_full.empty and 'quinteto_visitante' in pbp_df_full.columns:
                                dfp = pbp_df_full.copy()
                                if 'autoincremental_id' in dfp.columns and 'autoincremental_id_num' not in dfp.columns:
                                    dfp['autoincremental_id_num'] = pd.to_numeric(dfp['autoincremental_id'], errors='coerce').fillna(0)
                                if 'autoincremental_id_num' in dfp.columns:
                                    dfp = dfp.sort_values(by=['_id','numero_periodo','autoincremental_id_num'])
                                elif 'tiempo_segundos' in dfp.columns:
                                    dfp = dfp.sort_values(by=['_id','numero_periodo','tiempo_segundos'])
                                dfp = dfp.reset_index(drop=True)
                                if 'x_period' not in dfp.columns:
                                    tnum = pd.to_numeric(dfp.get('tiempo_segundos', np.nan), errors='coerce').fillna(0)
                                    pnum = pd.to_numeric(dfp.get('numero_periodo', 1), errors='coerce').fillna(1)
                                    dfp['x_period'] = (600 - tnum) + (pnum - 1) * 600
                                dfp['x_period'] = pd.to_numeric(dfp['x_period'], errors='coerce')
                                dfp = dfp[~dfp['x_period'].isna()].copy()
                                # Y-order desde planilla visitante
                                est_vis_df2 = tablas.get('estadisticas_equipovisitante', pd.DataFrame()).copy()
                                y_order_vis: List[str] = []
                                if not est_vis_df2.empty:
                                    dcol = _first_col(est_vis_df2, ['dorsal','numero','nro','número','numero_camiseta','n_camisa'])
                                    ncol = _first_col(est_vis_df2, ['nombre','jugador','nombre_jugador'])
                                    if dcol and ncol:
                                        tmp = est_vis_df2[[dcol, ncol]].copy()
                                        def _fmt_label_vis(r):
                                            try:
                                                dd = int(pd.to_numeric(r.get(dcol, ''), errors='coerce'))
                                                dd2 = f"{dd:02d}"
                                            except Exception:
                                                dd2 = str(r.get(dcol, '')).strip()
                                            nm = str(r.get(ncol, '')).strip()
                                            return (dd2 + '-' + nm).strip('-') if dd2 or nm else nm
                                        y_order_vis = [_fmt_label_vis(r) for _, r in tmp.iterrows()]
                                        seen = set()
                                        y_order_vis = [x for x in y_order_vis if not (x in seen or seen.add(x))]
                                # Normalizar quinteto_visitante
                                def _to_list2(v):
                                    if isinstance(v, list):
                                        return [str(x) for x in v]
                                    if isinstance(v, str):
                                        try:
                                            parsed = json.loads(v)
                                            if isinstance(parsed, list):
                                                return [str(x) for x in parsed]
                                        except Exception:
                                            pass
                                        parts = [p.strip() for p in re.split(r"\s*/\s*", v) if p.strip()]
                                        return parts if parts else [v]
                                    return []
                                dfp['quinteto_visitante'] = dfp['quinteto_visitante'].apply(_to_list2)
                                # Intervalos
                                intervals = []
                                for i in range(1, len(dfp)):
                                    r_prev = dfp.iloc[i-1]
                                    r_cur = dfp.iloc[i]
                                    try:
                                        per_prev = int(pd.to_numeric(r_prev.get('numero_periodo'), errors='coerce'))
                                        per_cur = int(pd.to_numeric(r_cur.get('numero_periodo'), errors='coerce'))
                                    except Exception:
                                        per_prev, per_cur = None, None
                                    if per_prev is None or per_cur is None or per_prev != per_cur:
                                        continue
                                    x1 = float(pd.to_numeric(r_prev.get('x_period'), errors='coerce'))
                                    x2 = float(pd.to_numeric(r_cur.get('x_period'), errors='coerce'))
                                    if x2 < x1:
                                        x1, x2 = x2, x1
                                    for p in (r_prev.get('quinteto_visitante') or []):
                                        intervals.append({'player': str(p), 'x1': x1, 'x2': x2})
                                intervals_df = pd.DataFrame(intervals)
                                # Eventos ENTRA/SALE (VISITANTE)
                                ev = pd.DataFrame()
                                if 'accion_tipo' in dfp.columns and 'Condicion' in dfp.columns:
                                    ev = dfp[dfp['accion_tipo'].isin(['CAMBIO-JUGADOR-ENTRA','CAMBIO-JUGADOR-SALE']) & (dfp['Condicion'].astype(str).str.upper()=='VISITANTE')].copy()
                                    if not ev.empty:
                                        ev['player'] = ev.get('nombre', '').astype(str)
                                if not intervals_df.empty:
                                    intervals_df['player'] = intervals_df['player'].astype(str)
                                    players_count = len(y_order_vis) if y_order_vis else intervals_df['player'].nunique()
                                    chart_height = max(380, 28 * players_count)
                                    halo_vis = (
                                        alt.Chart(intervals_df)
                                        .mark_rule(color='#000000', strokeWidth=9, opacity=0.15)
                                        .encode(
                                            x=alt.X('x1:Q', title='Tiempo (x_period)'),
                                            x2='x2:Q',
                                            y=alt.Y('player:N', sort=y_order_vis if y_order_vis else None, title='Jugador (Visitante)', axis=alt.Axis(labelFontSize=13))
                                        )
                                    )
                                    base = (
                                        alt.Chart(intervals_df)
                                        .mark_rule(color=color_visitante, strokeWidth=6)
                                        .encode(
                                            x=alt.X('x1:Q', title='Tiempo (x_period)'),
                                            x2='x2:Q',
                                            y=alt.Y('player:N', sort=y_order_vis if y_order_vis else None, title='Jugador (Visitante)', axis=alt.Axis(labelFontSize=13))
                                        )
                                        .properties(height=chart_height, title=alt.TitleParams(text=f'Jugadores en cancha - {visitante_title}', anchor='middle'))
                                    )
                                    # Reglas verticales y etiquetas de periodo
                                    period_changes = []
                                    prev = None
                                    for _, rp in dfp.iterrows():
                                        cur = rp.get('numero_periodo')
                                        if prev is not None and cur != prev:
                                            period_changes.append({'x_period': rp.get('x_period'), 'numero_periodo': cur})
                                        prev = cur
                                    chart_vis = halo_vis + base
                                    if period_changes:
                                        rules_df = pd.DataFrame(period_changes)
                                        rules = alt.Chart(rules_df).mark_rule(color='#333333', strokeDash=[8,4], strokeWidth=3).encode(
                                            x=alt.X('x_period:Q'),
                                            tooltip=[alt.Tooltip('numero_periodo:N', title='Inicio periodo')]
                                        )
                                        top_name = y_order_vis[0] if y_order_vis else intervals_df['player'].unique().tolist()[0]
                                        labels_df = rules_df.copy()
                                        labels_df['label'] = 'P ' + labels_df['numero_periodo'].astype(str)
                                        labels_df['y_label'] = top_name
                                        labels = alt.Chart(labels_df).mark_text(align='left', baseline='bottom', dx=6, dy=-8, color='#333333', fontSize=14).encode(
                                            x=alt.X('x_period:Q'), y=alt.Y('y_label:N'), text='label:N'
                                        )
                                        chart_vis = chart_vis + rules + labels
                                    # Marcas ENTRA/SALE
                                    if not ev.empty:
                                        enter = ev[ev['accion_tipo']=='CAMBIO-JUGADOR-ENTRA'].copy()
                                        sale = ev[ev['accion_tipo']=='CAMBIO-JUGADOR-SALE'].copy()
                                        if not enter.empty:
                                            enter_layer = (
                                                alt.Chart(enter)
                                                .mark_point(filled=True, size=140, color='#2e7d32', shape='circle')
                                                .encode(x=alt.X('x_period:Q'), y=alt.Y('player:N', sort=y_order_vis if y_order_vis else None))
                                            )
                                            chart_vis = chart_vis + enter_layer
                                        if not sale.empty:
                                            sale_layer = (
                                                alt.Chart(sale)
                                                .mark_point(filled=True, size=140, color='#c62828', shape='circle')
                                                .encode(x=alt.X('x_period:Q'), y=alt.Y('player:N', sort=y_order_vis if y_order_vis else None))
                                            )
                                            chart_vis = chart_vis + sale_layer
                                    # Marcas de tiros (VISITANTE)
                                    if 'accion_tipo' in dfp.columns and 'Condicion' in dfp.columns:
                                        shots = dfp[(dfp['Condicion'].astype(str).str.upper()=='VISITANTE') & (dfp['accion_tipo'].isin([
                                            'CANASTA-1P','TIRO1-FALLADO','CANASTA-2P','TIRO2-FALLADO','CANASTA-3P','TIRO3-FALLADO'
                                        ]))].copy()
                                        if not shots.empty:
                                            shots['player'] = shots.get('nombre','').astype(str)
                                            # Separación adicional en X para VISITANTE (igual que LOCAL)
                                            try:
                                                at = shots['accion_tipo'].astype(str)
                                                tipo_num = at.str.extract(r'(\d)P', expand=False).fillna('0').astype(int)
                                                is_made = at.str.startswith('CANASTA-')
                                                base_off = np.where(is_made, 0.25, -0.25)
                                                tipo_off = np.where(tipo_num == 1, 0.00, np.where(tipo_num == 2, 0.08, 0.16))
                                                xper = pd.to_numeric(shots.get('x_period', 0), errors='coerce').fillna(0)
                                                shots['__x_adj'] = xper + base_off + tipo_off
                                            except Exception:
                                                shots['__x_adj'] = shots.get('x_period', 0)
                                            def shot_layer_text(df_in, color_hex, glyph, dx_px):
                                                if df_in.empty:
                                                    return None
                                                return (
                                                    alt.Chart(df_in)
                                                    .mark_text(fontWeight='bold', fontSize=26, color=color_hex, stroke='black', strokeWidth=0.6, dx=dx_px)
                                                    .encode(
                                                        x=alt.X('__x_adj:Q'),
                                                        y=alt.Y('player:N', sort=y_order_vis if y_order_vis else None),
                                                        text=alt.value(glyph)
                                                    )
                                                )
                                            grn = '#2e7d32'
                                            red = '#c62828'
                                            l_m1 = shot_layer_text(shots[shots['accion_tipo']=='CANASTA-1P'], grn, '1', 8)
                                            l_x1 = shot_layer_text(shots[shots['accion_tipo']=='TIRO1-FALLADO'], red, '1', -8)
                                            l_m2 = shot_layer_text(shots[shots['accion_tipo']=='CANASTA-2P'], grn, '2', 8)
                                            l_x2 = shot_layer_text(shots[shots['accion_tipo']=='TIRO2-FALLADO'], red, '2', -8)
                                            l_m3 = shot_layer_text(shots[shots['accion_tipo']=='CANASTA-3P'], grn, '3', 8)
                                            l_x3 = shot_layer_text(shots[shots['accion_tipo']=='TIRO3-FALLADO'], red, '3', -8)
                                            for lyr in [l_m1,l_x1,l_m2,l_x2,l_m3,l_x3]:
                                                if lyr is not None:
                                                    chart_vis = chart_vis + lyr
                                    st.altair_chart(chart_vis, use_container_width=True)
                                    st.caption("1 verde: 1P convertido · 1 rojo: 1P fallado · 2 verde: 2P convertido · 2 rojo: 2P fallado · 3 verde: 3P convertido · 3 rojo: 3P fallado · ● Verde: entra · ● Rojo: sale")
                                else:
                                    st.info('No se pudieron derivar intervalos de presencia del PBP (VISITANTE).')
                        except Exception as e:
                            st.warning(f'No se pudo generar el timeline VISITANTE: {e}')
                    else:
                        st.info('Sin datos de quintetos VISITANTE para los filtros seleccionados')
                else:
                    st.info('No hay datos de quintetos para mostrar')


            # (Se elimina la pestaña Aux; el timeline fue integrado arriba en Quintetos)

            #     def two_digit(v: Any) -> str:
            #         try:
            #             iv = int(pd.to_numeric(v, errors='coerce'))
            #             return f"{iv:02d}"
            #         except Exception:
            #             return ''

            #     def add_dd_nombre(df: pd.DataFrame) -> pd.DataFrame:
            #         if df is None or df.empty:
            #             return df
            #         dcol = _first_col2(df, ['dorsal','numero','nro','número','numero_camiseta','n_camisa'])
            #         ncol = _first_col2(df, ['Nombre','nombre','Jugador','jugador','nombre_jugador','NombreJugador'])
            #         out = df.copy()
            #         if dcol and ncol:
            #             out['DD_NOMBRE'] = out.apply(lambda r: (two_digit(r.get(dcol)) + ' - ' + str(r.get(ncol)).strip()).strip(' -'), axis=1)
            #         elif ncol:
            #             out['DD_NOMBRE'] = out[ncol].astype(str).str.strip()
            #         else:
            #             out['DD_NOMBRE'] = ''
            #         return out

            #     plan_loc_v = add_dd_nombre(plan_loc)
            #     plan_vis_v = add_dd_nombre(plan_vis)

            #     col1, col2 = st.columns(2)
            #     with col1:
            #         st.markdown('Local (estadisticas_equipolocal)')
            #         st.dataframe(plan_loc_v, use_container_width=True, hide_index=True)
            #     with col2:
            #         st.markdown('Visitante (estadisticas_equipovisitante)')
            #         st.dataframe(plan_vis_v, use_container_width=True, hide_index=True)


else:
    st.info("Ingrese un ID de partido, presione 'Descargar y procesar' o use los datos ya descargados previamente.")
