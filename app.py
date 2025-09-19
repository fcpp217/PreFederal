import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import requests
import streamlit as st
import re
import altair as alt

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
            df_jug[col] = df_jug[col].astype("string").fillna("").str.strip()
    for df_stats in [df_loc, df_vis]:
        for col in ["_id", "idequipo", "dorsal", "nombre"]:
            if col in df_stats.columns:
                df_stats[col] = df_stats[col].astype("string").fillna("").str.strip()
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
    df_jug["nombre"] = np.where(
        df_jug.get("equipo_id", "") == -1,
        "SIN JUGADOR",
        np.where(
            df_jug["nombre_local"].notnull() & (df_jug["nombre_local"] != ""),
            df_jug["nombre_local"],
            df_jug["nombre_visitante"],
        )
    )
    if "dorsal" in df_jug.columns:
        dorsal_str = df_jug["dorsal"].astype("string").fillna("").str.strip()
        nombre_str = df_jug["nombre"].astype("string").fillna("").str.strip()
        nombre_str = nombre_str.mask(nombre_str.str.upper() == "NOMBRE", "")
        df_jug["nombre"] = (dorsal_str.str.zfill(2) + "-" + nombre_str).str.strip("-")
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

        for _, fila in df_partido.iterrows():
            jugador_key = (fila.get('equipo', ''), fila.get('nombre', ''))
            if jugador_key not in condicion_jugador:
                condicion_jugador[jugador_key] = fila.get('Condicion', '')

            accion = fila.get('accion_tipo')
            tiempo_actual = fila.get('tiempo_segundos', 0)
            numero_periodo = fila.get('numero_periodo')
            condicion_fija = condicion_jugador[jugador_key]

            if (
                fila_anterior is not None and
                fila_anterior.get('accion_tipo') != 'FINAL-PERIODO' and
                fila_anterior.get('numero_periodo') == numero_periodo
            ):
                delta_tiempo = abs(float(fila_anterior.get('tiempo_segundos', 0)) - float(tiempo_actual))
                delta_local = float(fila.get('puntosLocal', 0)) - float(fila_anterior.get('puntosLocal', 0))
                delta_visitante = float(fila.get('puntosVisitante', 0)) - float(fila_anterior.get('puntosVisitante', 0))

                for jug in list(jugadores_en_cancha):
                    cond_jug = condicion_jugador.get(jug, None)
                    if cond_jug is None:
                        continue
                    clave = (
                        pid, jug[0], jug[1],
                        cond_jug, fila.get('SituacionMarcador'),
                        fila.get('ultimos_dos_minutos'), numero_periodo
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

        for clave in tiempo_jugado:
            partido_id2, equipo, nombre, condicion, situacion, ult2min, periodo = clave
            if condicion != 'NEUTRAL':
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

    df_TiempoJugadores = pd.DataFrame(resultados)
    acciones_interes = [
        'TIRO2-FALLADO', 'REBOTE-DEFENSIVO', 'CANASTA-2P', 'PERDIDA',
        'RECUPERACION', 'TIRO3-FALLADO', 'ASISTENCIA', 'CANASTA-3P',
        'FALTA-COMETIDA', 'FALTA-RECIBIDA', 'REBOTE-OFENSIVO',
        'TIRO1-FALLADO', 'CANASTA-1P'
    ]
    df_acciones = df_tmp[df_tmp.get('accion_tipo', pd.Series(dtype=str)).isin(acciones_interes)] if 'accion_tipo' in df_tmp.columns else pd.DataFrame(columns=['accion_tipo'])
    if not df_acciones.empty:
        df_acc_ind = (
            df_acciones
            .groupby(['_id','equipo','nombre','Condicion','numero_periodo','SituacionMarcador','ultimos_dos_minutos'])['accion_tipo']
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        df_JugadoresFinal = df_TiempoJugadores.merge(
            df_acc_ind,
            on=['_id','equipo','nombre','Condicion','numero_periodo','SituacionMarcador','ultimos_dos_minutos'],
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
            situacion = fila.get('SituacionMarcador')
            ult2min = fila.get('ultimos_dos_minutos')

            quinteto_local = fila.get('quinteto_local')
            quinteto_visitante = fila.get('quinteto_visitante')
            if isinstance(quinteto_local, list):
                quinteto_local = tuple(sorted(quinteto_local))
            if isinstance(quinteto_visitante, list):
                quinteto_visitante = tuple(sorted(quinteto_visitante))
            if quinteto_local is not None and len(quinteto_local) != 5:
                quinteto_local = None
            if quinteto_visitante is not None and len(quinteto_visitante) != 5:
                quinteto_visitante = None

            if (
                fila_anterior is not None and
                fila_anterior.get('accion_tipo') != 'FINAL-PERIODO' and
                fila_anterior.get('numero_periodo') == numero_periodo
            ):
                delta_tiempo = abs(float(fila_anterior.get('tiempo_segundos', 0)) - tiempo_actual)
                delta_local = float(fila.get('puntosLocal', 0)) - float(fila_anterior.get('puntosLocal', 0))
                delta_visitante = float(fila.get('puntosVisitante', 0)) - float(fila_anterior.get('puntosVisitante', 0))
                if quinteto_local is not None:
                    clave_local = (partido_id3, quinteto_local, 'LOCAL', situacion, ult2min, numero_periodo)
                    tiempo_quinteto[clave_local] = tiempo_quinteto.get(clave_local, 0) + delta_tiempo
                    puntos_favor_q[clave_local] = puntos_favor_q.get(clave_local, 0) + delta_local
                    puntos_contra_q[clave_local] = puntos_contra_q.get(clave_local, 0) + delta_visitante
                if quinteto_visitante is not None:
                    clave_vis = (partido_id3, quinteto_visitante, 'VISITANTE', situacion, ult2min, numero_periodo)
                    tiempo_quinteto[clave_vis] = tiempo_quinteto.get(clave_vis, 0) + delta_tiempo
                    puntos_favor_q[clave_vis] = puntos_favor_q.get(clave_vis, 0) + delta_visitante
                    puntos_contra_q[clave_vis] = puntos_contra_q.get(clave_vis, 0) + delta_local

            if accion in acciones_interes:
                condicion = str(fila.get('Condicion', '')).upper()
                if quinteto_local is not None:
                    clave_local = (partido_id3, quinteto_local, 'LOCAL', situacion, ult2min, numero_periodo)
                else:
                    clave_local = None
                if quinteto_visitante is not None:
                    clave_vis = (partido_id3, quinteto_visitante, 'VISITANTE', situacion, ult2min, numero_periodo)
                else:
                    clave_vis = None

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
        raise ValueError("Partido no válido o incompleto")

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
            tiempo_num = pd.to_numeric(pbp_df.get('tiempo_segundos', np.nan), errors='coerce')
            periodo_num = pd.to_numeric(pbp_df.get('numero_periodo', 1), errors='coerce').fillna(1)
            pbp_df['x_period'] = (600 - tiempo_num.fillna(0)) + (periodo_num - 1) * 600

    return {
        'partido': partido_df,
        'jugada': jugada_df,
        'estadistica': estadistica_df,
        'estadisticas_equipolocal': local_df,
        'estadisticas_equipovisitante': visitante_df,
        'pbp': pbp_df,
        'jugadoresAgregado': jugadores_df,
        'qunitetosAgregado': quintetos_df,
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
                with st.spinner("Descargando y procesando datos..."):
                    tablas = descargar_y_transformar(partido_id_input.strip())
                # Persistir en sesión para evitar re-descarga al cambiar filtros
                st.session_state['tablas'] = tablas
            except Exception as e:
                # Mostrar un mensaje claro y corto, sin traza
                st.error("ID de Partido no encontrado")
                tablas = None
    else:
        tablas = st.session_state.get('tablas')

    if tablas is not None:
        # Mantener orden fijo con 'Estadisticas por jugador' primero para evitar cambios de pestaña al aplicar filtros
        # Ocultar pestañas pbp, jugadoresAgregado y qunitetosAgregado
        nombres = ["Estadisticas por jugador", "Resumen"]
        # Mostrar pestañas
        tabs = st.tabs(nombres)
        # Referencias por nombre
        t_estadistica = tabs[0]
        t_resumen = tabs[1]

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
                    sel_periodo = st.selectbox('Seleccionar periodo', opciones_periodo, index=0)
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
                        # x_period = (600 - tiempo_segundos) + (numero_periodo - 1) * 600
                        tiempo_num = pd.to_numeric(dfp.get('tiempo_segundos', np.nan), errors='coerce')
                        periodo_num = pd.to_numeric(dfp.get('numero_periodo', 1), errors='coerce').fillna(1)
                        # Si no hay tiempo_segundos, usar orden como aproximación
                        tiempo_num = tiempo_num.fillna(dfp['orden'])
                        dfp['x_period'] = (600 - tiempo_num) + (periodo_num - 1) * 600

                    # Asegurar x_period numérico exacto del pbp
                    dfp['x_period'] = pd.to_numeric(dfp['x_period'], errors='coerce')
                    dfp = dfp[~dfp['x_period'].isna()].copy()
                    # Selección interactiva sobre eje X
                    brush = alt.selection_interval(encodings=['x'], name='Seleccion')
                    # Series explícitas por equipo para evitar valores no presentes en pbp
                    dfp['puntosLocal_num'] = pd.to_numeric(dfp.get('puntosLocal', 0), errors='coerce').fillna(0)
                    dfp['puntosVisitante_num'] = pd.to_numeric(dfp.get('puntosVisitante', 0), errors='coerce').fillna(0)

                    line_local = (
                        alt.Chart(dfp)
                        .mark_line(point=False, color=color_local)
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
                        .mark_line(point=False, color=color_visitante)
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

                            # Selección por leyenda para tipo de canasta
                            sel_leg = alt.selection_point(fields=['accion_tipo'], bind='legend')

                            points = alt.Chart(ev).mark_point(filled=True, size=point_size).encode(
                                x=alt.X('x_period:Q', title='Tiempo', sort=None),
                                y=alt.Y('y_points:Q'),
                                color=alt.Color('accion_tipo:N', title='Tipos de canasta', scale=alt.Scale(
                                    domain=['CANASTA-1P','CANASTA-2P','CANASTA-3P'],
                                    range=['#fdd835','#1e88e5','#43a047']
                                ), legend=alt.Legend(orient='bottom', direction='horizontal', title='Tipo de canasta', titleAnchor='middle')),
                                opacity=alt.condition(sel_leg, alt.value(1.0), alt.value(0.2)),
                                order=alt.Order('orden:Q'),
                                tooltip=[
                                    alt.Tooltip('Equipo:N'),
                                    alt.Tooltip('accion_tipo:N', title='Acción'),
                                    alt.Tooltip('numero_periodo:Q', title='Periodo'),
                                    alt.Tooltip('tiempo_segundos:Q', title='Tiempo (s)'),
                                    alt.Tooltip('x_period:Q', title='x_period'),
                                    alt.Tooltip('y_points:Q', title='Puntos')
                                ]
                            ).add_params(sel_leg).transform_filter(brush)
                            chart = chart + points

                    # Leyenda inferior centrada
                    chart = chart.configure_legend(orient='bottom', direction='horizontal', titleAnchor='middle')
                    chart = chart.add_params(brush)
                    st.altair_chart(chart, use_container_width=True)

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
                            .properties(height=640, title=alt.TitleParams(text='Diferencia de puntos (local - visitante)', anchor='middle'))
                            .transform_filter(brush)
                        )
                        # Línea horizontal en 0
                        zero_rule = alt.Chart(pd.DataFrame({'y':[0]})).mark_rule(color='#aaaaaa').encode(y='y:Q')

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
                            # Mínimo dentro del rango seleccionado: última aparición por x_period
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
                            diff_chart = diff_chart + max_point + max_label + min_point + min_label
                        except Exception:
                            pass

                        diff_chart = diff_chart.add_params(brush)
                        st.altair_chart(diff_chart, use_container_width=True)
                    except Exception:
                        pass

                    # (Se elimina sección de estadística desde PBP a pedido del usuario)

            # Pestaña Estadisticas por jugador (desde jugadoresAgregado por jugador, con Totales y derivadas)
            with t_estadistica:
                part_df = tablas.get('partido', pd.DataFrame())
                local_title = str(part_df.iloc[0].get('local')) if not part_df.empty else 'Local'
                visitante_title = str(part_df.iloc[0].get('visitante')) if not part_df.empty else 'Visitante'
                jg = tablas.get('jugadoresAgregado', pd.DataFrame()).copy()
                if not jg.empty:
                    # Normalizar y detectar columnas
                    if 'Condicion' in jg.columns:
                        jg['Condicion'] = jg['Condicion'].astype(str).str.upper().fillna('')
                    if 'equipo' in jg.columns:
                        jg['equipo'] = jg['equipo'].astype(str).str.strip()

                    period_col = _first_col(jg, ['numero_periodo', 'periodo', 'Periodo'])
                    situ_col = _first_col(jg, ['SituacionMarcador', 'Situacion marcador', 'situacion_marcador', 'situacionMarcador', 'situacion', 'Situacion'])
                    u2m_col = _first_col(jg, ['ultimos_dos_minutos', 'ultimos2min', 'ultimos_dos', 'u2m'])

                    with st.form(key='estadistica_filters'):
                        fcols = st.columns(3)
                        with fcols[0]:
                            if period_col and period_col in jg.columns:
                                per_opts = ['TODOS'] + sorted(pd.to_numeric(jg[period_col], errors='coerce').dropna().astype(int).unique().tolist())
                                sel_per = st.selectbox('Número de periodo', per_opts, index=0, key='estad_sel_per')
                            else:
                                sel_per = 'TODOS'
                        with fcols[1]:
                            if situ_col and situ_col in jg.columns:
                                situ_vals = jg[situ_col].astype(str).fillna('').unique().tolist()
                                situ_opts = ['TODOS'] + sorted([s for s in situ_vals if s != ''])
                                sel_situ = st.selectbox('Situacion marcador', situ_opts, index=0, key='estad_sel_situ')
                            else:
                                sel_situ = 'TODOS'
                        with fcols[2]:
                            if u2m_col and u2m_col in jg.columns:
                                u2_vals = jg[u2m_col].astype(str).fillna('').unique().tolist()
                                u2_opts = ['TODOS'] + sorted([u for u in u2_vals if u != ''])
                                sel_u2m = st.selectbox('Momento del periodo', u2_opts, index=0, key='estad_sel_u2m')
                            else:
                                sel_u2m = 'TODOS'
                        submitted = st.form_submit_button('Aplicar filtros')
                        if submitted:
                            _stay_estadistica()

                    # Aplicar filtros
                    jg_f = jg.copy()
                    if sel_per != 'TODOS' and period_col and period_col in jg_f.columns:
                        try:
                            jg_f = jg_f[pd.to_numeric(jg_f[period_col], errors='coerce') == int(sel_per)]
                        except Exception:
                            pass
                    if sel_situ != 'TODOS' and situ_col and situ_col in jg_f.columns:
                        jg_f = jg_f[jg_f[situ_col].astype(str) == str(sel_situ)]
                    if sel_u2m != 'TODOS' and u2m_col and u2m_col in jg_f.columns:
                        jg_f = jg_f[jg_f[u2m_col].astype(str) == str(sel_u2m)]

                    def make_table(df_src: pd.DataFrame, condicion: str) -> pd.DataFrame:
                        df_t = df_src[df_src.get('Condicion', '').astype(str).str.upper() == condicion.upper()].copy()
                        if df_t.empty:
                            return pd.DataFrame()
                        name_col = _first_col(df_t, ['nombre', 'jugador', 'nombre_jugador']) or 'nombre'

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
                        # Pasar por _build_table para formato (conv/att y %)
                        return _build_table(agg)

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
                    st.dataframe(tbl_loc, use_container_width=True, hide_index=True, column_config=build_column_config(tbl_loc))

                    st.markdown(f"""
                        <div style='text-align:center;background:{color_visitante};color:{tc_visitante};padding:10px;border-radius:6px;margin:16px 0 8px;'>
                            <strong>VISITANTE - {visitante_title}</strong>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(tbl_vis, use_container_width=True, hide_index=True, column_config=build_column_config(tbl_vis))
                else:
                    st.info('No hay jugadoresAgregado para generar estadística')

            # Pestañas adicionales ocultas: no se renderizan
else:
    st.info("Ingrese un ID de partido, presione 'Descargar y procesar' o use los datos ya descargados previamente.")
