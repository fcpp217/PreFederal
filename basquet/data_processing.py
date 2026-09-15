"""Transformación de los datos crudos de la API en tablas listas para análisis.

Contiene el pipeline principal: enriquecer jugadas con equipo/condición,
reconstruir el tanteo jugada a jugada, derivar quintetos en cancha y
agregar el tiempo jugado / rendimiento por jugador y por quinteto.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .http_client import fetch_estadisticas, fetch_partido
from .utils import puntos_canasta, to_seconds

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
        # Tanteo acumulado por partido: suma acumulada de puntos por condición,
        # respetando el orden ya establecido por el sort previo (por partido y tiempo).
        pts = df["accion_tipo"].map(puntos_canasta)
        pts_local = pts.where(df["Condicion"] == "LOCAL", 0)
        pts_visitante = pts.where(df["Condicion"] == "VISITANTE", 0)
        df["puntosLocal"] = pts_local.groupby(df["_id"]).cumsum()
        df["puntosVisitante"] = pts_visitante.groupby(df["_id"]).cumsum()
        df["DifPuntos"] = df["puntosLocal"] - df["puntosVisitante"]

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

def descargar_y_transformar(partido_id: str) -> Dict[str, pd.DataFrame]:
    # Descargas: ambos endpoints son independientes, se piden en paralelo
    # para no pagar dos veces la latencia de red de forma secuencial.
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_partido = executor.submit(fetch_partido, partido_id)
        future_estadisticas = executor.submit(fetch_estadisticas, partido_id)
        part_payload = future_partido.result()
        est_payload = future_estadisticas.result()

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
