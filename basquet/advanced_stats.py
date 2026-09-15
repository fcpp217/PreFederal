"""Estadísticas avanzadas de equipo y de jugador (posesiones, eficiencia, etc.).

Replica las fórmulas de la planilla de referencia del club (hojas
`CargaDatosEq` / `CargaDatosJug` / `PartidoUnicoAvanzadas`):

- Posesiones = FGA + 0.44*FTA - OREB + TOV
- Eficiencia Ofensiva = Puntos propios / Posesiones propias
- Eficiencia Defensiva = Puntos del rival / Posesiones propias
- Net Rating = Eficiencia Ofensiva - Eficiencia Defensiva
- % Rebotes Defensivos = RebDef propio / (RebDef propio + RebOf rival)
- % Rebotes Ofensivos = RebOf propio / (RebOf propio + RebDef rival)
- % Rebotes Totales = RebTot propio / (RebTot propio + RebTot rival)
- % Asistencias / % Pérdidas / % Robos / % Bloqueos = esa estadística / Posesiones propias
- 3p/FG = intentos de 3P / (intentos de 3P + intentos de 2P)
- eFG% = (2P convertidos + 1.5 * 3P convertidos) / (intentos 2P + intentos 3P)
- TS% = Puntos / (2 * (intentos 2P + intentos 3P + 0.44 * intentos de tiro libre))
- FT% (tasa de tiros libres, tal como está en la planilla) = tiros libres convertidos / (intentos 2P + intentos 3P)

Los tiros libres no aportan a "Posesiones" salvo por el factor 0.44 (una
aproximación estándar de cuántas posesiones terminan en línea de tiros
libres), y no se cuentan como tiro de campo en ninguna de estas fórmulas.
"""

from typing import Dict

import numpy as np
import pandas as pd

from .utils import _first_col


def _sum_col(df: pd.DataFrame, col) -> float:
    if not col or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors='coerce').fillna(0).sum())


def totales_raw_equipo(df: pd.DataFrame) -> Dict[str, float]:
    """Suma las estadísticas "crudas" (conteos) de todos los jugadores de un equipo."""
    if df is None or df.empty:
        return {k: 0.0 for k in [
            'puntos', '2pm', '2pa', '3pm', '3pa', 'ftm', 'fta',
            'reb_def', 'reb_of', 'reb_tot', 'asistencias', 'perdidas',
            'robos', 'bloqueos_cometidos', 'bloqueos_recibidos',
        ]}
    cols = {
        'puntos': _first_col(df, ['puntos', 'pts']),
        '2pm': _first_col(df, ['canasta2p', 'canastas2p', 'conv2p']),
        '2pa': _first_col(df, ['tiro2p', 'tiros2p', 'int2p']),
        '3pm': _first_col(df, ['canasta3p', 'canastas3p', 'conv3p']),
        '3pa': _first_col(df, ['tiro3p', 'tiros3p', 'int3p']),
        'ftm': _first_col(df, ['canasta1p', 'canastas1p', 'conv1p', 'convierte1p']),
        'fta': _first_col(df, ['tiro1p', 'tiros1p', 'int1p']),
        'reb_def': _first_col(df, ['rebotedefensivo']),
        'reb_of': _first_col(df, ['reboteofensivo']),
        'asistencias': _first_col(df, ['asistencias']),
        'perdidas': _first_col(df, ['perdidas']),
        'robos': _first_col(df, ['recuperaciones']),
        'bloqueos_cometidos': _first_col(df, ['taponescometidos', 'tapones_cometidos', 'taponcometido', 'bloqueoscometidos']),
        'bloqueos_recibidos': _first_col(df, ['taponesrecibidos', 'tapones_recibidos', 'taponrecibido', 'bloqueosrecibidos']),
    }
    out = {k: _sum_col(df, c) for k, c in cols.items()}
    out['reb_tot'] = out['reb_def'] + out['reb_of']
    return out


def _posesiones(t: Dict[str, float]) -> float:
    fga = t['2pa'] + t['3pa']
    return fga + 0.44 * t['fta'] - t['reb_of'] + t['perdidas']


def calcular_avanzadas_equipo(propio: Dict[str, float], rival: Dict[str, float]) -> Dict[str, float]:
    """Métricas avanzadas de un equipo, usando también los totales del rival
    (necesarios para eficiencia defensiva y los % de rebote)."""
    poses = _posesiones(propio)
    poses_rival = _posesiones(rival)

    def div(a, b):
        return (a / b) if b else 0.0

    fga = propio['2pa'] + propio['3pa']
    reb_tot_rival = rival['reb_def'] + rival['reb_of']
    ef_of = div(propio['puntos'], poses)
    ef_def = div(rival['puntos'], poses)
    return {
        'Posesiones': poses,
        'Eficiencia Ofensiva': ef_of,
        'Eficiencia Defensiva': ef_def,
        'Net Rating': ef_of - ef_def,
        '% Rebotes Defensivos': div(propio['reb_def'], propio['reb_def'] + rival['reb_of']),
        '% Rebotes Ofensivos': div(propio['reb_of'], propio['reb_of'] + rival['reb_def']),
        '% Rebotes Totales': div(propio['reb_tot'], propio['reb_tot'] + reb_tot_rival),
        '% Asistencias': div(propio['asistencias'], poses),
        '% Pérdidas': div(propio['perdidas'], poses),
        '% Robos': div(propio['robos'], poses),
        '% Bloqueos': div(propio['bloqueos_cometidos'], poses),
        '3p/FG': div(propio['3pa'], fga),
        'eFG%': div(propio['2pm'] + 1.5 * propio['3pm'], fga),
        'TS%': div(propio['puntos'], 2 * (fga + 0.44 * propio['fta'])),
        'FT%': div(propio['ftm'], fga),
        '_poses_rival': poses_rival,
    }


def calcular_avanzadas_jugador(df: pd.DataFrame) -> pd.DataFrame:
    """3p/FG%, eFG%, TS% y FT% (tasa) por jugador, igual que CargaDatosJug."""
    if df is None or df.empty:
        return pd.DataFrame()
    nombre_col = _first_col(df, ['nombre', 'jugador', 'nombre_jugador'])
    can2 = _first_col(df, ['canasta2p', 'canastas2p', 'conv2p'])
    tir2 = _first_col(df, ['tiro2p', 'tiros2p', 'int2p'])
    can3 = _first_col(df, ['canasta3p', 'canastas3p', 'conv3p'])
    tir3 = _first_col(df, ['tiro3p', 'tiros3p', 'int3p'])
    tir1_made = _first_col(df, ['canasta1p', 'canastas1p', 'conv1p', 'convierte1p'])
    tir1_att = _first_col(df, ['tiro1p', 'tiros1p', 'int1p'])
    puntos_col = _first_col(df, ['puntos', 'pts'])

    def num(col):
        return pd.to_numeric(df[col], errors='coerce').fillna(0) if col and col in df.columns else pd.Series(0.0, index=df.index)

    m2, a2 = num(can2), num(tir2)
    m3, a3 = num(can3), num(tir3)
    m1, a1 = num(tir1_made), num(tir1_att)
    puntos = num(puntos_col)
    fga = a2 + a3

    out = pd.DataFrame({
        'Nombre': df[nombre_col] if nombre_col else '',
        '3p/FG%': np.where(fga > 0, a3 / fga, 0.0),
        'eFG%': np.where(fga > 0, (m2 + 1.5 * m3) / fga, 0.0),
        'TS%': np.where((fga + 0.44 * a1) > 0, puntos / (2 * (fga + 0.44 * a1)), 0.0),
        'FT%': np.where(fga > 0, m1 / fga, 0.0),
    })
    return out
