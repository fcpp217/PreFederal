"""Agregaciones de tiros de campo por tiempo de posesión.

Compartido entre la pestaña "Posesión" y el exportador de PDF, para no
duplicar la misma lógica en los dos lugares.
"""

from typing import List

import numpy as np
import pandas as pd

from .data_processing import ACCIONES_TIRO_DE_CAMPO

CONVERTIDAS = {'CANASTA-2P', 'CANASTA-3P'}
PUNTOS_POR_ACCION = {'CANASTA-2P': 2, 'CANASTA-3P': 3}


def preparar_tiros(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Filtra el pbp a los tiros de campo con bucket de posesión calculado."""
    if pbp_df.empty or 'accion_tipo' not in pbp_df.columns or 'bucket_posesion' not in pbp_df.columns:
        return pd.DataFrame()
    d = pbp_df[pbp_df['accion_tipo'].isin(ACCIONES_TIRO_DE_CAMPO)].copy()
    if d.empty:
        return d
    d = d[d['bucket_posesion'].notna()]
    d['Tipo'] = np.where(d['accion_tipo'].isin(['CANASTA-2P', 'TIRO2-FALLADO']), '2P', '3P')
    d['Convertido'] = d['accion_tipo'].isin(CONVERTIDAS)
    d['Puntos'] = d['accion_tipo'].map(PUNTOS_POR_ACCION).fillna(0)
    d['Condicion'] = d.get('Condicion', '').astype(str).str.upper()
    return d


def resumen_por_bucket(d: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Intentos, conversiones, % de efectividad y puntos por bucket de posesión."""
    if d.empty:
        return pd.DataFrame(columns=group_cols + ['Intentos', 'Convertidos', '%Efectividad', 'Puntos'])
    agg = d.groupby(group_cols, as_index=False).agg(
        Intentos=('accion_tipo', 'count'),
        Convertidos=('Convertido', 'sum'),
        Puntos=('Puntos', 'sum'),
    )
    agg['%Efectividad'] = np.where(agg['Intentos'] > 0, (agg['Convertidos'] / agg['Intentos'] * 100.0).round(1), 0.0)
    agg['Puntos'] = agg['Puntos'].astype(int)
    return agg


def resumen_por_bucket_y_tipo(d: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Igual que resumen_por_bucket, pero separando 2P y 3P en columnas propias
    (Intentados/Convertidos/% para cada uno) en vez de un total combinado."""
    tipos = ['2P', '3P']
    columnas_salida = list(group_cols)
    for tipo in tipos:
        columnas_salida += [f'{tipo} Intentados', f'{tipo} Convertidos', f'{tipo} %']
    if d.empty:
        return pd.DataFrame(columns=columnas_salida)

    agg = d.groupby(group_cols + ['Tipo'], as_index=False).agg(
        Intentos=('accion_tipo', 'count'),
        Convertidos=('Convertido', 'sum'),
    )
    base = agg[group_cols].drop_duplicates().reset_index(drop=True)
    for tipo in tipos:
        sub = agg[agg['Tipo'] == tipo][group_cols + ['Intentos', 'Convertidos']].rename(
            columns={'Intentos': f'{tipo} Intentados', 'Convertidos': f'{tipo} Convertidos'}
        )
        base = base.merge(sub, on=group_cols, how='left')
        base[f'{tipo} Intentados'] = base[f'{tipo} Intentados'].fillna(0).astype(int)
        base[f'{tipo} Convertidos'] = base[f'{tipo} Convertidos'].fillna(0).astype(int)
        base[f'{tipo} %'] = np.where(
            base[f'{tipo} Intentados'] > 0,
            (base[f'{tipo} Convertidos'] / base[f'{tipo} Intentados'] * 100.0).round(1),
            0.0,
        )
    return base[columnas_salida]
