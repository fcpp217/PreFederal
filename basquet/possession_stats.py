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
