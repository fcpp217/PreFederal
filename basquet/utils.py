"""Utilidades genéricas: acceso seguro a columnas/valores y conversiones."""

from typing import Any, List, Optional

import pandas as pd
import streamlit as st


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


def _stay_estadistica():
    try:
        st.session_state['force_estadistica'] = True
    except Exception:
        pass
