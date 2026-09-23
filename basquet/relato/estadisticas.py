"""Estadísticas armadas a partir de los eventos del relato."""

from typing import List

import pandas as pd

TIROS_CAMPO = ["CANASTA-2P", "TIRO2-FALLADO", "CANASTA-3P", "TIRO3-FALLADO"]

_CONTEOS = {
    "2P conv": ["CANASTA-2P"], "2P int": ["CANASTA-2P", "TIRO2-FALLADO"],
    "3P conv": ["CANASTA-3P"], "3P int": ["CANASTA-3P", "TIRO3-FALLADO"],
    "1P conv": ["CANASTA-1P"], "1P int": ["CANASTA-1P", "TIRO1-FALLADO"],
    "Reb. Of.": ["REBOTE-OFENSIVO"], "Reb. Def.": ["REBOTE-DEFENSIVO"],
    "Asist.": ["ASISTENCIA"], "Recup.": ["RECUPERACION"], "Pérdidas": ["PERDIDA"],
    "Tapas": ["TAPON"], "Faltas com.": ["FALTA-COMETIDA"], "Faltas rec.": ["FALTA-RECIBIDA"],
}


def _pct(conv: pd.Series, intentos: pd.Series) -> pd.Series:
    return (conv / intentos.where(intentos > 0) * 100).round(1)


def box_score(eventos: pd.DataFrame, condicion: str) -> pd.DataFrame:
    """Planilla por jugador de un equipo (LOCAL o VISITANTE), con una fila
    de totales al final. Las pérdidas sin dorsal (robo relatado sin la
    pérdida) solo suman al total del equipo."""
    d = eventos[eventos["Condicion"] == condicion].copy()
    if d.empty:
        return pd.DataFrame()
    d["dorsal"] = d["dorsal"].fillna("").astype(str)
    d["jugador"] = d["jugador"].fillna("").astype(str)

    filas: List[dict] = []
    for (dorsal, jugador), g in d.groupby(["dorsal", "jugador"], sort=False):
        fila = {"Dorsal": dorsal or "-", "Jugador": jugador or ("(sin dorsal)" if not dorsal else "")}
        for col, acciones in _CONTEOS.items():
            fila[col] = int(g["accion_tipo"].isin(acciones).sum())
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla = tabla.groupby(["Dorsal", "Jugador"], as_index=False).sum()
    tabla["_orden"] = pd.to_numeric(tabla["Dorsal"], errors="coerce")
    tabla = tabla.sort_values("_orden", na_position="last").drop(columns="_orden")

    total = tabla.drop(columns=["Dorsal", "Jugador"]).sum()
    total["Dorsal"], total["Jugador"] = "", "TOTAL"
    tabla = pd.concat([tabla, total.to_frame().T], ignore_index=True)

    num = [c for c in _CONTEOS]
    tabla[num] = tabla[num].astype(int)
    tabla.insert(2, "Puntos", tabla["2P conv"] * 2 + tabla["3P conv"] * 3 + tabla["1P conv"])
    for t in ("2P", "3P", "1P"):
        tabla[f"%{t}"] = _pct(tabla[f"{t} conv"], tabla[f"{t} int"])
    tabla["Reb. Tot."] = tabla["Reb. Of."] + tabla["Reb. Def."]
    orden = ["Dorsal", "Jugador", "Puntos", "2P conv", "2P int", "%2P", "3P conv", "3P int", "%3P",
             "1P conv", "1P int", "%1P", "Reb. Of.", "Reb. Def.", "Reb. Tot.", "Asist.", "Recup.",
             "Pérdidas", "Tapas", "Faltas com.", "Faltas rec."]
    return tabla[orden]


def tiros_por(eventos: pd.DataFrame, condicion: str, columna: str) -> pd.DataFrame:
    """Tiros de campo de un equipo agrupados por `columna` (zona, marca,
    tipo_tiro o bucket_posesion): intentos, convertidos, % y puntos por tiro."""
    d = eventos[(eventos["Condicion"] == condicion) & eventos["accion_tipo"].isin(TIROS_CAMPO)].copy()
    if d.empty:
        return pd.DataFrame()
    d[columna] = d[columna].fillna("(sin dato)")
    d["conv"] = d["accion_tipo"].str.startswith("CANASTA")
    d["pts"] = d["accion_tipo"].map({"CANASTA-2P": 2, "CANASTA-3P": 3}).fillna(0)
    g = d.groupby(columna).agg(Intentos=("conv", "size"), Convertidos=("conv", "sum"), Puntos=("pts", "sum"))
    g["%"] = _pct(g["Convertidos"], g["Intentos"])
    g["Pts por tiro"] = (g["Puntos"] / g["Intentos"]).round(2)
    return g.sort_values("Intentos", ascending=False).reset_index()
