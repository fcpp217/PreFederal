"""Definiciones de las métricas usadas en toda la app.

Contenido compartido entre la pestaña "Definiciones", los expanders de ayuda
en Posesión/Avanzadas, y la hoja "Definiciones" del Excel exportable, para
no mantener el mismo texto en varios lugares.
"""

from typing import List, Tuple

import pandas as pd

DEFINICIONES: "dict[str, List[Tuple[str, str]]]" = {
    "Estadísticas básicas (Resumen, Jugadores, Quintetos)": [
        ("Puntos", "Total de puntos anotados."),
        ("1P / 2P / 3P (conv/att)", "Tiros convertidos sobre tiros intentados: libres, dobles y triples."),
        ("%1P / %2P / %3P", "Porcentaje de efectividad: convertidos ÷ intentados × 100."),
        ("Rebote Def. / Rebote Of.", "Rebote tomado en el aro propio (defensivo) o en el aro rival, tras un tiro propio errado (ofensivo)."),
        ("Rebotes Totales", "Rebote Def. + Rebote Of."),
        ("Asist.", "Pase que termina directamente en una canasta convertida por un compañero."),
        ("Pérdidas", "Posesión perdida sin llegar a un tiro (pase robado, pasos, violación, etc.)."),
        ("Recup.", "Robo de pelota al rival."),
        ("Tiempo Jugado", "Minutos:segundos que el jugador estuvo en cancha."),
        ("+/-", "Diferencia de puntos del equipo mientras ese jugador (o quinteto) estuvo en cancha."),
    ],
    "Quintetos": [
        ("Quinteto", "Combinación de 5 jugadores en cancha al mismo tiempo."),
        ("Tiempo", "Minutos:segundos que ese quinteto estuvo junto en cancha."),
        ("PF / PC", "Puntos a favor / en contra del equipo mientras ese quinteto estuvo en cancha."),
        ("+/-", "PF − PC."),
    ],
    "Posesión (tiempo de posesión antes del tiro)": [
        ("0-8s / 9-16s / 17-24s", "Segundos desde que el equipo recuperó la pelota (rebote defensivo, robo, pérdida rival o canasta convertida) hasta el tiro. No incluye tiros libres."),
        ("Reb. Of.", "Tiro después de un rebote ofensivo propio: no se mide contra un reloj de 24s nuevo, porque esa jugada no arranca una posesión nueva."),
        ("A revisar (>24s)", "Más de 24s calculados: imposible en básquet real. Casi siempre indica que falta un evento en la carga del partido (rebote, robo o pérdida no registrado)."),
        ("Intentados / Convertidos / %", "Tiros de campo (2P o 3P) intentados y convertidos en ese momento de la posesión, y su % de efectividad."),
    ],
    "Avanzadas": [
        ("Posesiones", "Estimación de cuántas veces tuvo la pelota el equipo: Tiros de campo intentados + 0.44 × Tiros libres intentados − Rebotes ofensivos + Pérdidas."),
        ("Eficiencia Ofensiva", "Puntos propios por posesión propia (no está multiplicada por 100, a diferencia del \"Offensive Rating\" habitual)."),
        ("Eficiencia Defensiva", "Puntos del rival por posesión propia."),
        ("Net Rating", "Eficiencia Ofensiva − Eficiencia Defensiva."),
        ("% Rebotes Defensivos", "Rebotes defensivos propios ÷ (rebotes defensivos propios + rebotes ofensivos del rival): qué tan bien controlás el rebote cuando defendés."),
        ("% Rebotes Ofensivos", "Rebotes ofensivos propios ÷ (rebotes ofensivos propios + rebotes defensivos del rival): segundas oportunidades que generás."),
        ("% Rebotes Totales", "Rebotes totales propios ÷ (rebotes totales propios + rebotes totales del rival)."),
        ("% Asistencias / % Pérdidas / % Robos / % Bloqueos", "Esa estadística dividida por las posesiones propias: con cuántas de tus posesiones generás cada una."),
        ("3p/FG", "Proporción de tiros de campo intentados que fueron de 3 puntos: 3PA ÷ (2PA + 3PA)."),
        ("eFG%", "Porcentaje de tiro efectivo: pondera el triple 1.5 veces más que el doble, porque vale más. (2PM + 1.5×3PM) ÷ (2PA + 3PA)."),
        ("TS%", "Porcentaje de tiro real: eficiencia de anotación incluyendo tiros libres. Puntos ÷ (2 × (2PA + 3PA + 0.44 × TLA))."),
        ("FT%", "Ojo: acá NO es el % de acierto en la línea de tiros libres. Es una tasa (cuánto generás desde la línea por cada tiro de campo): Tiros libres convertidos ÷ (2PA + 3PA)."),
    ],
}


def definiciones_dataframe() -> pd.DataFrame:
    """Todas las definiciones como filas (Sección, Métrica, Definición) - para el Excel."""
    filas = []
    for seccion, items in DEFINICIONES.items():
        for nombre, texto in items:
            filas.append({"Sección": seccion, "Métrica": nombre, "Definición": texto})
    return pd.DataFrame(filas)


def render_definiciones_markdown(secciones=None) -> str:
    """Arma el texto en Markdown para una o más secciones (todas por default)."""
    claves = secciones if secciones is not None else list(DEFINICIONES.keys())
    partes = []
    for seccion in claves:
        items = DEFINICIONES.get(seccion, [])
        if not items:
            continue
        partes.append(f"**{seccion}**")
        for nombre, texto in items:
            partes.append(f"- **{nombre}**: {texto}")
        partes.append("")
    return "\n".join(partes)
