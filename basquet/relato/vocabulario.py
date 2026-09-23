"""Vocabulario cerrado del relato: zonas, marca, tipo de tiro, resultados y
palabras clave de cada evento.

Todas las expresiones se escriben ya normalizadas (minúsculas, sin tildes)
porque el parser normaliza el texto antes de buscarlas. Las zonas evitan a
propósito palabras que sean números para no confundirlas con dorsales.
"""

from typing import Dict, List, Tuple

# ------------------------------
# Zonas de tiro
# ------------------------------
# zona -> (valor del tiro, alias). El alias se busca como secuencia de
# palabras; gana siempre el alias más largo (ej. "triple frontal" antes que
# "frontal").
ZONAS: Dict[str, Tuple[int, List[str]]] = {
    "Aro": (2, ["aro", "debajo"]),
    "Pintura": (2, ["pintura", "zona pintada"]),
    "Base izquierda": (2, ["base izquierda", "base izquierdo", "fondo izquierda", "fondo izquierdo"]),
    "Base derecha": (2, ["base derecha", "base derecho", "fondo derecha", "fondo derecho"]),
    "Codo izquierdo": (2, ["codo izquierdo", "codo izquierda"]),
    "Codo derecho": (2, ["codo derecho", "codo derecha"]),
    "Frontal": (2, ["frontal", "media frontal"]),
    "Doble (sin zona)": (2, ["doble"]),
    "Esquina izquierda": (3, ["esquina izquierda", "esquina izquierdo"]),
    "Esquina derecha": (3, ["esquina derecha", "esquina derecho"]),
    "Ala izquierda": (3, ["ala izquierda", "ala izquierdo"]),
    "Ala derecha": (3, ["ala derecha", "ala derecho"]),
    "Triple frontal": (3, ["triple frontal", "triple arriba"]),
    "Triple (sin zona)": (3, ["triple"]),
}

# ------------------------------
# Distancia del defensor más cercano al momento del tiro
# ------------------------------
MARCAS: Dict[str, List[str]] = {
    "Encima": ["encima", "tapado", "pegado"],     # < 0,6 m
    "Cerca": ["cerca"],                           # 0,6 a 1,2 m
    "Abierto": ["abierto", "abierta"],            # 1,2 a 1,8 m
    "Solo": ["solo", "sola", "sin marca", "nadie"],  # > 1,8 m
}
MARCA_DESCRIPCION = {
    "Encima": "menos de 0,6 m",
    "Cerca": "0,6 a 1,2 m",
    "Abierto": "1,2 a 1,8 m",
    "Solo": "más de 1,8 m",
}

# ------------------------------
# Tipo de tiro (opcional)
# ------------------------------
TIPOS_TIRO: Dict[str, List[str]] = {
    "Recepción": ["recepcion", "catch"],
    "Bote": ["bote", "pull up", "pullup", "dribbling"],
    "Bandeja": ["bandeja", "doble paso"],
    "Volcada": ["volcada", "volcadura", "clavada"],
    "Flotadora": ["flotadora", "floja"],
    "Gancho": ["gancho"],
    "Palmeo": ["palmeo", "palmea"],
}
# Tipos que ya dicen dónde fue el tiro: si no se nombra zona, se asume "Aro".
TIPOS_EN_ARO = {"Bandeja", "Volcada", "Palmeo"}

# ------------------------------
# Resultado del tiro
# ------------------------------
RESULTADOS: Dict[str, List[str]] = {
    "adentro": ["adentro", "dentro", "anota", "convierte", "canasta", "gol", "mete", "metio"],
    "afuera": ["afuera", "fuera", "falla", "fallo", "errado", "erra", "erro", "no entra"],
}

# ------------------------------
# Palabras clave que inician un evento que no es un tiro de campo
# ------------------------------
CLAVES: Dict[str, List[str]] = {
    "libres": ["tiros libres", "tiro libre", "libres", "libre"],
    "rebote": ["rebote"],
    "perdida": ["perdida", "pierde"],
    "robo": ["robo", "roba", "recupera", "recuperacion"],
    "tapa": ["tapa", "tapon", "bloqueo"],
    "falta": ["falta"],
    "entra": ["entra"],
    "sale": ["sale"],
    "tiempo_muerto": ["tiempo muerto", "minuto"],
    "borrar": ["borrar", "borra", "anular", "anula"],
    "asiste": ["asiste", "asistencia", "pase de"],
    "sobre": ["sobre", "a"],
}

# Prefijo opcional para aclarar el equipo cuando el mismo dorsal existe en
# ambos (ej. "local siete").
EQUIPOS: Dict[str, List[str]] = {
    "LOCAL": ["local", "locales"],
    "VISITANTE": ["visita", "visitante", "visitantes"],
}

# ------------------------------
# Números en palabras (0-99)
# ------------------------------
_UNIDADES = {
    "cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
}
_ESPECIALES = {
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "veintiuno": 21, "veintiun": 21, "veintidos": 22, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27,
    "veintiocho": 28, "veintinueve": 29,
}
DECENAS = {
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90,
}
NUMEROS_SIMPLES = {**_UNIDADES, **_ESPECIALES}
# Una sola palabra de número en contexto donde no puede ser un dorsal.
UNIDADES_SUFIJO = {k: v for k, v in _UNIDADES.items() if v > 0}


def prompt_inicial(dorsales: List[str]) -> str:
    """Texto que se le pasa a Whisper como contexto para que reconozca mejor
    el vocabulario del relato y los dorsales del partido."""
    zonas = ", ".join(a[0] for _, a in ZONAS.values())
    marcas = ", ".join(a[0] for a in MARCAS.values())
    tipos = ", ".join(a[0] for a in TIPOS_TIRO.values())
    nums = ", ".join(sorted(set(dorsales), key=lambda d: (len(d), d)))
    return (
        f"Relato de básquet. Zonas: {zonas}. Marca: {marcas}. Tipo: {tipos}. "
        f"Resultado: adentro, afuera. Rebote, pérdida, robo, tapa, falta sobre, "
        f"libres, asiste, entra, sale, tiempo muerto, borrar. Dorsales: {nums}."
    )
