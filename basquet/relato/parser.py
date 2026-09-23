"""Convierte el texto de un relato en eventos de play-by-play.

Flujo:
1. `palabras_desde_texto` pasa el texto editable ("[mm:ss] frase" por línea)
   a una lista de palabras con su segundo dentro del audio.
2. `tokenizar` reconoce el vocabulario cerrado (zonas, marca, números, …).
3. `segmentar` agrupa los tokens en frases, una por evento relatado.
4. `resolver_eventos` asigna equipos (a partir del plantel y de quién tiene
   la pelota), deduce 2P/3P, rebote ofensivo/defensivo y el tiempo de
   posesión, y devuelve un DataFrame con los mismos `accion_tipo` que usa el
   play-by-play oficial para poder compararlos.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data_processing import bucket_posesion
from .vocabulario import (
    CLAVES,
    DECENAS,
    EQUIPOS,
    MARCAS,
    NUMEROS_SIMPLES,
    RESULTADOS,
    TIPOS_EN_ARO,
    TIPOS_TIRO,
    UNIDADES_SUFIJO,
    ZONAS,
)

LOCAL, VISITANTE = "LOCAL", "VISITANTE"

# Plantel: {"LOCAL": {"7": "Pérez"}, "VISITANTE": {...}}
Plantel = Dict[str, Dict[str, str]]


def _otro(equipo: Optional[str]) -> Optional[str]:
    if equipo == LOCAL:
        return VISITANTE
    if equipo == VISITANTE:
        return LOCAL
    return None


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9ñ\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def formatear_tiempo(segundos: float) -> str:
    segundos = max(0.0, float(segundos))
    return f"{int(segundos // 60):02d}:{segundos % 60:04.1f}"


# ------------------------------
# Texto <-> palabras con tiempo
# ------------------------------
_RE_LINEA = re.compile(r"^\s*\[(\d+):(\d+(?:[.,]\d+)?)\]\s*(.*)$")


def palabras_desde_texto(texto: str) -> List[Tuple[str, float]]:
    """Lee líneas "[mm:ss.s] frase" y reparte el tiempo entre las palabras
    de cada línea (interpolando hasta el inicio de la línea siguiente).
    Las líneas sin marca de tiempo heredan el tiempo de la anterior."""
    lineas: List[Tuple[Optional[float], List[str]]] = []
    for linea in str(texto).splitlines():
        m = _RE_LINEA.match(linea)
        if m:
            t = int(m.group(1)) * 60 + float(m.group(2).replace(",", "."))
            cuerpo = m.group(3)
        else:
            t, cuerpo = None, linea
        palabras = normalizar(cuerpo).split()
        if palabras:
            lineas.append((t, palabras))

    salida: List[Tuple[str, float]] = []
    t_prev = 0.0
    for i, (t, palabras) in enumerate(lineas):
        inicio = t if t is not None else t_prev
        siguiente = next((lt for lt, _ in lineas[i + 1:] if lt is not None), None)
        dur = (siguiente - inicio) if (siguiente is not None and siguiente > inicio) else 0.4 * len(palabras)
        dur = min(dur, 0.6 * len(palabras))
        for j, p in enumerate(palabras):
            salida.append((p, inicio + dur * j / len(palabras)))
        t_prev = inicio + dur
    return salida


# ------------------------------
# Tokenización
# ------------------------------
@dataclass
class Token:
    tipo: str  # NUM, ZONA, MARCA, TIPO, RES, CLAVE, EQUIPO
    valor: object
    t: float
    i0: int  # índice de la primera palabra
    i1: int  # índice siguiente a la última palabra


def _lexico() -> Dict[Tuple[str, ...], Tuple[str, object]]:
    lex: Dict[Tuple[str, ...], Tuple[str, object]] = {}

    def add(frase: str, tipo: str, valor: object) -> None:
        lex[tuple(frase.split())] = (tipo, valor)

    for clave, alias in CLAVES.items():
        for a in alias:
            add(a, "CLAVE", clave)
    for equipo, alias in EQUIPOS.items():
        for a in alias:
            add(a, "EQUIPO", equipo)
    for res, alias in RESULTADOS.items():
        for a in alias:
            add(a, "RES", res)
    for tipo, alias in TIPOS_TIRO.items():
        for a in alias:
            add(a, "TIPO", tipo)
    for marca, alias in MARCAS.items():
        for a in alias:
            add(a, "MARCA", marca)
    for zona, (_, alias) in ZONAS.items():
        for a in alias:
            add(a, "ZONA", zona)
    return lex


_LEXICO = _lexico()
_MAX_FRASE = max(len(k) for k in _LEXICO)


def _leer_numero(palabras: List[str], i: int) -> Optional[Tuple[int, int]]:
    """Devuelve (número, palabras consumidas) si en `i` empieza un dorsal."""
    p = palabras[i]
    if p.isdigit() and len(p) <= 3:
        return int(p), 1
    if p in NUMEROS_SIMPLES:
        return NUMEROS_SIMPLES[p], 1
    if p in DECENAS:
        if i + 2 < len(palabras) and palabras[i + 1] == "y" and palabras[i + 2] in UNIDADES_SUFIJO:
            return DECENAS[p] + UNIDADES_SUFIJO[palabras[i + 2]], 3
        return DECENAS[p], 1
    return None


def tokenizar(palabras: List[Tuple[str, float]]) -> List[Token]:
    textos = [p for p, _ in palabras]
    tokens: List[Token] = []
    i = 0
    while i < len(textos):
        encontrado = None
        for n in range(min(_MAX_FRASE, len(textos) - i), 0, -1):
            clave = tuple(textos[i:i + n])
            if clave in _LEXICO:
                encontrado = (_LEXICO[clave], n)
                break
        if encontrado:
            (tipo, valor), n = encontrado
            # "a" solo cuenta como "sobre" justo antes de un número
            if tipo == "CLAVE" and valor == "sobre" and textos[i] == "a":
                if not (i + 1 < len(textos) and _leer_numero(textos, i + 1)):
                    i += 1
                    continue
            tokens.append(Token(tipo, valor, palabras[i][1], i, i + n))
            i += n
            continue
        num = _leer_numero(textos, i)
        if num:
            valor, n = num
            tokens.append(Token("NUM", str(valor), palabras[i][1], i, i + n))
            i += n
            continue
        i += 1  # palabra de relleno
    return tokens


# ------------------------------
# Segmentación en frases
# ------------------------------
@dataclass
class Frase:
    tipo: str  # tiro, libres, rebote, perdida, robo, tapa, falta, cambio, tiempo_muerto
    t: float
    i0: int
    i1: int = 0
    jugador: Optional[str] = None
    equipo_hint: Optional[str] = None
    zona: Optional[str] = None
    marca: Optional[str] = None
    tipo_tiro: Optional[str] = None
    resultado: Optional[str] = None
    defensor: Optional[str] = None
    defensor_hint: Optional[str] = None
    asistente: Optional[str] = None
    tapador: Optional[str] = None
    receptor: Optional[str] = None  # falta: jugador que la recibe
    entra: Optional[str] = None
    sale: Optional[str] = None
    libres: List[str] = field(default_factory=list)
    esperando: Optional[str] = None  # a qué campo va el próximo número
    avisos: List[str] = field(default_factory=list)

    def cerrada(self) -> bool:
        if self.tipo == "tiro":
            return self.resultado is not None
        if self.tipo in ("rebote", "perdida", "robo", "tapa"):
            return self.jugador is not None
        if self.tipo == "falta":
            return self.jugador is not None and self.esperando is None
        if self.tipo == "libres":
            return self.jugador is not None and bool(self.libres)
        if self.tipo == "cambio":
            return self.entra is not None and self.sale is not None
        return True


_CLAVES_INICIO = {
    "libres": "libres", "rebote": "rebote", "perdida": "perdida", "robo": "robo",
    "tapa": "tapa", "falta": "falta", "entra": "cambio", "sale": "cambio",
    "tiempo_muerto": "tiempo_muerto",
}


def segmentar(tokens: List[Token]) -> List[Frase]:
    frases: List[Frase] = []
    actual: Optional[Frase] = None
    hint: Optional[str] = None
    hint_i0: Optional[int] = None  # palabra donde se dijo el equipo

    def cerrar() -> None:
        nonlocal actual
        if actual is not None:
            frases.append(actual)
        actual = None

    def nueva(tipo: str, tok: Token) -> Frase:
        nonlocal actual
        cerrar()
        i0 = hint_i0 if (hint is not None and hint_i0 is not None) else tok.i0
        actual = Frase(tipo=tipo, t=tok.t, i0=i0)
        return actual

    for k, tok in enumerate(tokens):
        sig = tokens[k + 1] if k + 1 < len(tokens) else None
        if actual is not None:
            actual.i1 = max(actual.i1, tok.i1)

        if tok.tipo == "EQUIPO":
            hint, hint_i0 = tok.valor, tok.i0
            if actual is not None and actual.tipo == "tiempo_muerto" and actual.equipo_hint is None:
                actual.equipo_hint = hint
                hint = None
            continue

        if tok.tipo == "CLAVE":
            clave = tok.valor
            if clave == "borrar":
                if actual is not None:
                    actual = None
                elif frases:
                    frases.pop()
                continue
            if clave == "asiste":
                if actual is not None and actual.tipo == "tiro":
                    actual.esperando = "asistente"
                continue
            if clave == "sobre":
                if actual is not None and actual.tipo == "falta":
                    actual.esperando = "receptor"
                continue
            if clave == "tapa" and actual is not None and actual.tipo == "tiro" \
                    and actual.resultado in (None, "afuera") and actual.tapador is None:
                actual.resultado = "afuera"
                actual.esperando = "tapador"
                continue
            if clave in ("entra", "sale") and actual is not None and actual.tipo == "cambio" \
                    and getattr(actual, clave) is None:
                actual.esperando = clave
                continue
            f = nueva(_CLAVES_INICIO[clave], tok)
            if f.tipo == "cambio":
                f.esperando = clave
            if f.tipo == "tiempo_muerto" and hint:
                f.equipo_hint, hint = hint, None
            continue

        if tok.tipo == "NUM":
            # "N zona" siempre arranca un tiro; "N tipo" solo si la frase actual
            # no está esperando todavía a su jugador (ej. "rebote doce, bote…").
            falta_jugador = actual is not None and actual.jugador is None \
                and actual.tipo not in ("tiro", "cambio", "tiempo_muerto")
            empieza_tiro = sig is not None and (sig.tipo == "ZONA" or (sig.tipo == "TIPO" and not falta_jugador))
            if actual is not None and actual.esperando and not empieza_tiro:
                campo = actual.esperando
                setattr(actual, campo, tok.valor)
                if campo == "defensor":
                    actual.defensor_hint = hint
                actual.esperando = None
                hint = None
                continue
            if actual is not None and not empieza_tiro and not actual.cerrada():
                if actual.jugador is None and actual.tipo not in ("cambio", "tiempo_muerto"):
                    actual.jugador, actual.equipo_hint, hint = tok.valor, hint, None
                    continue
                if actual.tipo == "tiro" and actual.defensor is None and sig is not None and sig.tipo == "MARCA":
                    actual.defensor, actual.defensor_hint, hint = tok.valor, hint, None
                    continue
                if actual.tipo == "tiro" and actual.defensor is None and actual.resultado is None:
                    actual.defensor, actual.defensor_hint, hint = tok.valor, hint, None
                    continue
            # Número que no completa la frase actual: arranca un tiro nuevo.
            f = nueva("tiro", tok)
            f.jugador, f.equipo_hint, hint = tok.valor, hint, None
            continue

        # ZONA / MARCA / TIPO / RES
        if actual is None or (actual.tipo != "tiro" and not (tok.tipo == "RES" and actual.tipo == "libres")):
            if tok.tipo == "RES" and actual is not None:
                continue  # resultado suelto fuera de un tiro: se ignora
            f = nueva("tiro", tok)
            f.avisos.append("Tiro sin dorsal del tirador")
        if actual.tipo == "libres":
            actual.libres.append(tok.valor)
            continue
        if tok.tipo == "ZONA":
            if actual.zona is not None and actual.cerrada():
                f = nueva("tiro", tok)
                f.avisos.append("Tiro sin dorsal del tirador")
            actual.zona = tok.valor
        elif tok.tipo == "MARCA":
            actual.marca = tok.valor
        elif tok.tipo == "TIPO":
            actual.tipo_tiro = tok.valor
        elif tok.tipo == "RES" and actual.resultado is None:
            actual.resultado = tok.valor
    cerrar()
    return frases


def lineas_desde_palabras(palabras: List[Tuple[str, float]]) -> str:
    """Arma el texto editable: una línea "[mm:ss.s] frase" por evento."""
    if not palabras:
        return ""
    frases = segmentar(tokenizar(palabras))
    cortes = sorted({f.i0 for f in frases} | {0})
    lineas = []
    for a, b in zip(cortes, cortes[1:] + [len(palabras)]):
        if a >= b:
            continue
        texto = " ".join(p for p, _ in palabras[a:b])
        lineas.append(f"[{formatear_tiempo(palabras[a][1])}] {texto}")
    return "\n".join(lineas)


# ------------------------------
# Resolución: equipos, posesión y eventos finales
# ------------------------------
COLUMNAS_EVENTOS = [
    "numero_periodo", "t_audio", "tiempo_audio", "frase", "accion_tipo", "Condicion", "dorsal",
    "jugador", "zona", "marca", "defensor_dorsal", "defensor", "tipo_tiro",
    "tiempo_posesion", "bucket_posesion", "revisar", "texto",
]


class _Resolver:
    def __init__(self, plantel: Plantel):
        self.plantel = plantel

    def equipo(self, dorsal: Optional[str], hint: Optional[str], preferido: Optional[str],
               avisos: List[str]) -> Optional[str]:
        if dorsal is None:
            return hint or preferido
        if hint:
            if self.plantel.get(hint) and dorsal not in self.plantel[hint]:
                avisos.append(f"#{dorsal} no figura en el plantel {hint.lower()}")
            return hint
        en = [e for e in (LOCAL, VISITANTE) if dorsal in self.plantel.get(e, {})]
        if len(en) == 1:
            return en[0]
        if len(en) == 2:
            if preferido is None:
                avisos.append(f"#{dorsal} está en los dos equipos")
            return preferido
        if any(self.plantel.values()):
            avisos.append(f"#{dorsal} no figura en ningún plantel")
        return preferido

    def nombre(self, equipo: Optional[str], dorsal: Optional[str]) -> str:
        if equipo is None or dorsal is None:
            return ""
        return self.plantel.get(equipo, {}).get(dorsal, "")


def resolver_eventos(frases: List[Frase], plantel: Plantel, periodo: int,
                     palabras: Optional[List[Tuple[str, float]]] = None) -> pd.DataFrame:
    r = _Resolver(plantel)
    filas: List[dict] = []
    ataque: Optional[str] = None
    inicio_posesion: Optional[float] = None
    tras_reb_of = False
    ultimo_fallo: Optional[str] = None  # equipo del último tiro errado
    ultima_falta: Optional[Tuple[Optional[str], Optional[str]]] = None  # (comete, recibe)
    ultimo_tipo: Optional[str] = None

    def cambiar_posesion(equipo: Optional[str], t: float) -> None:
        nonlocal ataque, inicio_posesion, tras_reb_of
        ataque, inicio_posesion, tras_reb_of = equipo, t, False

    # Palabras sueltas del vocabulario ("solo", "cerca", "aro"…) sin tirador ni
    # resultado no alcanzan para ser un tiro: se descartan.
    frases = [f for f in frases if not (f.tipo == "tiro" and f.jugador is None and f.resultado is None)]
    for n, f in enumerate(frases, start=1):
        avisos = list(f.avisos)
        texto = " ".join(p for p, _ in palabras[f.i0:max(f.i1, f.i0 + 1)]) if palabras else ""

        def fila(accion: str, equipo: Optional[str], dorsal: Optional[str], **extra) -> dict:
            base = {
                "numero_periodo": periodo, "t_audio": round(f.t, 1), "tiempo_audio": formatear_tiempo(f.t),
                "frase": n, "accion_tipo": accion, "Condicion": equipo, "dorsal": dorsal,
                "jugador": r.nombre(equipo, dorsal), "zona": None, "marca": None,
                "defensor_dorsal": None, "defensor": "", "tipo_tiro": None,
                "tiempo_posesion": np.nan, "bucket_posesion": None, "revisar": "", "texto": texto,
            }
            base.update(extra)
            return base

        nuevas: List[dict] = []
        if f.tipo == "tiro":
            eq = r.equipo(f.jugador, f.equipo_hint, ataque, avisos)
            zona = f.zona
            if zona is None and f.tipo_tiro in TIPOS_EN_ARO:
                zona = "Aro"
            if zona is None:
                avisos.append("Tiro sin zona (se asume doble)")
            valor = ZONAS[zona][0] if zona else 2
            if f.resultado is None:
                avisos.append("Tiro sin resultado (se asume errado)")
            convertido = f.resultado == "adentro"
            accion = f"CANASTA-{valor}P" if convertido else f"TIRO{valor}-FALLADO"
            eq_def = r.equipo(f.defensor, f.defensor_hint, _otro(eq), avisos) if f.defensor else None
            if eq != ataque:
                if eq is not None and ataque is not None:
                    avisos.append("Tiro del equipo que no tenía la pelota: ¿falta un evento?")
                # Sin saber cuándo empezó la posesión no se mide su tiempo.
                ataque, inicio_posesion, tras_reb_of = eq, None, False
            tp = (f.t - inicio_posesion) if (inicio_posesion is not None and not tras_reb_of) else np.nan
            nuevas.append(fila(
                accion, eq, f.jugador, zona=zona, marca=f.marca, tipo_tiro=f.tipo_tiro,
                defensor_dorsal=f.defensor, defensor=r.nombre(eq_def, f.defensor),
                tiempo_posesion=round(tp, 1) if not pd.isna(tp) else np.nan,
                bucket_posesion=bucket_posesion(tp, es_reb_ofensivo=tras_reb_of),
            ))
            if f.asistente:
                eq_as = r.equipo(f.asistente, None, eq, avisos)
                if not convertido:
                    avisos.append("Asistencia en un tiro errado")
                nuevas.append(fila("ASISTENCIA", eq_as, f.asistente))
            if f.tapador:
                eq_tap = r.equipo(f.tapador, None, _otro(eq), avisos)
                nuevas.append(fila("TAPON", eq_tap, f.tapador))
            if convertido:
                cambiar_posesion(_otro(eq), f.t)
                ultimo_fallo = None
            else:
                ultimo_fallo = eq

        elif f.tipo == "libres":
            pref = ultima_falta[1] if ultima_falta and ultima_falta[1] else (_otro(ultima_falta[0]) if ultima_falta else ataque)
            eq = r.equipo(f.jugador, f.equipo_hint, pref, avisos)
            if not f.libres:
                avisos.append("Libres sin resultado")
            for res in f.libres:
                nuevas.append(fila("CANASTA-1P" if res == "adentro" else "TIRO1-FALLADO", eq, f.jugador))
            if f.libres and f.libres[-1] == "adentro":
                cambiar_posesion(_otro(eq), f.t)
                ultimo_fallo = None
            elif f.libres:
                ultimo_fallo = eq

        elif f.tipo == "rebote":
            pref = _otro(ultimo_fallo) if ultimo_fallo else _otro(ataque)
            if ultimo_fallo is None:
                avisos.append("Rebote sin un tiro errado antes")
            eq = r.equipo(f.jugador, f.equipo_hint, pref, avisos)
            ofensivo = eq is not None and eq == (ultimo_fallo or ataque)
            nuevas.append(fila("REBOTE-OFENSIVO" if ofensivo else "REBOTE-DEFENSIVO", eq, f.jugador))
            if ofensivo:
                ataque, tras_reb_of = eq, True
            else:
                cambiar_posesion(eq, f.t)
            ultimo_fallo = None

        elif f.tipo == "perdida":
            eq = r.equipo(f.jugador, f.equipo_hint, ataque, avisos)
            nuevas.append(fila("PERDIDA", eq, f.jugador))
            cambiar_posesion(_otro(eq), f.t)

        elif f.tipo == "robo":
            viene_de_perdida = ultimo_tipo == "perdida"
            pref = ataque if viene_de_perdida else _otro(ataque)
            eq = r.equipo(f.jugador, f.equipo_hint, pref, avisos)
            if not viene_de_perdida:
                # Todo robo implica una pérdida del rival aunque no se relate.
                nuevas.append(fila("PERDIDA", _otro(eq), None))
            nuevas.append(fila("RECUPERACION", eq, f.jugador))
            cambiar_posesion(eq, f.t)

        elif f.tipo == "tapa":
            eq = r.equipo(f.jugador, f.equipo_hint, _otro(ataque), avisos)
            nuevas.append(fila("TAPON", eq, f.jugador))

        elif f.tipo == "falta":
            eq = r.equipo(f.jugador, f.equipo_hint, _otro(ataque), avisos)
            nuevas.append(fila("FALTA-COMETIDA", eq, f.jugador))
            eq_rec = None
            if f.receptor:
                eq_rec = r.equipo(f.receptor, None, _otro(eq), avisos)
                nuevas.append(fila("FALTA-RECIBIDA", eq_rec, f.receptor))
            ultima_falta = (eq, eq_rec)

        elif f.tipo == "cambio":
            en = [e for e in (LOCAL, VISITANTE)
                  if f.entra in plantel.get(e, {}) and f.sale in plantel.get(e, {})]
            if f.equipo_hint:
                eq = f.equipo_hint
            elif len(en) == 1:
                eq = en[0]
            else:
                eq = r.equipo(f.entra or f.sale, None, None, avisos)
            if f.entra is None or f.sale is None:
                avisos.append("Cambio incompleto (falta quién entra o quién sale)")
            if f.entra:
                nuevas.append(fila("CAMBIO-JUGADOR-ENTRA", eq, f.entra))
            if f.sale:
                nuevas.append(fila("CAMBIO-JUGADOR-SALE", eq, f.sale))

        elif f.tipo == "tiempo_muerto":
            nuevas.append(fila("TIEMPO-MUERTO-SOLICITADO", f.equipo_hint, None))

        if any(fl["Condicion"] is None for fl in nuevas if fl["accion_tipo"] != "TIEMPO-MUERTO-SOLICITADO"):
            avisos.append("No se pudo deducir el equipo")
        motivo = "; ".join(dict.fromkeys(avisos))
        for fl in nuevas:
            fl["revisar"] = motivo
        filas.extend(nuevas)
        ultimo_tipo = f.tipo

    return pd.DataFrame(filas, columns=COLUMNAS_EVENTOS)


def parsear_relato(texto: str, plantel: Plantel, periodo: int) -> pd.DataFrame:
    palabras = palabras_desde_texto(texto)
    frases = segmentar(tokenizar(palabras))
    return resolver_eventos(frases, plantel, periodo, palabras)
