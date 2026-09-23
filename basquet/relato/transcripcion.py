"""Transcripción del audio del relato con faster-whisper (corre local, sin
servicios pagos). El modelo se descarga la primera vez que se usa."""

from typing import List, Optional, Tuple

from .parser import normalizar

MODELOS = ["small", "base", "medium"]


def whisper_disponible() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def cargar_modelo(nombre: str):
    from faster_whisper import WhisperModel

    return WhisperModel(nombre, device="cpu", compute_type="int8")


def transcribir(modelo, audio, prompt: Optional[str] = None,
                progreso=None) -> List[Tuple[str, float]]:
    """Devuelve la lista de palabras con el segundo en que empieza cada una.

    `audio` puede ser una ruta o un archivo abierto (mp3, m4a, wav, ogg/opus…).
    `progreso(fraccion)` se llama a medida que avanza la transcripción.
    """
    segmentos, info = modelo.transcribe(
        audio,
        language="es",
        word_timestamps=True,
        vad_filter=True,
        initial_prompt=prompt,
        condition_on_previous_text=False,
    )
    palabras: List[Tuple[str, float]] = []
    for seg in segmentos:
        for w in seg.words or []:
            for parte in normalizar(w.word).split():
                palabras.append((parte, float(w.start)))
        if progreso and info.duration:
            progreso(min(1.0, seg.end / info.duration))
    return palabras
