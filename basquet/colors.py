"""Utilidades de color: parseo de nombres/hex, contraste y ajuste de tono."""

import re
from typing import Any

def _parse_color(c: Any, fallback: str) -> str:
    try:
        s = str(c).strip()
        if not s:
            return fallback
        # Mapeo de nombres en español a HEX
        name = s.lower().replace('ó','o').replace('á','a').replace('é','e').replace('í','i').replace('ú','u').strip()
        es_map = {
            'negro': '#000000',
            'blanco': '#ffffff',
            'rojo': '#e53935',
            'azul': '#1e88e5',
            'verde': '#43a047',
            'amarillo': '#fdd835',
            'naranja': '#fb8c00',
            'violeta': '#8e24aa',
            'morado': '#6a1b9a',
            'gris': '#9e9e9e',
            'gris claro': '#cfd8dc',
            'gris oscuro': '#616161',
            'celeste': '#03a9f4',
            'cian': '#00bcd4',
            'turquesa': '#26a69a',
            'bordo': '#7b1fa2',
            'granate': '#800000',
            'marron': '#6d4c41',
            'cafe': '#6d4c41',
            'rosa': '#ec407a',
            'magenta': '#d81b60',
            'lima': '#c0ca33',
            'oliva': '#827717',
            'dorado': '#b8860b',
            'plateado': '#b0bec5',
        }
        if name in es_map:
            return es_map[name]
        # Aceptar hex sin '#'
        if re.fullmatch(r"[0-9A-Fa-f]{6}", s):
            return f"#{s}"
        # Aceptar hex con '#'
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", s):
            return s
        # Cualquier otro formato no-hex: forzar fallback para asegurar contraste visible
        return fallback
    except Exception:
        return fallback


def _text_color_for_bg(hex_color: str) -> str:
    # Contraste WCAG aproximado para elegir blanco/negro
    try:
        c = hex_color.lstrip('#')
        if len(c) != 6:
            return "#ffffff"
        r = int(c[0:2], 16) / 255.0
        g = int(c[2:4], 16) / 255.0
        b = int(c[4:6], 16) / 255.0
        # luminancia relativa
        def lin(u: float) -> float:
            return u / 12.92 if u <= 0.03928 else ((u + 0.055) / 1.055) ** 2.4
        L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
        return "#000000" if (L > 0.55) else "#ffffff"
    except Exception:
        return "#ffffff"


def _clamp(v: int) -> int:
    return max(0, min(255, v))


def _adjust_color(hex_color: str, factor: float) -> str:
    """Aclarar u oscurecer un color HEX. factor>1 aclara, <1 oscurece."""
    try:
        c = hex_color.lstrip('#')
        if len(c) != 6:
            return hex_color
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)
        r = _clamp(int(r * factor))
        g = _clamp(int(g * factor))
        b = _clamp(int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color
