"""Constantes de configuración: URLs y headers para la API de estadísticas."""

PARTIDO_URL = "https://appaficioncabb.indalweb.net/envivonavegador/partido.ashx"
ESTADISTICAS_URL = "https://appaficioncabb.indalweb.net/envivonavegador/estadisticas.ashx"

DEFAULT_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://appaficioncabb.indalweb.net",
    "Referer": "https://appaficioncabb.indalweb.net/envivonavegador/",
    "X-Requested-With": "XMLHttpRequest",
}
