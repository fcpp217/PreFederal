"""Cliente HTTP para descargar datos de partido y estadísticas desde la API."""

import time
from typing import Any, Dict, Optional, Tuple

import requests

from .config import DEFAULT_REQUEST_HEADERS, ESTADISTICAS_URL, PARTIDO_URL


def _get_requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(DEFAULT_REQUEST_HEADERS)
    return s


def _post_with_retries(url: str, data: Dict[str, Any], timeout: Tuple[int, int] = (10, 120), retries: int = 4) -> requests.Response:
    last_exc: Optional[Exception] = None
    s = _get_requests_session()
    for attempt in range(max(1, retries)):
        try:
            return s.post(url, data=data, timeout=timeout)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            time.sleep(min(2 ** attempt, 8))
    if last_exc is not None:
        raise last_exc
    return s.post(url, data=data, timeout=timeout)


def fetch_partido(partido_id: str) -> Dict[str, Any]:
    resp = _post_with_retries(PARTIDO_URL, data={"id_partido": str(partido_id)})
    resp.raise_for_status()
    return resp.json()


def fetch_estadisticas(partido_id: str) -> Dict[str, Any]:
    resp = _post_with_retries(ESTADISTICAS_URL, data={"id_partido": str(partido_id)})
    resp.raise_for_status()
    return resp.json()
