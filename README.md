# Baskesta (Streamlit)

Aplicación Streamlit para descargar y transformar datos de un partido (sin base de datos), basada en la lógica de `sqlite_loader/download_to_sqlite.py`.

## Requisitos

- Python 3.9+
- Pip

## Instalación

Se recomienda usar un entorno virtual, pero no es obligatorio.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar

Desde la carpeta del proyecto `Baskesta/`:

```bash
streamlit run app.py
```

Luego, abrir el enlace local que muestre Streamlit (por ejemplo, http://localhost:8501).

## Uso

1. Ingrese el ID numérico del partido en la barra lateral.
2. Presione "Descargar y procesar".
3. La app descargará los datos del partido y las estadísticas, ejecutará las transformaciones en memoria (sin escribir en SQLite) y mostrará las tablas:
   - `partido`
   - `jugada`
   - `estadistica`
   - `estadisticas_equipolocal`
   - `estadisticas_equipovisitante`
   - `pbp` (play-by-play procesado)
   - `jugadoresAgregado`
   - `qunitetosAgregado`

## Notas

- No se guarda nada en disco (no se usa base de datos). Todo se procesa en memoria.
- Si el servicio de origen cambia su formato o campos, pueden aparecer columnas faltantes; en ese caso, abrir un issue o ajustar el código en `app.py`.
