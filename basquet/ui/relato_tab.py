"""Pestaña 'Relato': play-by-play propio a partir de un audio por período.

El relator graba un audio por período con frases cortas de gramática fija.
El audio se transcribe con Whisper y el texto queda editable (una línea por
evento, con su segundo dentro del audio): corregir el texto recalcula los
eventos y las estadísticas al instante. También se puede escribir o pegar el
texto sin subir audio.
"""

from typing import Dict

import pandas as pd
import streamlit as st

from ..relato.estadisticas import box_score, tiros_por
from ..relato.parser import COLUMNAS_EVENTOS, LOCAL, VISITANTE, lineas_desde_palabras, parsear_relato
from ..relato.transcripcion import MODELOS, cargar_modelo, transcribir, whisper_disponible
from ..relato.vocabulario import MARCA_DESCRIPCION, MARCAS, TIPOS_TIRO, ZONAS, prompt_inicial
from ..utils import _first_of

FORMATOS_AUDIO = ["mp3", "m4a", "wav", "ogg", "opus", "webm", "aac", "flac"]


@st.cache_resource(show_spinner=False)
def _modelo(nombre: str):
    return cargar_modelo(nombre)


def _norm_dorsal(v) -> str:
    s = "" if v is None or pd.isna(v) else str(v).strip()
    return str(int(float(s))) if s.replace(".", "", 1).isdigit() else s


def _plantel_inicial(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "dorsal" not in df.columns:
        return pd.DataFrame({"dorsal": pd.Series(dtype=str), "nombre": pd.Series(dtype=str)})
    d = df[["dorsal"] + (["nombre"] if "nombre" in df.columns else [])].copy()
    d["dorsal"] = d["dorsal"].map(_norm_dorsal)
    if "nombre" not in d.columns:
        d["nombre"] = ""
    d["nombre"] = d["nombre"].astype(str).str.strip()
    return d[d["dorsal"] != ""].drop_duplicates("dorsal").reset_index(drop=True)


def _a_dict(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for _, r in df.iterrows():
        dorsal = _norm_dorsal(r.get("dorsal"))
        if dorsal:
            nombre = r.get("nombre")
            out[dorsal] = "" if nombre is None or pd.isna(nombre) else str(nombre).strip()
    return out


def _guia() -> None:
    with st.expander("📖 Cómo relatar", expanded=False):
        st.markdown(
            "Una frase corta por evento, **siempre en el mismo orden**. Los números son dorsales. "
            "Si el mismo dorsal existe en los dos equipos y hay dudas, anteponé `local` o `visita`."
        )
        st.markdown(
            "| Evento | Qué decir | Ejemplo |\n|---|---|---|\n"
            "| Tiro | tirador · zona · [defensor · marca] · [tipo] · resultado · [asiste N] | "
            "*siete, esquina izquierda, doce encima, adentro, asiste cuatro* |\n"
            "| Tiro tapado | … · tapa N | *cinco, bandeja, tapa doce* |\n"
            "| Libres | libres · tirador · resultado de cada uno | *libres siete, adentro, afuera* |\n"
            "| Rebote | rebote · N (ofensivo/defensivo se deduce) | *rebote doce* |\n"
            "| Pérdida / robo | pérdida · N · [robo · M] | *pérdida ocho, robo nueve* |\n"
            "| Tapa suelta | tapa · N | *tapa doce* |\n"
            "| Falta | falta · N · [sobre · M] | *falta doce sobre siete* |\n"
            "| Cambio | entra · N · sale · M | *entra veintiuno, sale cuatro* |\n"
            "| Tiempo muerto | tiempo muerto · local/visita | *tiempo muerto visita* |\n"
            "| Corregir | borrar (anula la frase anterior) | *…adentro. borrar* |"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Zonas de 2**")
            st.markdown("\n".join(f"- {a[0]}" for z, (v, a) in ZONAS.items() if v == 2))
        with c2:
            st.markdown("**Zonas de 3**")
            st.markdown("\n".join(f"- {a[0]}" for z, (v, a) in ZONAS.items() if v == 3))
        with c3:
            st.markdown("**Marca (defensor más cercano)**")
            st.markdown("\n".join(f"- {MARCAS[m][0]}: {d}" for m, d in MARCA_DESCRIPCION.items()))
            st.markdown("**Tipo de tiro (opcional)**")
            st.markdown(", ".join(a[0] for a in TIPOS_TIRO.values()))
        st.caption(
            "El 2P/3P sale de la zona. Si en el tiro no se nombra la zona pero sí bandeja, volcada o "
            "palmeo, se asume aro. El tiempo de posesión se mide en el audio desde el rebote defensivo, "
            "robo, pérdida o canasta anterior hasta el tiro, así que solo es válido si se relata en vivo."
        )


def _estilo_revisar(df: pd.DataFrame):
    def fila(r):
        color = "background-color: rgba(230, 80, 60, 0.18)" if r.get("revisar") else ""
        return [color] * len(r)
    return df.style.apply(fila, axis=1)


def render_relato(tablas: Dict[str, pd.DataFrame]) -> None:
    part_df = tablas.get("partido", pd.DataFrame())
    row = part_df.iloc[0] if not part_df.empty else {}
    nombres = {
        LOCAL: str(_first_of(row, ["local"], "Local")),
        VISITANTE: str(_first_of(row, ["visitante"], "Visitante")),
    }

    st.caption(
        "Play-by-play propio a partir de un audio por período, independiente del play-by-play oficial. "
        "Se puede subir el audio y transcribirlo, o escribir/pegar el relato directamente."
    )
    _guia()

    # Plantel
    with st.expander("👥 Plantel (dorsal → jugador)", expanded=False):
        st.caption("Se toma del partido cargado. Se puede corregir o completar acá.")
        c1, c2 = st.columns(2)
        planteles = {}
        for col, cond, key in ((c1, LOCAL, "estadisticas_equipolocal"), (c2, VISITANTE, "estadisticas_equipovisitante")):
            with col:
                st.markdown(f"**{nombres[cond]}**")
                editado = st.data_editor(
                    _plantel_inicial(tablas.get(key, pd.DataFrame())),
                    num_rows="dynamic", use_container_width=True, hide_index=True,
                    key=f"relato_plantel_{cond}",
                )
                planteles[cond] = _a_dict(editado)

    # Transcripción
    hay_whisper = whisper_disponible()
    if hay_whisper:
        modelo_nombre = st.selectbox(
            "Modelo de transcripción", MODELOS, index=0, key="relato_modelo",
            help="small: buen equilibrio. base: más rápido, más errores. medium: más preciso y bastante más lento.",
        )
    else:
        modelo_nombre = None
        st.warning("No está instalado `faster-whisper`: se puede escribir el relato pero no transcribir audio.")

    n_periodos = st.number_input("Cantidad de períodos", min_value=1, max_value=8, value=4, step=1, key="relato_n_periodos")
    etiquetas = [f"Período {p}" if p <= 4 else f"Alargue {p - 4}" for p in range(1, int(n_periodos) + 1)]
    sub = st.tabs(etiquetas)
    dorsales = sorted(set(planteles[LOCAL]) | set(planteles[VISITANTE]))
    eventos_periodos = []

    for p, tab in enumerate(sub, start=1):
        with tab:
            key_txt = f"relato_texto_{p}"
            audio = st.file_uploader(f"Audio del {etiquetas[p - 1].lower()}", type=FORMATOS_AUDIO, key=f"relato_audio_{p}")
            if audio is not None:
                st.audio(audio)
                if hay_whisper and st.button("Transcribir", key=f"relato_btn_{p}", type="primary"):
                    barra = st.progress(0.0, text="Cargando modelo…")
                    try:
                        modelo = _modelo(modelo_nombre)
                        barra.progress(0.0, text="Transcribiendo…")
                        audio.seek(0)
                        palabras = transcribir(
                            modelo, audio, prompt=prompt_inicial(dorsales),
                            progreso=lambda f: barra.progress(f, text=f"Transcribiendo… {f:.0%}"),
                        )
                        st.session_state[key_txt] = lineas_desde_palabras(palabras)
                        barra.empty()
                    except Exception as e:
                        barra.empty()
                        st.error(f"No se pudo transcribir el audio: {e}")

            texto = st.text_area(
                "Relato (una línea por evento: [mm:ss] frase). Corregí acá y se recalcula todo.",
                key=key_txt, height=260,
                placeholder="[00:05.0] siete esquina izquierda doce encima adentro asiste cuatro\n[00:21.3] rebote doce",
            )
            ev = parsear_relato(texto or "", planteles, p)
            eventos_periodos.append(ev)
            if ev.empty:
                continue
            n_rev = int((ev.drop_duplicates("frase")["revisar"] != "").sum())
            st.caption(f"{ev['frase'].nunique()} frases → {len(ev)} eventos. "
                       + (f"⚠️ {n_rev} frases a revisar (en rojo)." if n_rev else "Sin avisos."))
            ver = ev.drop(columns=["numero_periodo", "t_audio"]).copy()
            ver["Condicion"] = ver["Condicion"].map(nombres).fillna("")
            st.dataframe(_estilo_revisar(ver), use_container_width=True, hide_index=True)

    eventos = pd.concat([e for e in eventos_periodos if not e.empty], ignore_index=True) \
        if any(not e.empty for e in eventos_periodos) else pd.DataFrame(columns=COLUMNAS_EVENTOS)
    st.session_state["relato_eventos"] = eventos
    if eventos.empty:
        st.info("Subí y transcribí un audio (o escribí el relato) para ver las estadísticas.")
        return

    st.divider()
    st.subheader("Estadísticas del relato")
    for cond in (LOCAL, VISITANTE):
        st.markdown(f"**{nombres[cond]}**")
        bs = box_score(eventos, cond)
        if bs.empty:
            st.caption("Sin eventos.")
        else:
            st.dataframe(bs, use_container_width=True, hide_index=True)

    st.subheader("Tiros de campo")
    agrupar = st.radio(
        "Agrupar por", ["zona", "marca", "tipo_tiro", "bucket_posesion"], horizontal=True, key="relato_agrupar",
        format_func={"zona": "Zona", "marca": "Marca del defensor", "tipo_tiro": "Tipo de tiro",
                     "bucket_posesion": "Tiempo de posesión"}.get,
    )
    c1, c2 = st.columns(2)
    for col, cond in ((c1, LOCAL), (c2, VISITANTE)):
        with col:
            st.markdown(f"**{nombres[cond]}**")
            t = tiros_por(eventos, cond, agrupar)
            if t.empty:
                st.caption("Sin tiros.")
            else:
                st.dataframe(t, use_container_width=True, hide_index=True)

    salida = eventos.copy()
    salida["equipo"] = salida["Condicion"].map(nombres)
    st.download_button(
        "⬇️ Descargar play-by-play del relato (CSV)",
        salida.to_csv(index=False).encode("utf-8-sig"),
        file_name="relato_pbp.csv", mime="text/csv", key="relato_csv",
    )
