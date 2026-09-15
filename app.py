import streamlit as st

from basquet.data_processing import descargar_y_transformar
from basquet.ui.estadisticas_tab import render_estadisticas
from basquet.ui.posesion_tab import render_posesion
from basquet.ui.quintetos_tab import render_quintetos
from basquet.ui.resumen_tab import render_resumen

# ------------------------------
# Config UI
# ------------------------------
st.set_page_config(page_title="Estadística", layout="wide")

# ------------------------------
# UI
# ------------------------------
st.title("Estadísticas Basquet")

# Entrada en el cuerpo principal, centrada y angosta
st.subheader("Entrada")
wrap = st.container()
colL, colMid, colR = wrap.columns([2, 1, 2])
with colMid:
    partido_id_input = st.text_input("ID de partido", value="", placeholder="Ej: 123456", max_chars=6)
    ejecutar = st.button("Buscar partido", type="primary")

if (ejecutar or ('tablas' in st.session_state)):
    if ejecutar:
        if not partido_id_input.strip().isdigit():
            st.error("Ingrese un ID numérico válido.")
        else:
            try:
                # Limpiar cache de Streamlit para asegurar datos frescos
                st.cache_data.clear()

                # Resetear filtros de las 3 pestañas al buscar un nuevo partido
                for k in ['res_sel_per', 'estad_sel_per', 'estad_sel_situ', 'estad_sel_u2m', 'q_sel_per', 'q_sel_situ', 'q_sel_u2m', 'pos_sel_per']:
                    try:
                        if k in st.session_state:
                            del st.session_state[k]
                    except Exception:
                        pass

                # Limpiar datos anteriores antes de buscar nuevos
                if 'tablas' in st.session_state:
                    del st.session_state['tablas']

                with st.spinner("Descargando y procesando datos..."):
                    tablas = descargar_y_transformar(partido_id_input.strip())
                # Persistir en sesión para evitar re-descarga al cambiar filtros
                st.session_state['tablas'] = tablas
                # Forzar reinicio visual de filtros a 'TODOS'
                st.session_state['res_sel_per'] = 'TODOS'
                st.session_state['estad_sel_per'] = 'TODOS'
                st.session_state['estad_sel_situ'] = 'TODOS'
                st.session_state['estad_sel_u2m'] = 'TODOS'
                st.session_state['q_sel_per'] = 'TODOS'
                st.session_state['q_sel_situ'] = 'TODOS'
                st.session_state['q_sel_u2m'] = 'TODOS'
                st.session_state['pos_sel_per'] = 'TODOS'
            except Exception as e:
                # Mostrar error y detalle para diagnóstico
                st.error("ID de Partido no encontrado")
                tablas = None
    else:
        tablas = st.session_state.get('tablas')

    if tablas is not None:
        # Mantener orden fijo pero mostrando primero 'Resumen' al abrir la app
        nombres = ["Resumen", "Estadisticas por jugador", "Estadistica por Quintetos", "Posesión"]
        # Mostrar pestañas
        tabs = st.tabs(nombres)
        # Referencias por nombre
        t_resumen = tabs[0]
        t_estadistica = tabs[1]
        t_quintetos = tabs[2]
        t_posesion = tabs[3]

        with t_resumen:
            render_resumen(tablas)

        with t_estadistica:
            render_estadisticas(tablas)

        with t_quintetos:
            render_quintetos(tablas)

        with t_posesion:
            render_posesion(tablas)
else:
    st.info("Ingrese un ID de partido, presione 'Descargar y procesar' o use los datos ya descargados previamente.")
