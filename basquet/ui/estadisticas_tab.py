"""Pestaña 'Estadisticas por jugador': box score de cada equipo con derivadas."""

import re
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from ..colors import _parse_color, _text_color_for_bg
from ..pdf_export import render_pdf_button
from ..tables import _build_table, build_column_config
from ..utils import _first_col, _first_of


def render_estadisticas(tablas: Dict[str, pd.DataFrame]) -> None:
        # Botón de descarga PDF
        render_pdf_button(tablas, key='estadistica')
    
        part_df = tablas.get('partido', pd.DataFrame())
        local_title = str(part_df.iloc[0].get('local')) if not part_df.empty else 'Local'
        visitante_title = str(part_df.iloc[0].get('visitante')) if not part_df.empty else 'Visitante'
        jg = tablas.get('jugadoresAgregado', pd.DataFrame()).copy()
        # Planillas por equipo para detectar titulares
        plan_loc = tablas.get('estadisticas_equipolocal', pd.DataFrame()).copy()
        plan_vis = tablas.get('estadisticas_equipovisitante', pd.DataFrame()).copy()
        def titular_sets(planilla_df: pd.DataFrame):
            if planilla_df is None or planilla_df.empty:
                return set()
            dcol = _first_col(planilla_df, ['dorsal','numero','nro','número','numero_camiseta','n_camisa'])
            tcol = _first_col(planilla_df, ['quintetotitular','quinteto_titular','QuintetoTitular','quintetoTitular','titular','es_titular'])
            if not dcol or dcol not in planilla_df.columns or not tcol or tcol not in planilla_df.columns:
                return set()
            dfp2 = planilla_df[[dcol, tcol]].copy()
            def is_true(v):
                try:
                    if isinstance(v, (int, float)):
                        return float(v) != 0.0
                    s = str(v).strip().lower()
                    return s in ('si','sí','true','t','1','x','s')
                except Exception:
                    return False
            dfp2['__titular'] = dfp2[tcol].apply(is_true)
            tit = set(pd.to_numeric(dfp2.loc[dfp2['__titular'], dcol], errors='coerce').dropna().astype(int).tolist())
            return tit
        tit_set_loc = titular_sets(plan_loc)
        tit_set_vis = titular_sets(plan_vis)
        if not jg.empty:
            # Ya no necesitamos arreglar nombres "NOMBRE" aquí, ya están formateados correctamente
            # en la generación de jugadoresAgregado
            # Normalizar y detectar columnas
            if 'Condicion' in jg.columns:
                jg['Condicion'] = jg['Condicion'].astype(str).str.upper().fillna('')
            if 'equipo' in jg.columns:
                jg['equipo'] = jg['equipo'].astype(str).str.strip()
    
            period_col = _first_col(jg, ['numero_periodo', 'periodo', 'Periodo'])
            situ_col = _first_col(jg, ['SituacionMarcador', 'Situacion marcador', 'situacion_marcador', 'situacionMarcador', 'situacion', 'Situacion'])
            u2m_col = _first_col(jg, ['ultimos_dos_minutos', 'ultimos2min', 'ultimos_dos', 'u2m'])
    
            # Filtros en formulario para evitar cambio de pestaña al aplicar
            st.markdown("### Filtros")
            with st.form(key='estadisticas_filters'):
                cols_f = st.columns(3)
                with cols_f[0]:
                    if period_col and period_col in jg.columns:
                        per_opts = ['TODOS'] + sorted(pd.to_numeric(jg[period_col], errors='coerce').dropna().astype(int).unique().tolist())
                        sel_per = st.selectbox('Periodo', per_opts, index=0, key='estad_sel_per')
                    else:
                        sel_per = 'TODOS'
                with cols_f[1]:
                    if situ_col and situ_col in jg.columns:
                        situ_vals = jg[situ_col].astype(str).fillna('').unique().tolist()
                        situ_opts = ['TODOS'] + sorted([s for s in situ_vals if s != ''])
                        sel_situ = st.selectbox('Situacion', situ_opts, index=0, key='estad_sel_situ')
                    else:
                        sel_situ = 'TODOS'
                with cols_f[2]:
                    if u2m_col and u2m_col in jg.columns:
                        u2m_vals = jg[u2m_col].astype(str).fillna('').unique().tolist()
                        u2m_opts = ['TODOS'] + sorted([s for s in u2m_vals if s != ''])
                        sel_u2m = st.selectbox('Últimos 2 min', u2m_opts, index=0, key='estad_sel_u2m')
                    else:
                        sel_u2m = 'TODOS'
                apply_filters = st.form_submit_button('Aplicar filtros')
    
            jg_f = jg.copy()
            if apply_filters:
                if sel_per != 'TODOS' and period_col and period_col in jg_f.columns:
                    try:
                        jg_f = jg_f[pd.to_numeric(jg_f[period_col], errors='coerce') == int(sel_per)]
                    except Exception:
                        pass
                if sel_situ != 'TODOS' and situ_col and situ_col in jg_f.columns:
                    jg_f = jg_f[jg_f[situ_col].astype(str) == str(sel_situ)]
                if sel_u2m != 'TODOS' and u2m_col and u2m_col in jg_f.columns:
                    jg_f = jg_f[jg_f[u2m_col].astype(str) == str(sel_u2m)]
    
            # Top-5 jugadores por minutos en últimos 2 min (Q4+) usando quintetosAgregado
            def compute_top5_u2m_names(cond: str, u2m_enabled: bool) -> set:
                if not u2m_enabled:
                    return set()
                qg_all = tablas.get('quintetosAgregado', pd.DataFrame()).copy()
                if qg_all is None or qg_all.empty:
                    return set()
                qg_all['Condicion'] = qg_all.get('Condicion','').astype(str).str.upper().fillna('')
                base = qg_all[qg_all['Condicion']==cond.upper()].copy()
                if base.empty:
                    return set()
                # Normalizar columnas
                per_col = _first_col(base, ['numero_periodo','periodo','Periodo']) or 'numero_periodo'
                u2m_col = _first_col(base, ['ultimos_dos_minutos','ultimos2min','ultimos_dos','u2m','Ultimos 2 min','Últimos dos min']) or 'ultimos_dos_minutos'
                base['__per'] = pd.to_numeric(base.get(per_col, 0), errors='coerce').fillna(0)
                def _norm_txt(x: Any) -> str:
                    s = str(x)
                    s = s.replace('Ú','U').replace('ú','u').replace('ó','o').replace('í','i').replace('á','a').replace('é','e')
                    return s.strip().lower()
                base['__u2m'] = base.get(u2m_col, '').apply(_norm_txt).isin(['ultimos dos min','u2m','true','1','si','sí','s'])
                # Parse tiempo_jugado en formato mm:ss a segundos
                def parse_secs(v: Any) -> float:
                    try:
                        s = str(v)
                        if ':' in s:
                            mm, ss = s.split(':', 1)
                            return float(int(mm) * 60 + int(ss))
                        return float(pd.to_numeric(v, errors='coerce'))
                    except Exception:
                        return 0.0
                base['__t'] = base.get('tiempo_jugado', 0).apply(parse_secs)
                filt = base[(base['__per']>=4) & (base['__u2m'])]
                if filt.empty:
                    return set()
                # Acumular minutos por jugador a partir del quinteto
                acc = {}
                for _, r in filt.iterrows():
                    t = float(r['__t'])
                    q = r.get('quinteto')
                    members = []
                    if isinstance(q, (list, tuple)):
                        members = [str(x).strip() for x in q]
                    else:
                        members = [s.strip() for s in re.split(r"\s*/\s*", str(q)) if s.strip()]
                    for m in members:
                        acc[m] = acc.get(m, 0.0) + t
                # Top-5 por tiempo
                top5 = sorted(acc.items(), key=lambda x: x[1], reverse=True)[:5]
                return set([str(k) for k,_ in top5])
    
            # Usar el filtro de Quintetos para decidir si marcamos 🔥
            def _norm_txt2(x: Any) -> str:
                s = str(x)
                return s.replace('Ú','U').replace('ú','u').replace('ó','o').replace('í','i').replace('á','a').replace('é','e').strip().lower()
            u2m_enabled_flag = False
            try:
                u2m_enabled_flag = (_norm_txt2(st.session_state.get('q_sel_u2m', '')) == 'ultimos dos min') or (_norm_txt2(locals().get('sel_u2m_q', '')) == 'ultimos dos min')
            except Exception:
                u2m_enabled_flag = False
            top5_u2m_loc = compute_top5_u2m_names('LOCAL', u2m_enabled_flag)
            top5_u2m_vis = compute_top5_u2m_names('VISITANTE', u2m_enabled_flag)
    
            def make_table(df_src: pd.DataFrame, condicion: str) -> pd.DataFrame:
                df_t = df_src[df_src.get('Condicion', '').astype(str).str.upper() == condicion.upper()].copy()
                if df_t.empty:
                    return pd.DataFrame()
                name_col = _first_col(df_t, ['nombre', 'jugador', 'nombre_jugador']) or 'nombre'
                # Formatear el nombre que se mostrará en la tabla como "DD-Nombre" si hay dorsal (sin espacios)
                dorsal_candidates = ['dorsal','numero','nro','número','numero_camiseta','n_camisa']
                dcol = _first_col(df_t, dorsal_candidates)
                def two_digit(v: Any) -> str:
                    try:
                        iv = int(pd.to_numeric(v, errors='coerce'))
                        return f"{iv:02d}"
                    except Exception:
                        return ''
                def fmt_name_row(r: pd.Series) -> str:
                    nm = str(r.get(name_col, '')).strip()
                    if dcol and dcol in r.index:
                        dd = two_digit(r.get(dcol))
                        return f"{dd}-{nm}" if dd else nm
                    return nm
                df_t[name_col] = df_t.apply(fmt_name_row, axis=1)
    
                # Armar set de nombres titulares desde la planilla con el mismo formato "DD-Nombre" (sin espacios)
                plan_df = plan_loc if condicion.upper()=='LOCAL' else plan_vis
                tit_name_set = set()
                try:
                    if plan_df is not None and not plan_df.empty:
                        plan_dcol = _first_col(plan_df, dorsal_candidates)
                        name_cands = ['Nombre','nombre','Jugador','jugador','nombre_jugador','NombreJugador']
                        plan_name_col = _first_col(plan_df, name_cands)
                        tit_cands = ['quintetotitular','quinteto_titular','QuintetoTitular','quintetoTitular','titular','es_titular']
                        plan_tit_col = _first_col(plan_df, tit_cands)
                        if plan_dcol and plan_name_col and plan_tit_col and \
                           plan_dcol in plan_df.columns and plan_name_col in plan_df.columns and plan_tit_col in plan_df.columns:
                            tmp = plan_df[[plan_dcol, plan_name_col, plan_tit_col]].copy()
                            def truthy(v: Any) -> bool:
                                try:
                                    if isinstance(v, (int, float)):
                                        return float(v) != 0.0
                                    s = str(v).strip().lower()
                                    return s in ('si','sí','true','t','1','x','s')
                                except Exception:
                                    return False
                            tmp['__tit'] = tmp[plan_tit_col].apply(truthy)
                            tmp['__dd'] = tmp[plan_dcol].apply(two_digit)
                            tmp['__nn'] = tmp[plan_name_col].astype(str).str.strip()
                            tmp['__disp'] = np.where(tmp['__dd']!='', tmp['__dd'] + '-' + tmp['__nn'], tmp['__nn'])
                            tit_name_set = set(tmp.loc[tmp['__tit']==True, '__disp'].astype(str).str.lower().tolist())
                except Exception:
                    tit_name_set = set()
    
                # Mapeo estricto de la tabla provista (izquierda -> derecha)
                strict_map = [
                    ('ASISTENCIA', 'asistencias'),
                    ('CANASTA-1P', 'canasta1p'),
                    ('CANASTA-2P', 'canasta2p'),
                    ('CANASTA-3P', 'canasta3p'),
                    ('FALTA-COMETIDA', 'faltascometidas'),
                    ('FALTA-RECIBIDA', 'faltasrecibidas'),
                    ('PERDIDA', 'perdidas'),
                    ('REBOTE-DEFENSIVO', 'rebotedefensivo'),
                    ('REBOTE-OFENSIVO', 'reboteofensivo'),
                    ('RECUPERACION', 'recuperaciones'),
                    ('TIRO1-FALLADO', 'tiro1fallado'),
                    ('TIRO2-FALLADO', 'tiro2fallado'),
                    ('TIRO3-FALLADO', 'tiro3fallado'),
                    ('tiempo_jugado', 'tiempo_jugado'),
                ]
                # Construir columnas destino sumando desde la fuente estricta
                for src, dst in strict_map:
                    if src in df_t.columns:
                        df_t[dst] = pd.to_numeric(df_t[src], errors='coerce').fillna(0)
                    else:
                        # si no existe en la fuente, crear en 0
                        df_t[dst] = 0
    
                cols_sum = [dst for _, dst in strict_map]
                # Agregar columna de plus-minus desde candidatos
                pm_cands = ['diferencia', 'plusminus', 'plus_minus', '+-']
                pm_series = None
                for pmc in pm_cands:
                    if pmc in df_t.columns:
                        s = pd.to_numeric(df_t[pmc], errors='coerce').fillna(0)
                        pm_series = s if pm_series is None else (pm_series + s)
                if pm_series is None:
                    df_t['+-'] = 0
                else:
                    df_t['+-'] = pm_series
                cols_sum = cols_sum + ['+-']
                # Agrupar por jugador
                agg = df_t.groupby(name_col, as_index=False)[cols_sum].sum()
                # Columna Titular (estrella y 🔥 si aplica) por nombre (comparación exacta con el formato "DD-Nombre")
                try:
                    def _norm_name(s: Any) -> str:
                        s = re.sub(r"\s+", " ", str(s).strip())
                        return s.lower()
                    tit_name_norm = set((_norm_name(x) for x in tit_name_set)) if tit_name_set else set()
                    # set de fuego (top-5 u2m) según condición
                    top_u2m = top5_u2m_loc if condicion.upper()=='LOCAL' else top5_u2m_vis
                    top_u2m_norm = set((_norm_name(x) for x in top_u2m)) if top_u2m else set()
                    def is_tit(nm: Any) -> str:
                        s = str(nm).strip()
                        if s == 'Totales':
                            return ''
                        by_name = (_norm_name(s) in tit_name_norm) if tit_name_norm else False
                        has_fire = (_norm_name(s) in top_u2m_norm) if top_u2m_norm else False
                        return ('⭐' if by_name else '') + ('🔥' if has_fire else '')
                    agg['Titular'] = agg[name_col].map(is_tit)
                except Exception:
                    agg['Titular'] = ''
                # Derivadas simples
                agg['rebotetotal'] = agg['rebotedefensivo'] + agg['reboteofensivo']
                agg['puntos'] = agg['canasta1p'] + 2*agg['canasta2p'] + 3*agg['canasta3p']
    
                # Intentos: sumar directamente desde columnas fuente por jugador, sin crear columnas intermedias
                idx = agg[name_col]
                def sum_by(col: str):
                    if col in df_t.columns:
                        return df_t.groupby(name_col)[col].sum().reindex(idx).fillna(0).values
                    else:
                        return np.zeros(len(idx))
                conv1 = sum_by('CANASTA-1P')
                fall1 = sum_by('TIRO1-FALLADO')
                conv2 = sum_by('CANASTA-2P')
                fall2 = sum_by('TIRO2-FALLADO')
                conv3 = sum_by('CANASTA-3P')
                fall3 = sum_by('TIRO3-FALLADO')
    
                agg['tiro1p'] = conv1 + fall1
                agg['tiro2p'] = conv2 + fall2
                agg['tiro3p'] = conv3 + fall3
                # Fila Totales
                totals = agg.select_dtypes(include=[np.number]).sum(numeric_only=True)
                total_row = {col: '' for col in agg.columns}
                total_row[name_col] = 'Totales'
                for c in totals.index:
                    total_row[c] = totals[c]
                # Sin total para '+-'
                if '+-' in total_row:
                    total_row['+-'] = ''
                agg = pd.concat([agg, pd.DataFrame([total_row])], ignore_index=True)
                # Asegurar que 'Titular' quede vacío en la fila Totales
                if 'Titular' in agg.columns:
                    agg.loc[agg[name_col]=='Totales','Titular'] = ''
                # Pasar por _build_table para formato (conv/att y %)
                out = _build_table(agg)
                # Reordenar para que 'Titular' sea la 2da columna (sin título)
                name_display = None
                for cand in ['Nombre', 'nombre', 'Jugador', name_col]:
                    if cand in out.columns:
                        name_display = cand
                        break
                if name_display and 'Titular' in out.columns:
                    cols = out.columns.tolist()
                    # remover y reinsertar
                    cols.remove('Titular')
                    insert_idx = cols.index(name_display) + 1 if name_display in cols else 1
                    cols.insert(insert_idx, 'Titular')
                    out = out[cols]
                return out
    
            tbl_loc = make_table(jg_f, 'LOCAL')
            tbl_vis = make_table(jg_f, 'VISITANTE')
    
            # Usar mismos colores que en Resumen
            row = part_df.iloc[0] if not part_df.empty else {}
            color_local_raw = _first_of(row, ['color_local', 'local_color', 'colorLocal', 'colorlocal'], '#1f77b4')
            color_visitante_raw = _first_of(row, ['color_visitante', 'visitante_color', 'colorVisitante', 'colorvisitante'], '#ff7f0e')
            color_local = _parse_color(color_local_raw, '#1f77b4')
            color_visitante = _parse_color(color_visitante_raw, '#ff7f0e')
            tc_local = _text_color_for_bg(color_local)
            tc_visitante = _text_color_for_bg(color_visitante)
    
            # Títulos con colores de equipo
            st.markdown(f"""
            <div style='background:{color_local}; color:{tc_local}; padding:12px; border-radius:8px; text-align:center; font-weight:700; margin:16px 0 8px;'>
                🏀 LOCAL - {local_title}
            </div>
            """, unsafe_allow_html=True)
            # Colorear estrella con color del equipo
            def style_star(df: pd.DataFrame, color_hex: str):
                if isinstance(df, pd.DataFrame) and 'Titular' in df.columns:
                    styler = df.style
                    fn = lambda v: f'color: {color_hex}' if str(v) == '⭐' else ''
                    if hasattr(styler, 'map'):
                        return styler.map(fn, subset=['Titular'])
                    if hasattr(styler, 'applymap'):
                        return styler.applymap(fn, subset=['Titular'])
                    return styler
                return df
            # (Eliminado post-proceso de fueguito en jugadores)
            styled_loc = style_star(tbl_loc, color_local)
            st.dataframe(styled_loc, use_container_width=True, hide_index=True, column_config=build_column_config(tbl_loc))
            # Leyenda solo para estrella
            st.caption("⭐ Quinteto titular")
            # (Timeline LOCAL movido a pestaña Quintetos)
    
            st.markdown(f"""
            <div style='background:{color_visitante}; color:{tc_visitante}; padding:12px; border-radius:8px; text-align:center; font-weight:700; margin:16px 0 8px;'>
                🏀 VISITANTE - {visitante_title}
            </div>
            """, unsafe_allow_html=True)
            styled_vis = style_star(tbl_vis, color_visitante)
            st.dataframe(styled_vis, use_container_width=True, hide_index=True, column_config=build_column_config(tbl_vis))
            st.caption("⭐ Quinteto titular")
            # (Timeline VISITANTE movido a pestaña Quintetos)
        else:
            st.info('No hay jugadoresAgregado para generar estadística')
    
    # Pestaña Quintetos (agregado de quintetos)
