"""Pestaña 'Estadistica por Quintetos': rendimiento de las combinaciones de 5 jugadores."""

import json
import re
from typing import Any, Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ..colors import _parse_color, _text_color_for_bg
from ..pdf_export import render_pdf_button
from ..tables import build_column_config
from ..utils import _first_col, _first_of, _stay_estadistica


def render_quintetos(tablas: Dict[str, pd.DataFrame]) -> None:
        render_pdf_button(tablas, key='quintetos')
    
        part_df = tablas.get('partido', pd.DataFrame())
        local_title = str(part_df.iloc[0].get('local')) if not part_df.empty else 'Local'
        visitante_title = str(part_df.iloc[0].get('visitante')) if not part_df.empty else 'Visitante'
        qg = tablas.get('quintetosAgregado', pd.DataFrame()).copy()
        if not qg.empty:
            # Normalizar
            if 'Condicion' in qg.columns:
                qg['Condicion'] = qg['Condicion'].astype(str).str.upper().fillna('')
            # Formateo de quinteto a string estandar para agrupar y mostrar
            if 'quinteto' in qg.columns:
                def fmt_quinteto_group(v):
                    if isinstance(v, (list, tuple)):
                        return ' / '.join([str(x) for x in v])
                    return str(v)
                qg['quinteto'] = qg['quinteto'].apply(fmt_quinteto_group)
    
            period_col = _first_col(qg, ['numero_periodo', 'periodo', 'Periodo'])
            situ_col = _first_col(qg, ['SituacionMarcador', 'Situacion marcador', 'situacion_marcador', 'situacionMarcador', 'situacion', 'Situacion'])
            u2m_col = _first_col(qg, ['ultimos_dos_minutos', 'ultimos2min', 'ultimos_dos', 'u2m'])
    
            with st.form(key='quintetos_filters'):
                fcols = st.columns(3)
                with fcols[0]:
                    if period_col and period_col in qg.columns:
                        per_opts = ['TODOS'] + sorted(pd.to_numeric(qg[period_col], errors='coerce').dropna().astype(int).unique().tolist())
                        sel_per_q = st.selectbox('Número de periodo', per_opts, index=0, key='q_sel_per')
                    else:
                        sel_per_q = 'TODOS'
                with fcols[1]:
                    if situ_col and situ_col in qg.columns:
                        situ_vals = qg[situ_col].astype(str).fillna('').unique().tolist()
                        situ_opts = ['TODOS'] + sorted([s for s in situ_vals if s != ''])
                        sel_situ_q = st.selectbox('Situacion marcador', situ_opts, index=0, key='q_sel_situ')
                    else:
                        sel_situ_q = 'TODOS'
                with fcols[2]:
                    if u2m_col and u2m_col in qg.columns:
                        u2_vals = qg[u2m_col].astype(str).fillna('').unique().tolist()
                        u2_opts = ['TODOS'] + sorted([u for u in u2_vals if u != ''])
                        sel_u2m_q = st.selectbox('Momento del periodo', u2_opts, index=0, key='q_sel_u2m')
                    else:
                        sel_u2m_q = 'TODOS'
                submitted_q = st.form_submit_button('Aplicar filtros')
                if submitted_q:
                    _stay_estadistica()
    
            # Aplicar filtros
            qg_f = qg.copy()
            if sel_per_q != 'TODOS' and period_col and period_col in qg_f.columns:
                try:
                    qg_f = qg_f[pd.to_numeric(qg_f[period_col], errors='coerce') == int(sel_per_q)]
                except Exception:
                    pass
            if sel_situ_q != 'TODOS' and situ_col and situ_col in qg_f.columns:
                qg_f = qg_f[qg_f[situ_col].astype(str) == str(sel_situ_q)]
            if sel_u2m_q != 'TODOS' and u2m_col and u2m_col in qg_f.columns:
                qg_f = qg_f[qg_f[u2m_col].astype(str) == str(sel_u2m_q)]
    
            # Construcción de tablas por Condicion, agrupando por quinteto
            def make_quintetos_table(
                df_src: pd.DataFrame,
                condicion: str,
                mark_init: Optional[set]=None,
                mark_pm_max: Optional[set]=None,
                mark_pm_min: Optional[set]=None,
            ) -> pd.DataFrame:
                df_t = df_src[df_src.get('Condicion', '').astype(str).str.upper() == condicion.upper()].copy()
                if df_t.empty:
                    return pd.DataFrame()
                name_col = 'quinteto'
                # Seleccionar métricas disponibles
                base_cols = []
                for c in ['tiempo_jugado', 'puntos_favor', 'puntos_contra', 'diferencia']:
                    if c in df_t.columns:
                        df_t[c] = pd.to_numeric(df_t[c], errors='coerce').fillna(0)
                        base_cols.append(c)
                # Variables de Favor/Contra
                fav_cols = [c for c in df_t.columns if c.endswith('_Favor')]
                con_cols = [c for c in df_t.columns if c.endswith('_Contra')]
                sum_cols = base_cols + fav_cols + con_cols
                if not sum_cols:
                    return pd.DataFrame()
                keep_cols = [name_col] + sum_cols
                agg = (
                    df_t
                    .groupby([name_col], as_index=False)[keep_cols]
                    .sum(numeric_only=True)
                )
                # Agregar columna de marca (símbolos) antes de formatear nombres
                init_sym = '⭐'    # quinteto inicial (igual que en tabla de jugadores)
                pm_up_sym = '🟢↑' # mayor +/-
                pm_dn_sym = '🔴↓' # menor +/-
                def mark_symbol(qs: str) -> str:
                    sym = ''
                    q = str(qs)
                    if mark_init and q in mark_init:
                        sym += init_sym
                    if mark_pm_max and q in mark_pm_max:
                        sym += pm_up_sym
                    if mark_pm_min and q in mark_pm_min:
                        sym += pm_dn_sym
                    return sym
                try:
                    agg['Marca'] = agg[name_col].apply(mark_symbol)
                except Exception:
                    agg['Marca'] = ''
                # Ordenar por tiempo_jugado desc si existe antes de agregar Totales
                if 'tiempo_jugado' in agg.columns:
                    agg = agg.sort_values(by=['tiempo_jugado'], ascending=False).reset_index(drop=True)
                # Eficiencias de 1P/2P/3P a Favor y en Contra
                def get_col(df, *cands):
                    for c in cands:
                        if c in df.columns:
                            return df[c]
                    return pd.Series([0]*len(df))
                # Favor
                c1F = pd.to_numeric(get_col(agg, 'CANASTA-1P_Favor', 'CANASTA_1P_Favor'), errors='coerce').fillna(0)
                m1F = pd.to_numeric(get_col(agg, 'TIRO1-FALLADO_Favor', 'TIRO1_FALLADO_Favor'), errors='coerce').fillna(0)
                a1F = c1F + m1F
                c2F = pd.to_numeric(get_col(agg, 'CANASTA-2P_Favor', 'CANASTA_2P_Favor'), errors='coerce').fillna(0)
                m2F = pd.to_numeric(get_col(agg, 'TIRO2-FALLADO_Favor', 'TIRO2_FALLADO_Favor'), errors='coerce').fillna(0)
                a2F = c2F + m2F
                c3F = pd.to_numeric(get_col(agg, 'CANASTA-3P_Favor', 'CANASTA_3P_Favor'), errors='coerce').fillna(0)
                m3F = pd.to_numeric(get_col(agg, 'TIRO3-FALLADO_Favor', 'TIRO3_FALLADO_Favor'), errors='coerce').fillna(0)
                a3F = c3F + m3F
                # Contra
                c1C = pd.to_numeric(get_col(agg, 'CANASTA-1P_Contra', 'CANASTA_1P_Contra'), errors='coerce').fillna(0)
                m1C = pd.to_numeric(get_col(agg, 'TIRO1-FALLADO_Contra', 'TIRO1_FALLADO_Contra'), errors='coerce').fillna(0)
                a1C = c1C + m1C
                c2C = pd.to_numeric(get_col(agg, 'CANASTA-2P_Contra', 'CANASTA_2P_Contra'), errors='coerce').fillna(0)
                m2C = pd.to_numeric(get_col(agg, 'TIRO2-FALLADO_Contra', 'TIRO2_FALLADO_Contra'), errors='coerce').fillna(0)
                a2C = c2C + m2C
                c3C = pd.to_numeric(get_col(agg, 'CANASTA-3P_Contra', 'CANASTA_3P_Contra'), errors='coerce').fillna(0)
                m3C = pd.to_numeric(get_col(agg, 'TIRO3-FALLADO_Contra', 'TIRO3_FALLADO_Contra'), errors='coerce').fillna(0)
                a3C = c3C + m3C
    
                def pct(conv, att):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        p = np.where(att > 0, (conv / att) * 100.0, 0.0)
                    return pd.Series(p).round(0).astype(int)
    
                agg['1P Favor'] = (c1F.astype(int).astype(str) + '/' + a1F.astype(int).astype(str))
                agg['%1P Favor'] = pct(c1F, a1F).astype(str) + '%'
                agg['1P Contra'] = (c1C.astype(int).astype(str) + '/' + a1C.astype(int).astype(str))
                agg['%1P Contra'] = pct(c1C, a1C).astype(str) + '%'
                agg['2P Favor'] = (c2F.astype(int).astype(str) + '/' + a2F.astype(int).astype(str))
                agg['%2P Favor'] = pct(c2F, a2F).astype(str) + '%'
                agg['2P Contra'] = (c2C.astype(int).astype(str) + '/' + a2C.astype(int).astype(str))
                agg['%2P Contra'] = pct(c2C, a2C).astype(str) + '%'
                agg['3P Favor'] = (c3F.astype(int).astype(str) + '/' + a3F.astype(int).astype(str))
                agg['%3P Favor'] = pct(c3F, a3F).astype(str) + '%'
                agg['3P Contra'] = (c3C.astype(int).astype(str) + '/' + a3C.astype(int).astype(str))
                agg['%3P Contra'] = pct(c3C, a3C).astype(str) + '%'
                # Diferencias de % (Favor - Contra)
                # No agregar columnas de diferencias de % en la tabla
                # Derivar plus/minus de puntos
                if 'puntos_favor' in agg.columns and 'puntos_contra' in agg.columns:
                    try:
                        agg['+-'] = pd.to_numeric(agg['puntos_favor'], errors='coerce').fillna(0) - pd.to_numeric(agg['puntos_contra'], errors='coerce').fillna(0)
                    except Exception:
                        agg['+-'] = 0
                # Totales
                totals = agg.select_dtypes(include=[np.number]).sum(numeric_only=True)
                total_row = {col: '' for col in agg.columns}
                total_row[name_col] = 'Totales'
                for c in totals.index:
                    total_row[c] = totals[c]
                agg = pd.concat([agg, pd.DataFrame([total_row])], ignore_index=True)
                # Formatos
                if 'tiempo_jugado' in agg.columns:
                    def fmt_t(s):
                        try:
                            s = float(s)
                        except Exception:
                            return str(s)
                        return f"{int(s//60)}:{int(s%60):02d}" if s == s and s != '' else ''
                    agg['tiempo_jugado'] = agg['tiempo_jugado'].apply(fmt_t)
                    agg = agg.rename(columns={'tiempo_jugado': 'Tiempo Jugado'})
                # Acortar quinteto a apellidos
                def short_quinteto(qs: str) -> str:
                    try:
                        s = str(qs)
                        if s == 'Totales':
                            return s
                        parts = [p.strip() for p in re.split(r"\s*/\s*|\s*-\s*", s) if p.strip()]
                        apes = []
                        for p in parts:
                            # Tomar después de guión en caso de '00-Nombre Apellido'
                            if '-' in p:
                                p = p.split('-', 1)[-1].strip()
                            # Si viene 'APELLIDO, NOMBRE'
                            if ',' in p:
                                apes.append(p.split(',', 1)[0].strip())
                            else:
                                toks = [t for t in p.split(' ') if t]
                                apes.append(toks[-1] if toks else p)
                        return ' - '.join(apes)
                    except Exception:
                        return str(qs)
                agg[name_col] = agg[name_col].apply(short_quinteto)
                # Reordenar para que 'Marca' sea segunda columna (sin título)
                if 'Marca' in agg.columns:
                    cols = agg.columns.tolist()
                    if name_col in cols:
                        cols.remove('Marca')
                        insert_idx = 1
                        cols.insert(insert_idx, 'Marca')
                        agg = agg[cols]
                # Orden de columnas: como jugadores: Quinteto, Tiempo, Puntos a favor/en contra, Diferencia, eficiencias, luego pares Favor/Contra
                preferred = [
                    name_col, 'Marca', 'Tiempo Jugado', 'puntos_favor', 'puntos_contra', 'diferencia', '+-',
                    '1P Favor', '%1P Favor', '1P Contra', '%1P Contra',
                    '2P Favor', '%2P Favor', '2P Contra', '%2P Contra',
                    '3P Favor', '%3P Favor', '3P Contra', '%3P Contra',
                ]
                base_priority = [
                    'REBOTE-DEFENSIVO', 'REBOTE-OFENSIVO', 'ASISTENCIA', 'PERDIDA', 'RECUPERACION',
                    'FALTA-COMETIDA', 'FALTA-RECIBIDA',
                    'TIRO1-FALLADO', 'CANASTA-1P',
                    'TIRO2-FALLADO', 'CANASTA-2P',
                    'TIRO3-FALLADO', 'CANASTA-3P'
                ]
                pair_cols = []
                for base in base_priority:
                    fcol = f'{base}_Favor'
                    ccol = f'{base}_Contra'
                    if fcol in agg.columns:
                        pair_cols.append(fcol)
                    if ccol in agg.columns:
                        pair_cols.append(ccol)
                ordered = [c for c in preferred if c in agg.columns] + [c for c in pair_cols if c not in preferred]
                ordered += [c for c in agg.columns if c not in ordered]
                agg = agg[ordered]
                # Renombrar columnas a etiquetas amigables
                def nice_label(col: str) -> str:
                    mapping_base = {
                        'puntos_favor': 'Puntos a favor',
                        'puntos_contra': 'Puntos en contra',
                        'diferencia': 'Diferencia de puntos',
                        'REBOTE-DEFENSIVO': 'Rebote Def.',
                        'REBOTE-OFENSIVO': 'Rebote Of.',
                        'ASISTENCIA': 'Asist.',
                        'PERDIDA': 'Pérdidas',
                        'RECUPERACION': 'Recup.',
                        'FALTA-COMETIDA': 'Falta Cometida',
                        'FALTA-RECIBIDA': 'Falta Recibida',
                        'TIRO1-FALLADO': 'Libres Fallados',
                        'CANASTA-1P': 'Libres Convertidos',
                        'TIRO2-FALLADO': 'Dobles Fallados',
                        'CANASTA-2P': 'Dobles Convertidos',
                        'TIRO3-FALLADO': 'Triples Fallados',
                        'CANASTA-3P': 'Triples Convertidos',
                    }
                    if col in (name_col, 'Tiempo Jugado'):
                        return 'Quinteto' if col == name_col else col
                    if col in ('1P Favor','%1P Favor','1P Contra','%1P Contra','%1P Dif',
                               '2P Favor','%2P Favor','2P Contra','%2P Contra','%2P Dif',
                               '3P Favor','%3P Favor','3P Contra','%3P Contra','%3P Dif'):
                        return col
                    if col in ('puntos_favor','puntos_contra','diferencia','+-'):
                        return mapping_base.get(col, col)
                    # Favor/Contra metrics
                    if col.endswith('_Favor') or col.endswith('_Contra'):
                        base = col.replace('_Favor','').replace('_Contra','')
                        side = 'Favor' if col.endswith('_Favor') else 'Contra'
                        base_label = mapping_base.get(base, base.title())
                        return f"{base_label} {side}"
                    return col
                agg = agg.rename(columns={c: nice_label(c) for c in agg.columns})
                return agg
    
            # Calcular marcas de quinteto inicial y más minutos en últimos 2 min (Q4+)
            def compute_marks(df: pd.DataFrame, condicion: str):
                base = df[df.get('Condicion', '').astype(str).str.upper() == condicion.upper()].copy()
                if base.empty:
                    return set(), set()
                # Asegurar columnas (detectar nombres alternativos)
                per_col = _first_col(base, ['numero_periodo','periodo','Periodo']) or 'numero_periodo'
                u2m_col = _first_col(base, ['ultimos_dos_minutos','ultimos2min','ultimos_dos','u2m','Ultimos 2 min','Últimos dos min']) or 'ultimos_dos_minutos'
                per = pd.to_numeric(base.get(per_col, 0), errors='coerce').fillna(0)
                base['__per'] = per
                # ultimos_dos_minutos puede venir como texto 'Últimos dos min'. Hacer comparación robusta sin acentos ni mayúsculas
                def _norm_txt(x: Any) -> str:
                    s = str(x)
                    s = s.replace('Ú','U').replace('ú','u').replace('ó','o').replace('í','i').replace('á','a').replace('é','e')
                    return s.strip().lower()
                base['__u2m'] = base.get(u2m_col, '').apply(_norm_txt).isin(['ultimos dos min','u2m','true','1','si','sí','s'])
                # Parse tiempo_jugado a segundos (mm:ss o numérico)
                def _to_secs(v: Any) -> float:
                    try:
                        s = str(v)
                        if ':' in s:
                            mm, ss = s.split(':', 1)
                            return float(int(mm) * 60 + int(ss))
                        return float(pd.to_numeric(v, errors='coerce'))
                    except Exception:
                        return 0.0
                base['__t'] = base.get('tiempo_jugado', 0).apply(_to_secs)
                # Función para extraer dorsales de un quinteto (como conjunto de dos dígitos)
                def two_digit(v):
                    try:
                        iv = int(pd.to_numeric(v, errors='coerce'))
                        return f"{iv:02d}"
                    except Exception:
                        return None
                def quinteto_dorsales(qv):
                    s = ''
                    if isinstance(qv, (list, tuple)):
                        parts = [str(x) for x in qv]
                        s = ' '.join(parts)
                    else:
                        s = str(qv)
                    # Buscar números de 1 o 2 dígitos y también prefijos 'DD-'
                    nums = set()
                    for m in re.findall(r"\b(\d{1,2})\b", s):
                        td = two_digit(m)
                        if td:
                            nums.add(td)
                    for m in re.findall(r"\b(\d{2})-", s):
                        td = two_digit(m)
                        if td:
                            nums.add(td)
                    return nums
                # Obtener dorsales titulares desde planillas
                plan_df = tablas.get('estadisticas_equipolocal' if condicion.upper()=='LOCAL' else 'estadisticas_equipovisitante', pd.DataFrame())
                starters = set()
                if not plan_df.empty:
                    dcol = None
                    for c in ['dorsal','numero','nro','número','numero_camiseta','n_camisa']:
                        if c in plan_df.columns:
                            dcol = c
                            break
                    tcol = None
                    for c in ['quintetotitular','quinteto_titular','QuintetoTitular','quintetoTitular','titular','es_titular']:
                        if c in plan_df.columns:
                            tcol = c
                            break
                    if dcol and tcol:
                        tmp = plan_df[[dcol, tcol]].copy()
                        tmp['__tit'] = tmp[tcol].astype(str).str.lower().isin(['true','1','si','sí','t','x'])
                        starters = set([two_digit(v) for v in tmp.loc[tmp['__tit'], dcol].tolist() if two_digit(v) is not None])
                init_set = set()
                if starters and len(starters) >= 5:
                    # Encontrar fila cuyo quinteto contenga todos los dorsales titulares
                    for _, r in base.iterrows():
                        qs = r.get('quinteto')
                        nums = quinteto_dorsales(qs)
                        if starters.issubset(nums):
                            init_set.add(str(qs))
                # Últimos 2 min: periodo >=4 y flag u2m -> más tiempo jugado
                u2 = base[(base['__per'] >= 4) & (base['__u2m'])]
                u2m_set = set()
                if not u2.empty:
                    idx2 = u2.groupby('quinteto')['__t'].sum().sort_values(ascending=False).head(1).index
                    u2m_set = set([str(x) for x in idx2.tolist()])
                return init_set, u2m_set
    
            init_loc, u2m_loc = compute_marks(qg, 'LOCAL')
            init_vis, u2m_vis = compute_marks(qg, 'VISITANTE')
    
            # Marcar mayor y menor +/- por equipo
            def mark_pm(df: pd.DataFrame, condicion: str) -> tuple[set, set]:
                b = df[df.get('Condicion','').astype(str).str.upper()==condicion.upper()].copy()
                if b.empty:
                    return set(), set()
                # Si no existe 'diferencia', derivar como puntos_favor - puntos_contra
                if 'diferencia' not in b.columns:
                    try:
                        b['diferencia'] = pd.to_numeric(b.get('puntos_favor',0), errors='coerce').fillna(0) - pd.to_numeric(b.get('puntos_contra',0), errors='coerce').fillna(0)
                    except Exception:
                        b['diferencia'] = 0
                g = b.groupby('quinteto')['diferencia'].sum()
                if g.empty:
                    return set(), set()
                srt = g.sort_values(ascending=False)
                top = srt.head(1).index
                bottom = srt.tail(1).index
                return set([str(x) for x in top.tolist()]), set([str(x) for x in bottom.tolist()])
    
            maxpm_loc, minpm_loc = mark_pm(qg, 'LOCAL')
            maxpm_vis, minpm_vis = mark_pm(qg, 'VISITANTE')
    
            # Pasar conjuntos combinados a la tabla
            tbl_loc_q = make_quintetos_table(qg_f, 'LOCAL', init_loc, maxpm_loc, minpm_loc)
            tbl_vis_q = make_quintetos_table(qg_f, 'VISITANTE', init_vis, maxpm_vis, minpm_vis)
            # Post-proceso: marcar 🔥 en quintetos usando SOLO la tabla mostrada y filtro de esta pestaña
            def _norm_txt_q(x: Any) -> str:
                s = str(x)
                return s.replace('Ú','U').replace('ú','u').replace('ó','o').replace('í','i').replace('á','a').replace('é','e').strip().lower()
            u2m_sel_q = _norm_txt_q(st.session_state.get('q_sel_u2m','TODOS'))
            def _parse_secs_q(v: Any) -> float:
                try:
                    s = str(v)
                    if ':' in s:
                        mm, ss = s.split(':', 1)
                        return float(int(mm)*60 + int(ss))
                    return float(pd.to_numeric(v, errors='coerce'))
                except Exception:
                    return 0.0
            def add_fire_quinteto(df_in: pd.DataFrame) -> pd.DataFrame:
                if not isinstance(df_in, pd.DataFrame) or df_in.empty:
                    return df_in
                if u2m_sel_q != 'ultimos dos min':
                    return df_in
                if 'Tiempo Jugado' not in df_in.columns:
                    return df_in
                work = df_in.copy()
                work['__sec'] = work['Tiempo Jugado'].apply(_parse_secs_q)
                # excluir Totales
                name_col_cur = 'Quinteto' if 'Quinteto' in work.columns else None
                if name_col_cur is None:
                    return df_in
                mask = work[name_col_cur].astype(str).str.strip().ne('Totales')
                if not mask.any():
                    return df_in
                idx_max = work[mask]['__sec'].idxmax()
                if 'Marca' not in work.columns:
                    work['Marca'] = ''
                work.at[idx_max, 'Marca'] = str(work.at[idx_max, 'Marca']) + '🔥'
                return work
            tbl_loc_q = add_fire_quinteto(tbl_loc_q)
            tbl_vis_q = add_fire_quinteto(tbl_vis_q)
    
            # Usar mismos colores que en Resumen
            row = part_df.iloc[0] if not part_df.empty else {}
            color_local_raw = _first_of(row, ['color_local', 'local_color', 'colorLocal', 'colorlocal'], '#1f77b4')
            color_visitante_raw = _first_of(row, ['color_visitante', 'visitante_color', 'colorVisitante', 'colorvisitante'], '#ff7f0e')
            color_local = _parse_color(color_local_raw, '#1f77b4')
            color_visitante = _parse_color(color_visitante_raw, '#ff7f0e')
            tc_local = _text_color_for_bg(color_local)
            tc_visitante = _text_color_for_bg(color_visitante)
    
            # Títulos y tablas con colores de equipo
            st.markdown(f"""
            <div style='background:{color_local}; color:{tc_local}; padding:12px; border-radius:8px; text-align:center; font-weight:700; margin:16px 0 8px;'>
                🏀 LOCAL - {local_title}
            </div>
            """, unsafe_allow_html=True)
            if not tbl_loc_q.empty:
                st.dataframe(tbl_loc_q, use_container_width=True, hide_index=True, column_config=build_column_config(tbl_loc_q))
                st.caption("⭐ Quinteto inicial   ·   🟢↑ Mayor +/-   ·   🔴↓ Menor +/-")
                # Timeline de presencia (LOCAL)
                try:
                    pbp_df_full = tablas.get('pbp', pd.DataFrame()).copy()
                    if not pbp_df_full.empty and 'quinteto_local' in pbp_df_full.columns:
                        dfp = pbp_df_full.copy()
                        # Orden temporal consistente
                        if 'autoincremental_id' in dfp.columns and 'autoincremental_id_num' not in dfp.columns:
                            dfp['autoincremental_id_num'] = pd.to_numeric(dfp['autoincremental_id'], errors='coerce').fillna(0)
                        if 'autoincremental_id_num' in dfp.columns:
                            dfp = dfp.sort_values(by=['_id','numero_periodo','autoincremental_id_num'])
                        elif 'tiempo_segundos' in dfp.columns:
                            dfp = dfp.sort_values(by=['_id','numero_periodo','tiempo_segundos'])
                        dfp = dfp.reset_index(drop=True)
                        # x_period con soporte de prórrogas (OT)
                        if 'x_period' not in dfp.columns:
                            tnum = pd.to_numeric(dfp.get('tiempo_segundos', np.nan), errors='coerce').fillna(0)
                            pnum = pd.to_numeric(dfp.get('numero_periodo', 1), errors='coerce').fillna(1)
                            dfp['x_period'] = np.where(
                                pnum <= 4,
                                (600 - tnum) + (pnum - 1) * 600,
                                2400 + (300 - tnum) + (pnum - 4) * 300
                            )
                        dfp['x_period'] = pd.to_numeric(dfp['x_period'], errors='coerce')
                        dfp = dfp[~dfp['x_period'].isna()].copy()
                        # Y-order desde planilla local
                        est_loc_df2 = tablas.get('estadisticas_equipolocal', pd.DataFrame()).copy()
                        y_order_loc: List[str] = []
                        if not est_loc_df2.empty:
                            dcol = _first_col(est_loc_df2, ['dorsal','numero','nro','número','numero_camiseta','n_camisa'])
                            ncol = _first_col(est_loc_df2, ['nombre','jugador','nombre_jugador'])
                            if dcol and ncol:
                                tmp = est_loc_df2[[dcol, ncol]].copy()
                                def _fmt_label_loc(r):
                                    try:
                                        dd = int(pd.to_numeric(r.get(dcol, ''), errors='coerce'))
                                        dd2 = f"{dd:02d}"
                                    except Exception:
                                        dd2 = str(r.get(dcol, '')).strip()
                                    nm = str(r.get(ncol, '')).strip()
                                    return (dd2 + '-' + nm).strip('-') if dd2 or nm else nm
                                y_order_loc = [_fmt_label_loc(r) for _, r in tmp.iterrows()]
                                seen = set()
                                y_order_loc = [x for x in y_order_loc if not (x in seen or seen.add(x))]
                        # Normalizar quinteto_local en dfp
                        def _to_list(v):
                            if isinstance(v, list):
                                return [str(x) for x in v]
                            if isinstance(v, str):
                                try:
                                    parsed = json.loads(v)
                                    if isinstance(parsed, list):
                                        return [str(x) for x in parsed]
                                except Exception:
                                    pass
                                parts = [p.strip() for p in re.split(r"\s*/\s*", v) if p.strip()]
                                return parts if parts else [v]
                            return []
                        dfp['quinteto_local'] = dfp['quinteto_local'].apply(_to_list)
                        # Intervalos [x1,x2] por jugador del quinteto previo
                        intervals = []
                        for i in range(1, len(dfp)):
                            r_prev = dfp.iloc[i-1]
                            r_cur = dfp.iloc[i]
                            try:
                                per_prev = int(pd.to_numeric(r_prev.get('numero_periodo'), errors='coerce'))
                                per_cur = int(pd.to_numeric(r_cur.get('numero_periodo'), errors='coerce'))
                            except Exception:
                                per_prev, per_cur = None, None
                            if per_prev is None or per_cur is None or per_prev != per_cur:
                                continue
                            x1 = float(pd.to_numeric(r_prev.get('x_period'), errors='coerce'))
                            x2 = float(pd.to_numeric(r_cur.get('x_period'), errors='coerce'))
                            if x2 < x1:
                                x1, x2 = x2, x1
                            for p in (r_prev.get('quinteto_local') or []):
                                intervals.append({'player': str(p), 'x1': x1, 'x2': x2})
                        intervals_df = pd.DataFrame(intervals)
                        # Eventos ENTRA/SALE (LOCAL) -> círculos verde/rojo
                        ev = pd.DataFrame()
                        if 'accion_tipo' in dfp.columns and 'Condicion' in dfp.columns:
                            ev = dfp[dfp['accion_tipo'].isin(['CAMBIO-JUGADOR-ENTRA','CAMBIO-JUGADOR-SALE']) & (dfp['Condicion'].astype(str).str.upper()=='LOCAL')].copy()
                            if not ev.empty:
                                ev['player'] = ev.get('nombre', '').astype(str)
                        if not intervals_df.empty:
                            intervals_df['player'] = intervals_df['player'].astype(str)
                            # Alto suficiente: ~28px por jugador
                            players_count = len(y_order_loc) if y_order_loc else intervals_df['player'].nunique()
                            chart_height = max(380, 28 * players_count)
                            # Halo para mejorar contraste + línea principal
                            halo_loc = (
                                alt.Chart(intervals_df)
                                .mark_rule(color='#000000', strokeWidth=9, opacity=0.15)
                                .encode(
                                    x=alt.X('x1:Q', title='Tiempo (x_period)'),
                                    x2='x2:Q',
                                    y=alt.Y('player:N', sort=y_order_loc if y_order_loc else None, title='Jugador (Local)', axis=alt.Axis(labelFontSize=13))
                                )
                            )
                            base = (
                                alt.Chart(intervals_df)
                                .mark_rule(color=color_local, strokeWidth=6)
                                .encode(
                                    x=alt.X('x1:Q', title='Tiempo (x_period)'),
                                    x2='x2:Q',
                                    y=alt.Y('player:N', sort=y_order_loc if y_order_loc else None, title='Jugador (Local)', axis=alt.Axis(labelFontSize=13))
                                )
                                .properties(height=chart_height, title=alt.TitleParams(text=f'Jugadores en cancha - {local_title}', anchor='middle'))
                            )
                            # Reglas verticales y etiquetas de periodo
                            period_changes = []
                            prev = None
                            for _, rp in dfp.iterrows():
                                cur = rp.get('numero_periodo')
                                if prev is not None and cur != prev:
                                    period_changes.append({'x_period': rp.get('x_period'), 'numero_periodo': cur})
                                prev = cur
                            chart_loc = halo_loc + base
                            if period_changes:
                                rules_df = pd.DataFrame(period_changes)
                                rules = alt.Chart(rules_df).mark_rule(color='#333333', strokeDash=[8,4], strokeWidth=3).encode(
                                    x=alt.X('x_period:Q'),
                                    tooltip=[alt.Tooltip('numero_periodo:N', title='Inicio periodo')]
                                )
                                # Etiqueta en la parte superior con texto "P <n>"
                                top_name = y_order_loc[0] if y_order_loc else intervals_df['player'].unique().tolist()[0]
                                labels_df = rules_df.copy()
                                labels_df['label'] = 'P ' + labels_df['numero_periodo'].astype(str)
                                labels_df['y_label'] = top_name
                                labels = alt.Chart(labels_df).mark_text(align='left', baseline='bottom', dx=6, dy=-8, color='#333333', fontSize=14).encode(
                                    x=alt.X('x_period:Q'), y=alt.Y('y_label:N'), text='label:N'
                                )
                                chart_loc = chart_loc + rules + labels
                            # Marcas ENTRA (verde) / SALE (rojo) como círculos
                            if not ev.empty:
                                enter = ev[ev['accion_tipo']=='CAMBIO-JUGADOR-ENTRA'].copy()
                                sale = ev[ev['accion_tipo']=='CAMBIO-JUGADOR-SALE'].copy()
                                if not enter.empty:
                                    enter_layer = (
                                        alt.Chart(enter)
                                        .mark_point(filled=True, size=140, color='#2e7d32', shape='circle')
                                        .encode(x=alt.X('x_period:Q'), y=alt.Y('player:N', sort=y_order_loc if y_order_loc else None))
                                    )
                                    chart_loc = chart_loc + enter_layer
                                if not sale.empty:
                                    sale_layer = (
                                        alt.Chart(sale)
                                        .mark_point(filled=True, size=140, color='#c62828', shape='circle')
                                        .encode(x=alt.X('x_period:Q'), y=alt.Y('player:N', sort=y_order_loc if y_order_loc else None))
                                    )
                                    chart_loc = chart_loc + sale_layer
                            # Marcas de tiros (LOCAL)
                            if 'accion_tipo' in dfp.columns and 'Condicion' in dfp.columns:
                                shots = dfp[(dfp['Condicion'].astype(str).str.upper()=='LOCAL') & (dfp['accion_tipo'].isin([
                                    'CANASTA-1P','TIRO1-FALLADO','CANASTA-2P','TIRO2-FALLADO','CANASTA-3P','TIRO3-FALLADO'
                                ]))].copy()
                                if not shots.empty:
                                    shots['player'] = shots.get('nombre','').astype(str)
                                    # Separación adicional: ajustar levemente el tiempo (x) según tipo y si fue convertido/fallado
                                    try:
                                        at = shots['accion_tipo'].astype(str)
                                        tipo_num = at.str.extract(r'(\d)P', expand=False).fillna('0').astype(int)
                                        is_made = at.str.startswith('CANASTA-')
                                        base_off = np.where(is_made, 0.25, -0.25)
                                        tipo_off = np.where(tipo_num == 1, 0.00, np.where(tipo_num == 2, 0.08, 0.16))
                                        xper = pd.to_numeric(shots.get('x_period', 0), errors='coerce').fillna(0)
                                        shots['__x_adj'] = xper + base_off + tipo_off
                                    except Exception:
                                        shots['__x_adj'] = shots.get('x_period', 0)
                                    # Definir capas por tipo usando glifos de texto (✔ para convertidos, ✚ para fallados) con offset para no taparse
                                    def shot_layer_text(df_in, color_hex, glyph, dx_px):
                                        if df_in.empty:
                                            return None
                                        return (
                                            alt.Chart(df_in)
                                            .mark_text(fontWeight='bold', fontSize=26, color=color_hex, stroke='black', strokeWidth=0.6, dx=dx_px)
                                            .encode(
                                                x=alt.X('__x_adj:Q'),
                                                y=alt.Y('player:N', sort=y_order_loc if y_order_loc else None),
                                                text=alt.value(glyph)
                                            )
                                        )
                                    grn = '#2e7d32'
                                    red = '#c62828'
                                    # Offsets: convertidos +8px, fallados -8px (para mismo tiempo)
                                    l_m1 = shot_layer_text(shots[shots['accion_tipo']=='CANASTA-1P'], grn, '1', 8)
                                    l_x1 = shot_layer_text(shots[shots['accion_tipo']=='TIRO1-FALLADO'], red, '1', -8)
                                    l_m2 = shot_layer_text(shots[shots['accion_tipo']=='CANASTA-2P'], grn, '2', 8)
                                    l_x2 = shot_layer_text(shots[shots['accion_tipo']=='TIRO2-FALLADO'], red, '2', -8)
                                    l_m3 = shot_layer_text(shots[shots['accion_tipo']=='CANASTA-3P'], grn, '3', 8)
                                    l_x3 = shot_layer_text(shots[shots['accion_tipo']=='TIRO3-FALLADO'], red, '3', -8)
                                    for lyr in [l_m1,l_x1,l_m2,l_x2,l_m3,l_x3]:
                                        if lyr is not None:
                                            chart_loc = chart_loc + lyr
                            st.altair_chart(chart_loc, use_container_width=True)
                            st.caption("1 verde: 1P convertido · 1 rojo: 1P fallado · 2 verde: 2P convertido · 2 rojo: 2P fallado · 3 verde: 3P convertido · 3 rojo: 3P fallado · ● Verde: entra · ● Rojo: sale")
                        else:
                            st.info('No se pudieron derivar intervalos de presencia del PBP (LOCAL).')
                except Exception as e:
                    st.warning(f'No se pudo generar el timeline LOCAL: {e}')
            else:
                st.info('Sin datos de quintetos LOCAL para los filtros seleccionados')
    
            st.markdown(f"""
            <div style='background:{color_visitante}; color:{tc_visitante}; padding:12px; border-radius:8px; text-align:center; font-weight:700; margin:16px 0 8px;'>
                🏀 VISITANTE - {visitante_title}
            </div>
            """, unsafe_allow_html=True)
            if not tbl_vis_q.empty:
                st.dataframe(tbl_vis_q, use_container_width=True, hide_index=True, column_config=build_column_config(tbl_vis_q))
                st.caption("⭐ Quinteto inicial   ·   🟢↑ Mayor +/-   ·   🔴↓ Menor +/-")
                # Timeline de presencia (VISITANTE)
                try:
                    pbp_df_full = tablas.get('pbp', pd.DataFrame()).copy()
                    if not pbp_df_full.empty and 'quinteto_visitante' in pbp_df_full.columns:
                        dfp = pbp_df_full.copy()
                        if 'autoincremental_id' in dfp.columns and 'autoincremental_id_num' not in dfp.columns:
                            dfp['autoincremental_id_num'] = pd.to_numeric(dfp['autoincremental_id'], errors='coerce').fillna(0)
                        if 'autoincremental_id_num' in dfp.columns:
                            dfp = dfp.sort_values(by=['_id','numero_periodo','autoincremental_id_num'])
                        elif 'tiempo_segundos' in dfp.columns:
                            dfp = dfp.sort_values(by=['_id','numero_periodo','tiempo_segundos'])
                        dfp = dfp.reset_index(drop=True)
                        if 'x_period' not in dfp.columns:
                            tnum = pd.to_numeric(dfp.get('tiempo_segundos', np.nan), errors='coerce').fillna(0)
                            pnum = pd.to_numeric(dfp.get('numero_periodo', 1), errors='coerce').fillna(1)
                            dfp['x_period'] = (600 - tnum) + (pnum - 1) * 600
                        dfp['x_period'] = pd.to_numeric(dfp['x_period'], errors='coerce')
                        dfp = dfp[~dfp['x_period'].isna()].copy()
                        # Y-order desde planilla visitante
                        est_vis_df2 = tablas.get('estadisticas_equipovisitante', pd.DataFrame()).copy()
                        y_order_vis: List[str] = []
                        if not est_vis_df2.empty:
                            dcol = _first_col(est_vis_df2, ['dorsal','numero','nro','número','numero_camiseta','n_camisa'])
                            ncol = _first_col(est_vis_df2, ['nombre','jugador','nombre_jugador'])
                            if dcol and ncol:
                                tmp = est_vis_df2[[dcol, ncol]].copy()
                                def _fmt_label_vis(r):
                                    try:
                                        dd = int(pd.to_numeric(r.get(dcol, ''), errors='coerce'))
                                        dd2 = f"{dd:02d}"
                                    except Exception:
                                        dd2 = str(r.get(dcol, '')).strip()
                                    nm = str(r.get(ncol, '')).strip()
                                    return (dd2 + '-' + nm).strip('-') if dd2 or nm else nm
                                y_order_vis = [_fmt_label_vis(r) for _, r in tmp.iterrows()]
                                seen = set()
                                y_order_vis = [x for x in y_order_vis if not (x in seen or seen.add(x))]
                        # Normalizar quinteto_visitante
                        def _to_list2(v):
                            if isinstance(v, list):
                                return [str(x) for x in v]
                            if isinstance(v, str):
                                try:
                                    parsed = json.loads(v)
                                    if isinstance(parsed, list):
                                        return [str(x) for x in parsed]
                                except Exception:
                                    pass
                                parts = [p.strip() for p in re.split(r"\s*/\s*", v) if p.strip()]
                                return parts if parts else [v]
                            return []
                        dfp['quinteto_visitante'] = dfp['quinteto_visitante'].apply(_to_list2)
                        # Intervalos
                        intervals = []
                        for i in range(1, len(dfp)):
                            r_prev = dfp.iloc[i-1]
                            r_cur = dfp.iloc[i]
                            try:
                                per_prev = int(pd.to_numeric(r_prev.get('numero_periodo'), errors='coerce'))
                                per_cur = int(pd.to_numeric(r_cur.get('numero_periodo'), errors='coerce'))
                            except Exception:
                                per_prev, per_cur = None, None
                            if per_prev is None or per_cur is None or per_prev != per_cur:
                                continue
                            x1 = float(pd.to_numeric(r_prev.get('x_period'), errors='coerce'))
                            x2 = float(pd.to_numeric(r_cur.get('x_period'), errors='coerce'))
                            if x2 < x1:
                                x1, x2 = x2, x1
                            for p in (r_prev.get('quinteto_visitante') or []):
                                intervals.append({'player': str(p), 'x1': x1, 'x2': x2})
                        intervals_df = pd.DataFrame(intervals)
                        # Eventos ENTRA/SALE (VISITANTE)
                        ev = pd.DataFrame()
                        if 'accion_tipo' in dfp.columns and 'Condicion' in dfp.columns:
                            ev = dfp[dfp['accion_tipo'].isin(['CAMBIO-JUGADOR-ENTRA','CAMBIO-JUGADOR-SALE']) & (dfp['Condicion'].astype(str).str.upper()=='VISITANTE')].copy()
                            if not ev.empty:
                                ev['player'] = ev.get('nombre', '').astype(str)
                        if not intervals_df.empty:
                            intervals_df['player'] = intervals_df['player'].astype(str)
                            players_count = len(y_order_vis) if y_order_vis else intervals_df['player'].nunique()
                            chart_height = max(380, 28 * players_count)
                            halo_vis = (
                                alt.Chart(intervals_df)
                                .mark_rule(color='#000000', strokeWidth=9, opacity=0.15)
                                .encode(
                                    x=alt.X('x1:Q', title='Tiempo (x_period)'),
                                    x2='x2:Q',
                                    y=alt.Y('player:N', sort=y_order_vis if y_order_vis else None, title='Jugador (Visitante)', axis=alt.Axis(labelFontSize=13))
                                )
                            )
                            base = (
                                alt.Chart(intervals_df)
                                .mark_rule(color=color_visitante, strokeWidth=6)
                                .encode(
                                    x=alt.X('x1:Q', title='Tiempo (x_period)'),
                                    x2='x2:Q',
                                    y=alt.Y('player:N', sort=y_order_vis if y_order_vis else None, title='Jugador (Visitante)', axis=alt.Axis(labelFontSize=13))
                                )
                                .properties(height=chart_height, title=alt.TitleParams(text=f'Jugadores en cancha - {visitante_title}', anchor='middle'))
                            )
                            # Reglas verticales y etiquetas de periodo
                            period_changes = []
                            prev = None
                            for _, rp in dfp.iterrows():
                                cur = rp.get('numero_periodo')
                                if prev is not None and cur != prev:
                                    period_changes.append({'x_period': rp.get('x_period'), 'numero_periodo': cur})
                                prev = cur
                            chart_vis = halo_vis + base
                            if period_changes:
                                rules_df = pd.DataFrame(period_changes)
                                rules = alt.Chart(rules_df).mark_rule(color='#333333', strokeDash=[8,4], strokeWidth=3).encode(
                                    x=alt.X('x_period:Q'),
                                    tooltip=[alt.Tooltip('numero_periodo:N', title='Inicio periodo')]
                                )
                                top_name = y_order_vis[0] if y_order_vis else intervals_df['player'].unique().tolist()[0]
                                labels_df = rules_df.copy()
                                labels_df['label'] = 'P ' + labels_df['numero_periodo'].astype(str)
                                labels_df['y_label'] = top_name
                                labels = alt.Chart(labels_df).mark_text(align='left', baseline='bottom', dx=6, dy=-8, color='#333333', fontSize=14).encode(
                                    x=alt.X('x_period:Q'), y=alt.Y('y_label:N'), text='label:N'
                                )
                                chart_vis = chart_vis + rules + labels
                            # Marcas ENTRA/SALE
                            if not ev.empty:
                                enter = ev[ev['accion_tipo']=='CAMBIO-JUGADOR-ENTRA'].copy()
                                sale = ev[ev['accion_tipo']=='CAMBIO-JUGADOR-SALE'].copy()
                                if not enter.empty:
                                    enter_layer = (
                                        alt.Chart(enter)
                                        .mark_point(filled=True, size=140, color='#2e7d32', shape='circle')
                                        .encode(x=alt.X('x_period:Q'), y=alt.Y('player:N', sort=y_order_vis if y_order_vis else None))
                                    )
                                    chart_vis = chart_vis + enter_layer
                                if not sale.empty:
                                    sale_layer = (
                                        alt.Chart(sale)
                                        .mark_point(filled=True, size=140, color='#c62828', shape='circle')
                                        .encode(x=alt.X('x_period:Q'), y=alt.Y('player:N', sort=y_order_vis if y_order_vis else None))
                                    )
                                    chart_vis = chart_vis + sale_layer
                            # Marcas de tiros (VISITANTE)
                            if 'accion_tipo' in dfp.columns and 'Condicion' in dfp.columns:
                                shots = dfp[(dfp['Condicion'].astype(str).str.upper()=='VISITANTE') & (dfp['accion_tipo'].isin([
                                    'CANASTA-1P','TIRO1-FALLADO','CANASTA-2P','TIRO2-FALLADO','CANASTA-3P','TIRO3-FALLADO'
                                ]))].copy()
                                if not shots.empty:
                                    shots['player'] = shots.get('nombre','').astype(str)
                                    # Separación adicional en X para VISITANTE (igual que LOCAL)
                                    try:
                                        at = shots['accion_tipo'].astype(str)
                                        tipo_num = at.str.extract(r'(\d)P', expand=False).fillna('0').astype(int)
                                        is_made = at.str.startswith('CANASTA-')
                                        base_off = np.where(is_made, 0.25, -0.25)
                                        tipo_off = np.where(tipo_num == 1, 0.00, np.where(tipo_num == 2, 0.08, 0.16))
                                        xper = pd.to_numeric(shots.get('x_period', 0), errors='coerce').fillna(0)
                                        shots['__x_adj'] = xper + base_off + tipo_off
                                    except Exception:
                                        shots['__x_adj'] = shots.get('x_period', 0)
                                    def shot_layer_text(df_in, color_hex, glyph, dx_px):
                                        if df_in.empty:
                                            return None
                                        return (
                                            alt.Chart(df_in)
                                            .mark_text(fontWeight='bold', fontSize=26, color=color_hex, stroke='black', strokeWidth=0.6, dx=dx_px)
                                            .encode(
                                                x=alt.X('__x_adj:Q'),
                                                y=alt.Y('player:N', sort=y_order_vis if y_order_vis else None),
                                                text=alt.value(glyph)
                                            )
                                        )
                                    grn = '#2e7d32'
                                    red = '#c62828'
                                    l_m1 = shot_layer_text(shots[shots['accion_tipo']=='CANASTA-1P'], grn, '1', 8)
                                    l_x1 = shot_layer_text(shots[shots['accion_tipo']=='TIRO1-FALLADO'], red, '1', -8)
                                    l_m2 = shot_layer_text(shots[shots['accion_tipo']=='CANASTA-2P'], grn, '2', 8)
                                    l_x2 = shot_layer_text(shots[shots['accion_tipo']=='TIRO2-FALLADO'], red, '2', -8)
                                    l_m3 = shot_layer_text(shots[shots['accion_tipo']=='CANASTA-3P'], grn, '3', 8)
                                    l_x3 = shot_layer_text(shots[shots['accion_tipo']=='TIRO3-FALLADO'], red, '3', -8)
                                    for lyr in [l_m1,l_x1,l_m2,l_x2,l_m3,l_x3]:
                                        if lyr is not None:
                                            chart_vis = chart_vis + lyr
                            st.altair_chart(chart_vis, use_container_width=True)
                            st.caption("1 verde: 1P convertido · 1 rojo: 1P fallado · 2 verde: 2P convertido · 2 rojo: 2P fallado · 3 verde: 3P convertido · 3 rojo: 3P fallado · ● Verde: entra · ● Rojo: sale")
                        else:
                            st.info('No se pudieron derivar intervalos de presencia del PBP (VISITANTE).')
                except Exception as e:
                    st.warning(f'No se pudo generar el timeline VISITANTE: {e}')
            else:
                st.info('Sin datos de quintetos VISITANTE para los filtros seleccionados')
        else:
            st.info('No hay datos de quintetos para mostrar')
    
    
    # (Se elimina la pestaña Aux; el timeline fue integrado arriba en Quintetos)
    
    #     def two_digit(v: Any) -> str:
    #         try:
    #             iv = int(pd.to_numeric(v, errors='coerce'))
    #             return f"{iv:02d}"
    #         except Exception:
    #             return ''
    
    #     def add_dd_nombre(df: pd.DataFrame) -> pd.DataFrame:
    #         if df is None or df.empty:
    #             return df
    #         dcol = _first_col2(df, ['dorsal','numero','nro','número','numero_camiseta','n_camisa'])
    #         ncol = _first_col2(df, ['Nombre','nombre','Jugador','jugador','nombre_jugador','NombreJugador'])
    #         out = df.copy()
    #         if dcol and ncol:
    #             out['DD_NOMBRE'] = out.apply(lambda r: (two_digit(r.get(dcol)) + ' - ' + str(r.get(ncol)).strip()).strip(' -'), axis=1)
    #         elif ncol:
    #             out['DD_NOMBRE'] = out[ncol].astype(str).str.strip()
    #         else:
    #             out['DD_NOMBRE'] = ''
    #         return out
    
    #     plan_loc_v = add_dd_nombre(plan_loc)
    #     plan_vis_v = add_dd_nombre(plan_vis)
    
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.markdown('Local (estadisticas_equipolocal)')
    #         st.dataframe(plan_loc_v, use_container_width=True, hide_index=True)
    #     with col2:
    #         st.markdown('Visitante (estadisticas_equipovisitante)')
    #         st.dataframe(plan_vis_v, use_container_width=True, hide_index=True)
