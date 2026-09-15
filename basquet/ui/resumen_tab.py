"""Pestaña 'Resumen': marcador, evolución de puntos y comparativas de equipo."""

from typing import Dict

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ..colors import _adjust_color, _parse_color, _text_color_for_bg
from ..pdf_export import render_pdf_button
from ..utils import _first_col, _first_of, to_seconds


def render_resumen(tablas: Dict[str, pd.DataFrame]) -> None:
    # Botón de descarga PDF
    render_pdf_button(tablas, key='resumen')
    
    part_df = tablas.get('partido', pd.DataFrame())
    pbp_df = tablas.get('pbp', pd.DataFrame())
    est_loc_df = tablas.get('estadisticas_equipolocal', pd.DataFrame())
    est_vis_df = tablas.get('estadisticas_equipovisitante', pd.DataFrame())
    # Derivar valores base aún si 'partido' está vacío
    row = part_df.iloc[0] if not part_df.empty else {}
    # Nombres (variantes)
    local_name = str(_first_of(row, [
        'local', 'equipo_local', 'nombre_local', 'localnombre', 'nombreLocal', 'equipoLocal', 'club_local', 'clubLocal'
    ], ''))
    visitante_name = str(_first_of(row, [
        'visitante', 'equipo_visitante', 'nombre_visitante', 'visitantenombre', 'nombreVisitante', 'equipoVisitante', 'club_visitante', 'clubVisitante'
    ], ''))
    if (not local_name or local_name.strip() == '') and not est_loc_df.empty:
        local_name = str(_first_of(est_loc_df.iloc[0], ['equipo', 'nombre_equipo'], 'Local'))
    if (not visitante_name or visitante_name.strip() == '') and not est_vis_df.empty:
        visitante_name = str(_first_of(est_vis_df.iloc[0], ['equipo', 'nombre_equipo'], 'Visitante'))
    if not local_name:
        local_name = 'Local'
    if not visitante_name:
        visitante_name = 'Visitante'
    
    # Colores
    color_local_raw = _first_of(row, ['color_local', 'local_color', 'colorLocal', 'colorlocal'], '#1f77b4')
    color_visitante_raw = _first_of(row, ['color_visitante', 'visitante_color', 'colorVisitante', 'colorvisitante'], '#ff7f0e')
    color_local = _parse_color(color_local_raw, '#1f77b4')
    color_visitante = _parse_color(color_visitante_raw, '#ff7f0e')
    tc_local = _text_color_for_bg(color_local)
    tc_visitante = _text_color_for_bg(color_visitante)
    
    # Tanteos con fallback a pbp
    tanteo_local = _first_of(row, ['tanteo_local', 'puntos_local', 'marcador_local', 'score_local', 'tanteoLocal', 'marcadorLocal'], None)
    tanteo_visitante = _first_of(row, ['tanteo_visitante', 'puntos_visitante', 'marcador_visitante', 'score_visitante', 'tanteoVisitante', 'marcadorVisitante'], None)
    if (tanteo_local is None or (hasattr(pd, 'isna') and pd.isna(tanteo_local)) or tanteo_local == '') or \
       (tanteo_visitante is None or (hasattr(pd, 'isna') and pd.isna(tanteo_visitante)) or tanteo_visitante == ''):
        if not pbp_df.empty:
            dfp = pbp_df.copy()
            if 'autoincremental_id' in dfp.columns and 'autoincremental_id_num' not in dfp.columns:
                dfp['autoincremental_id_num'] = pd.to_numeric(dfp['autoincremental_id'], errors='coerce').fillna(0)
            if 'autoincremental_id_num' in dfp.columns:
                dfp = dfp.sort_values(by=['_id','numero_periodo','autoincremental_id_num'])
            elif 'tiempo_segundos' in dfp.columns:
                dfp = dfp.sort_values(by=['_id','numero_periodo','tiempo_segundos'])
            last = dfp.iloc[-1]
            tanteo_local = last.get('puntosLocal', tanteo_local)
            tanteo_visitante = last.get('puntosVisitante', tanteo_visitante)
    
    # Resumen por periodo para tarjetas (si hay pbp)
    local_line = ''
    visit_line = ''
    if not pbp_df.empty:
        try:
            dfpp = pbp_df.copy()
            if 'autoincremental_id' in dfpp.columns and 'autoincremental_id_num' not in dfpp.columns:
                dfpp['autoincremental_id_num'] = pd.to_numeric(dfpp['autoincremental_id'], errors='coerce').fillna(0)
            if 'autoincremental_id_num' in dfpp.columns:
                dfpp = dfpp.sort_values(by=['_id','numero_periodo','autoincremental_id_num'])
            else:
                if 'tiempo_segundos' not in dfpp.columns and 'tiempo_partido' in dfpp.columns:
                    dfpp['tiempo_segundos'] = dfpp['tiempo_partido'].apply(to_seconds)
                if 'tiempo_segundos' in dfpp.columns:
                    dfpp = dfpp.sort_values(by=['_id','numero_periodo','tiempo_segundos'])
            # Asegurar numéricos
            dfpp['puntosLocal_num'] = pd.to_numeric(dfpp.get('puntosLocal', 0), errors='coerce').fillna(0)
            dfpp['puntosVisitante_num'] = pd.to_numeric(dfpp.get('puntosVisitante', 0), errors='coerce').fillna(0)
            # Último por periodo
            per_local = {}
            per_visit = {}
            if 'numero_periodo' in dfpp.columns:
                for per in sorted(pd.to_numeric(dfpp['numero_periodo'], errors='coerce').dropna().astype(int).unique().tolist()):
                    sub = dfpp[pd.to_numeric(dfpp['numero_periodo'], errors='coerce') == per]
                    if not sub.empty:
                        last = sub.iloc[-1]
                        per_local[per] = int(float(last.get('puntosLocal_num', 0)))
                        per_visit[per] = int(float(last.get('puntosVisitante_num', 0)))
            if per_local:
                local_line = ' | '.join([f"{per_local.get(p, 0)}" for p in sorted(per_local.keys())])
            if per_visit:
                visit_line = ' | '.join([f"{per_visit.get(p, 0)}" for p in sorted(per_visit.keys())])
        except Exception:
            pass
    
    # Render con columnas y colores de equipo - tarjetas más compactas
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style='background:{color_local}; color:{tc_local}; padding:12px 16px; border-radius:10px; text-align:center;'>
            <div style='font-size:12px; opacity:0.85; margin-bottom:4px;'>🏀 LOCAL</div>
            <div style='font-size:20px; font-weight:700; margin-bottom:2px;'>{local_name}</div>
            <div style='font-size:36px; font-weight:800;'>{int(tanteo_local) if (tanteo_local is not None and str(tanteo_local).strip() != '' and not pd.isna(tanteo_local)) else '-'}</div>
            {f"<div style='font-size:11px; opacity:0.8; margin-top:4px;'>{local_line}</div>" if local_line else ""}
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style='background:{color_visitante}; color:{tc_visitante}; padding:12px 16px; border-radius:10px; text-align:center;'>
            <div style='font-size:12px; opacity:0.85; margin-bottom:4px;'>🏀 VISITANTE</div>
            <div style='font-size:20px; font-weight:700; margin-bottom:2px;'>{visitante_name}</div>
            <div style='font-size:36px; font-weight:800;'>{int(tanteo_visitante) if (tanteo_visitante is not None and str(tanteo_visitante).strip() != '' and not pd.isna(tanteo_visitante)) else '-'}</div>
            {f"<div style='font-size:11px; opacity:0.8; margin-top:4px;'>{visit_line}</div>" if visit_line else ""}
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico de evolución de puntos desde PBP
    if not pbp_df.empty:
            # Selector de periodo (aplica a todo lo siguiente en esta pestaña)
            try:
                periodos = sorted(pd.to_numeric(pbp_df['numero_periodo'], errors='coerce').dropna().astype(int).unique().tolist())
            except Exception:
                periodos = []
            opciones_periodo = ['TODOS'] + periodos
            sel_periodo = st.selectbox('Seleccionar periodo', opciones_periodo, index=0, key='res_sel_per')
            # Espaciado
            st.write("")
            chart_height = 600
            point_size = 120
    
            dfp = pbp_df.copy()
            # Filtrar por periodo si corresponde
            if sel_periodo != 'TODOS':
                try:
                    sel_val = int(sel_periodo)
                    dfp = dfp[pd.to_numeric(dfp['numero_periodo'], errors='coerce') == sel_val].copy()
                except Exception:
                    pass
            # Orden temporal por autoincremental_id si existe, sino por tiempo
            if 'autoincremental_id' in dfp.columns and 'autoincremental_id_num' not in dfp.columns:
                dfp['autoincremental_id_num'] = pd.to_numeric(dfp['autoincremental_id'], errors='coerce').fillna(0)
            if 'autoincremental_id_num' in dfp.columns:
                dfp = dfp.sort_values(by=['_id','numero_periodo','autoincremental_id_num'])
            elif 'tiempo_segundos' in dfp.columns:
                dfp = dfp.sort_values(by=['_id','numero_periodo','tiempo_segundos'])
            dfp = dfp.reset_index(drop=True)
            dfp['orden'] = np.arange(len(dfp))
    
            # Eje X: usar x_period de pbp si existe; sino calcular
            if 'x_period' not in dfp.columns:
                # x_period: Q1-Q4 (10 min) y OT (5 min). Si no hay tiempo, usar 'orden'.
                tiempo_num = pd.to_numeric(dfp.get('tiempo_segundos', np.nan), errors='coerce')
                periodo_num = pd.to_numeric(dfp.get('numero_periodo', 1), errors='coerce').fillna(1)
                tiempo_num = tiempo_num.fillna(dfp['orden'])
                dfp['x_period'] = np.where(
                    periodo_num <= 4,
                    (600 - tiempo_num) + (periodo_num - 1) * 600,
                    2400 + (300 - tiempo_num) + (periodo_num - 5) * 300
                )
    
            # Asegurar x_period numérico exacto del pbp
            dfp['x_period'] = pd.to_numeric(dfp['x_period'], errors='coerce')
            dfp = dfp[~dfp['x_period'].isna()].copy()
            # Selección interactiva sobre eje X
            brush = alt.selection_interval(encodings=['x'], name='Seleccion')
            # Series explícitas por equipo para evitar valores no presentes en pbp
            dfp['puntosLocal_num'] = pd.to_numeric(dfp.get('puntosLocal', 0), errors='coerce').fillna(0)
            dfp['puntosVisitante_num'] = pd.to_numeric(dfp.get('puntosVisitante', 0), errors='coerce').fillna(0)
    
            # Ajustar color de línea si el color de equipo es blanco para que se vea
            def _is_white(c: str) -> bool:
                s = str(c).strip().lower()
                return s in ('#fff', '#ffffff', 'white')
            local_line_color = '#cfd8dc' if _is_white(color_local) else color_local
            visit_line_color = '#cfd8dc' if _is_white(color_visitante) else color_visitante
    
            line_local = (
                alt.Chart(dfp)
                .mark_line(point=False, color=local_line_color)
                .encode(
                    x=alt.X('x_period:Q', title='Tiempo', sort=None),
                    y=alt.Y('puntosLocal_num:Q', title='Puntos acumulados'),
                    order=alt.Order('orden:Q'),
                    tooltip=[
                        alt.Tooltip('numero_periodo:Q', title='Periodo'),
                        alt.Tooltip('tiempo_segundos:Q', title='Tiempo (s)'),
                        alt.Tooltip('puntosLocal_num:Q', title=local_name)
                    ]
                )
                .properties(height=chart_height, title=alt.TitleParams(text='Puntos local vs visitante', anchor='middle'))
                .transform_filter(brush)
            )
            line_visit = (
                alt.Chart(dfp)
                .mark_line(point=False, color=visit_line_color)
                .encode(
                    x=alt.X('x_period:Q', title='Tiempo', sort=None),
                    y=alt.Y('puntosVisitante_num:Q', title='Puntos acumulados'),
                    order=alt.Order('orden:Q'),
                    tooltip=[
                        alt.Tooltip('numero_periodo:Q', title='Periodo'),
                        alt.Tooltip('tiempo_segundos:Q', title='Tiempo (s)'),
                        alt.Tooltip('puntosVisitante_num:Q', title=visitante_name)
                    ]
                )
                .properties(height=chart_height)
                .transform_filter(brush)
            )
            line = line_local + line_visit
    
            # Reglas verticales al cambio de periodo
            period_changes = []
            prev = None
            for _, rowp in dfp.iterrows():
                cur = rowp.get('numero_periodo')
                if prev is not None and cur != prev:
                    period_changes.append({'x_period': rowp.get('x_period'), 'numero_periodo': cur})
                prev = cur
            chart = line
            if period_changes:
                rules_df = pd.DataFrame(period_changes)
                # Reglas más visibles
                rules = alt.Chart(rules_df).mark_rule(color='#333333', strokeDash=[8,4], strokeWidth=3).encode(
                    x=alt.X('x_period:Q'),
                    tooltip=[alt.Tooltip('numero_periodo:N', title='Inicio periodo')]
                ).transform_filter(brush)
                # Etiquetas del número de periodo (debajo y corridas)
                try:
                    rules_df['label'] = 'P ' + rules_df['numero_periodo'].astype(str)
                    # Posicionar texto al 10% del eje Y
                    y_min = float(min(dfp['puntosLocal_num'].min(), dfp['puntosVisitante_num'].min())) if len(dfp) else 0.0
                    y_max = float(max(dfp['puntosLocal_num'].max(), dfp['puntosVisitante_num'].max())) if len(dfp) else 1.0
                    y_pos = y_min + 0.1 * (y_max - y_min)
                    rules_df['y_pos'] = y_pos
                    labels = alt.Chart(rules_df).mark_text(align='left', baseline='bottom', dx=6, color='#333333', fontSize=16).encode(
                        x=alt.X('x_period:Q'), y=alt.Y('y_pos:Q'), text='label:N'
                    ).transform_filter(brush)
                    chart = chart + rules + labels
                except Exception:
                    chart = chart + rules
    
            # Puntos de canastas con tonos por 1P/2P/3P y color por equipo
            if 'accion_tipo' in dfp.columns:
                ev = dfp[dfp['accion_tipo'].isin(['CANASTA-1P','CANASTA-2P','CANASTA-3P'])].copy()
                if not ev.empty:
                    ev['Equipo'] = np.where(ev.get('Condicion','').astype(str).str.upper()=='LOCAL', local_name, visitante_name)
                    ev['y_points'] = np.where(ev.get('Condicion','').astype(str).str.upper()=='LOCAL',
                                              pd.to_numeric(ev.get('puntosLocal', 0), errors='coerce').fillna(0),
                                              pd.to_numeric(ev.get('puntosVisitante', 0), errors='coerce').fillna(0))
                    ev['orden'] = pd.to_numeric(ev['orden'], errors='coerce').fillna(0)
    
                    # Leyenda externa removida; crearemos una leyenda interna personalizada
                    points = alt.Chart(ev).mark_point(filled=True, size=point_size).encode(
                        x=alt.X('x_period:Q', title='Tiempo', sort=None),
                        y=alt.Y('y_points:Q'),
                        color=alt.Color('accion_tipo:N', scale=alt.Scale(
                            domain=['CANASTA-1P','CANASTA-2P','CANASTA-3P'],
                            range=['#fdd835','#1e88e5','#43a047']
                        ), legend=None),
                        opacity=alt.value(1.0),
                        order=alt.Order('orden:Q'),
                        tooltip=[
                            alt.Tooltip('Equipo:N'),
                            alt.Tooltip('accion_tipo:N', title='Acción'),
                            alt.Tooltip('numero_periodo:Q', title='Periodo'),
                            alt.Tooltip('tiempo_segundos:Q', title='Tiempo (s)'),
                            alt.Tooltip('x_period:Q', title='x_period'),
                            alt.Tooltip('y_points:Q', title='Puntos')
                        ]
                    ).transform_filter(brush)
                    chart = chart + points
    
                    # Leyenda interna (esquina superior izquierda dentro del gráfico)
                    try:
                        x_min = float(dfp['x_period'].min()) if len(dfp) else 0.0
                        x_max = float(dfp['x_period'].max()) if len(dfp) else 1.0
                        y_min_v = float(min(dfp['puntosLocal_num'].min(), dfp['puntosVisitante_num'].min())) if len(dfp) else 0.0
                        y_max_v = float(max(dfp['puntosLocal_num'].max(), dfp['puntosVisitante_num'].max())) if len(dfp) else 1.0
                        x_pad = (x_max - x_min) * 0.02
                        y_range = (y_max_v - y_min_v) if (y_max_v - y_min_v) != 0 else 1.0
                        base_x = x_min + x_pad
                        base_y = y_max_v - 0.06 * y_range
                        step = 0.07 * y_range
                        legend_df = pd.DataFrame({
                            'accion_tipo': ['CANASTA-1P','CANASTA-2P','CANASTA-3P'],
                            'lx': [base_x, base_x, base_x],
                            'ly': [base_y, base_y - step, base_y - 2*step],
                            'label': ['1 Punto','2 Puntos','3 Puntos']
                        })
                        leg_points = alt.Chart(legend_df).mark_point(filled=True, size=point_size*0.6).encode(
                            x=alt.X('lx:Q'), y=alt.Y('ly:Q'),
                            color=alt.Color('accion_tipo:N', scale=alt.Scale(
                                domain=['CANASTA-1P','CANASTA-2P','CANASTA-3P'],
                                range=['#fdd835','#1e88e5','#43a047']
                            ), legend=None)
                        ).transform_filter(brush)
                        leg_text = alt.Chart(legend_df).mark_text(align='left', dx=10, dy=4, color='#333333', fontSize=14).encode(
                            x=alt.X('lx:Q'), y=alt.Y('ly:Q'), text='label:N'
                        ).transform_filter(brush)
                        chart = chart + leg_points + leg_text
                    except Exception:
                        pass
    
            # Sin leyenda externa; ya colocamos una interna personalizada
            chart = chart.add_params(brush)
            # Guardar para renderizar en columnas
            chart_full = chart
    
            # Segundo gráfico: Diferencia de puntos (sin puntos de canasta)
            try:
                dfp['DifPuntos_num'] = dfp['puntosLocal_num'] - dfp['puntosVisitante_num']
                # Dominio simétrico alrededor de 0
                dmin = float(dfp['DifPuntos_num'].min()) if not dfp.empty else -1.0
                dmax = float(dfp['DifPuntos_num'].max()) if not dfp.empty else 1.0
                dabs = max(abs(dmin), abs(dmax)) if not dfp.empty else 1.0
                y_domain = [-dabs, dabs]
    
                diff_line = (
                    alt.Chart(dfp)
                    .mark_line(point=False, color='#43a047')
                    .encode(
                        x=alt.X('x_period:Q', title='Tiempo', sort=None),
                        y=alt.Y('DifPuntos_num:Q', title='Diferencia de puntos', scale=alt.Scale(domain=y_domain))
                    )
                    .properties(height=chart_height, title=alt.TitleParams(text='Diferencia de puntos (local - visitante)', anchor='middle'))
                    .transform_filter(brush)
                )
                # Línea horizontal en 0 (más oscura y punteada)
                zero_rule = (
                    alt.Chart(pd.DataFrame({'y':[0]}))
                    .mark_rule(color='#333333', strokeDash=[6,3], strokeWidth=2)
                    .encode(y='y:Q')
                )
    
                # Reglas y etiquetas de cambio de periodo igual que en el primer gráfico
                diff_chart = diff_line + zero_rule
                if period_changes:
                    rules_df2 = pd.DataFrame(period_changes)
                    rules2 = alt.Chart(rules_df2).mark_rule(color='#333333', strokeDash=[8,4], strokeWidth=3).encode(
                        x=alt.X('x_period:Q'),
                        tooltip=[alt.Tooltip('numero_periodo:N', title='Inicio periodo')]
                    ).transform_filter(brush)
                    try:
                        rules_df2['label'] = 'P ' + rules_df2['numero_periodo'].astype(str)
                        # Posicionar al 10% del eje Y simétrico
                        y_pos2 = y_domain[0] + 0.1 * (y_domain[1] - y_domain[0])
                        rules_df2['y_pos'] = y_pos2
                        labels2 = alt.Chart(rules_df2).mark_text(align='left', baseline='bottom', dx=6, color='#333333', fontSize=14).encode(
                            x=alt.X('x_period:Q'), y=alt.Y('y_pos:Q'), text='label:N'
                        ).transform_filter(brush)
                        diff_chart = diff_chart + rules2 + labels2
                    except Exception:
                        diff_chart = diff_chart + rules2
    
                # Marcas de máximo y mínimo de diferencia con etiquetas (dentro del brush actual)
                try:
                    # Máximo dentro del rango seleccionado: última aparición por x_period
                    if dmax >= 0:
                        max_layer = (
                            alt.Chart(dfp)
                            .transform_filter(brush)
                            .transform_joinaggregate(maxDiff='max(DifPuntos_num)')
                            .transform_filter('datum.DifPuntos_num == datum.maxDiff')
                            .transform_window(rn='row_number()', sort=[alt.SortField('x_period', order='descending')])
                            .transform_filter('datum.rn == 1')
                        )
                        max_point = max_layer.mark_point(filled=True, size=140, color='#000000').encode(
                            x='x_period:Q', y='DifPuntos_num:Q'
                        )
                        max_label = (
                            max_layer
                            .transform_calculate(label="'Max dif Local = ' + toString(datum.DifPuntos_num)")
                            .mark_text(color='#000000', fontSize=16, dy=-12)
                            .encode(x='x_period:Q', y='DifPuntos_num:Q', text='label:N')
                        )
                        diff_chart = diff_chart + max_point + max_label
                    # Mínimo dentro del rango seleccionado: última aparición por x_period
                    if dmin <= 0:
                        min_layer = (
                            alt.Chart(dfp)
                            .transform_filter(brush)
                            .transform_joinaggregate(minDiff='min(DifPuntos_num)')
                            .transform_filter('datum.DifPuntos_num == datum.minDiff')
                            .transform_window(rn='row_number()', sort=[alt.SortField('x_period', order='descending')])
                            .transform_filter('datum.rn == 1')
                        )
                        min_point = min_layer.mark_point(filled=True, size=140, color='#000000').encode(
                            x='x_period:Q', y='DifPuntos_num:Q'
                        )
                        min_label = (
                            min_layer
                            .transform_calculate(label="'Max dif Visitante = ' + toString(datum.DifPuntos_num)")
                            .mark_text(color='#000000', fontSize=16, dy=18)
                            .encode(x='x_period:Q', y='DifPuntos_num:Q', text='label:N')
                        )
                        diff_chart = diff_chart + min_point + min_label
                except Exception as e:
                    st.warning(f"No se pudieron calcular las marcas de máximo/mínimo: {e}")
    
                # Marcar tiempos muertos solicitados con puntos del color del equipo
                try:
                    tm = dfp[dfp.get('accion_tipo', '').astype(str) == 'TIEMPO-MUERTO-SOLICITADO'].copy()
                    if not tm.empty:
                        # Asegurar columnas necesarias
                        if 'DifPuntos_num' not in tm.columns:
                            tm['DifPuntos_num'] = pd.to_numeric(tm.get('puntosLocal', 0), errors='coerce').fillna(0) - pd.to_numeric(tm.get('puntosVisitante', 0), errors='coerce').fillna(0)
                        tm['EquipoTM'] = np.where(tm.get('Condicion', '').astype(str).str.upper()=='LOCAL', local_name, visitante_name)
                        tm_color = alt.Scale(domain=[local_name, visitante_name], range=[color_local, color_visitante])
                        tm_points = (
                            alt.Chart(tm)
                            .mark_point(filled=True, size=180, stroke='#000', strokeWidth=0.5)
                            .encode(
                                x=alt.X('x_period:Q'),
                                y=alt.Y('DifPuntos_num:Q'),
                                color=alt.Color('EquipoTM:N', scale=tm_color, legend=alt.Legend(orient='top-left', title='Tiempo muerto')),
                                tooltip=[
                                    alt.Tooltip('EquipoTM:N', title='Equipo'),
                                    alt.Tooltip('numero_periodo:Q', title='Periodo'),
                                    alt.Tooltip('tiempo_segundos:Q', title='Tiempo (s)'),
                                    alt.Tooltip('DifPuntos_num:Q', title='Dif puntos')
                                ]
                            )
                            .transform_filter(brush)
                        )
                        diff_chart = diff_chart + tm_points
                except Exception:
                    pass
    
                diff_chart = diff_chart.add_params(brush)
                # Mostrar ambos gráficos en la misma línea (layout original)
                diff_chart_full = diff_chart
                col_a, col_b = st.columns(2)
                with col_a:
                    st.altair_chart(chart_full, use_container_width=True)
                with col_b:
                    st.altair_chart(diff_chart_full, use_container_width=True)
            except Exception:
                pass
    
        # (Se elimina sección de estadística desde PBP a pedido del usuario)
    
    # Tercer gráfico: Proporción de tiempo por estado del marcador (Local ganando / Empate / Visitante ganando)
    try:
        if not pbp_df.empty:
            # Usar el mismo dfp (filtrado y ordenado) si existe; en su defecto, construirlo
            if 'dfp' not in locals():
                dfp = pbp_df.copy()
                # Quitar columnas legacy con underscore si existen
                dfp = dfp.drop(columns=['puntos_local', 'puntos_visitante'], errors='ignore')
                if 'autoincremental_id' in dfp.columns and 'autoincremental_id_num' not in dfp.columns:
                    dfp['autoincremental_id_num'] = pd.to_numeric(dfp['autoincremental_id'], errors='coerce').fillna(0)
                if 'autoincremental_id_num' in dfp.columns:
                    dfp = dfp.sort_values(by=['_id','numero_periodo','autoincremental_id_num'])
                elif 'tiempo_segundos' in dfp.columns:
                    dfp = dfp.sort_values(by=['_id','numero_periodo','tiempo_segundos'])
                dfp = dfp.reset_index(drop=True)
                dfp['orden'] = np.arange(len(dfp))
                if 'x_period' not in dfp.columns:
                    tiempo_num = pd.to_numeric(dfp.get('tiempo_segundos', np.nan), errors='coerce')
                    periodo_num = pd.to_numeric(dfp.get('numero_periodo', 1), errors='coerce').fillna(1)
                    tiempo_num = tiempo_num.fillna(dfp['orden'])
                    dfp['x_period'] = (600 - tiempo_num) + (periodo_num - 1) * 600
                # Asegurar conversión on-the-fly cuando se use
    
            # Aplicar filtro de periodo si corresponde (para mantener coherencia con los gráficos anteriores)
            if 'sel_periodo' in locals() and sel_periodo != 'TODOS':
                try:
                    sel_val = int(sel_periodo)
                    dfp = dfp[pd.to_numeric(dfp['numero_periodo'], errors='coerce') == sel_val].copy()
                except Exception:
                    pass
    
            # Asegurar tipos y ordenar SIEMPRE por autoincremental_id_num cuando exista
            work = dfp.copy()
            work['numero_periodo'] = pd.to_numeric(work.get('numero_periodo', 1), errors='coerce').fillna(1).astype(int)
            work['x_period'] = pd.to_numeric(work.get('x_period', 0), errors='coerce').fillna(0)
            if 'autoincremental_id_num' not in work.columns and 'autoincremental_id' in work.columns:
                work['autoincremental_id_num'] = pd.to_numeric(work['autoincremental_id'], errors='coerce').fillna(0)
            if 'autoincremental_id_num' in work.columns:
                work = work.sort_values(by=['numero_periodo', 'autoincremental_id_num']).reset_index(drop=True)
            else:
                work = work.sort_values(by=['numero_periodo', 'x_period']).reset_index(drop=True)
    
            t_local = 0.0
            t_empate = 0.0
            t_visit = 0.0
            # Acumular por periodo, incluyendo desde el inicio del periodo hasta la primera jugada (empate)
            # y desde la última jugada hasta el final del periodo usando el signo de la última diferencia
            for per, df_per in work.groupby('numero_periodo'):
                df_per = df_per.sort_values(by=['x_period']).reset_index(drop=True)
                # Inicio y fin absolutos del periodo en x_period
                try:
                    per_int = int(per)
                except Exception:
                    per_int = 1
                x_start = (per_int - 1) * 600.0
                x_end = per_int * 600.0
                # Si hay al menos un evento, sumar tramo inicial como empate hasta la primera jugada
                if len(df_per) > 0:
                    first = df_per.iloc[0]
                    first_x = float(first.get('x_period', x_start))
                    if first_x > x_start:
                        t_empate += (first_x - x_start)
                    # Entre jugadas
                    prev = None
                    for _, r in df_per.iterrows():
                        if prev is not None:
                            if str(prev.get('accion_tipo', '')).upper() != 'FINAL-PERIODO':
                                dt = float(r.get('x_period', 0)) - float(prev.get('x_period', 0))
                                if dt < 0:
                                    dt = abs(float(r.get('tiempo_segundos', 0)) - float(prev.get('tiempo_segundos', 0)))
                                try:
                                    pl_prev = float(prev.get('puntosLocal', 0))
                                    pv_prev = float(prev.get('puntosVisitante', 0))
                                except Exception:
                                    pl_prev = float(pd.to_numeric(prev.get('puntosLocal', 0), errors='coerce'))
                                    pv_prev = float(pd.to_numeric(prev.get('puntosVisitante', 0), errors='coerce'))
                                dif_prev = pl_prev - pv_prev
                                if dif_prev > 0:
                                    t_local += dt
                                elif dif_prev < 0:
                                    t_visit += dt
                                else:
                                    t_empate += dt
                        prev = r
                    # Tramo final hasta el fin del periodo
                    last = df_per.iloc[-1]
                    last_x = float(last.get('x_period', x_end))
                    if x_end > last_x:
                        dif_last = float(last.get('DifPuntos_num', 0))
                        dt_end = x_end - last_x
                        if dif_last > 0:
                            t_local += dt_end
                        elif dif_last < 0:
                            t_visit += dt_end
                        else:
                            t_empate += dt_end
    
            total_t = t_local + t_empate + t_visit
            if total_t > 0:
                # Datos para barra apilada (etiquetas con nombre de equipo en MAYÚSCULAS)
                estados = [f'{local_name.upper()} GANANDO', 'EMPATE', f'{visitante_name.upper()} GANANDO']  # orden fijo solicitado
                segundos = [t_local, t_empate, t_visit]
                # Ajustar color de Empate si coincide con algún color de equipo (gris) para no confundir
                empate_base = '#9e9e9e'
                empate_color = empate_base
                if str(color_local).lower() == empate_base or str(color_visitante).lower() == empate_base:
                    # Aclarar un poco el gris de empate para distinguir
                    empate_color = _adjust_color(empate_base, 1.2)
                colores = [color_local, empate_color, color_visitante]
                fracciones = [s / total_t for s in segundos]
                porcentajes = [f * 100.0 for f in fracciones]
                # Color de texto por segmento para asegurar contraste (etiquetas legibles)
                text_colors = [
                    _text_color_for_bg(color_local),
                    _text_color_for_bg(empate_color),
                    _text_color_for_bg(color_visitante)
                ]
                # Trazo para equipos con color blanco para que se vean los límites
                local_is_white = str(color_local).lower() in ('#ffffff', '#fff', 'white')
                visit_is_white = str(color_visitante).lower() in ('#ffffff', '#fff', 'white')
                strokes = [
                    '#000000' if local_is_white else None,
                    None,
                    '#000000' if visit_is_white else None
                ]
                start_fracs = [0.0, fracciones[0], fracciones[0] + fracciones[1]]
                mid_fracs = [start_fracs[i] + fracciones[i] / 2.0 for i in range(3)]
                # Color de etiqueta: negro por defecto, blanco si el color del equipo es negro
                def is_black(c: str) -> bool:
                    c = str(c).strip().lower()
                    return c in ('#000000', '#000', 'black')
                label_colors = [
                    '#ffffff' if is_black(color_local) else '#000000',  # Local
                    '#000000',                                         # Empate
                    '#ffffff' if is_black(color_visitante) else '#000000'  # Visitante
                ]
                df_bar = pd.DataFrame({
                    'Estado': estados,
                    'EstadoOrden': [0, 1, 2],  # asegurar orden en la pila (izq->der): Local, Empate, Visitante
                    'Segundos': segundos,
                    'Fraccion': fracciones,
                    'Porcentaje': porcentajes,
                    'StartFrac': start_fracs,
                    'MidFrac': mid_fracs,
                    'Stroke': strokes,
                    'TextColor': text_colors,
                    'LabelColor': label_colors,
                    'y': [' ', ' ', ' ']  # única fila para que sea una barra única
                })
    
                # Separador visual entre filas
                st.write("")
    
                bar = (
                    alt.Chart(df_bar)
                    .mark_bar()
                    .encode(
                        y=alt.Y('y:N', axis=None, scale=alt.Scale(paddingInner=0, paddingOuter=0)),
                        x=alt.X('Segundos:Q', stack='normalize', axis=None),
                        color=alt.Color(
                            'Estado:N',
                            scale=alt.Scale(domain=estados, range=colores),
                            legend=alt.Legend(orient='top', direction='horizontal', title=None, labelLimit=1000)
                        ),
                        order=alt.Order('EstadoOrden:Q'),
                        stroke=alt.Color('Stroke:N', scale=None, legend=None),
                        tooltip=[
                            alt.Tooltip('Estado:N'),
                            alt.Tooltip('Porcentaje:Q', format='.0f', title='Porcentaje (%)')
                        ]
                    )
                    .properties(height=180, title=alt.TitleParams(text='Tiempo por estado del marcador', anchor='middle'))
                )
                # Etiquetas de porcentaje centradas dentro de cada segmento
                labels = (
                    alt.Chart(df_bar)
                    .mark_text(fontSize=18, fontWeight=700, align='center', stroke='#dddddd', strokeWidth=0.35)
                    .encode(
                        y=alt.Y('y:N'),
                        x=alt.X('MidFrac:Q', axis=None, scale=alt.Scale(domain=[0,1])),
                        color=alt.Color('LabelColor:N', scale=None, legend=None),
                        order=alt.Order('EstadoOrden:Q'),
                        detail='Estado:N',
                        text=alt.Text('Fraccion:Q', format='.0%')
                    )
                )
                # Mostrar la barra en una nueva fila de columnas, ocupando la primera columna
                row2_col1, row2_col2 = st.columns(2)
                with row2_col1:
                    st.altair_chart(bar + labels, use_container_width=True)
                with row2_col2:
                    # Resumen textual: mejor racha pura (X-0) y mayor sequía para cada equipo, con rango temporal
                    try:
                        timeline = dfp[['x_period','puntosLocal','puntosVisitante','numero_periodo','tiempo_segundos']].copy()
                        if 'autoincremental_id_num' not in dfp.columns and 'autoincremental_id' in dfp.columns:
                            dfp['autoincremental_id_num'] = pd.to_numeric(dfp['autoincremental_id'], errors='coerce').fillna(0)
                        if 'autoincremental_id_num' in dfp.columns:
                            timeline = timeline.join(dfp['autoincremental_id_num'])
                            timeline = timeline.sort_values(by=['numero_periodo','autoincremental_id_num']).reset_index(drop=True)
                        else:
                            timeline = timeline.sort_values(by=['numero_periodo','x_period']).reset_index(drop=True)
                        # Convertir a numérico para cálculos
                        timeline['puntosLocal'] = pd.to_numeric(timeline['puntosLocal'], errors='coerce').fillna(0)
                        timeline['puntosVisitante'] = pd.to_numeric(timeline['puntosVisitante'], errors='coerce').fillna(0)
                        if len(timeline) >= 2:
                            # Longest drought per team: track start/end rows
                            first_x = float(timeline.iloc[0]['x_period'])
                            last_x = float(timeline.iloc[-1]['x_period'])
                            last_change_L = first_x
                            last_change_V = first_x
                            drought_start_L = timeline.iloc[0]
                            drought_start_V = timeline.iloc[0]
                            drought_best_L = (0.0, timeline.iloc[0], timeline.iloc[0])
                            drought_best_V = (0.0, timeline.iloc[0], timeline.iloc[0])
                            prev_PL = float(timeline.iloc[0]['puntosLocal'])
                            prev_PV = float(timeline.iloc[0]['puntosVisitante'])
                            for i in range(1, len(timeline)):
                                rowi = timeline.iloc[i]
                                x = float(rowi['x_period'])
                                PL = float(rowi['puntosLocal'])
                                PV = float(rowi['puntosVisitante'])
                                if PL != prev_PL:
                                    dur = x - last_change_L
                                    if dur > drought_best_L[0]:
                                        drought_best_L = (dur, drought_start_L, rowi)
                                    last_change_L = x
                                    drought_start_L = rowi
                                if PV != prev_PV:
                                    dur = x - last_change_V
                                    if dur > drought_best_V[0]:
                                        drought_best_V = (dur, drought_start_V, rowi)
                                    last_change_V = x
                                    drought_start_V = rowi
                                prev_PL, prev_PV = PL, PV
                            # considerar sequía hasta el final del juego
                            end_dummy = timeline.iloc[-1]
                            dur = last_x - last_change_L
                            if dur > drought_best_L[0]:
                                drought_best_L = (dur, drought_start_L, end_dummy)
                            dur = last_x - last_change_V
                            if dur > drought_best_V[0]:
                                drought_best_V = (dur, drought_start_V, end_dummy)
    
                            # Best pure scoring run per team: X-0 (oponente no anota en ese tramo)
                            def best_pure_run(favor_is_local: bool):
                                prev = timeline.iloc[0]
                                prev_PL = float(prev['puntosLocal'])
                                prev_PV = float(prev['puntosVisitante'])
                                run_pts = 0.0
                                run_start = None
                                best_pts = 0.0
                                best_start = None
                                best_end = None
                                for i in range(1, len(timeline)):
                                    cur = timeline.iloc[i]
                                    PL = float(cur['puntosLocal'])
                                    PV = float(cur['puntosVisitante'])
                                    dL = PL - prev_PL
                                    dV = PV - prev_PV
                                    # Determinar si anota el equipo a analizar y el otro NO
                                    if favor_is_local:
                                        scores = (dL > 0 and dV == 0)
                                        rival_scores = (dV > 0)
                                        inc = dL
                                    else:
                                        scores = (dV > 0 and dL == 0)
                                        rival_scores = (dL > 0)
                                        inc = dV
                                    if scores:
                                        if run_start is None:
                                            run_start = timeline.iloc[i-1]
                                            run_pts = 0.0
                                        run_pts += inc
                                    if rival_scores or (dL > 0 and dV > 0):
                                        # Cierra la racha pura si estaba abierta
                                        if run_start is not None and run_pts > best_pts:
                                            best_pts, best_start, best_end = run_pts, run_start, timeline.iloc[i]
                                        run_start = None
                                        run_pts = 0.0
                                    prev_PL, prev_PV = PL, PV
                                # cerrar al final
                                if run_start is not None and run_pts > best_pts:
                                    best_pts, best_start, best_end = run_pts, run_start, timeline.iloc[-1]
                                return int(best_pts), best_start, best_end
    
                            best_L_pts, sL, eL = best_pure_run(True)
                            best_V_pts, sV, eV = best_pure_run(False)
    
                            def fmt_when(srow, erow):
                                try:
                                    p1 = int(srow['numero_periodo'])
                                    t1 = float(srow['tiempo_segundos']) if pd.notna(srow['tiempo_segundos']) else 0.0
                                    p2 = int(erow['numero_periodo'])
                                    t2 = float(erow['tiempo_segundos']) if pd.notna(erow['tiempo_segundos']) else 0.0
                                    def mmss(x):
                                        m = int(x // 60)
                                        s = int(x % 60)
                                        return f"{m}:{s:02d}"
                                    return f"P{p1} {mmss(t1)} → P{p2} {mmss(t2)}"
                                except Exception:
                                    return ""
    
                            def fmt_drought_tuple(dt):
                                sec = dt[0]
                                srow = dt[1]
                                erow = dt[2]
                                m = int(sec // 60)
                                s = int(sec % 60)
                                return f"{m}m {s:02d}s ({fmt_when(srow, erow)})"
    
                            # Tarjetas con color de equipo
                            card_style = "border-radius:10px;padding:12px 14px;margin:6px 0;box-shadow:0 1px 5px rgba(0,0,0,.08)"
                            # Calcular puntos titulares/suplentes por equipo cruzando PBP con planillas (respetando filtro de periodo ya aplicado a dfp)
                            try:
                                pbp_pts = dfp.copy()
                                # Columnas candidatas
                                dorsal_candidates = ['dorsal','numero','nro','número','numero_camiseta','n_camisa']
                                pbp_dorsal_col = _first_col(pbp_pts, dorsal_candidates)
                                if pbp_dorsal_col is None or pbp_dorsal_col not in pbp_pts.columns:
                                    raise ValueError('PBP sin columna de dorsal reconocible')
                                # Normalizar Condicion
                                if 'Condicion' in pbp_pts.columns:
                                    pbp_pts['Condicion'] = pbp_pts['Condicion'].astype(str).str.upper().fillna('')
                                else:
                                    pbp_pts['Condicion'] = ''
                                # Mapear puntos por tipo de canasta
                                at = pbp_pts.get('accion_tipo', '').astype(str).str.upper()
                                pts_event = np.where(at == 'CANASTA-1P', 1,
                                              np.where(at == 'CANASTA-2P', 2,
                                              np.where(at == 'CANASTA-3P', 3, 0)))
                                pbp_pts['__puntos'] = pts_event
                                # Agregar por Condicion y dorsal
                                jg_grp = (
                                    pbp_pts.groupby(['Condicion', pbp_dorsal_col], as_index=False)['__puntos']
                                    .sum()
                                    .rename(columns={pbp_dorsal_col: '__dorsal'})
                                )
                            except Exception:
                                jg_grp = pd.DataFrame(columns=['Condicion','__dorsal','__puntos'])
    
                            # Obtener planillas por equipo y detectar titularidad y dorsal
                            def titular_sets(planilla_df: pd.DataFrame):
                                if planilla_df is None or planilla_df.empty:
                                    return set(), set()
                                dcol = _first_col(planilla_df, ['dorsal','numero','nro','número','numero_camiseta','n_camisa'])
                                tcol = _first_col(planilla_df, ['quintetotitular','quinteto_titular','QuintetoTitular','quintetoTitular','titular','es_titular'])
                                if not dcol or dcol not in planilla_df.columns or not tcol or tcol not in planilla_df.columns:
                                    return set(), set()
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
                                supl = set(pd.to_numeric(dfp2.loc[~dfp2['__titular'], dcol], errors='coerce').dropna().astype(int).tolist())
                                return tit, supl
    
                            tit_loc, supl_loc = titular_sets(est_loc_df)
                            tit_vis, supl_vis = titular_sets(est_vis_df)
    
                            # Función para sumar puntos por conjunto de dorsales
                            def sum_pts(cond: str, dorsales: set) -> float:
                                if jg_grp is None or jg_grp.empty or not dorsales:
                                    return 0.0
                                tmp = jg_grp[jg_grp['Condicion'].astype(str).str.upper()==cond.upper()].copy()
                                if tmp.empty:
                                    return 0.0
                                tmp['__d'] = pd.to_numeric(tmp['__dorsal'], errors='coerce').astype('Int64')
                                s = tmp[tmp['__d'].isin(list(dorsales))]['__puntos'].sum()
                                try:
                                    return float(s)
                                except Exception:
                                    return 0.0
    
                            # Calcular puntos por equipo
                            pts_tit_loc = sum_pts('LOCAL', tit_loc)
                            pts_sup_loc = sum_pts('LOCAL', supl_loc)
                            pts_tit_vis = sum_pts('VISITANTE', tit_vis)
                            pts_sup_vis = sum_pts('VISITANTE', supl_vis)
    
                            # Puntos en la zona (def: apariciones de CANASTA-2P * 2) respetando filtros de periodo aplicados a dfp
                            try:
                                df_zone = dfp.copy()
                                if 'accion_tipo' in df_zone.columns:
                                    at2 = df_zone['accion_tipo'].astype(str).str.upper()
                                    df_zone['__is_c2p'] = (at2 == 'CANASTA-2P')
                                else:
                                    df_zone['__is_c2p'] = False
                                if 'Condicion' in df_zone.columns:
                                    df_zone['Condicion'] = df_zone['Condicion'].astype(str).str.upper().fillna('')
                                # Detectar columna de zona
                                zona_candidates = ['zona', 'zona_tiro', 'tiro_zona', 'zona_accion', 'zona_tiro_codigo', 'zonaCodigo', 'zona_codigo']
                                zona_col = None
                                for zc in zona_candidates:
                                    if zc in df_zone.columns:
                                        zona_col = zc
                                        break
                                if zona_col is not None:
                                    zvals = df_zone[zona_col].astype(str).fillna('')
                                    df_zone['__is_z1'] = zvals.str.upper().str.startswith('Z1-')
                                else:
                                    df_zone['__is_z1'] = False
                                mask_loc = (df_zone['Condicion']=='LOCAL') & df_zone['__is_c2p'] & df_zone['__is_z1']
                                mask_vis = (df_zone['Condicion']=='VISITANTE') & df_zone['__is_c2p'] & df_zone['__is_z1']
                                cnt_loc = int(df_zone[mask_loc].shape[0])
                                cnt_vis = int(df_zone[mask_vis].shape[0])
                                pzona_loc = 2 * cnt_loc
                                pzona_vis = 2 * cnt_vis
                            except Exception:
                                pzona_loc = 0
                                pzona_vis = 0
    
                            # Tarjetas con colores de equipo
                            st.markdown(f"""
                            <div style='background:{color_local}; color:{tc_local}; padding:16px; border-radius:10px; margin-bottom:12px;'>
                                <div style='font-weight:700; font-size:18px; margin-bottom:10px;'>{local_name}</div>
                                <div style='margin:4px 0;'>Puntos titulares: <strong>{int(pts_tit_loc)}</strong></div>
                                <div style='margin:4px 0;'>Puntos suplentes: <strong>{int(pts_sup_loc)}</strong></div>
                                <div style='margin:4px 0;'>Puntos en la zona (Z1): <strong>{int(pzona_loc)}</strong></div>
                                <div style='margin:4px 0;'>Mejor racha: <strong>{best_L_pts}-0</strong> ({fmt_when(sL, eL) if sL is not None else ''})</div>
                                <div style='margin:4px 0;'>Mayor sequía: <strong>{fmt_drought_tuple(drought_best_L)}</strong></div>
                            </div>
                            """, unsafe_allow_html=True)
    
                            st.markdown(f"""
                            <div style='background:{color_visitante}; color:{tc_visitante}; padding:16px; border-radius:10px;'>
                                <div style='font-weight:700; font-size:18px; margin-bottom:10px;'>{visitante_name}</div>
                                <div style='margin:4px 0;'>Puntos titulares: <strong>{int(pts_tit_vis)}</strong></div>
                                <div style='margin:4px 0;'>Puntos suplentes: <strong>{int(pts_sup_vis)}</strong></div>
                                <div style='margin:4px 0;'>Puntos en la zona (Z1): <strong>{int(pzona_vis)}</strong></div>
                                <div style='margin:4px 0;'>Mejor racha: <strong>{best_V_pts}-0</strong> ({fmt_when(sV, eV) if sV is not None else ''})</div>
                                <div style='margin:4px 0;'>Mayor sequía: <strong>{fmt_drought_tuple(drought_best_V)}</strong></div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info('Sin suficientes eventos para resumen de rachas')
                    except Exception as e:
                        st.warning(f"No se pudo calcular el resumen de rachas: {e}")
            else:
                st.info('No se pudo calcular el tiempo por estado del marcador (total de tiempo = 0).')
    except Exception as e:
        st.warning(f"No se pudo renderizar la barra de estados del marcador: {e}")
    
    # (Se eliminó la fila 3 comparativa en Resumen a pedido del usuario)
    
    # Comparativa de totales (desde "estadisticas por jugador")
    try:
        jg_tot = tablas.get('jugadoresAgregado', pd.DataFrame()).copy()
        if isinstance(jg_tot, pd.DataFrame) and not jg_tot.empty:
            # Respetar filtro de periodo de la pestaña (si existe columna y selección)
            if 'sel_periodo' in locals() and sel_periodo != 'TODOS':
                if 'numero_periodo' in jg_tot.columns:
                    try:
                        sel_val = int(sel_periodo)
                        jg_tot = jg_tot[pd.to_numeric(jg_tot['numero_periodo'], errors='coerce') == sel_val].copy()
                    except Exception:
                        pass
            # Normalizar Condicion
            if 'Condicion' in jg_tot.columns:
                jg_tot['Condicion'] = jg_tot['Condicion'].astype(str).str.upper().fillna('')
            # Mapeo estricto como en la tabla de jugadores
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
            ]
            for src, dst in strict_map:
                if src in jg_tot.columns:
                    jg_tot[dst] = pd.to_numeric(jg_tot[src], errors='coerce').fillna(0)
                elif dst not in jg_tot.columns:
                    jg_tot[dst] = 0
            # Derivadas
            jg_tot['rebotetotal'] = jg_tot['rebotedefensivo'] + jg_tot['reboteofensivo']
            jg_tot['puntos'] = jg_tot['canasta1p'] + 2*jg_tot['canasta2p'] + 3*jg_tot['canasta3p']
            # Plus-minus
            pm_cands = ['diferencia', 'plusminus', 'plus_minus', '+-']
            pm_series = None
            for pmc in pm_cands:
                if pmc in jg_tot.columns:
                    s = pd.to_numeric(jg_tot[pmc], errors='coerce').fillna(0)
                    pm_series = s if pm_series is None else (pm_series + s)
            if pm_series is None:
                jg_tot['pm'] = 0
            else:
                jg_tot['pm'] = pm_series
    
            # Sumar por condicion
            vars_keep = {
                'puntos': 'Puntos',
                'rebotedefensivo': 'Rebote Def.',
                'reboteofensivo': 'Rebote Of.',
                'rebotetotal': 'Rebotes Totales',
                'asistencias': 'Asistencias',
                'perdidas': 'Pérdidas',
                'recuperaciones': 'Recuperaciones',
                'pm': '+/-',
            }
            agg_tot = (
                jg_tot.groupby('Condicion')[list(vars_keep.keys())]
                .sum(numeric_only=True)
                .reset_index()
            )
            # Asegurar presencia de ambas filas LOCAL y VISITANTE con 0s si faltan
            needed = ['LOCAL', 'VISITANTE']
            for cond in needed:
                if cond not in agg_tot['Condicion'].astype(str).tolist():
                    row0 = {**{k: 0.0 for k in vars_keep.keys()}, 'Condicion': cond}
                    agg_tot = pd.concat([agg_tot, pd.DataFrame([row0])], ignore_index=True)
            # Preparar tidy para Altair
            def equipo_name(cond):
                return local_name if str(cond).upper()=='LOCAL' else (visitante_name if str(cond).upper()=='VISITANTE' else str(cond))
            rows = []
            for _, r in agg_tot.iterrows():
                eq = equipo_name(r['Condicion'])
                for k, label in vars_keep.items():
                    rows.append({'Equipo': eq, 'Variable': label, 'Valor': float(r.get(k, 0))})
            df_vars = pd.DataFrame(rows)
            if not df_vars.empty:
                # Colores por equipo y helper de chart
                color_scale = alt.Scale(domain=[local_name, visitante_name], range=[color_local, color_visitante])
    
                # Construir agregados por equipo para porcentajes
                def ensure_cols(df, cols):
                    for c in cols:
                        if c not in df.columns:
                            df[c] = 0
                    return df
                jg_cnt = jg_tot.copy()
                jg_cnt = ensure_cols(jg_cnt, ['CANASTA-1P','TIRO1-FALLADO','CANASTA-2P','TIRO2-FALLADO','CANASTA-3P','TIRO3-FALLADO'])
                grp_cnt = (
                    jg_cnt.groupby('Condicion')[['CANASTA-1P','TIRO1-FALLADO','CANASTA-2P','TIRO2-FALLADO','CANASTA-3P','TIRO3-FALLADO']]
                    .sum(numeric_only=True)
                    .reset_index()
                )
                # Asegurar filas para ambos equipos
                for cond in ['LOCAL','VISITANTE']:
                    if cond not in grp_cnt['Condicion'].astype(str).tolist():
                        grp_cnt = pd.concat([
                            grp_cnt,
                            pd.DataFrame([{'Condicion': cond, 'CANASTA-1P':0,'TIRO1-FALLADO':0,'CANASTA-2P':0,'TIRO2-FALLADO':0,'CANASTA-3P':0,'TIRO3-FALLADO':0}])
                        ], ignore_index=True)
    
                def equipo_name(cond):
                    return local_name if str(cond).upper()=='LOCAL' else (visitante_name if str(cond).upper()=='VISITANTE' else str(cond))
    
                def bar_chart_for(variable_label, valores_dict, y_title='Total', is_percent=False):
                    data = [{'Equipo': k, 'Valor': valores_dict.get(k, 0)} for k in [local_name, visitante_name]]
                    dfc = pd.DataFrame(data)
                    enc_y = alt.Y('Valor:Q', title=y_title, scale=alt.Scale(domain=[0,100])) if is_percent else alt.Y('Valor:Q', title=y_title)
                    # Barras
                    bars = (
                        alt.Chart(dfc)
                        .mark_bar(stroke='#000000', strokeWidth=1)
                        .encode(
                            x=alt.X('Equipo:N', title=None, sort=[local_name, visitante_name], axis=alt.Axis(labelAngle=315)),
                            y=enc_y,
                            color=alt.Color('Equipo:N', scale=color_scale, legend=None),
                            tooltip=[alt.Tooltip('Equipo:N'), alt.Tooltip('Valor:Q', format='.0f' if not is_percent else '.0f')]
                        )
                    )
                    # Etiquetas
                    if is_percent:
                        text = (
                            alt.Chart(dfc)
                            .transform_calculate(label="toString(round(datum.Valor)) + '%'")
                            .mark_text(dy=-6, fontSize=16, fontWeight='bold')
                            .encode(
                                x=alt.X('Equipo:N', sort=[local_name, visitante_name], axis=alt.Axis(labelAngle=315)),
                                y=enc_y,
                                text='label:N',
                                color=alt.value('#000000')
                            )
                        )
                    else:
                        text = (
                            alt.Chart(dfc)
                            .mark_text(dy=-6, fontSize=16, fontWeight='bold')
                            .encode(
                                x=alt.X('Equipo:N', sort=[local_name, visitante_name], axis=alt.Axis(labelAngle=315)),
                                y=enc_y,
                                text=alt.Text('Valor:Q', format='.0f'),
                                color=alt.value('#000000')
                            )
                        )
                    return (bars + text).properties(height=340, title=variable_label)
    
                # Calcular valores por equipo
                def get_total(cond, col):
                    try:
                        return float(agg_tot.loc[agg_tot['Condicion'].astype(str)==cond, col].sum())
                    except Exception:
                        return 0.0
                # Puntos y rebotes/asist/perd/recup
                vals_puntos = {equipo_name('LOCAL'): get_total('LOCAL','puntos'), equipo_name('VISITANTE'): get_total('VISITANTE','puntos')}
                vals_rd = {equipo_name('LOCAL'): get_total('LOCAL','rebotedefensivo'), equipo_name('VISITANTE'): get_total('VISITANTE','rebotedefensivo')}
                vals_ro = {equipo_name('LOCAL'): get_total('LOCAL','reboteofensivo'), equipo_name('VISITANTE'): get_total('VISITANTE','reboteofensivo')}
                vals_rt = {equipo_name('LOCAL'): get_total('LOCAL','rebotetotal'), equipo_name('VISITANTE'): get_total('VISITANTE','rebotetotal')}
                vals_ast = {equipo_name('LOCAL'): get_total('LOCAL','asistencias'), equipo_name('VISITANTE'): get_total('VISITANTE','asistencias')}
                vals_per = {equipo_name('LOCAL'): get_total('LOCAL','perdidas'), equipo_name('VISITANTE'): get_total('VISITANTE','perdidas')}
                vals_rec = {equipo_name('LOCAL'): get_total('LOCAL','recuperaciones'), equipo_name('VISITANTE'): get_total('VISITANTE','recuperaciones')}
    
                # Porcentajes
                def pct(conv, fall):
                    a = conv + fall
                    return (conv / a * 100.0) if a > 0 else 0.0
                def get_pct(cond, conv_col, miss_col):
                    row = grp_cnt[grp_cnt['Condicion'].astype(str)==cond]
                    c = float(row[conv_col].sum()) if not row.empty else 0.0
                    m = float(row[miss_col].sum()) if not row.empty else 0.0
                    return pct(c, m)
                vals_1p = {
                    equipo_name('LOCAL'): get_pct('LOCAL','CANASTA-1P','TIRO1-FALLADO'),
                    equipo_name('VISITANTE'): get_pct('VISITANTE','CANASTA-1P','TIRO1-FALLADO')
                }
                vals_2p = {
                    equipo_name('LOCAL'): get_pct('LOCAL','CANASTA-2P','TIRO2-FALLADO'),
                    equipo_name('VISITANTE'): get_pct('VISITANTE','CANASTA-2P','TIRO2-FALLADO')
                }
                vals_3p = {
                    equipo_name('LOCAL'): get_pct('LOCAL','CANASTA-3P','TIRO3-FALLADO'),
                    equipo_name('VISITANTE'): get_pct('VISITANTE','CANASTA-3P','TIRO3-FALLADO')
                }
    
                # Render en grilla 5 por fila
                st.write("")
                st.subheader('Comparativas por variable (Local vs Visitante)')
                row1 = st.columns(5)
                with row1[0]:
                    st.altair_chart(bar_chart_for('Puntos', vals_puntos, 'Total'), use_container_width=True)
                with row1[1]:
                    st.altair_chart(bar_chart_for('%1P', vals_1p, '%', is_percent=True), use_container_width=True)
                with row1[2]:
                    st.altair_chart(bar_chart_for('%2P', vals_2p, '%', is_percent=True), use_container_width=True)
                with row1[3]:
                    st.altair_chart(bar_chart_for('%3P', vals_3p, '%', is_percent=True), use_container_width=True)
                with row1[4]:
                    st.altair_chart(bar_chart_for('Reb. Def.', vals_rd, 'Total'), use_container_width=True)
    
                row2 = st.columns(5)
                with row2[0]:
                    st.altair_chart(bar_chart_for('Reb. Of.', vals_ro, 'Total'), use_container_width=True)
                with row2[1]:
                    st.altair_chart(bar_chart_for('Reb. Tot.', vals_rt, 'Total'), use_container_width=True)
                with row2[2]:
                    st.altair_chart(bar_chart_for('Asistencias', vals_ast, 'Total'), use_container_width=True)
                with row2[3]:
                    st.altair_chart(bar_chart_for('Pérdidas', vals_per, 'Total'), use_container_width=True)
                with row2[4]:
                    st.altair_chart(bar_chart_for('Recuperaciones', vals_rec, 'Total'), use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo construir la comparativa de totales: {e}")
    
    # Pestaña Estadisticas por jugador (desde jugadoresAgregado por jugador, con Totales y derivadas)
