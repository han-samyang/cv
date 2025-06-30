from unittest import result
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from scipy.signal import find_peaks, savgol_filter
import base64
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage

# scipy 버전에 따른 trapz 함수 import (호환성 처리)
try:
    from scipy.integrate import trapezoid as integrate_trapz
except ImportError:
    try:
        from scipy.integrate import trapz as integrate_trapz
    except ImportError:
        try:
            from numpy import trapezoid as integrate_trapz
        except AttributeError:
            from numpy import trapz as integrate_trapz

# 페이지 설정
st.set_page_config(
    page_title="🔋 CV 데이터 분석 - 배터리 전해액 첨가제 개발",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일링
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    margin: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

def smooth_data(data, window_length=11, polyorder=3):
    """Savitzky-Golay 필터를 사용한 데이터 스무딩"""
    try:
        if len(data) < window_length:
            window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
            if window_length < 3:
                return data
            if window_length < polyorder:
                polyorder = window_length - 1 if window_length > 1 else 1
        return savgol_filter(data, window_length, polyorder)
    except Exception as e:
        return data

def detect_cycles(voltage_array, min_drop=0.5, min_points=30):
    """사이클 탐지 함수"""
    cycles = []
    i = 0
    cycle_id = 1
    n = len(voltage_array)

    while i < n - 2:
        start_idx = i
        while i < n - 1 and voltage_array[i + 1] < voltage_array[i]:
            i += 1
        min_idx = i
        while i < n - 1 and voltage_array[i + 1] > voltage_array[i]:
            i += 1
        end_idx = i

        if min_idx > start_idx and end_idx > min_idx:
            v_start = voltage_array[start_idx]
            v_min = voltage_array[min_idx]
            v_end = voltage_array[end_idx]
            total_points = end_idx - start_idx + 1

            if (v_start - v_min >= min_drop) and (total_points >= min_points):
                cycles.append({
                    'cycle': cycle_id,
                    'reduction_start_idx': start_idx,
                    'reduction_end_idx': min_idx,
                    'oxidation_end_idx': end_idx,
                    'reduction_start_voltage': v_start,
                    'reduction_end_voltage': v_min,
                    'oxidation_end_voltage': v_end,
                    'total_points': total_points
                })
                cycle_id += 1
        else:
            i += 1
    return cycles

def find_cv_cycles_and_segments(df):
    """CV 데이터를 사이클별로 분리"""
    try:
        voltage = df['voltage'].values
        cycles_data = detect_cycles(voltage)
        
        cycles = {}
        
        for cycle_info in cycles_data:
            cycle_id = cycle_info['cycle']
            
            # 환원 구간
            reduction_start = cycle_info['reduction_start_idx']
            reduction_end = cycle_info['reduction_end_idx']
            reduction_df = df.iloc[reduction_start:reduction_end+1].copy()
            
            cycles[f"Cycle_{cycle_id}_1_Reduction"] = {
                'data': reduction_df,
                'type': 'reduction',
                'cycle': cycle_id,
                'segment': 1
            }
            
            # 산화 구간
            oxidation_end = cycle_info['oxidation_end_idx']
            oxidation_df = df.iloc[reduction_end:oxidation_end+1].copy()
            
            cycles[f"Cycle_{cycle_id}_2_Oxidation"] = {
                'data': oxidation_df,
                'type': 'oxidation',
                'cycle': cycle_id,
                'segment': 2
            }
        
        if not cycles:
            cycles["Full_Cycle_1_Complete"] = {
                'data': df.copy(),
                'type': 'complete',
                'cycle': 1,
                'segment': 1
            }
        
        return cycles
        
    except Exception as e:
        st.error(f"사이클 분리 중 오류: {str(e)}")
        return {}

def detect_reduction_potentials(voltage, current):
    """환원전위 탐지 - 1차 미분 절댓값 변화 최대 방법만 사용"""
    try:
        # 1.5V 이하 구간만 사용
        valid_mask = voltage <= 1.5
        
        # 유효한 데이터가 너무 적으면 범위 확장
        if np.sum(valid_mask) < 20:
            valid_mask = voltage <= 2.0  # 2V 이하로 확장
        
        if np.sum(valid_mask) < 10:
            valid_mask = voltage <= voltage.max()  # 전체 데이터 사용
        
        voltage_clean = voltage[valid_mask]
        current_clean = current[valid_mask]
        
        if len(voltage_clean) < 10:
            voltage_clean = voltage
            current_clean = current
        
        # 데이터 스무딩
        window_length = min(11, len(current_clean)//3*2+1 if len(current_clean)//3*2+1 >= 5 else 5)
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 5:
            window_length = 5
        
        y_smooth = smooth_data(current_clean, window_length=window_length, polyorder=3)
        
        # 1차 미분 계산
        dydx = np.gradient(y_smooth, voltage_clean)
        
        results = []
        
        # 방법 1: 1차 미분값의 절댓값을 취하고, 절댓값 변화가 가장 큰 포인트
        abs_dydx = np.abs(dydx)
        
        # 1차 미분 절댓값의 변화량 계산
        abs_dydx_changes = []
        for i in range(1, len(abs_dydx)):
            change = abs(abs_dydx[i] - abs_dydx[i-1])
            abs_dydx_changes.append(change)
        
        if len(abs_dydx_changes) > 0:
            max_abs_change_idx = np.argmax(abs_dydx_changes) + 1
            max_abs_change_idx = min(max_abs_change_idx, len(voltage_clean) - 1)
            results.append(('1차 미분 절댓값 변화 최대', voltage_clean[max_abs_change_idx], current_clean[max_abs_change_idx]))
        
        # 결과가 없으면 전류 최소점 사용 (무조건 결과 보장)
        if len(results) == 0:
            idx_min_current = np.argmin(current_clean)
            results.append(('전류 최소점', voltage_clean[idx_min_current], current_clean[idx_min_current]))
        
        # 결과 데이터프레임 생성
        df_results = pd.DataFrame(results, columns=['탐지방법', '환원전위(V)', '전류(A)'])
        df_results['전류밀도(mA)'] = df_results['전류(A)'].abs() * 1000
        
        # 첫 번째 방법 우선
        best_result = {
            'voltage': results[0][1],
            'current': results[0][2],
            'method': results[0][0],
            'current_density': abs(results[0][2]) * 1000,
            'all_methods': df_results
        }
        
        return best_result
        
    except Exception as e:
        # 예외 발생 시에도 결과 보장
        min_idx = np.argmin(current)
        return {
            'voltage': voltage[min_idx],
            'current': current[min_idx],
            'method': '예외처리-최소점',
            'current_density': abs(current[min_idx]) * 1000,
            'all_methods': pd.DataFrame([['예외처리-최소점', voltage[min_idx], current[min_idx], abs(current[min_idx]) * 1000]], 
                                      columns=['탐지방법', '환원전위(V)', '전류(A)', '전류밀도(mA)'])
        }

def find_reduction_potential_first_cycle(cycles):
    """첫 번째 사이클의 환원 구간에서 환원전위 찾기"""
    try:
        first_cycle_reduction = None
        for name, cycle_info in cycles.items():
            if cycle_info['type'] == 'reduction' and cycle_info['cycle'] == 1:
                first_cycle_reduction = cycle_info['data']
                break
        
        if first_cycle_reduction is None or len(first_cycle_reduction) < 10:
            return None
            
        voltage = first_cycle_reduction['voltage'].values
        current = first_cycle_reduction['current'].values
        
        return detect_reduction_potentials(voltage, current)
        
    except Exception as e:
        st.error(f"첫 번째 사이클 환원전위 계산 중 오류: {str(e)}")
        return None

def calculate_capacity(df, scan_rate_mv_s=1):
    """용량 계산"""
    try:
        voltage = df['voltage'].values
        current = df['current'].values
        capacity_mah_g = abs(integrate_trapz(current, voltage)) / (scan_rate_mv_s / 1000) * 1000 / 3600
        return capacity_mah_g
    except Exception as e:
        return 0.0

def classify_sheet_name(sheet_name):
    """시트 분류 함수"""
    name_lower = sheet_name.lower()
    if 'ref' in name_lower or 'reference' in name_lower or '데이터_1' in name_lower:
        return 'Reference'
    return 'Additive'

def load_cv_data_from_excel(uploaded_file):
    """Excel 파일에서 CV 데이터 로드"""
    datasets = {}
    
    try:
        if uploaded_file.name.endswith('.csv'):
            content = uploaded_file.getvalue().decode('utf-8')
            df = pd.read_csv(io.StringIO(content))
            
            processed_df = process_cv_dataframe(df, uploaded_file.name)
            if processed_df is not None:
                sample_name = uploaded_file.name.split('.')[0]
                sample_type = classify_sheet_name(sample_name)
                datasets[f"{sample_type} - {sample_name}"] = processed_df
            
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            for sheet_name in sheet_names:
                if '정보' in sheet_name.lower() or 'info' in sheet_name.lower():
                    continue
                
                try:
                    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                    processed_df = process_cv_dataframe(df, sheet_name)
                    
                    if processed_df is not None:
                        sample_type = classify_sheet_name(sheet_name)
                        datasets[f"{sample_type} - {sheet_name}"] = processed_df
                            
                except Exception as e:
                    continue
        
        return datasets
        
    except Exception as e:
        st.error(f"파일 로딩 중 오류 발생: {str(e)}")
        return {}

def process_cv_dataframe(df, sheet_name):
    """CV 데이터프레임 처리"""
    try:
        if df.empty:
            return None
            
        original_columns = df.columns.tolist()
        normalized_columns = [str(col).strip().lower() for col in df.columns]
        
        voltage_col = None
        current_col = None
        
        voltage_keywords = ['voltage', 'volt', 'potential', 'v', '전압', '전위', 'e']
        current_keywords = ['current', 'amp', 'i', '전류', 'a']
        
        for i, norm_col in enumerate(normalized_columns):
            if voltage_col is None and any(keyword in norm_col for keyword in voltage_keywords):
                voltage_col = original_columns[i]
            if current_col is None and any(keyword in norm_col for keyword in current_keywords):
                current_col = original_columns[i]
        
        if voltage_col is None or current_col is None:
            return None
        
        df_processed = df[[voltage_col, current_col]].copy()
        df_processed.columns = ['voltage', 'current']
        
        df_processed = df_processed.dropna()
        
        df_processed = df_processed[
            pd.to_numeric(df_processed['voltage'], errors='coerce').notna() &
            pd.to_numeric(df_processed['current'], errors='coerce').notna()
        ]
        
        df_processed['voltage'] = pd.to_numeric(df_processed['voltage'])
        df_processed['current'] = pd.to_numeric(df_processed['current'])
        
        if len(df_processed) < 10:
            return None
        
        return df_processed
        
    except Exception as e:
        return None

def analyze_cv_data(datasets):
    """CV 데이터 분석 메인 함수"""
    try:
        results = []
        
        for name, df in datasets.items():
            cycles = find_cv_cycles_and_segments(df)
            reduction_potential = find_reduction_potential_first_cycle(cycles)
            capacity = calculate_capacity(df)
            
            voltage_range = f"{df['voltage'].min():.3f} ~ {df['voltage'].max():.3f} V"
            current_range = f"{df['current'].min():.2e} ~ {df['current'].max():.2e} A"
            
            sample_type = "Reference" if name.startswith("Reference") else "Additive"
            clean_name = name.replace("Reference - ", "").replace("Additive - ", "")
            
            result = {
                'sample_name': clean_name,
                'sample_type': sample_type,
                'full_df': df,
                'cycles': cycles,
                'reduction_potential': reduction_potential,
                'capacity_mah_g': capacity,
                'voltage_range': voltage_range,
                'current_range': current_range
            }
            
            results.append(result)
        
        # 환원전위 개선도 계산 = 첨가제 - ref
        reference_samples = [r for r in results if r['sample_type'] == 'Reference']
        
        if reference_samples:
            reference = reference_samples[0]
            
            for result in results:
                if (result['reduction_potential'] is not None and 
                    reference['reduction_potential'] is not None):
                    
                    # 개선도 = 첨가제 환원전위 - ref 환원전위
                    voltage_shift = (
                        result['reduction_potential']['voltage'] - 
                        reference['reduction_potential']['voltage']
                    )
                    
                    if result['sample_type'] == 'Reference' and result == reference:
                        result['improvement'] = "기준"
                    else:
                        result['improvement'] = f"{voltage_shift:+.3f}V"
                else:
                    result['improvement'] = "분석 불가"
        else:
            for result in results:
                result['improvement'] = "Reference 없음"
        
        return results
    
    except Exception as e:
        st.error(f"데이터 분석 중 오류: {str(e)}")
        return []

def create_overlay_cv_plot(datasets, selected_samples, show_peaks=True, auto_range=True):
    """오버레이 CV 그래프 생성"""
    try:
        fig = go.Figure()
        
        # 더 많은 색상과 스타일 정의
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        line_styles = {
            'full': dict(width=3, dash='solid'),
            'reduction': dict(width=4, dash='solid'),
            'oxidation': dict(width=3, dash='dash')
        }
        
        # 전체 범위 계산
        all_currents = []
        all_voltages = []
        
        for sample_name, analysis_result in datasets.items():
            df = analysis_result['full_df']
            all_currents.extend(df['current'].tolist())
            all_voltages.extend(df['voltage'].tolist())
        
        if all_currents and all_voltages:
            y_min = min(all_currents)
            y_max = max(all_currents)
            x_min = min(all_voltages)
            x_max = max(all_voltages)
            y_margin = abs(y_max - y_min) * 0.05 if y_max != y_min else 0.1
            x_margin = abs(x_max - x_min) * 0.05 if x_max != x_min else 0.1
        else:
            y_min, y_max = -1, 1
            x_min, x_max = -1, 1
            y_margin, x_margin = 0.1, 0.1

        # 색상 인덱스 관리
        color_idx = 0
        sample_color_map = {}
        
        for item in selected_samples:
            sample_name = item['sample']
            plot_type = item['type']
            
            # 샘플별 고유 색상 할당
            if sample_name not in sample_color_map:
                sample_color_map[sample_name] = colors[color_idx % len(colors)]
                color_idx += 1
            
            color = sample_color_map[sample_name]
            
            if sample_name in datasets:
                analysis_result = datasets[sample_name]
                
                if plot_type == "full":
                    # 전체 CV 곡선
                    df = analysis_result['full_df']
                    fig.add_trace(go.Scatter(
                        x=df['voltage'],
                        y=df['current'],
                        mode='lines',
                        name=f"{sample_name} (전체)",
                        line=dict(color=color, **line_styles['full']),
                        opacity=0.8
                    ))
                
                elif plot_type == "reduction":
                    # 환원 구간만 - 첫 번째 사이클의 1번 구간만
                    cycles = analysis_result['cycles']
                    for key, cycle_info in cycles.items():
                        if cycle_info['type'] == 'reduction' and cycle_info['cycle'] == 1:  # 첫 번째 사이클만
                            reduction_data = cycle_info['data']
                            fig.add_trace(go.Scatter(
                                x=reduction_data['voltage'],
                                y=reduction_data['current'],
                                mode='lines',
                                name=f"{sample_name} (C1-환원)",
                                line=dict(color=color, **line_styles['reduction']),
                                opacity=0.9
                            ))
                            break  # 첫 번째 사이클만 처리
                
                elif plot_type == "oxidation":
                    # 산화 구간만
                    cycles = analysis_result['cycles']
                    for key, cycle_info in cycles.items():
                        if cycle_info['type'] == 'oxidation':
                            oxidation_data = cycle_info['data']
                            cycle_num = cycle_info['cycle']
                            fig.add_trace(go.Scatter(
                                x=oxidation_data['voltage'],
                                y=oxidation_data['current'],
                                mode='lines',
                                name=f"{sample_name} (C{cycle_num}-산화)",
                                line=dict(color=color, **line_styles['oxidation']),
                                opacity=0.7
                            ))
                
                # 환원전위 피크 표시
                if show_peaks and analysis_result.get('reduction_potential'):
                    rp = analysis_result['reduction_potential']
                    fig.add_trace(go.Scatter(
                        x=[rp['voltage']],
                        y=[rp['current']],
                        mode='markers',
                        name=f"{sample_name} - {rp['method']} 환원전위",
                        marker=dict(
                            color=color,
                            size=12,
                            symbol='circle',
                            line=dict(width=2, color='white')
                        ),
                        showlegend=False,
                        hovertemplate=f'<b>{sample_name}</b><br>' +
                                    f'환원전위: {rp["voltage"]:.3f} V<br>' +
                                    f'전류: {rp["current"]:.2e} A<br>' +
                                    f'방법: {rp["method"]}<br>' +
                                    '<extra></extra>'
                    ))
        
        # 레이아웃 설정
        layout_config = {
            'title': 'CV 오버레이 분석 - 환원전위 비교',
            'xaxis_title': 'Voltage vs Li/Li⁺ (V)',
            'yaxis_title': 'Current (A)',
            'hovermode': 'closest',
            'height': 600,
            'template': 'plotly_white',
            'legend': dict(
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        }
        
        if not auto_range:
            layout_config['yaxis'] = dict(
                range=[y_min - y_margin, y_max + y_margin],
                autorange=False
            )
            layout_config['xaxis'] = dict(
                range=[x_min - x_margin, x_max + x_margin],
                autorange=False
            )
        
        # 환원전위 그래프의 전압 범위 1.5V ~ 0V로 제한
        if any('reduction' in item['type'] for item in selected_samples):
            layout_config['xaxis'] = dict(range=[0, 1.5])
        
        fig.update_layout(**layout_config)
        return fig
    
    except Exception as e:
        st.error(f"오버레이 그래프 생성 중 오류: {str(e)}")
        return go.Figure()

def create_reduction_potential_comparison_chart(detailed_df):
    """환원전위 비교 차트 생성"""
    # 이 함수는 더 이상 사용되지 않음
    pass

def create_manual_input_table():
    """ref 환원전위 수동 입력 UI 추가 + 수동 계산 표 생성"""
    st.subheader("수동 환원전위 입력 및 계산")
    
    # 세션 상태 초기화
    if 'manual_data' not in st.session_state:
        st.session_state.manual_data = []
    
    # Reference 환원전위 입력
    col1, col2 = st.columns(2)
    with col1:
        ref_potential = st.number_input("Reference 환원전위 (V)", value=0.0, format="%.4f", key="ref_potential")
    
    # 샘플 데이터 입력
    with col2:
        st.write("샘플 데이터 입력:")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sample_name = st.text_input("샘플명", key="sample_name_input")
    with col2:
        sample_type = st.selectbox("타입", ["Additive", "Reference"], key="sample_type_input")
    with col3:
        sample_potential = st.number_input("환원전위 (V)", format="%.4f", key="sample_potential_input")
    with col4:
        if st.button("추가", key="add_sample_btn"):
            if sample_name:
                # 개선도 계산: 첨가제 - ref
                if sample_type == "Reference":
                    improvement = "기준"
                else:
                    improvement = f"{sample_potential - ref_potential:+.3f}V"
                
                st.session_state.manual_data.append({
                    'sample_name': sample_name,
                    'sample_type': sample_type,
                    'reduction_potential': sample_potential,
                    'improvement': improvement
                })
                st.success("추가되었습니다!")
    
    # 입력된 데이터 표시
    if st.session_state.manual_data:
        df_manual = pd.DataFrame(st.session_state.manual_data)
        st.dataframe(df_manual, use_container_width=True)
        
        # 데이터 초기화 버튼
        if st.button("데이터 초기화", key="clear_manual_data"):
            st.session_state.manual_data = []
            st.rerun()

def export_to_excel_new_format(analysis_results):
    """엑셀 다운로드 구조 변경 - 그래프와 데이터 연결"""
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet1: 분석결과 표
            summary_data = []
            for result in analysis_results:
                rp = result.get('reduction_potential')
                summary_data.append({
                    '샘플 타입': result['sample_type'],
                    '샘플명': result['sample_name'],
                    '전압 범위 (V)': result['voltage_range'],
                    '전류 범위 (A)': result['current_range'],
                    '환원전위 (V)': rp['voltage'] if rp else 'N/A',
                    '피크전류 (A)': f"{rp['current']:.2e}" if rp else 'N/A',
                    '전류밀도 (mA)': f"{rp['current_density']:.2f}" if rp else 'N/A',
                    '용량 (mAh/g)': f"{result['capacity_mah_g']:.2f}",
                    '개선도': result.get('improvement', 'N/A')
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='분석결과_요약', index=False)
            
            # Sheet2: 첫 사이클 환원 구간 통합 데이터 (그래프용)
            chart_data = []
            max_length = 0
            
            # 모든 샘플의 첫 번째 사이클 환원 구간 데이터 수집
            sample_data_dict = {}
            for result in analysis_results:
                cycles = result.get('cycles', {})
                for key, cycle_info in cycles.items():
                    if cycle_info['type'] == 'reduction' and cycle_info['cycle'] == 1:
                        data = cycle_info['data'].reset_index(drop=True)
                        sample_name = result['sample_name']
                        sample_data_dict[sample_name] = data
                        max_length = max(max_length, len(data))
                        break
            
            # 통합 데이터프레임 생성 (Excel 차트가 인식할 수 있도록)
            chart_df = pd.DataFrame()
            
            for sample_name, data in sample_data_dict.items():
                # 데이터 길이를 맞추기 위해 빈 값으로 패딩
                voltage_col = data['voltage'].tolist() + [None] * (max_length - len(data))
                current_col = data['current'].tolist() + [None] * (max_length - len(data))
                
                chart_df[f'{sample_name}_전압(V)'] = voltage_col
                chart_df[f'{sample_name}_전류(A)'] = current_col
            
            chart_df.to_excel(writer, sheet_name='첫사이클_환원구간_데이터', index=False)
            
            # Excel 네이티브 차트 생성
            try:
                from openpyxl.chart import ScatterChart, Reference, Series
                
                workbook = writer.book
                worksheet = workbook['첫사이클_환원구간_데이터']
                
                # 산점도 차트 생성
                chart = ScatterChart()
                chart.title = "첫 번째 사이클 환원 구간 비교"
                chart.style = 2
                chart.width = 15
                chart.height = 10
                
                # X축 설정 (전압)
                chart.x_axis.title = '전압 (V)'
                chart.x_axis.tickLblPos = "low"  # 축 숫자 표시
                chart.x_axis.majorUnit = 0.2  # 주 눈금 간격
                chart.x_axis.minorUnit = 0.1  # 보조 눈금 간격
                # 축에 숫자 표
                # Y축 설정 (전류)
                chart.y_axis.title = '전류 (A)'
                chart.y_axis.tickLblPos = "low"  # 축 숫자 표시
                
                # 각 샘플별로 시리즈 추가
                colors = ['0000FF', 'FF0000', '00FF00', 'FF8000', '8000FF']
                for i, sample_name in enumerate(sample_data_dict.keys()):
                    voltage_col = (i * 2) + 1  # A, C, E, G...
                    current_col = (i * 2) + 2  # B, D, F, H...
                    
                    # 데이터 범위 설정 (None 값 제외)
                    data_length = len(sample_data_dict[sample_name])
                    
                    xvalues = Reference(worksheet, min_col=voltage_col, min_row=2, 
                                      max_row=data_length + 1)
                    values = Reference(worksheet, min_col=current_col, min_row=2, 
                                     max_row=data_length + 1)
                    
                    series = Series(values, xvalues, title=sample_name)
                    
                    # 선 스타일 설정
                    series.graphicalProperties.line.solidFill = colors[i % len(colors)]
                    series.graphicalProperties.line.width = 20000  # 선 굵기
                    
                    chart.series.append(series)
                
                # 차트를 워크시트에 추가
                chart_col = len(sample_data_dict) * 2 + 2  # 데이터 옆에 배치
                worksheet.add_chart(chart, f'{chr(65 + chart_col)}2')
                
            except Exception as e:
                st.warning(f"Excel 네이티브 차트 생성 중 오류: {str(e)}")
            
            # Sheet3~: 샘플별 상세 데이터 및 차트
            for i, result in enumerate(analysis_results):
                sheet_name = f"{result['sample_name'][:25]}_상세"
                
                # 각 사이클 데이터 정리
                cycles_data = []
                for key, cycle_info in result['cycles'].items():
                    cycle_df = cycle_info['data'].copy()
                    cycle_df['사이클'] = cycle_info['cycle']
                    cycle_df['구간타입'] = cycle_info['type']
                    cycles_data.append(cycle_df)
                
                if cycles_data:
                    all_cycles_df = pd.concat(cycles_data, ignore_index=True)
                    all_cycles_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # 개별 샘플 Excel 네이티브 차트 생성
                    try:
                        from openpyxl.chart import ScatterChart, Reference, Series
                        
                        workbook = writer.book
                        worksheet = workbook[sheet_name]
                        
                        chart = ScatterChart()
                        chart.title = f'{result["sample_name"]} - 모든 사이클'
                        chart.style = 2
                        chart.width = 15
                        chart.height = 10
                        
                        # X축 설정 (전압)
                        chart.x_axis.title = '전압 (V)'
                        chart.x_axis.tickLblPos = "low"  # 축 숫자 표시
                        chart.x_axis.majorUnit = 0.5  # 주 눈금 간격
                        chart.x_axis.minorUnit = 0.1  # 보조 눈금 간격
                        chart.x_axis.number_format = '0.0' 
                        
                        # Y축 설정 (전류)
                        chart.y_axis.title = '전류 (A)'
                        chart.y_axis.tickLblPos = "low"  # 축 숫자 표시
                        chart.y_axis.number_format = '0.0' 
                        # 데이터가 있는 행 수 계산
                        data_rows = len(all_cycles_df)
                        
                        # 전압(A열)과 전류(B열) 데이터 참조
                        xvalues = Reference(worksheet, min_col=1, min_row=2, max_row=data_rows + 1)
                        values = Reference(worksheet, min_col=2, min_row=2, max_row=data_rows + 1)
                        
                        series = Series(values, xvalues, title=f'{result["sample_name"]} CV')
                        series.graphicalProperties.line.solidFill = '0000FF'
                        series.graphicalProperties.line.width = 15000
                        
                        chart.series.append(series)
                        chart.legend = None
                        # 차트를 워크시트에 추가 (F열부터)
                        worksheet.add_chart(chart, 'F2')
                        
                    except Exception as e:
                        st.warning(f"{sheet_name} Excel 차트 생성 중 오류: {str(e)}")
        
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        st.error(f"Excel 파일 생성 중 오류: {str(e)}")
        return None

def main():
    # 헤더
    st.markdown('<h1 class="main-header">🔋 CV 데이터 분석 - 배터리 전해액 첨가제 개발</h1>', unsafe_allow_html=True)
    st.markdown("**수학적 접근법 기반 환원전위 탐지 및 성능 비교**")
    
    # 사이드바
    with st.sidebar:
        st.header("📁 데이터 업로드")
        uploaded_files = st.file_uploader(
            "CV 데이터 파일 선택",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="CSV 또는 Excel 파일을 업로드하세요."
        )
        
        st.markdown("---")
        st.header("⚙️ 분석 설정")
        
        show_peaks = st.checkbox("환원전위 표시", value=True)
        
        st.subheader("📊 그래프 범위 설정")
        auto_range = st.checkbox("Y축 자동 범위 조정", value=True)
        
        # Reference 샘플 환원전위 설정
        st.subheader("Reference 샘플 환원전위 설정")
        use_manual_ref = st.checkbox("Reference 샘플 환원전위 수동 입력", value=False)
        manual_ref_potential = None
        if use_manual_ref:
            manual_ref_potential = st.number_input(
                "Reference 환원전위 (V)",
                value=0.0,
                format="%.4f",
                help="Reference 샘플의 환원전위를 수동으로 입력하세요"
            )
        
        st.header("📊 그래프 설정")
        plot_height = st.slider("그래프 높이", 400, 800, 600, 50)
    
    # 메인 컨텐츠
    if uploaded_files:
        all_datasets = {}
        loading_success = True
        
        for uploaded_file in uploaded_files:
            file_datasets = load_cv_data_from_excel(uploaded_file)
            if file_datasets:
                all_datasets.update(file_datasets)
            else:
                loading_success = False
        
        if all_datasets and loading_success:
            st.success(f"✅ {len(all_datasets)}개의 CV 데이터가 성공적으로 로드되었습니다!")
            
            with st.spinner("CV 데이터 분석 중..."):
                analysis_results = analyze_cv_data(all_datasets)
            
            if analysis_results:
                analyzed_datasets = {}
                for result in analysis_results:
                    key = f"{result['sample_type']} - {result['sample_name']}"
                    analyzed_datasets[key] = result
                
                # CV 그래프 시각화
                st.header("📊 CV 그래프 시각화")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("샘플 선택")
                    
                    # 오버레이 방식으로 변경
                    st.markdown("**오버레이 그래프 선택:**")
                    available_samples = list(analyzed_datasets.keys())
                    
                    # 다중 선택 체크박스
                    selected_samples = []
                    
                    st.markdown("**전체 CV 곡선:**")
                    for sample in available_samples:
                        if st.checkbox(f"전체 - {sample}", key=f"full_{sample}"):
                            selected_samples.append({"sample": sample, "type": "full"})
                    
                    st.markdown("**환원 구간:**")
                    for sample in available_samples:
                        if st.checkbox(f"환원 - {sample}", key=f"reduction_{sample}"):
                            selected_samples.append({"sample": sample, "type": "reduction"})
                    
                    st.markdown("**산화 구간:**")
                    for sample in available_samples:
                        if st.checkbox(f"산화 - {sample}", key=f"oxidation_{sample}"):
                            selected_samples.append({"sample": sample, "type": "oxidation"})
                
                with col2:
                    if selected_samples:
                        # 전체 데이터 범위 정보
                        all_currents = []
                        all_voltages = []
                        
                        for sample_name, analysis_result in analyzed_datasets.items():
                            df = analysis_result['full_df']
                            all_currents.extend(df['current'].tolist())
                            all_voltages.extend(df['voltage'].tolist())
                        
                        if all_currents and all_voltages:
                            st.info(f"📊 전체 데이터 범위: 전압 {min(all_voltages):.3f}~{max(all_voltages):.3f}V, "
                                   f"전류 {min(all_currents):.2e}~{max(all_currents):.2e}A")
                        
                        # 오버레이 그래프 생성
                        fig = create_overlay_cv_plot(analyzed_datasets, selected_samples, show_peaks, auto_range)
                        fig.update_layout(height=plot_height)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if not auto_range:
                            st.success(f"🎯 전체 범위 모드 활성: Y축 {min(all_currents):.2e} ~ {max(all_currents):.2e}")
                    
                    else:
                        st.info("👆 왼쪽에서 표시할 그래프를 선택하세요")
                
                # 선택된 샘플들의 환원전위 상세 테이블
                if selected_samples:
                    st.subheader("🔬 선택된 샘플 환원전위 상세 분석")
                    selected_sample_names = list(set([item["sample"] for item in selected_samples]))
                    
                    if selected_sample_names:
                        # 선택된 샘플들의 상세 분석 결과 테이블
                        detailed_analysis_data = []
                        
                        for sample_name in selected_sample_names:
                            if sample_name in analyzed_datasets:
                                analysis_result = analyzed_datasets[sample_name]
                                rp = analysis_result.get('reduction_potential')
                                
                                if rp:
                                    detailed_analysis_data.append({
                                        '샘플명': analysis_result['sample_name'],
                                        '샘플타입': analysis_result['sample_type'],
                                        '환원전위(V)': f"{rp['voltage']:.3f}",
                                        '전류(A)': f"{rp['current']:.2e}",
                                        '전류밀도(mA)': f"{rp['current_density']:.2f}",
                                        '개선도': analysis_result.get('improvement', 'N/A')
                                    })
                        
                        if detailed_analysis_data:
                            detailed_df = pd.DataFrame(detailed_analysis_data)
                            st.dataframe(detailed_df, use_container_width=True)
                                            
                # Excel 다운로드
                st.subheader("📥 결과 다운로드")
                if st.button("📊 Excel 파일 생성"):
                    excel_data = export_to_excel_new_format(analysis_results)
                    if excel_data:
                        st.download_button(
                            label="📥 Excel 파일 다운로드",
                            data=excel_data,
                            file_name="CV_Analysis_Results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        
        else:
            st.error("❌ 유효한 CV 데이터를 찾을 수 없습니다.")
    
    else:
        st.info("📁 CV 데이터 파일을 업로드하여 분석을 시작하세요.")
        
        # 사용법 안내
        with st.expander("📖 사용법 안내"):
            st.markdown("""
            ### 📋 CV 데이터 분석 도구 사용법
            
            1. **파일 업로드**: CSV 또는 Excel 파일을 업로드하세요
            2. **분석 설정**: 사이드바에서 환원전위 표시 및 그래프 설정을 조정하세요
            3. **결과 확인**: 분석 결과와 그래프를 확인하세요
            4. **다운로드**: Excel 형식으로 결과를 다운로드하세요
            
            ### 🔬 분석 기능
            
            - **환원전위 탐지**: 1.5V-0V 구간에서 환원전위 탐지
            - **개선도 계산**: 첨가제 환원전위 - Reference 환원전위
            - **CV 곡선 오버레이**: 전체/환원/산화 구간별 곡선 비교
            - **Excel 출력**: 분석결과 + 그래프가 포함된 Excel 파일
            """)
    
    # ref 환원전위 수동 입력 UI 추가
    st.markdown("---")
    create_manual_input_table()
    
    # 환원전위 탐지 원리 설명
    with st.expander("🧠 환원전위 탐지 원리"):
        st.markdown("""
        ### 🔬 환원전위 탐지 방법 (1.5V-0V 구간)
        
        **1. 1차 미분 절댓값 변화 최대**
        - 전류-전압 곡선의 기울기 변화가 가장 큰 지점
        - 환원반응이 가장 활발하게 시작되는 전위
        
        **핵심 특징:**
        - 1.5V 이하 구간에서만 분석
        - 데이터 스무딩으로 노이즈 제거
        - 첨가제 vs Reference 성능 비교
        """)

if __name__ == "__main__":
    main()