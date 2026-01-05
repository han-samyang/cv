import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import simpson
import io
import base64
from typing import Dict, List, Tuple, Optional

# scipy의 trapezoid 사용 (numpy.trapz는 최신 버전에서 제거됨)
try:
    from scipy.integrate import trapezoid
except ImportError:
    from scipy.integrate import trapz as trapezoid

# 페이지 설정
st.set_page_config(
    page_title="CV 데이터 분석 - 환원 피크 탐지",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .analysis-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .peak-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #ff4444;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class CVAnalyzer:
    """CV 데이터 분석 클래스"""
    
    def __init__(self):
        self.data_sheets = {}
        self.analysis_results = {}
    
    def load_excel_data(self, uploaded_file) -> Dict[str, pd.DataFrame]:
        """엑셀 파일에서 여러 시트 로드"""
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            loaded_sheets = {}
            
            for sheet_name in sheet_names:
                # "정보" 시트는 무시
                if "정보" in sheet_name.lower():
                    continue
                
                try:
                    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                    if len(df) > 10:  # 최소 데이터 요구사항
                        loaded_sheets[sheet_name] = df
                        st.success(f"✅ '{sheet_name}' 시트 로드 완료 ({len(df)}개 행)")
                except Exception as e:
                    st.warning(f"⚠️ '{sheet_name}' 시트 로드 실패: {e}")
            
            return loaded_sheets
            
        except Exception as e:
            st.error(f"❌ 엑셀 파일 로드 실패: {e}")
            return {}
    
    def find_voltage_current_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """전압과 전류 컬럼 찾기 - 더 견고한 방법"""
        voltage_col = None
        current_col = None
        
        # 방법 1: 컬럼명에서 전압과 전류 찾기
        for col in df.columns:
            col_str = str(col).lower()
            if voltage_col is None and ('전압' in col_str or 'voltage' in col_str):
                voltage_col = col
            elif current_col is None and ('전류' in col_str or 'current' in col_str):
                current_col = col
                current_col = col
        
        # 방법 2: 순서로 찾기 (G=6번째, H=7번째)
        if not voltage_col or not current_col:
            columns_list = list(df.columns)
            
            if not voltage_col and len(columns_list) >= 7:
                voltage_col = columns_list[6]  # G열
                st.write(f"📍 G열을 전압으로 선택: {voltage_col}")
                
            if not current_col and len(columns_list) >= 8:
                current_col = columns_list[7]  # H열  
                st.write(f"📍 H열을 전류로 선택: {current_col}")
        
        return voltage_col, current_col
    
    def analyze_cycle_numbers(self, df: pd.DataFrame) -> Dict:
        """C열 사이클 횟수 분석 - 어떤 값들이 있는지 확인"""
        try:
            # C열 = 사이클 횟수 (인덱스 2)
            cycle_col = df.columns[2] if len(df.columns) > 2 else None
            if not cycle_col:
                st.error("C열(사이클 횟수)을 찾을 수 없습니다.")
                return {}
            
            # 사이클 횟수의 고유값들 확인
            cycle_data = pd.to_numeric(df[cycle_col], errors='coerce')
            unique_cycles = cycle_data.dropna().unique()
            unique_cycles = sorted(unique_cycles)
            
            # 각 사이클별 개수 확인
            cycle_counts = {}
            for cycle in unique_cycles:
                count = (cycle_data == cycle).sum()
                cycle_counts[cycle] = count
            
            return {
                'cycle_column': cycle_col,
                'unique_cycles': unique_cycles,
                'cycle_counts': cycle_counts,
                'total_rows': len(df)
            }
            
        except Exception as e:
            st.error(f"사이클 횟수 분석 중 오류: {e}")
            return {}
    
    def extract_cycle_data(self, df: pd.DataFrame, target_cycle: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """지정된 사이클 횟수의 데이터만 추출"""
        try:
            
            # C열 = 사이클 횟수 (인덱스 2)
            cycle_col = df.columns[2] if len(df.columns) > 2 else None
            if not cycle_col:
                st.error("C열(사이클 횟수)을 찾을 수 없습니다.")
                return np.array([]), np.array([]), 0
            
            # 지정된 사이클 횟수인 행들만 필터링
            cycle_data = pd.to_numeric(df[cycle_col], errors='coerce')
            cycle_mask = (cycle_data == target_cycle) & (~cycle_data.isna())
            
            cycle_count = cycle_mask.sum()
            total_count = len(df)
            
            # 사이클 데이터 확인
            
            if cycle_count == 0:
                st.warning(f"⚠️ 사이클 횟수가 {target_cycle}인 데이터가 없습니다.")
                return np.array([]), np.array([]), 0
            
            # 해당 사이클 데이터에서 전압, 전류 추출
            voltage_col, current_col = self.find_voltage_current_columns(df)
            if not voltage_col or not current_col:
                st.error("전압 또는 전류 컬럼을 찾을 수 없습니다.")
                return np.array([]), np.array([]), 0
            
            # 해당 사이클인 행들의 전압, 전류 데이터
            df_cycle = df[cycle_mask]
            
            voltage_data = pd.to_numeric(df_cycle[voltage_col], errors='coerce')
            current_data = pd.to_numeric(df_cycle[current_col], errors='coerce')
            
            # 유효한 데이터만 선택
            valid_mask = ~(voltage_data.isna() | current_data.isna())
            voltage = voltage_data[valid_mask].values
            current = current_data[valid_mask].values
            
            valid_count = len(voltage)
            
            if valid_count > 0:
                # 데이터 범위 확인
                return voltage, current, valid_count
            
        except Exception as e:
            st.error(f"사이클 {target_cycle} 데이터 추출 중 오류: {e}")
            return np.array([]), np.array([]), 0
    
    def find_reduction_inflection_point(self, voltage: np.ndarray, current: np.ndarray) -> Optional[Dict]:
        """사이클 데이터에서 전류가 하강하다가 상승하는 변곡점 찾기"""
        if len(voltage) < 20:
            return None
        
        try:
            # 변곡점 탐지 시작
            
            # 스무딩 적용 (노이즈 제거)
            if len(current) > 15:
                window_length = min(15, len(current) // 4)
                if window_length % 2 == 0:
                    window_length += 1
                i_smooth = savgol_filter(current, window_length, 3)
            else:
                i_smooth = current.copy()
            
            # 1차 미분 계산 (전류 변화율)
            di_dt = np.gradient(i_smooth)
            
            # 중간 구간에서만 탐지 (시작과 끝 제외)
            search_start = len(voltage) // 10
            search_end = len(voltage) - len(voltage) // 10
            
            # 탐지 범위 설정
            
            # 변곡점 찾기: 음의 기울기에서 양의 기울기로 변하는 지점
            inflection_candidates = []
            
            for i in range(search_start + 2, search_end - 2):
                # 연속된 몇 개 점에서 추세 확인
                prev_trend = np.mean(di_dt[i-2:i])    # 이전 추세
                curr_trend = np.mean(di_dt[i:i+2])    # 현재 추세
                
                # 이전에는 감소, 현재는 증가하는 경우
                if prev_trend < -1e-8 and curr_trend > 1e-8:
                    inflection_candidates.append({
                        'index': i,
                        'voltage': voltage[i],
                        'current': i_smooth[i],
                        'derivative': di_dt[i],
                        'score': abs(i_smooth[i])  # 전류 절댓값을 점수로 사용
                    })
            
            if not inflection_candidates:
                # 대안: 사이클 데이터에서 전류가 가장 음수인 지점
                min_idx = np.argmin(i_smooth[search_start:search_end]) + search_start
                # 변곡점이 없으면 최소값 사용
                return {
                    'voltage': voltage[min_idx],
                    'current': i_smooth[min_idx],
                    'current_density_ma': abs(i_smooth[min_idx]) * 1000,
                    'method': '사이클 최소값',
                    'original_index': min_idx,
                    'cycle_info': f"사이클 데이터 {len(voltage)}개 포인트",
                    'derivative': di_dt[min_idx] if min_idx < len(di_dt) else 0
                }
            
            # 가장 적합한 변곡점 선택 (전류 절댓값이 가장 큰 지점)
            best_candidate = max(inflection_candidates, key=lambda x: x['score'])
            
            # 디버깅 정보
            # 최적 변곡점 선택됨
            
            return {
                'voltage': best_candidate['voltage'],
                'current': best_candidate['current'],
                'current_density_ma': abs(best_candidate['current']) * 1000,
                'method': '사이클 변곡점',
                'original_index': best_candidate['index'],
                'cycle_info': f"사이클 데이터 {len(voltage)}개 포인트",
                'derivative': best_candidate['derivative']
            }
            
        except Exception as e:
            st.error(f"변곡점 탐지 중 오류: {e}")
            return None
    
    def find_slope_change_points(self, voltage: np.ndarray, current: np.ndarray, peak_idx: int) -> tuple:
        """피크 양쪽에서 기울기 변화점 찾기 (x1,y1), (x2,y2)"""
        try:
            # 1. 스무딩 및 기울기 계산
            if len(current) > 10:
                window_length = min(9, len(current) // 3)
                if window_length % 2 == 0:
                    window_length += 1
                i_smooth = savgol_filter(current, window_length, 2)
                slope = savgol_filter(current, window_length, 2, deriv=1)
            else:
                i_smooth = current.copy()
                slope = np.gradient(i_smooth, voltage)
            
            # 2. 기울기 변화율 (2차 미분)
            slope_change = np.gradient(slope)
            
            # 3. 피크에서의 기준값들
            peak_slope_mag = abs(slope[peak_idx])
            slope_threshold = peak_slope_mag * 0.1  # 기울기 임계값
            
            # 4. 왼쪽 기울기 변화점 찾기 - 기울기 변화가 가장 큰 지점
            max_search_left = min(40, len(voltage) // 3)
            x1_idx = max(0, peak_idx - max_search_left)  # 기본값
            
            # 왼쪽 구간에서 기울기 변화율(2차 미분)이 최대인 지점 찾기
            max_slope_change = 0
            candidate_indices = []
            
            # 먼저 모든 후보 지점들을 수집
            for i in range(peak_idx - 5, max(0, peak_idx - max_search_left), -1):
                if i > 2 and i < len(slope_change) - 2:
                    # 주변 점들을 고려한 평균 기울기 변화율 계산
                    avg_slope_change = np.mean([abs(slope_change[i-1]), 
                                              abs(slope_change[i]), 
                                              abs(slope_change[i+1])])
                    candidate_indices.append((i, avg_slope_change))
            
            # 기울기 변화가 가장 큰 지점 선택
            if candidate_indices:
                # 기울기 변화량으로 정렬하여 상위 후보들 선택
                candidate_indices.sort(key=lambda x: x[1], reverse=True)
                
                # 상위 3개 후보 중에서 피크에서 가장 먼 점 선택 (더 넓은 baseline)
                top_candidates = candidate_indices[:3]
                x1_idx = min(top_candidates, key=lambda x: abs(x[0] - peak_idx))[0]
                max_slope_change = candidate_indices[0][1]
            
            # 5. 오른쪽 기울기 변화점 찾기 - 기울기 변화가 가장 큰 지점
            max_search_right = min(40, len(voltage) // 3)
            x2_idx = min(len(voltage) - 1, peak_idx + max_search_right)  # 기본값
            
            # 오른쪽 구간에서 기울기 변화율(2차 미분)이 최대인 지점 찾기
            max_slope_change = 0
            candidate_indices = []
            
            # 먼저 모든 후보 지점들을 수집
            for i in range(peak_idx + 5, min(len(voltage), peak_idx + max_search_right)):
                if i > 2 and i < len(slope_change) - 2:
                    # 주변 점들을 고려한 평균 기울기 변화율 계산
                    avg_slope_change = np.mean([abs(slope_change[i-1]), 
                                              abs(slope_change[i]), 
                                              abs(slope_change[i+1])])
                    candidate_indices.append((i, avg_slope_change))
            
            # 기울기 변화가 가장 큰 지점 선택
            if candidate_indices:
                # 기울기 변화량으로 정렬하여 상위 후보들 선택
                candidate_indices.sort(key=lambda x: x[1], reverse=True)
                
                # 상위 3개 후보 중에서 피크에서 가장 먼 점 선택 (더 넓은 baseline)
                top_candidates = candidate_indices[:3]
                x2_idx = min(top_candidates, key=lambda x: abs(x[0] - peak_idx))[0]
                max_slope_change = candidate_indices[0][1]
            
            # 6. (x1,y1), (x2,y2) 좌표 계산
            x1, y1 = voltage[x1_idx], i_smooth[x1_idx]
            x2, y2 = voltage[x2_idx], i_smooth[x2_idx]
            
            # 7. Baseline 매개변수 계산 (두 점을 지나는 직선)
            if abs(x2 - x1) > 1e-10:
                baseline_slope = (y2 - y1) / (x2 - x1)
                baseline_intercept = y1 - baseline_slope * x1
            else:
                baseline_slope = 0
                baseline_intercept = (y1 + y2) / 2
            
            return x1_idx, x2_idx, baseline_slope, baseline_intercept, (x1, y1), (x2, y2)
            
        except Exception as e:
            # 오류 시 기본값 반환
            search_range = min(15, len(voltage) // 6)
            x1_idx = max(0, peak_idx - search_range)
            x2_idx = min(len(voltage) - 1, peak_idx + search_range)
            
            y1 = current[x1_idx] if x1_idx < len(current) else 0
            y2 = current[x2_idx] if x2_idx < len(current) else 0
            x1, x2 = voltage[x1_idx], voltage[x2_idx]
            
            baseline_slope = (y2 - y1) / (x2 - x1) if abs(x2 - x1) > 1e-10 else 0
            baseline_intercept = y1 - baseline_slope * x1
            
            return x1_idx, x2_idx, baseline_slope, baseline_intercept, (x1, y1), (x2, y2)

    def calculate_peak_integration(self, voltage: np.ndarray, current: np.ndarray, 
                                 peak_idx: int, window_size: int = 50) -> Dict:
        """기울기 변화점 기반 피크 적분 계산 - baseline 아래 면적 - 곡선 아래 면적"""
        try:
            # 1. 피크 양쪽에서 기울기 변화점 찾기
            x1_idx, x2_idx, baseline_slope, baseline_intercept, (x1, y1), (x2, y2) = \
                self.find_slope_change_points(voltage, current, peak_idx)
            
            # 2. 적분 영역 설정 (x1부터 x2까지)
            start_idx = x1_idx
            end_idx = x2_idx
            
            v_region = voltage[start_idx:end_idx+1]
            i_region = current[start_idx:end_idx+1]
            
            if len(v_region) < 3:
                return {'area_trapz': 0, 'area_simpson': 0, 'integration_points': 0}
            
            # 3. 스무딩 적용
            if len(i_region) > 10:
                window_length = min(9, len(i_region) // 3)
                if window_length % 2 == 0:
                    window_length += 1
                try:
                    i_smooth = savgol_filter(i_region, window_length, 2)
                except:
                    i_smooth = i_region.copy()
            else:
                i_smooth = i_region.copy()
            
            # 4. 순수하게 (x1,y1)과 (x2,y2) 두 점만을 연결한 baseline
            # 피크는 baseline 계산에 전혀 관여하지 않음
            baseline = baseline_slope * v_region + baseline_intercept
            
            # 5. 올바른 면적 계산: baseline 아래 사각형 면적 - 곡선 아래 면적
            # Step 1: x1부터 x2까지 baseline 아래 사각형 면적
            baseline_area = trapezoid(baseline, v_region)
            
            # Step 2: x1부터 x2까지 실제 곡선 아래 면적  
            curve_area = trapezoid(i_smooth, v_region)
            
            # Step 3: 피크 면적 = |baseline 면적 - 곡선 면적|
            # baseline이 곡선보다 위에 있으므로 양수가 나와야 함
            peak_area = baseline_area - curve_area
            area_trapz = abs(peak_area)  # 혹시 음수면 절댓값
            
            # Simpson 적분으로도 계산
            if len(i_smooth) > 2:
                try:
                    baseline_area_simpson = simpson(baseline, v_region)
                    curve_area_simpson = simpson(i_smooth, v_region)
                    area_simpson = abs(baseline_area_simpson - curve_area_simpson)
                except:
                    area_simpson = area_trapz
            else:
                area_simpson = area_trapz
            
            # 6. 시각화용 데이터 준비
            # baseline 아래에 있는 곡선 부분만 적분 영역으로 표시
            integration_voltage = []
            integration_current = []
            integration_baseline = []
            
            for v, curr, base in zip(v_region, i_smooth, baseline):
                if curr < base:  # baseline 아래 부분
                    integration_voltage.append(v)
                    integration_current.append(curr)
                    integration_baseline.append(base)
            
            return {
                'area_trapz': area_trapz,
                'area_simpson': area_simpson,
                'integration_points': len(v_region),
                'voltage_range': f"{v_region.min():.4f}~{v_region.max():.4f}V",
                'window_size': end_idx - start_idx + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'integration_voltage': np.array(integration_voltage),  # 실제 적분 영역 전압
                'integration_current': np.array(integration_current),  # 실제 적분 영역 전류
                'baseline_voltage': v_region,  # baseline 전압 (전체 영역)
                'baseline_current': baseline,   # baseline 전류 (전체 영역)
                'peak_below_baseline': len(integration_voltage) > 0,  # baseline 아래 peak 존재
                'slope_based_detection': True,  # 기울기 변화점 기반 검출
                'baseline_params': {
                    'slope': baseline_slope,
                    'intercept': baseline_intercept
                },
                'slope_change_points': {
                    'left': (x1, y1),
                    'right': (x2, y2),
                    'left_idx': x1_idx,
                    'right_idx': x2_idx
                },
                'method': 'max_slope_change_baseline',  # 방법 식별
                'calculation_details': {
                    'baseline_area': baseline_area,
                    'curve_area': curve_area,
                    'peak_area': peak_area,
                    'peak_voltage': voltage[peak_idx],
                    'distance_left': abs(x1_idx - peak_idx),
                    'distance_right': abs(x2_idx - peak_idx)
                }
            }
            
        except Exception as e:
            st.error(f"적분 계산 중 오류: {e}")
            return {
                'area_trapz': 0.0,
                'area_simpson': 0.0,
                'integration_points': 0,
                'voltage_range': 'N/A',
                'window_size': 0,
                'start_idx': 0,
                'end_idx': 0,
                'integration_voltage': np.array([]),
                'integration_current': np.array([]),
                'baseline_voltage': np.array([]),
                'baseline_current': np.array([]),
                'peak_below_baseline': False
            }

    def calculate_peak_integration_fixed_window(self, voltage: np.ndarray, current: np.ndarray, 
                                              peak_idx: int, window_size: int) -> Dict:
        """고정 윈도우 + 새로운 파이프라인 적분 계산"""
        try:
            # 1. 고정 윈도우 크기 적용
            start_idx = max(0, peak_idx - window_size // 2)
            end_idx = min(len(voltage) - 1, peak_idx + window_size // 2)
            
            # 2. 적분 영역의 전압, 전류 데이터
            v_region = voltage[start_idx:end_idx+1]
            i_region = current[start_idx:end_idx+1]
            
            if len(v_region) < 3:
                return {'area_trapz': 0, 'area_simpson': 0, 'integration_points': 0}
            
            # 3. 스무딩 적용
            if len(i_region) > 10:
                window_length = min(9, len(i_region) // 3)
                if window_length % 2 == 0:
                    window_length += 1
                try:
                    i_smooth = savgol_filter(i_region, window_length, 2)
                except:
                    i_smooth = i_region.copy()
            else:
                i_smooth = i_region.copy()
            
            # 4. Baseline 정의 (양 끝점을 연결한 직선)
            if len(v_region) > 1:
                baseline_slope = (i_smooth[-1] - i_smooth[0]) / (v_region[-1] - v_region[0])
                baseline_intercept = i_smooth[0] - baseline_slope * v_region[0]
                baseline = baseline_slope * v_region + baseline_intercept
            else:
                baseline = i_smooth.copy()
                baseline_slope = 0
                baseline_intercept = i_smooth[0]
            
            # 5. (baseline - y)+ 계산
            diff = baseline - i_smooth
            positive_diff = np.maximum(diff, 0)  # 양수인 부분만
            
            # 6. trapz 적분으로 면적 계산
            area_trapz = trapezoid(positive_diff, v_region)
            
            if len(positive_diff) > 2:
                try:
                    area_simpson = simpson(positive_diff, v_region)
                except:
                    area_simpson = area_trapz
            else:
                area_simpson = area_trapz
            
            # 7. 시각화용 데이터
            integration_voltage = []
            integration_current = []
            integration_baseline = []
            
            for v, curr, base, pos_diff in zip(v_region, i_smooth, baseline, positive_diff):
                if pos_diff > 1e-10:
                    integration_voltage.append(v)
                    integration_current.append(curr)
                    integration_baseline.append(base)
            
            return {
                'area_trapz': area_trapz,
                'area_simpson': area_simpson,
                'integration_points': len(v_region),
                'voltage_range': f"{v_region.min():.4f}~{v_region.max():.4f}V",
                'window_size': window_size,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'integration_voltage': np.array(integration_voltage),
                'integration_current': np.array(integration_current),
                'baseline_voltage': v_region,
                'baseline_current': baseline,
                'peak_below_baseline': len(integration_voltage) > 0,
                'slope_based_detection': False,  # 고정 윈도우 방법
                'relative_current': positive_diff,
                'baseline_params': {
                    'slope': baseline_slope,
                    'intercept': baseline_intercept
                },
                'method': 'fixed_window_baseline'
            }
            
        except Exception as e:
            st.error(f"고정 윈도우 적분 계산 오류: {e}")
            return {'area_trapz': 0, 'area_simpson': 0, 'integration_points': 0}
    
    def analyze_cv_data(self, sheet_name: str, df: pd.DataFrame) -> Optional[Dict]:
        """CV 데이터 분석 - 사이클 횟수 기반"""
        try:
            # 분석 시작
            
            # 먼저 어떤 사이클 횟수들이 있는지 확인
            cycle_analysis = self.analyze_cycle_numbers(df)
            
            if not cycle_analysis:
                st.warning(f"⚠️ '{sheet_name}': 사이클 횟수 분석 실패")
                return None
            
            unique_cycles = cycle_analysis['unique_cycles']
            cycle_counts = cycle_analysis['cycle_counts']
            
            # 사이클 0이 있는지 확인
            if 0 in unique_cycles:
                target_cycle = 0
                st.info(f"🎯 **사이클 0 발견!** ({cycle_counts[0]}개 포인트)")
            else:
                # 사이클 0이 없으면 가장 작은 사이클 번호 선택
                target_cycle = min(unique_cycles)
                st.warning(f"⚠️ 사이클 0이 없어서 가장 작은 사이클 {target_cycle} 선택 ({cycle_counts[target_cycle]}개 포인트)")
            
            # 사용자가 원한다면 다른 사이클 선택 가능
            with st.expander("🔧 사이클 번호 선택 (고급)"):
                selected_cycle = st.selectbox(
                    f"{sheet_name} 분석할 사이클 선택:",
                    options=unique_cycles,
                    index=list(unique_cycles).index(target_cycle) if target_cycle in unique_cycles else 0,
                    key=f"cycle_select_{sheet_name}"
                )
                if selected_cycle != target_cycle:
                    target_cycle = selected_cycle
                    st.info(f"✅ 사이클 {target_cycle} 선택됨")
            
            # 선택된 사이클의 데이터 추출
            voltage, current, data_count = self.extract_cycle_data(df, target_cycle)
            
            if data_count == 0:
                st.warning(f"⚠️ '{sheet_name}': 사이클 {target_cycle} 데이터가 없습니다.")
                return None
            
            # 전압, 전류 컬럼명 찾기 (정보 표시용)
            voltage_col, current_col = self.find_voltage_current_columns(df)
            
            # 데이터 범위 표시
            v_min, v_max = voltage.min(), voltage.max()
            i_min, i_max = current.min(), current.max()
            
# 분석 준비 완료
            # 데이터 범위 정보
            
            # 환원 변곡점 탐지 (선택된 사이클 데이터에서)
            peak_info = self.find_reduction_inflection_point(voltage, current)
            
            if not peak_info:
                st.warning(f"⚠️ '{sheet_name}': 환원 변곡점을 찾을 수 없습니다.")
                return None
            
            # 적분값 계산 (기울기 기반 자동 영역 검출)
            integration_info = self.calculate_peak_integration(voltage, current, peak_info['original_index'])
            peak_info.update(integration_info)
            
            return {
                'sheet_name': sheet_name,
                'voltage': voltage,  # 선택된 사이클 데이터만
                'current': current,  # 선택된 사이클 데이터만
                'voltage_col': voltage_col if voltage_col else "전압(V)",
                'current_col': current_col if current_col else "전류(A)",
                'peak': peak_info,
                'data_points': data_count,
                'voltage_range': (v_min, v_max),
                'current_range': (i_min, i_max),
                'cycle_filter': f"사이클 횟수 = {target_cycle}",
                'cycle_analysis': cycle_analysis,
                'selected_cycle': target_cycle
            }
            
        except Exception as e:
            st.error(f"❌ '{sheet_name}' 분석 중 오류: {e}")
            return None
            
            # 환원 변곡점 탐지 (지정된 사이클 데이터에서)
            peak_info = self.find_reduction_inflection_point(voltage, current)
            
            if not peak_info:
                st.warning(f"⚠️ '{sheet_name}': 환원 변곡점을 찾을 수 없습니다.")
                return None
            
            # 적분값 계산 (기울기 기반 자동 영역 검출)
            integration_info = self.calculate_peak_integration(voltage, current, peak_info['original_index'])
            peak_info.update(integration_info)
            
            return {
                'sheet_name': sheet_name,
                'voltage': voltage,  # 지정된 사이클 데이터만
                'current': current,  # 지정된 사이클 데이터만
                'voltage_col': voltage_col if voltage_col else "전압(V)",
                'current_col': current_col if current_col else "전류(A)",
                'peak': peak_info,
                'data_points': data_count,
                'voltage_range': (v_min, v_max),
                'current_range': (i_min, i_max),
                'step_filter': f"사이클 번호 = {target_cycle}"
            }
            
        except Exception as e:
            st.error(f"❌ '{sheet_name}' 분석 중 오류: {e}")
            return None

def create_cv_plot(result: Dict, show_peak: bool = True) -> go.Figure:
    """첫 번째 사이클 CV 곡선 그래프 생성"""
    fig = go.Figure()
    
    # 첫 번째 사이클 CV 곡선
    fig.add_trace(go.Scatter(
        x=result['voltage'],
        y=result['current'],
        mode='lines',
        name=f"{result['sheet_name']} (1st Cycle)",
        line=dict(color='blue', width=2),
        hovertemplate="전압: %{x:.5f} V<br>" +
                    "전류: %{y:.6f} A<br>" +
                    "<extra></extra>"
    ))
    
    # 환원 변곡점 표시
    if show_peak and result['peak']:
        fig.add_trace(go.Scatter(
            x=[result['peak']['voltage']],
            y=[result['peak']['current']],
            mode='markers',
            name="환원 변곡점",
            marker=dict(
                color='red',
                size=12,
                symbol='circle',
                line=dict(color='darkred', width=2)
            ),
            hovertemplate="<b>🎯 환원 변곡점</b><br>" +
                        f"전압: {result['peak']['voltage']:.5f} V<br>" +
                        f"전류: {result['peak']['current']:.6f} A<br>" +
                        f"전류밀도: {result['peak']['current_density_ma']:.3f} mA<br>" +
                        f"적분값: {result['peak']['area_simpson']:.6f} A·V<br>" +
                        f"탐지방법: {result['peak']['method']}<br>" +
                        "<extra></extra>"
        ))
        
        # 기울기 변화점들 표시 (x1,y1), (x2,y2) - 최대 기울기 변화 지점
        if 'slope_change_points' in result['peak']:
            x1, y1 = result['peak']['slope_change_points']['left']
            x2, y2 = result['peak']['slope_change_points']['right']
            
            # 피크로부터의 거리 정보
            if 'calculation_details' in result['peak']:
                dist_left = result['peak']['calculation_details']['distance_left']
                dist_right = result['peak']['calculation_details']['distance_right']
            else:
                dist_left = dist_right = 0
            
            fig.add_trace(go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode='markers',
                name="최대 기울기 변화점",
                marker=dict(
                    color='orange',
                    size=12,
                    symbol='diamond',
                    line=dict(color='darkorange', width=2)
                ),
                hovertemplate="<b>🔸 최대 기울기 변화점</b><br>" +
                            "전압: %{x:.5f} V<br>" +
                            "전류: %{y:.6f} A<br>" +
                            f"피크로부터 거리: 왼쪽 {dist_left}점, 오른쪽 {dist_right}점<br>" +
                            "<extra></extra>"
            ))
        
        # Baseline 표시 - 순수하게 (x1,y1)과 (x2,y2) 두 점을 연결한 직선
        if ('baseline_params' in result['peak'] and 'slope_change_points' in result['peak']):
            
            # 기울기 변화점들
            x1, y1 = result['peak']['slope_change_points']['left']
            x2, y2 = result['peak']['slope_change_points']['right']
            
            # Baseline 매개변수
            baseline_slope = result['peak']['baseline_params']['slope']
            baseline_intercept = result['peak']['baseline_params']['intercept']
            
            # 두 점 사이와 약간 확장된 범위에서 baseline 그리기
            voltage_range = x2 - x1
            extension = voltage_range * 0.3  # 양쪽으로 30% 확장
            
            extended_v_min = max(result['voltage'].min(), x1 - extension)
            extended_v_max = min(result['voltage'].max(), x2 + extension)
            
            extended_v = np.linspace(extended_v_min, extended_v_max, 100)
            extended_baseline = baseline_slope * extended_v + baseline_intercept
            
            fig.add_trace(go.Scatter(
                x=extended_v,
                y=extended_baseline,
                mode='lines',
                name=f'Baseline: (x1,y1)↔(x2,y2)',
                line=dict(color='green', width=3, dash='dash'),
                hovertemplate='<b>Baseline (두 기울기 변화점 연결)</b><br>' +
                             'Voltage: %{x:.6f} V<br>' +
                             'Current: %{y:.6f} A<br>' +
                             f'기울기: {baseline_slope:.2e} A/V<br>' +
                             f'점1: ({x1:.4f}V, {y1:.2e}A)<br>' +
                             f'점2: ({x2:.4f}V, {y2:.2e}A)<extra></extra>'
            ))
        
        # 적분 영역 색칠 (baseline 아래 peak 부분만)
        if ('integration_voltage' in result['peak'] and 'integration_current' in result['peak'] and
            'baseline_params' in result['peak'] and result['peak']['peak_below_baseline']):
            
            # 실제 적분에 기여한 영역만 색칠
            int_v = result['peak']['integration_voltage']
            int_c = result['peak']['integration_current']
            
            if len(int_v) > 0:
                # baseline 매개변수를 사용해서 정확한 baseline 계산
                baseline_slope = result['peak']['baseline_params']['slope']
                baseline_intercept = result['peak']['baseline_params']['intercept']
                baseline_interp = baseline_slope * int_v + baseline_intercept
                
                # 적분 영역 색칠 (baseline과 곡선 사이)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([int_v, int_v[::-1]]),
                    y=np.concatenate([int_c, baseline_interp[::-1]]),
                    fill='toself',
                    mode='none',
                    name=f'적분 영역 (면적: {result["peak"]["area_trapz"]:.2e} A·V)',
                    fillcolor='rgba(255, 100, 100, 0.4)',
                    hoverinfo='skip',
                    showlegend=True
                ))
                
                # 적분 영역 경계선 표시
                fig.add_trace(go.Scatter(
                    x=[int_v[0], int_v[0], int_v[-1], int_v[-1]],
                    y=[min(result['current']) * 1.1, max(result['current']) * 1.1, 
                       max(result['current']) * 1.1, min(result['current']) * 1.1],
                    mode='lines',
                    name='적분 경계',
                    line=dict(color='red', width=1, dash='dot'),
                    hoverinfo='skip',
                    showlegend=False
                ))
    
    # 적절한 축 범위 설정 (첫 번째 사이클 기준)
    v_min, v_max = result['voltage_range']
    i_min, i_max = result['current_range']
    
    # 축 범위에 여유 공간 추가
    v_margin = (v_max - v_min) * 0.05
    i_margin = (i_max - i_min) * 0.05
    
    # 그래프 레이아웃
    fig.update_layout(
        title=dict(
            text=f"🔋 {result['sheet_name']} - 첫 번째 사이클 CV 분석",
            x=0.5,
            font=dict(size=18, color='#1f77b4')
        ),
        xaxis=dict(
            title="전압 (V)",
            range=[v_min - v_margin, v_max + v_margin],
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray'
        ),
        yaxis=dict(
            title="전류 (A)",
            range=[i_min - i_margin, i_max + i_margin],
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray'
        ),
        height=500,
        hovermode='closest',
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        margin=dict(r=150),
        showlegend=True
    )
    
    return fig

def create_results_table(analysis_results: List[Dict]) -> pd.DataFrame:
    """분석 결과 테이블 생성 - 사이클 정보 포함"""
    table_data = []
    
    for result in analysis_results:
        if result['peak']:
            table_data.append({
                '샘플명': result['sheet_name'],
                '환원전위 (V)': f"{result['peak']['voltage']:.5f}",
                '피크전류 (A)': f"{result['peak']['current']:.6f}",
                '전류밀도 (mA)': f"{result['peak']['current_density_ma']:.3f}",
                '적분값 (A·V)': f"{result['peak']['area_simpson']:.6f}",
                '사이클정보': result['peak'].get('step_info', 'N/A'),
                '적분범위': result['peak']['voltage_range'],
                '전압범위': f"{result['voltage_range'][0]:.3f}~{result['voltage_range'][1]:.3f}V"
            })
    
    return pd.DataFrame(table_data)

def create_overlay_plot(analysis_results: List[Dict]) -> go.Figure:
    """모든 샘플의 결과를 오버레이한 그래프 생성"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, result in enumerate(analysis_results):
        if result is None:
            continue
            
        color = colors[i % len(colors)]
        sheet_name = result['sheet_name']
        
        # CV 곡선 그리기
        fig.add_trace(go.Scatter(
            x=result['voltage'],
            y=result['current'],
            mode='lines',
            name=f'{sheet_name}',
            line=dict(color=color, width=2),
            hovertemplate=f'<b>{sheet_name}</b><br>' +
                         'Voltage: %{x:.6f} V<br>' +
                         'Current: %{y:.6f} A<extra></extra>'
        ))
        
        # 환원전위 포인트 표시
        peak = result['peak']
        fig.add_trace(go.Scatter(
            x=[peak['voltage']],
            y=[peak['current']],
            mode='markers',
            name=f'{sheet_name} 환원전위',
            marker=dict(
                color=color,
                size=12,
                symbol='star',
                line=dict(color='black', width=2)
            ),
            hovertemplate=f'<b>{sheet_name} 환원전위</b><br>' +
                         f'Voltage: {peak["voltage"]:.6f} V<br>' +
                         f'Current: {peak["current"]:.6f} A<br>' +
                         f'Current Density: {peak["current_density_ma"]:.3f} mA<br>' +
                         f'Integration: {peak["area_simpson"]:.6f} A·V<extra></extra>'
        ))
        
        # 오버레이에서는 baseline과 적분 영역 색칠 안 함 (깔끔한 표시를 위해)
    
    fig.update_layout(
        title=dict(
            text='🔋 모든 샘플 CV 분석 결과 오버레이',
            font=dict(size=20, color='#1f77b4'),
            x=0.5
        ),
        xaxis_title='전압 (V)',
        yaxis_title='전류 (A)',
        hovermode='closest',
        template='plotly_white',
        width=1000,
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

def export_results_to_excel(analysis_results: List[Dict]) -> bytes:
    """분석 결과를 그래프와 함께 엑셀로 내보내기"""
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 1. 요약 테이블
            summary_df = create_results_table(analysis_results)
            summary_df.to_excel(writer, sheet_name='분석결과_요약', index=False)
            
            # 2. 0번 사이클만 추출한 통합 데이터
            cycle_0_data = []
            for result in analysis_results:
                if result is None:
                    continue
                cycle_0_data.append({
                    '샘플명': result['sheet_name'],
                    '환원전위(V)': result['peak']['voltage'],
                    '환원전류(A)': result['peak']['current'],
                    '전류밀도(mA)': result['peak']['current_density_ma'],
                    '적분값(A·V)': result['peak']['area_simpson'],
                    '적분점수': result['peak']['integration_points'],
                    '선택사이클': result.get('selected_cycle', 0)
                })
            
            cycle_0_summary = pd.DataFrame(cycle_0_data)
            cycle_0_summary.to_excel(writer, sheet_name='사이클별_요약', index=False)
            
            # 3. 각 샘플의 상세 데이터 (0번 사이클 로우데이터)
            for result in analysis_results:
                if result is None:
                    continue
                    
                sheet_name = f"{result['sheet_name']}_사이클{result.get('selected_cycle', 0)}"
                sheet_name = sheet_name[:31]  # 엑셀 시트명 길이 제한
                
                # 로우 데이터
                data_df = pd.DataFrame({
                    '전압(V)': result['voltage'],
                    '전류(A)': result['current'],
                    '환원전위표시': ['✓' if abs(v - result['peak']['voltage']) < 1e-6 else '' 
                                  for v in result['voltage']]
                })
                
                data_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 워크시트에 추가 정보 기록
                worksheet = writer.sheets[sheet_name]
                
                # 환원전위 정보 추가
                info_row = len(data_df) + 3
                worksheet[f'A{info_row}'] = '=== 환원 변곡점 정보 ==='
                worksheet[f'A{info_row + 1}'] = f'환원전위(V): {result["peak"]["voltage"]:.6f}'
                worksheet[f'A{info_row + 2}'] = f'환원전류(A): {result["peak"]["current"]:.6f}'
                worksheet[f'A{info_row + 3}'] = f'전류밀도(mA): {result["peak"]["current_density_ma"]:.3f}'
                worksheet[f'A{info_row + 4}'] = f'적분값(A·V): {result["peak"]["area_simpson"]:.6f}'
                worksheet[f'A{info_row + 5}'] = f'적분범위: {result["peak"]["voltage_range"]}'
                worksheet[f'A{info_row + 6}'] = f'탐지방법: {result["peak"]["method"]}'
                
                # 엑셀 차트 생성
                try:
                    from openpyxl.chart import ScatterChart, Reference, Series
                    
                    chart = ScatterChart()
                    chart.title = f'{result["sheet_name"]} CV 곡선'
                    chart.style = 2
                    chart.x_axis.title = '전압 (V)'
                    chart.y_axis.title = '전류 (A)'
                    
                    # 데이터 범위 설정
                    xvalues = Reference(worksheet, min_col=1, min_row=2, max_row=len(data_df)+1)
                    yvalues = Reference(worksheet, min_col=2, min_row=2, max_row=len(data_df)+1)
                    
                    series = Series(yvalues, xvalues, title="CV 곡선")
                    chart.series.append(series)
                    
                    # 차트 위치 설정
                    chart.width = 15
                    chart.height = 10
                    worksheet.add_chart(chart, f'E1')
                    
                except Exception as chart_error:
                    print(f"차트 생성 중 오류: {chart_error}")
        
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        print(f"엑셀 내보내기 중 오류: {e}")
        return b""
        
        output.seek(0)
        return output.read()
        
    except Exception as e:
        st.error(f"엑셀 내보내기 오류: {e}")
        return b""

def main():
    # 헤더
    st.markdown('<h1 class="main-header">🔋 CV 분석 플랫폼</h1>', unsafe_allow_html=True)
    st.markdown("**순환전압전류법(CV) 데이터에서 환원 변곡점을 자동 검출하고 분석합니다**")
    st.markdown("---")
    
    # 사이드바
    with st.sidebar:
        st.header("📁 데이터 업로드")
        st.info("💡 **데이터 형식**: C열에 사이클 번호가 포함된 엑셀 파일")
        uploaded_file = st.file_uploader(
            "CV 데이터 파일 선택",
            type=['xlsx', 'xls'],
            help="여러 시트(샘플)가 포함된 엑셀 파일을 업로드하세요."
        )
        
        st.header("⚙️ 분석 설정")
        
        # 적분 방법 선택
        integration_method = st.radio(
            "적분 영역 검출 방법",
            ["기울기 변화 기반 (추천)", "고정 윈도우"],
            help="기울기 변화를 기반으로 최적의 적분 영역을 자동 검출합니다."
        )
        
        if integration_method == "고정 윈도우":
            integration_window = st.slider("적분 윈도우 크기 (데이터 점)", 10, 60, 30, 5)
        else:
            integration_window = 30
        
        # 표시 옵션을 더 간단하게
        st.header("� 표시 설정")
        show_peaks = st.checkbox("환원 변곡점 표시", value=True)
        show_data_preview = st.checkbox("데이터 미리보기", value=False)
        plot_height = st.slider("그래프 높이", 400, 800, 500, 50)
        
        # 디버깅 옵션은 기본값으로 설정
        show_debug = True
    
    # 메인 컨텐츠
    if uploaded_file:
        analyzer = CVAnalyzer()
        
        # 데이터 로딩
        with st.spinner("📊 엑셀 파일 로딩 중..."):
            data_sheets = analyzer.load_excel_data(uploaded_file)
        
        if data_sheets:
# 데이터 로드 완료
            
            # 각 시트 분석
            analysis_results = []
            
            with st.spinner("🔍 CV 데이터 분석 중..."):
                for sheet_name, df in data_sheets.items():
                    # 분석 진행
                    
                    if show_data_preview:
                        st.write("📋 데이터 미리보기:")
                        st.dataframe(df.head(10))
                    
                    result = analyzer.analyze_cv_data(sheet_name, df)
                    if result:
                        # 적분 방법에 따라 다시 계산 (고정 윈도우 방법인 경우만)
                        if integration_method == "고정 윈도우":
                            # 기존 결과 초기화하고 고정 윈도우로 재계산
                            integration_info = analyzer.calculate_peak_integration_fixed_window(
                                result['voltage'], result['current'], 
                                result['peak']['original_index'], integration_window
                            )
                            result['peak'].update(integration_info)
                        # 기울기 변화 기반 방법은 이미 analyze_cv_data에서 계산됨
                        
                        analysis_results.append(result)
                        
                        # 분석 완료
                        pass
                    else:
                        if show_debug:
                            st.error(f"❌ {sheet_name} 분석 실패!")
                    
                    st.markdown("---")
            
            if analysis_results:
# 전체 분석 완료
                
                # 모든 샘플 오버레이 그래프
                if len(analysis_results) > 1:
                    st.markdown('<h2 class="section-header">🔄 모든 샘플 오버레이 비교</h2>', unsafe_allow_html=True)
                    overlay_fig = create_overlay_plot(analysis_results)
                    st.plotly_chart(overlay_fig, use_container_width=True)
                
                # 적분 방법 설명
                st.markdown("""
                <div class="analysis-box">
                    <h4>📐 적분 계산 방법</h4>
                    <ul>
                        <li><strong>Baseline:</strong> 적분 구간의 시작점과 끝점을 연결한 직선</li>
                        <li><strong>적분 영역:</strong> Baseline 아래로 내려간 Peak 부분만 계산</li>
                        <li><strong>면적 계산:</strong> Baseline과 실제 곡선 사이의 면적 (수치적분)</li>
                        <li><strong>시각화:</strong> 초록색 점선(Baseline), 빨간 음영(적분 영역)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # 통합 결과 테이블
                st.markdown('<h2 class="section-header">📊 통합 분석 결과 테이블</h2>', unsafe_allow_html=True)
                summary_table = create_results_table(analysis_results)
                st.dataframe(summary_table, use_container_width=True)
                
                # 엑셀 다운로드 버튼
                excel_data = export_results_to_excel(analysis_results)
                st.download_button(
                    label="📥 엑셀 파일 다운로드 (차트 포함)",
                    data=excel_data,
                    file_name="CV_분석결과_with_charts.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # 각 시트별 상세 그래프 표시
                st.markdown('<h2 class="section-header">📈 개별 샘플 상세 결과</h2>', unsafe_allow_html=True)
                
                for result in analysis_results:
                    st.markdown(f"### 📊 {result['sheet_name']}")
                    
                    # CV 그래프
                    fig = create_cv_plot(result, show_peaks)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 분석 결과 메트릭
                    if result['peak']:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("🎯 환원전위", f"{result['peak']['voltage']:.5f} V")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("⚡ 피크전류", f"{result['peak']['current']:.6f} A")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("📊 전류밀도", f"{result['peak']['current_density_ma']:.3f} mA")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("🔺 적분값", f"{result['peak']['area_simpson']:.6f} A·V")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 상세 정보
                        with st.expander(f"🔍 {result['sheet_name']} 상세 정보"):
                            st.markdown(f"""
                            <div class="peak-info">
                                <h4>🎯 환원 변곡점 정보</h4>
                                <ul>
                                    <li><strong>환원전위:</strong> {result['peak']['voltage']:.6f} V</li>
                                    <li><strong>피크전류:</strong> {result['peak']['current']:.6f} A</li>
                                    <li><strong>전류밀도:</strong> {result['peak']['current_density_ma']:.3f} mA</li>
                                    <li><strong>적분값:</strong> {result['peak']['area_simpson']:.6f} A·V</li>
                                    <li><strong>적분범위:</strong> {result['peak']['voltage_range']}</li>
                                    <li><strong>적분점수:</strong> {result['peak']['integration_points']}개</li>
                                    <li><strong>탐지방법:</strong> {result['peak']['method']}</li>
                                    <li><strong>사이클정보:</strong> {result['peak'].get('step_info', 'N/A')}</li>
                                </ul>
                                
                                <h4>📊 사이클 데이터 정보</h4>
                                <ul>
                                    <li><strong>사이클 데이터:</strong> {result['data_points']}개 포인트</li>
                                    <li><strong>전압범위:</strong> {result['voltage_range'][0]:.5f} ~ {result['voltage_range'][1]:.5f} V</li>
                                    <li><strong>전류범위:</strong> {result['current_range'][0]:.6f} ~ {result['current_range'][1]:.6f} A</li>
                                    <li><strong>필터조건:</strong> {result.get('step_filter', result.get('cycle_filter', 'N/A'))}</li>
                                    <li><strong>시트명:</strong> {result['sheet_name']}</li>
                                    <li><strong>사용컬럼:</strong> {result['voltage_col']}, {result['current_col']}</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                
                # 전체 분석 결과 테이블
                st.markdown('<h2 class="section-header">📋 전체 분석 결과 요약</h2>', unsafe_allow_html=True)
                
                # 전류 밀도 설명
                with st.expander("ℹ️ 전류 밀도란?"):
                    st.markdown("""
                    **전류 밀도 (Current Density)**는 전극 표면적 단위당 흐르는 전류의 크기를 나타냅니다.
                    
                    - **단위**: mA (밀리암페어) - 피크 전류 × 1000
                    - **의미**: 환원 반응의 강도를 표현하는 지표
                    - **특징**: 값이 클수록 더 강한 환원 반응이 일어남을 의미
                    - **계산**: 피크 전류(A)를 밀리암페어(mA) 단위로 변환한 값
                    """)
                
                results_df = create_results_table(analysis_results)
                st.dataframe(results_df, use_container_width=True)
                
                # 결과 내보내기
                st.markdown('<h3 class="section-header">💾 결과 내보내기</h3>', unsafe_allow_html=True)
                
                if st.button("📥 Excel 파일로 다운로드", type="primary"):
                    excel_data = export_results_to_excel(analysis_results)
                    if excel_data:
                        st.download_button(
                            label="📁 CV_환원변곡점_분석결과.xlsx 다운로드",
                            data=excel_data,
                            file_name="CV_환원변곡점_분석결과.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            else:
                st.error("❌ 분석 가능한 CV 데이터가 없습니다.")
        
        else:
            st.error("❌ 유효한 시트를 찾을 수 없습니다.")
    
    else:
        # 시작 안내
        st.markdown('<h2 class="section-header">� 시작하기</h2>', unsafe_allow_html=True)
        
        # 간단한 3단계 안내
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="analysis-box" style="text-align: center;">
                <h3>1️⃣</h3>
                <h4>파일 업로드</h4>
                <p>사이드바에서 CV 데이터<br>엑셀 파일을 선택하세요</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="analysis-box" style="text-align: center;">
                <h3>2️⃣</h3>
                <h4>자동 분석</h4>
                <p>각 시트별로 환원 변곡점을<br>자동으로 검출합니다</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="analysis-box" style="text-align: center;">
                <h3>3️⃣</h3>
                <h4>결과 확인</h4>
                <p>그래프와 표로 결과를<br>확인하고 다운로드하세요</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 데이터 형식 안내
        with st.expander("📋 데이터 형식 요구사항"):
            st.markdown("""
            **필수 컬럼:**
            - **C열**: 사이클 번호 (필터링 기준)
            - **G열**: 전압 데이터 (V)
            - **H열**: 전류 데이터 (A)
            
            **파일 형식:**
            - Excel 파일 (.xlsx, .xls)
            - 시트별로 하나의 샘플 데이터
            - '정보' 시트는 자동으로 무시됩니다
            """)
        
        st.success("👆 **사이드바에서 파일을 업로드하여 분석을 시작하세요!**")

if __name__ == "__main__":
    main()
