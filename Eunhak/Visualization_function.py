import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def abbreviate_number(num):
    if num >= 1_000_000:
        return f'{num/1_000_000:.1f}M'
    elif num >= 1_000:
        return f'{num/1_000:.1f}K'
    else:
        return str(num)

def visualize_continuous_with_target(df, cols_to_visualize, bins=None, figure_size=(12, 6), target_column='임신 성공 여부'):
    """
    연속형 변수의 분포와 target 비율을 함께 시각화합니다.

    Parameters:
    - df: DataFrame
    - cols_to_visualize: 시각화할 변수들의 리스트 (연속형 혹은 이산형)
    - bins: 연속형 변수를 나눌 bin의 개수. 기본값이 None이면 각 컬럼마다 Scott's Rule로 결정.
    - figure_size: 그래프의 크기 (기본값 (12,6))
    - target_column: target 열 이름 (기본값 '임신 성공 여부')
    """
    import numpy as np

    # Scott's Rule 함수 정의 (연속형 변수용)
    def scotts_rule(data):
        data = np.asarray(data)
        n = data.size
        if n == 0:
            return 0
        sigma = data.std(ddof=1)
        bin_width = 3.5 * sigma / np.cbrt(n)
        if bin_width == 0:
            return 1
        num_bins = int(np.ceil((data.max() - data.min()) / bin_width))
        return num_bins

    # 1. 전체 컬럼에 대해 전역 최소/최대 target ratio 값을 구하기
    global_min = float('inf')
    global_max = float('-inf')
    for col in cols_to_visualize:
        bins_for_col = scotts_rule(df[col]) if bins is None else bins
        # 이산형 변수인지 확인: 고유값 수가 bins_for_col 이하인 경우
        if df[col].nunique() <= bins_for_col:
            binned = df[col].astype(str)
        else:
            binned = pd.cut(df[col], bins=bins_for_col)
        counts = binned.value_counts(sort=False)
        if counts.empty:
            continue
        ratios = df.groupby(binned)[target_column].mean().loc[counts.index]
        col_min = ratios.min()
        col_max = ratios.max()
        if col_min < global_min:
            global_min = col_min
        if col_max > global_max:
            global_max = col_max

    if global_min == float('inf') or global_max == float('-inf'):
        global_min, global_max = 0, 1

    margin = (global_max - global_min) * 0.1
    y_lower = global_min - margin
    y_upper = global_max + margin

    # 2. 각 컬럼별로 시각화
    for col in cols_to_visualize:
        bins_for_col = scotts_rule(df[col]) if bins is None else bins
        # 이산형 변수 판단: 고유값 수가 bins_for_col 이하라면
        if df[col].nunique() <= bins_for_col:
            binned = df[col].astype(str)
            cat = binned.astype('category')
            categories = list(cat.cat.categories)
            try:
                sorted_categories = sorted(categories, key=lambda x: float(x))
            except ValueError:
                sorted_categories = categories
            counts = binned.value_counts().reindex(sorted_categories)
            ratios = df.groupby(binned)[target_column].mean().reindex(sorted_categories)

            midpoints = np.arange(len(sorted_categories))
            bar_widths = np.full_like(midpoints, 0.8, dtype=float)
            xtick_labels = sorted_categories
        else:
            binned = pd.cut(df[col], bins=bins_for_col)
            counts = binned.value_counts(sort=False)
            ratios = df.groupby(binned)[target_column].mean().loc[counts.index]
            bin_intervals = binned.cat.categories
            bin_edges = [interval.left for interval in bin_intervals] + [bin_intervals[-1].right]
            bin_edges = np.array(bin_edges)
            bar_widths = np.diff(bin_edges)
            midpoints = bin_edges[:-1] + bar_widths / 2
            xtick_labels = [abbreviate_number(edge) for edge in bin_edges]

        fig, ax1 = plt.subplots(figsize=figure_size)
        ax2 = ax1.twinx()

        bars = ax1.bar(midpoints, counts.values, width=bar_widths, color='skyblue', alpha=0.7, align='center')
        ax1.set_xlabel(col, fontsize=12)
        ax1.set_ylabel('Count', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')

        num_ticks = len(xtick_labels)
        dynamic_fontsize = max(6, 12 - (num_ticks - 11) * 0.5)
        # x축 tick 설정
        if df[col].nunique() <= bins_for_col:
            ax1.set_xticks(midpoints)
            ax1.set_xticklabels(xtick_labels, fontsize=dynamic_fontsize, rotation=45)
        else:
            ax1.set_xticks(bin_edges)
            ax1.set_xticklabels(xtick_labels, fontsize=dynamic_fontsize, rotation=45)

        ax2.plot(midpoints, ratios.values, color='red', marker='o', linewidth=0.5, markersize=2)
        ax2.set_ylabel(f'{target_column} Ratio (%)', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(y_lower, y_upper)

        for bar in bars:
            height = bar.get_height()
            if height == 0:
                continue
            ax1.annotate(abbreviate_number(height),
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=9, color='black')

        for x, ratio in zip(midpoints, ratios.values):
            ax2.annotate(f'{ratio * 100:.1f}',
                         xy=(x, ratio),
                         xytext=(0, 8),
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=9, color='yellow')

        plt.title(f"{col} - Distribution and {target_column} Ratio (bins: {bins_for_col})", fontsize=12)
        plt.show()


def visualize_with_target(df, cols_to_visualize, threshold=10, figure_size=(12, 6), target_column='임신 성공 여부'):
    # 1. 전체 컬럼에 대한 ratio의 전역 최소/최대값 구하기
    global_min = float('inf')
    global_max = float('-inf')
    for col in cols_to_visualize:
        series_str = df[col].astype(str)
        counts = series_str.value_counts(sort=False)
        counts = counts[counts >= threshold]
        if counts.empty:
            continue  # 해당 컬럼은 건너뜀
        ratios = df.groupby(series_str)[target_column].mean().loc[counts.index]
        col_min = ratios.min()
        col_max = ratios.max()
        if col_min < global_min:
            global_min = col_min
        if col_max > global_max:
            global_max = col_max

    # 전역 최소/최대값이 갱신되지 않았다면 기본값 설정 (예: 확률은 0~1)
    if global_min == float('inf') or global_max == float('-inf'):
        global_min, global_max = 0, 1

    # 약간의 여유분(margin) 추가 (전체 범위의 10% 정도)
    margin = (global_max - global_min) * 0.1
    y_lower = global_min - margin
    y_upper = global_max + margin


    for col in cols_to_visualize:
        # 원본 데이터의 값을 문자열로 변환 (혼합형 문제 해결)
        series_str = df[col].astype(str)

        # 각 범주의 빈도수와 target 비율 계산
        counts = series_str.value_counts(sort=False)
        # 빈도수가 임계치 미만인 범주는 제거
        counts = counts[counts >= threshold]
        if counts.empty:
            print(f"{col}에 빈도수가 {threshold}개 이상인 값이 없습니다.")
            continue

        # 그룹별 target 비율 계산 후, 빈도수 기준 필터 적용
        ratios = df.groupby(series_str)[target_column].mean().loc[counts.index]

        # 정렬: 숫자로 변환 가능한 값과 그렇지 않은 값을 분리
        numeric_keys = []
        non_numeric_keys = []
        for key in counts.index:
            try:
                numeric_val = float(key)
                numeric_keys.append((numeric_val, key))
            except ValueError:
                non_numeric_keys.append(key)

        # 숫자 범주는 오름차순 정렬, 문자 범주는 사전순 정렬
        sorted_numeric_keys = [k for _, k in sorted(numeric_keys, key=lambda x: x[0])]
        sorted_non_numeric_keys = sorted(non_numeric_keys)
        sorted_keys = sorted_numeric_keys + sorted_non_numeric_keys

        # 정렬된 순서에 따라 counts와 ratios 재정렬
        counts = counts.loc[sorted_keys]
        ratios = ratios.loc[sorted_keys]

        # 시각화
        fig, ax1 = plt.subplots(figsize=figure_size)
        ax2 = ax1.twinx()  # 오른쪽 y축 생성

        # 막대그래프: 빈도수
        bars = ax1.bar(counts.index, counts.values, color='skyblue', alpha=0.7)
        ax1.set_xlabel(col, fontsize=12)
        ax1.set_ylabel('Count', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.tick_params(axis='x', rotation=45)

        # 선 그래프: '임신 성공 여부' 비율
        ax2.plot(ratios.index, ratios.values, color='red', marker='o', linewidth=0.5, markersize=2)
        ax2.set_ylabel(f'{target_column} Ratio (%)', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(y_lower, y_upper)  # 모든 그래프에 동일한 y축 범위 적용

        # 막대 위에 count 값 표시
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(abbreviate_number(height),
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3),  # 약간 위쪽에 표시
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=9, color='black')

        # 선 위에 ratio 값 표시
        # ax2의 선 그래프의 x 좌표를 직접 추출하여 사용
        line = ax2.lines[0]
        for x, ratio in zip(line.get_xdata(), ratios.values):
            ax2.annotate(f'{ratio * 100:.1f}',
                         xy=(x, ratio),
                         xytext=(0, 8),  # 약간 위쪽에 표시
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=9, color='yellow')

        plt.title(f"{col} - Frequency and {target_column} Ratio (빈도수 {threshold} 이상)", fontsize=12)
        plt.show()
