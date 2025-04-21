import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class PreProcessor:
    def __init__(self):
        # MinMaxScaler 초기화
        self.scaler = MinMaxScaler()

    def add_high_diff_feature(self, train, test, col, base_value_key, target_col='임신 성공 확률'):
        """
        회귀 타스크(임신 성공 확률)가 연속값이므로, 0.5를 기준으로 이진화한 후,
        train 데이터의 이진 target 분포를 기반으로 pivot table을 생성하여,
        0의 비율 보정, Diff, Scaled_Diff를 계산하고,
        기준 고유값(base_value_key)의 Scaled_Diff보다 큰 고유값에 대해 파생변수를 생성합니다.
        """
        # target을 0.5 기준으로 이진화하여 임시 컬럼 생성
        train['_binary_target'] = (train[target_col] >= 0.5).astype(int)
        
        # target 컬럼의 클래스 비율 계산 (0 / 1)
        n_normal = len(train[train['_binary_target'] == 0])
        n_abnormal = len(train[train['_binary_target'] == 1])
        ratio = n_normal / n_abnormal if n_abnormal != 0 else 1.0
        
        # pivot table 생성: 각 고유값별로 _binary_target의 빈도수를 계산
        result = train.pivot_table(index=col, columns='_binary_target', aggfunc='size', fill_value=-1)
        if 0 in result.columns and 1 in result.columns:
            result[0] = result[0] / ratio
            result['Diff'] = (result[0] - result[1]).abs()
            result['Scaled_Diff'] = self.scaler.fit_transform(result[['Diff']])
            
            # 기준 고유값(base_value_key)의 Scaled_Diff 값 확인
            if base_value_key in result.index:
                base_value = result.loc[base_value_key, 'Scaled_Diff']
            else:
                base_value = None

            if base_value is not None:
                # 기준 고유값보다 Scaled_Diff가 큰 고유값들 선택
                high_diff_indices = result[result['Scaled_Diff'] > base_value].index
                new_col_name = f"{col}_high_diff"
                train[new_col_name] = train[col].isin(high_diff_indices).astype(int)
                test[new_col_name] = test[col].isin(high_diff_indices).astype(int)
                
        # 임시로 추가한 이진 타깃 컬럼 삭제
        train.drop(columns=['_binary_target'], inplace=True)
        return train, test

    def apply_high_diff_features(self, train, test, col_base_map, target_col='임신 성공 확률'):
        """
        col_base_map: {컬럼명: 기준값(base_value_key)} 형태의 딕셔너리.
        각 범주형 변수에 대해 add_high_diff_feature()를 적용하여 파생변수를 생성
        """
        for col, base_value_key in col_base_map.items():
            train, test = self.add_high_diff_feature(train, test, col, base_value_key, target_col=target_col)
        return train, test

    def add_high_diff_feature_numeric(self, train, test, col, target_col='임신 성공 확률'):
        """
        수치형 변수에 대해, 회귀 타스크의 target을 0.5 기준으로 이진화하여 pivot table을 생성하고,
        0의 비율 보정, Diff, Scaled_Diff를 계산한 후,
        Scaled_Diff가 상위 20%(quantile 0.75) 이상인 고유값에 대해 파생변수를 생성합니다.
        """
        # target을 0.5 기준으로 이진화하여 임시 컬럼 생성
        train['_binary_target'] = (train[target_col] >= 0.5).astype(int)
        
        n_normal = len(train[train['_binary_target'] == 0])
        n_abnormal = len(train[train['_binary_target'] == 1])
        ratio = n_normal / n_abnormal if n_abnormal != 0 else 1.0
        
        result = train.pivot_table(index=col, columns='_binary_target', aggfunc='size', fill_value=-1)
        if 0 in result.columns and 1 in result.columns:
            result[0] = result[0] / ratio
            result['Diff'] = (result[0] - result[1]).abs()
            result['Scaled_Diff'] = self.scaler.fit_transform(result[['Diff']])
            
            # 상위 20% 기준의 threshold 계산
            threshold = result['Scaled_Diff'].quantile(0.75)
            high_diff_indices = result[result['Scaled_Diff'] > threshold].index
            new_col_name = f"{col}_high_diff"

            train[new_col_name] = train[col].isin(high_diff_indices).astype(int)
            test[new_col_name] = test[col].isin(high_diff_indices).astype(int)
            
        train.drop(columns=['_binary_target'], inplace=True)
        return train, test

    def apply_high_diff_numeric_features(self, train, test, num_cols, target_col='임신 성공 확률'):
        """
        num_cols: 파생변수를 생성할 수치형 변수 목록.
        각 수치형 변수에 대해 add_high_diff_feature_numeric()를 적용하여 파생변수를 생성
        """
        for col in num_cols:
            train, test = self.add_high_diff_feature_numeric(train, test, col, target_col=target_col)
        return train, test

    def all_process(self, train, test):
        """
        전처리 전체 프로세스 실행 함수
        현재는 수치형 변수에 대해 파생변수를 생성하도록 되어 있습니다.
        """
        # 예시: col_base_map이 필요하다면 주석 해제하여 범주형 변수도 적용 가능
        # col_base_map = {
        #     '배아 이식 경과일': 2.0,
        #     '총 임신 횟수': '3회',
        #     ...
        # }
        # train, test = self.apply_high_diff_features(train, test, col_base_map, target_col='임신 성공 확률')
        
        # 수치형 변수: 상위 20% 기준
        tmp_num_cols = [
            '이전 IVF 시술 횟수', 
            '이전 DI 시술 횟수', 
            '이전 총 임신 횟수', 
            '이전 총 임신 성공 횟수', 
            '이식된 배아 수', 
            '미세주입(ICSI) 배아 이식 수'
        ]
        
        train, test = self.apply_high_diff_numeric_features(train, test, tmp_num_cols, target_col='임신 성공 확률')
        
        return train, test
