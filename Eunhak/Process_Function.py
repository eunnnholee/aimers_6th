import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RareCategoryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10, columns=None):
        """
        Parameters:
        threshold : int
            해당 변수의 값의 빈도수가 이 값 이하이면 희귀 범주로 간주.
        columns : list or None
            변환할 열 리스트. None이면 객체형(object) 또는 범주형(category) 열 전체에 적용.
        """
        self.threshold = threshold
        self.columns = columns

    def fit(self, X, y=None):
        # columns가 지정되지 않으면, object나 category 타입의 모든 열을 사용
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

        self.frequent_categories_ = {}
        # 각 열에 대해 train 데이터의 빈도수를 계산 후, threshold보다 큰 경우만 보존
        for col in self.columns:
            freq = X[col].value_counts()
            # 빈도수가 threshold 이하이면 희귀 범주로 판단 (여기서는 > threshold인 경우만 보존)
            self.frequent_categories_[col] = freq[freq > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        # fit 단계에서 선정한 범주 이외의 값은 모두 "others"로 대체
        for col in self.columns:
            X[col] = X[col].apply(lambda x: x if x in self.frequent_categories_[col] else "others")
        return X

