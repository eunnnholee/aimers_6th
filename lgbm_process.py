# -*- coding: utf-8 -*-
"""
preprocessing.py

이 모듈은 주어진 데이터프레임에 대해 전처리를 진행하는 여러 함수를 포함합니다.
주피터 노트북 등에서 아래와 같이 import하여 사용할 수 있습니다.

    from preprocessing import lgbm_42_process, drop_columns, 특정시술유형, 시술횟수, encoding, numeric_process, ...

또는 전체 실행 예시:

    import pandas as pd
    from preprocessing import lgbm_42_process
    train = pd.read_csv("train.csv").drop(columns=["ID"])
    test = pd.read_csv("test.csv").drop(columns=["ID"])
    train, test = lgbm_42_process(train, test)
    print(train.shape, test.shape)
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MultiLabelBinarizer
# from category_encoders import TargetEncoder, CountEncoder
from prince import MCA
# from sklearn.preprocessing import StandardScaler  # 필요시 사용

def drop_columns(df):
    cols = [
        '불임 원인 - 여성 요인',  # 고유값 1
        '불임 원인 - 정자 면역학적 요인',  # train, test 모두 '1'인 데이터 1개 >> 신뢰할 수 없음
        '난자 해동 경과일',
    ]
    df = df.drop(cols, axis=1)
    return df

def 특정시술유형(train, test):
    def categorize_procedure(proc):
        tokens = [token.strip() for token in proc.split(",") if token.strip() and not token.strip().isdigit()]
        # 우선순위에 따른 범주화
        if tokens.count("Unknown") >= 1:
            return "Unknown"
        if tokens.count("AH") >= 1:
            return "AH"
        if tokens.count("BLASTOCYST") >= 1:
            return "BLASTOCYST"
        if tokens.count("ICSI") >= 2 or tokens.count("IVF") >= 2:
            return "2ICSI_2IVF"
        if tokens.count("IVF") >= 1 and tokens.count("ICSI") >= 1:
            return "IVF_ICSI"
        if tokens == "ICSI":
            return "ICSI"
        if tokens == "IVF":
            return "IVF"
        return ",".join(tokens) if tokens else None

    for df in [train, test]:
        df['특정 시술 유형'] = df['특정 시술 유형'].str.replace(" / ", ",")
        df['특정 시술 유형'] = df['특정 시술 유형'].str.replace(":", ",")
        df['특정 시술 유형'] = df['특정 시술 유형'].str.replace(" ", "")

    counts = train['특정 시술 유형'].value_counts()
    allowed_categories = counts[counts >= 100].index.tolist()

    # allowed_categories에 속하지 않는 값은 "Unknown"으로 대체
    train.loc[~train['특정 시술 유형'].isin(allowed_categories), '특정 시술 유형'] = "Unknown"
    test.loc[~test['특정 시술 유형'].isin(allowed_categories), '특정 시술 유형'] = "Unknown"

    train['특정 시술 유형'] = train['특정 시술 유형'].apply(categorize_procedure)
    test['특정 시술 유형'] = test['특정 시술 유형'].apply(categorize_procedure)

    train['시술유형_통합'] = train['시술 유형'].astype(str) + '_' + train['특정 시술 유형'].astype(str)
    test['시술유형_통합'] = test['시술 유형'].astype(str) + '_' + test['특정 시술 유형'].astype(str)

    drop_cols = ['시술 유형', '특정 시술 유형']
    train = train.drop(drop_cols, axis=1)
    test = test.drop(drop_cols, axis=1)

    return train, test

def 시술횟수(df_train):
    for col in [col for col in df_train.columns if '횟수' in col]:
        df_train[col] = df_train[col].replace({'6회 이상':'6회'})
        df_train[col] = df_train[col].str[0].astype(int)
    df_train['시술_임신'] = df_train['총 임신 횟수'] - df_train['총 시술 횟수']
    df_train = df_train.drop('총 시술 횟수', axis=1)
    return df_train

def encoding(train, test, cols_to_encoding, method='Ordinal'):
    if method == 'Ordinal':
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        train[cols_to_encoding] = ordinal_encoder.fit_transform(train[cols_to_encoding])
        test[cols_to_encoding] = ordinal_encoder.transform(test[cols_to_encoding])
    elif method == 'Target':
        target_encoder = TargetEncoder(cols=cols_to_encoding, smoothing=2)
        train[cols_to_encoding] = target_encoder.fit_transform(train[cols_to_encoding], train['임신 성공 여부'])
        test[cols_to_encoding] = target_encoder.transform(test[cols_to_encoding])
    elif method == 'Count':
        count_encoder = CountEncoder(cols=cols_to_encoding)
        train[cols_to_encoding] = count_encoder.fit_transform(train[cols_to_encoding])
        test[cols_to_encoding] = count_encoder.transform(test[cols_to_encoding])
    elif method == 'OneHot':
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        train_encoded = onehot_encoder.fit_transform(train[cols_to_encoding])
        test_encoded = onehot_encoder.transform(test[cols_to_encoding])
        encoded_feature_names = onehot_encoder.get_feature_names_out(cols_to_encoding)
        train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_feature_names, index=train.index)
        test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_feature_names, index=test.index)
        train = train.drop(columns=cols_to_encoding).join(train_encoded_df)
        test = test.drop(columns=cols_to_encoding).join(test_encoded_df)
    else:
        print('잘못 입력')
    return train, test

def numeric_process(train, test, cols_to_numeric=None, value='zero'):
    if cols_to_numeric is None:
        cols_to_numeric = [
            '임신 시도 또는 마지막 임신 경과 연수',
            '배란 자극 여부',
            '단일 배아 이식 여부',
            '착상 전 유전 검사 사용 여부',
            '착상 전 유전 진단 사용 여부',
            '총 생성 배아 수',
            '미세주입된 난자 수',
            '미세주입에서 생성된 배아 수',
            '이식된 배아 수',
            '미세주입 배아 이식 수',
            '저장된 배아 수',
            '미세주입 후 저장된 배아 수',
            '해동된 배아 수',
            '해동 난자 수',
            '수집된 신선 난자 수',
            '저장된 신선 난자 수',
            '혼합된 난자 수',
            '파트너 정자와 혼합된 난자 수',
            '기증자 정자와 혼합된 난자 수',
            '동결 배아 사용 여부',
            '신선 배아 사용 여부',
            '기증 배아 사용 여부',
            '대리모 여부',
            'PGD 시술 여부',
            'PGS 시술 여부',
            '난자 채취 경과일',
            '난자 혼합 경과일',
            '배아 이식 경과일',
            '배아 해동 경과일',
        ]
    if value == 'mean':
        imputer_mean = SimpleImputer(strategy='mean')
        train[cols_to_numeric] = imputer_mean.fit_transform(train[cols_to_numeric])
        test[cols_to_numeric] = imputer_mean.transform(test[cols_to_numeric])
    elif value == 'median':
        imputer_median = SimpleImputer(strategy='median')
        train[cols_to_numeric] = imputer_median.fit_transform(train[cols_to_numeric])
        test[cols_to_numeric] = imputer_median.transform(test[cols_to_numeric])
    elif value == 'zero':
        train[cols_to_numeric] = train[cols_to_numeric].fillna(0)
        test[cols_to_numeric] = test[cols_to_numeric].fillna(0)
    else:
        train[cols_to_numeric] = train[cols_to_numeric].fillna(value).astype(float)
        test[cols_to_numeric] = test[cols_to_numeric].fillna(value).astype(float)
    return train, test

def 배란유도유형(df_train, df_test):
    mapping = {
        '기록되지 않은 시행': 1,
        '알 수 없음': 0,
        '세트로타이드 (억제제)': 0,
        '생식선 자극 호르몬': 0,
    }
    df_train['배란 유도 유형'] = df_train['배란 유도 유형'].replace(mapping)
    df_test['배란 유도 유형'] = df_test['배란 유도 유형'].replace(mapping)
    return df_train, df_test

def 난자기증자나이(df_train, df_test):
    mapping = {
        '만20세 이하': 20,
        '만21-25세': 25,
        '만26-30세': 30,
        '만31-35세': 35,
        '알 수 없음': 20,  # 만20세 이하와 동일하게 처리
    }
    df_train['난자 기증자 나이'] = df_train['난자 기증자 나이'].replace(mapping)
    df_test['난자 기증자 나이'] = df_test['난자 기증자 나이'].replace(mapping)
    return df_train, df_test

def 난자출처(df_train, df_test):
    mapping = {
        '기증 제공': 1,
        '본인 제공': 0,
        '알 수 없음': -1,
    }
    df_train['난자 출처'] = df_train['난자 출처'].replace(mapping)
    df_test['난자 출처'] = df_test['난자 출처'].replace(mapping)
    return df_train, df_test

def 난자채취경과일(df_train):
    condition_DI = df_train['시술 유형'] == 'DI'
    condition_IVF = df_train['시술 유형'] == 'IVF'
    df_train.loc[condition_DI, '난자 채취 경과일'] = df_train.loc[condition_DI, '난자 채취 경과일'].fillna(0)
    df_train.loc[condition_IVF, '난자 채취 경과일'] = df_train.loc[condition_IVF, '난자 채취 경과일'].fillna(1)
    return df_train

def 배아생성주요이유(df_train, df_test):
    df_train['배아 생성 주요 이유'] = df_train['배아 생성 주요 이유'].fillna('DI')
    df_test['배아 생성 주요 이유'] = df_test['배아 생성 주요 이유'].fillna('DI')

    df_train['배아 생성 이유 리스트'] = df_train['배아 생성 주요 이유'].apply(lambda x: [reason.strip() for reason in x.split(',')])
    df_test['배아 생성 이유 리스트'] = df_test['배아 생성 주요 이유'].apply(lambda x: [reason.strip() for reason in x.split(',')])

    mlb = MultiLabelBinarizer()
    train_one_hot = pd.DataFrame(
        mlb.fit_transform(df_train['배아 생성 이유 리스트']),
        columns=mlb.classes_,
        index=df_train.index
    )
    train_one_hot.columns = ['배아생성이유_' + col for col in train_one_hot.columns]

    test_one_hot = pd.DataFrame(
        mlb.transform(df_test['배아 생성 이유 리스트']),
        columns=mlb.classes_,
        index=df_test.index
    )
    test_one_hot.columns = ['배아생성이유_' + col for col in test_one_hot.columns]

    df_train = pd.concat([df_train, train_one_hot], axis=1)
    df_test = pd.concat([df_test, test_one_hot], axis=1)

    cols_to_drop = [
        '배아 생성 주요 이유',
        '배아 생성 이유 리스트',
        '배아생성이유_연구용',
        '배아생성이유_DI'
    ]
    df_train = df_train.drop(cols_to_drop, axis=1, errors='ignore')
    df_test = df_test.drop(cols_to_drop, axis=1, errors='ignore')

    cols = ['배아생성이유_기증용',
            '배아생성이유_난자 저장용',
            '배아생성이유_배아 저장용',
            '배아생성이유_현재 시술용']

    df_train[cols] = df_train[cols].div(df_train[cols].sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    df_test[cols] = df_test[cols].div(df_test[cols].sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    return df_train, df_test

def 단일배아이식여부(df_train, df_val):
    df_train['단일 배아 이식 여부'] = df_train['단일 배아 이식 여부'].fillna(0)
    df_val['단일 배아 이식 여부'] = df_val['단일 배아 이식 여부'].fillna(0)
    return df_train, df_val

def 총생성배아수(df_train, df_val):
    df_train['총 생성 배아 수'] = df_train['총 생성 배아 수'].fillna(-1).clip(upper=25)
    df_val['총 생성 배아 수'] = df_val['총 생성 배아 수'].fillna(-1).clip(upper=25)
    return df_train, df_val

def 저장된배아수(df_train, df_val):
    return df_train, df_val

def 출산성공율(df_train, df_val):
    df_train['출산성공율'] = df_train['총 출산 횟수'] / (df_train['총 임신 횟수'] + 1)
    df_val['출산성공율'] = df_val['총 출산 횟수'] / (df_val['총 임신 횟수'] + 1)
    return df_train, df_val

def 출산실패경험유무(df_train):
    condition1 = df_train['총 임신 횟수'] >= 1
    condition2 = df_train['총 출산 횟수'] - df_train['총 임신 횟수']
    df_train['출산실패경험유무'] = (condition1 & condition2).astype(int)
    return df_train

def 미세주입된난자수(df_train, df_val):
    df_train['미세주입된 난자 수'] = df_train['미세주입된 난자 수'].fillna(1).clip(upper=8)
    df_val['미세주입된 난자 수'] = df_val['미세주입된 난자 수'].fillna(1).clip(upper=8)
    return df_train, df_val

def 유전검사_비정상(df_train):
    condition1 = (df_train['착상 전 유전 검사 사용 여부'] == 1.0) & (df_train['PGS 시술 여부'].isna())
    condition2 = (df_train['착상 전 유전 진단 사용 여부'] == 1.0) & (df_train['PGD 시술 여부'].isna())
    df_train['유전검사_비정상'] = (condition1 | condition2).astype(int)
    return df_train

def MCA_reasons1(df_train, df_val, n_components=2, seed=42):
    columns_to_mca = [
        "남성 주 불임 원인",
        "남성 부 불임 원인",
        "여성 주 불임 원인",
        "여성 부 불임 원인",
        "부부 주 불임 원인",
        "부부 부 불임 원인",
    ]
    df_train_mca = df_train[columns_to_mca].copy()
    df_val_mca = df_val[columns_to_mca].copy()
    mca = MCA(n_components=n_components, random_state=seed)
    mca.fit(df_train_mca)
    df_train_transformed = pd.DataFrame(mca.transform(df_train_mca))
    df_val_transformed = pd.DataFrame(mca.transform(df_val_mca))
    df_train_transformed.rename(columns={i: f"MCA1_{i+1}" for i in range(n_components)}, inplace=True)
    df_val_transformed.rename(columns={i: f"MCA1_{i+1}" for i in range(n_components)}, inplace=True)
    df_train_final = df_train.drop(columns=columns_to_mca).join(df_train_transformed)
    df_val_final = df_val.drop(columns=columns_to_mca).join(df_val_transformed)
    return df_train_final, df_val_final

def MCA_reasons2(df_train, df_val, n_components=2, seed=42):
    columns_to_mca = [
        '불임 원인 - 난관 질환',
        '불임 원인 - 남성 요인',
        '불임 원인 - 배란 장애',
        '불임 원인 - 자궁경부 문제',
        '불임 원인 - 자궁내막증',
        '불임 원인 - 정자 농도',
        '불임 원인 - 정자 운동성',
        '불임 원인 - 정자 형태',
    ]
    df_train_mca = df_train[columns_to_mca].copy()
    df_val_mca = df_val[columns_to_mca].copy()
    mca = MCA(n_components=n_components, random_state=seed)
    mca.fit(df_train_mca)
    df_train_transformed = pd.DataFrame(mca.transform(df_train_mca))
    df_val_transformed = pd.DataFrame(mca.transform(df_val_mca))
    df_train_transformed.rename(columns={i: f"MCA2_{i+1}" for i in range(n_components)}, inplace=True)
    df_val_transformed.rename(columns={i: f"MCA2_{i+1}" for i in range(n_components)}, inplace=True)
    df_train_final = df_train.drop(columns=columns_to_mca).join(df_train_transformed)
    df_val_final = df_val.drop(columns=columns_to_mca).join(df_val_transformed)
    return df_train_final, df_val_final

def 이식된배아수(df_train, df_test):
    df_train["이식된 배아 수"] = df_train["이식된 배아 수"].fillna(-1)
    df_test["이식된 배아 수"] = df_test["이식된 배아 수"].fillna(-1)
    return df_train, df_test

def 기증자정자와혼합된난자수(df_train, df_test):
    df_train["기증자 정자와 혼합된 난자 수"] = df_train["기증자 정자와 혼합된 난자 수"].fillna(2)
    df_test["기증자 정자와 혼합된 난자 수"] = df_test["기증자 정자와 혼합된 난자 수"].fillna(2)
    return df_train, df_test

def lgbm_process(train, val, seed=42):
    # 기본 전처리 단계
    train, val = drop_columns(train), drop_columns(val)
    train, val = 특정시술유형(train, val)
    train, val = 시술횟수(train), 시술횟수(val)

    cols_to_encoding = [
        "시술 시기 코드",
        "시술 당시 나이",
        "배란 유도 유형",
        "클리닉 내 총 시술 횟수",
        "IVF 시술 횟수",
        "DI 시술 횟수",
        "총 임신 횟수",
        "IVF 임신 횟수",
        "DI 임신 횟수",
        "총 출산 횟수",
        "IVF 출산 횟수",
        "DI 출산 횟수",
        "난자 출처",
        "정자 출처",
        "난자 기증자 나이",
        "정자 기증자 나이",
        '시술유형_통합',
    ]
    train, val = encoding(train, val, cols_to_encoding=cols_to_encoding, method='Ordinal')
    train, val = 단일배아이식여부(train, val)
    train, val = 배란유도유형(train, val)
    train, val = 배아생성주요이유(train, val)
    train, val = numeric_process(train, val, value='zero')

    return train, val


