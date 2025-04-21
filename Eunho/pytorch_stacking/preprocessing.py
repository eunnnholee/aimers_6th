# data_preprocessing.py
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer

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

    for df_ in [train, test]:
        df_['특정 시술 유형'] = df_['특정 시술 유형'].str.replace(" / ", ",")
        df_['특정 시술 유형'] = df_['특정 시술 유형'].str.replace(":", ",")
        df_['특정 시술 유형'] = df_['특정 시술 유형'].str.replace(" ", "")

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

def 기증자정자와혼합된난자수(df_train, df_test):
    df_train["기증자 정자와 혼합된 난자 수"] = df_train["기증자 정자와 혼합된 난자 수"].fillna(2)
    df_test["기증자 정자와 혼합된 난자 수"] = df_test["기증자 정자와 혼합된 난자 수"].fillna(2)
    return df_train, df_test

def label_encoding(train, test, cols):
    encoder = LabelEncoder()
    for col in cols:
        train[col] = encoder.fit_transform(train[col])
        test[col] = encoder.transform(test[col])
    return train, test

def type_to_category(train, test, cols):
    train[cols] = train[cols].astype('category')
    test[cols] = test[cols].astype('category')
    return train, test

def impute_nan(train, test):
    cols_to_impute = [
        '임신 시도 또는 마지막 임신 경과 연수',
    ]
    imputer = SimpleImputer(strategy='mean')
    train[cols_to_impute] = imputer.fit_transform(train[cols_to_impute])
    test[cols_to_impute] = imputer.transform(test[cols_to_impute])

    cols_to_impute = [
        '난자 채취 경과일',
        '난자 혼합 경과일',
        '배아 이식 경과일',
        '배아 해동 경과일',
        '착상 전 유전 검사 사용 여부',
        'PGD 시술 여부',
        'PGS 시술 여부',
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
    ]
    train[cols_to_impute] = train[cols_to_impute].fillna(0)
    test[cols_to_impute] = test[cols_to_impute].fillna(0)

    return train, test

def num_feature_scailing(train, test, seed=777):
    numeric_cols = train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [col for col in train.columns if pd.api.types.is_categorical_dtype(train[col])]
    cols_to_scale = [
        col for col in numeric_cols
        if col not in cat_cols and col != '임신 성공 여부'
    ]

    arr_train = train[cols_to_scale].to_numpy().astype(np.float32)
    arr_test = test[cols_to_scale].to_numpy().astype(np.float32)

    np.random.seed(seed)
    random.seed(seed)
    noise = (
        np.random.default_rng(0)
        .normal(0.0, 1e-5, arr_train.shape)
        .astype(arr_train.dtype)
    )
    transformer = QuantileTransformer(
        n_quantiles=max(min(len(train[cols_to_scale]) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9,
    ).fit(arr_train + noise)

    train[cols_to_scale] = transformer.transform(arr_train)
    test[cols_to_scale] = transformer.transform(arr_test)
    return train, test

def drop_single_value_columns(df_train, df_test):
    cols_to_drop = [col for col in df_train.columns if df_train[col].nunique() == 1]
    return df_train.drop(columns=cols_to_drop), df_test.drop(columns=cols_to_drop)

def all_process(train, val):
    # 기본 전처리 단계
    train, val = drop_columns(train), drop_columns(val)
    train, val = 특정시술유형(train, val)
    train, val = 시술횟수(train), 시술횟수(val)
    train, val = 단일배아이식여부(train, val)
    train, val = 배란유도유형(train, val)
    train, val = 배아생성주요이유(train, val)

    cols_to_encoding = [
        "시술 시기 코드",
        "시술 당시 나이",
        "배란 유도 유형",
        # "클리닉 내 총 시술 횟수",
        # "IVF 시술 횟수",
        # "DI 시술 횟수",
        # "총 임신 횟수",
        # "IVF 임신 횟수",
        # "DI 임신 횟수",
        # "총 출산 횟수",
        # "IVF 출산 횟수",
        # "DI 출산 횟수",
        "난자 출처",
        "정자 출처",
        "난자 기증자 나이",
        "정자 기증자 나이",
        '시술유형_통합',
    ]
    
    # train, val = label_encoding(train, val, cols=cols_to_encoding)
    train, val = type_to_category(train, val, cols=cols_to_encoding)

    train, val = impute_nan(train, val)
    train, val = num_feature_scailing(train, val) 
    train, val = drop_single_value_columns(train, val)

    return train, val