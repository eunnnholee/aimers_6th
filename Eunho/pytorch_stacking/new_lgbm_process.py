from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings(action='ignore')

def drop_cols_with_na(train_df, val_df):
    # 나중에 결측치 대체하면서 반영할 예정

    cat_cols_with_na = [
        '이전 총 임신 횟수',
        '이전 총 임신 성공 횟수',

        '총 생성 배아 수', ## 여기부터 100% DI
        '저장된 배아 수',
        '채취된 신선 난자 수',
        '수정 시도된 난자 수'
    ]

    numeric_cols_with_na = [
        '이식된 배아 수', ## only DI
        '미세주입(ICSI) 배아 이식 수',
        '배아 이식 후 경과일',
    ]
    train_df = train_df.drop(columns=cat_cols_with_na)
    train_df = train_df.drop(columns=numeric_cols_with_na)
    val_df = val_df.drop(columns=cat_cols_with_na)
    val_df = val_df.drop(columns=numeric_cols_with_na)
    return train_df, val_df


def 시술유형(train, test):
    train['세부 시술 유형'] = train['세부 시술 유형'].fillna("Unknown")
    test['세부 시술 유형'] = test['세부 시술 유형'].fillna("Unknown")

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
        df['세부 시술 유형'] = df['세부 시술 유형'].str.replace(" / ", ",")
        df['세부 시술 유형'] = df['세부 시술 유형'].str.replace(":", ",")
        df['세부 시술 유형'] = df['세부 시술 유형'].str.replace(" ", "")

    counts = train['세부 시술 유형'].value_counts()
    allowed_categories = counts[counts >= 100].index.tolist()

    # allowed_categories에 속하지 않는 값은 "Unknown"으로 대체
    train.loc[~train['세부 시술 유형'].isin(allowed_categories), '세부 시술 유형'] = "Unknown"
    test.loc[~test['세부 시술 유형'].isin(allowed_categories), '세부 시술 유형'] = "Unknown"

    train['세부 시술 유형'] = train['세부 시술 유형'].apply(categorize_procedure)
    test['세부 시술 유형'] = test['세부 시술 유형'].apply(categorize_procedure)

    train['시술유형_통합'] = train['시술 유형'].astype(str) + '_' + train['세부 시술 유형'].astype(str)
    test['시술유형_통합'] = test['시술 유형'].astype(str) + '_' + test['세부 시술 유형'].astype(str)

    drop_cols = ['시술 유형', '세부 시술 유형']
    train = train.drop(drop_cols, axis=1)
    test = test.drop(drop_cols, axis=1)

    return train, test

def 횟수_to_int(df_train, df_val):
    for col in [col for col in df_train.columns if '횟수' in col]:
        df_train[col] = df_train[col].replace({'6회 이상': '6회'})
        df_val[col] = df_val[col].replace({'6회 이상': '6회'})

        df_train[col] = df_train[col].str[0].astype(int)
        df_val[col] = df_val[col].str[0].astype(int)

    return df_train, df_val

def 임신_IVF(df_train, df_val):
    for col in [col for col in df_train.columns if '횟수' in col]:
        df_train[col] = df_train[col].replace({'6회 이상': '6회'})
        df_val[col] = df_val[col].replace({'6회 이상': '6회'})
        mode_value = df_train[col].mode()[0]

        df_train[col] = df_train[col].fillna(mode_value)
        df_val[col] = df_val[col].fillna(mode_value)

        # 문자열의 첫 글자를 추출 후 int형으로 변환
        df_train[col] = df_train[col].str[0].astype(int)
        df_val[col] = df_val[col].str[0].astype(int)

    df_train['임신_IVF'] = df_train['이전 총 임신 횟수'] - df_train['이전 IVF 시술 횟수']
    df_val['임신_IVF'] = df_val['이전 총 임신 횟수'] - df_val['이전 IVF 시술 횟수']
    # df_train = df_train.drop('이전 시술 횟수', axis=1)
    return df_train, df_val


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

    for col in cols_to_impute:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

    return train, test

def num_feature_scailing(train, test, seed=777):
    cat_cols = [col for col in train.columns if pd.api.types.is_object_dtype(train[col])]       ## category -> object
    numeric_cols = [col for col in train.columns if col not in cat_cols and col != '임신 성공 확률']
    # bin_cols 들도 동일하게 스케일링

    arr_train = train[numeric_cols].to_numpy()  # DataFrame -> NumPy
    arr_train = arr_train.astype(np.float32)
    arr_test = test[numeric_cols].to_numpy()
    arr_test = arr_test.astype(np.float32)

    np.random.seed(seed)
    random.seed(seed)
    noise = (
        np.random.default_rng(0)
        .normal(0.0, 1e-5, arr_train.shape)
        .astype(arr_train.dtype)
    )
    preprocessing = QuantileTransformer(
        n_quantiles=max(min(len(train[numeric_cols]) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9,
    ).fit(arr_train + noise)

    train[numeric_cols] = preprocessing.transform(arr_train)
    test[numeric_cols] = preprocessing.transform(arr_test)
    return train, test

def drop_single_value_columns(df_train, df_test):
    cols_to_drop = [col for col in df_train.columns if df_train[col].nunique() == 1]
    return df_train.drop(columns=cols_to_drop), df_test.drop(columns=cols_to_drop)

def 이전_총_임신_성공_횟수(train, test):
    train['이전 총 임신 횟수'] = train['이전 총 임신 횟수'].fillna(train['이전 총 임신 횟수'].mode()[0])
    test['이전 총 임신 횟수'] = test['이전 총 임신 횟수'].fillna(train['이전 총 임신 횟수'].mode()[0])

    train['이전 총 임신 성공 횟수'] = train['이전 총 임신 성공 횟수'].fillna(train['이전 총 임신 성공 횟수'].mode()[0])
    test['이전 총 임신 성공 횟수'] = test['이전 총 임신 성공 횟수'].fillna(train['이전 총 임신 성공 횟수'].mode()[0])

def 독립범주로보기(train, test):
    cols = ['총 생성 배아 수', '저장된 배아 수', '채취된 신선 난자 수', '수정 시도된 난자 수']
    for col in cols:
        train[col] = train[col].fillna('NAN')
        test[col] = test[col].fillna('NAN')
        
def numeric_nan(train, val):
    # '이식된 배아 수'의 최빈값을 학습 데이터에서 계산 후 결측치 대체
    mode_embryo = train['이식된 배아 수'].mode()[0]
    train['이식된 배아 수'] = train['이식된 배아 수'].fillna(mode_embryo)
    val['이식된 배아 수'] = val['이식된 배아 수'].fillna(mode_embryo)
    
    # 미세주입(ICSI) 배아 이식 수: 시술 유형에 따라 결측치 대체
    mode_embryo_미세 = train['미세주입(ICSI) 배아 이식 수'].mode()[0]
    train.loc[(train['시술 유형'] == 'IVF') & (train['미세주입(ICSI) 배아 이식 수'].isna()), '미세주입(ICSI) 배아 이식 수'] = mode_embryo_미세
    val.loc[(val['시술 유형'] == 'IVF') & (val['미세주입(ICSI) 배아 이식 수'].isna()), '미세주입(ICSI) 배아 이식 수'] = mode_embryo_미세
    train.loc[(train['시술 유형'] == 'DI') & (train['미세주입(ICSI) 배아 이식 수'].isna()), '미세주입(ICSI) 배아 이식 수'] = 4
    val.loc[(val['시술 유형'] == 'DI') & (val['미세주입(ICSI) 배아 이식 수'].isna()), '미세주입(ICSI) 배아 이식 수'] = 4

    # 배아 이식 후 경과일: 시술 유형에 따라 결측치 대체
    mode_embryo_배아 = train['배아 이식 후 경과일'].mode()[0]
    train.loc[(train['시술 유형'] == 'IVF') & (train['배아 이식 후 경과일'].isna()), '배아 이식 후 경과일'] = 8
    val.loc[(val['시술 유형'] == 'IVF') & (val['배아 이식 후 경과일'].isna()), '배아 이식 후 경과일'] = 8
    train.loc[(train['시술 유형'] == 'DI') & (train['배아 이식 후 경과일'].isna()), '배아 이식 후 경과일'] = mode_embryo_배아
    val.loc[(val['시술 유형'] == 'DI') & (val['배아 이식 후 경과일'].isna()), '배아 이식 후 경과일'] = mode_embryo_배아
    return train, val

def transform_obj(train, val):
    cols = ['이전 IVF 시술 횟수', 
            '이전 DI 시술 횟수', 
            '이전 총 임신 횟수', 
            '이전 총 임신 성공 횟수', 
            '이식된 배아 수', 
            '미세주입(ICSI) 배아 이식 수']
    for col in cols:
        train[col] = train[col].astype('object')
        val[col] = val[col].astype('object')
        

def all_process(train, val):
    # train, val = drop_cols_with_na(train, val)

    # 범주형 변수 결측값 전처리 새로 추가
    이전_총_임신_성공_횟수(train, val)
    독립범주로보기(train, val)
    
    # 연속형 변수 결측값 전처리 새로 추가
    train, val = numeric_nan(train, val)
    
    # 기본 전처리 단계
    train, val = 횟수_to_int(train, val)

    train, val = 시술유형(train, val)
    # train, val = 임신_IVF(train, val)

    train, val = 단일배아이식여부(train, val)

    cols_to_encoding = [
        "환자 시술 당시 나이",
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

        '해동된 배아 수', # 원래 int였는데 범주형으로 바뀜
        
        # 추가 
        "이전 IVF 시술 횟수",
        "이전 DI 시술 횟수",
        

    ]
    # train, val = label_encoding(train, val, cols=cols_to_encoding)
    # train, val = type_to_category(train, val, cols=cols_to_encoding)

    # train, val = impute_nan(train, val)
    
    # 기존의 이산형 변수 object형으로 변환
    # transform_obj(train, val)         ## 점수 떨어짐 -> 버림
    
    train, val = num_feature_scailing(train, val)

    train, val = drop_single_value_columns(train, val)

    return train, val