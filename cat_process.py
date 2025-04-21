import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def drop_columns(df):
    cols = [
        '불임 원인 - 여성 요인', # 고유값 1
        '불임 원인 - 정자 면역학적 요인' # '1'인 데이터가 train, test에서 모두 1개 >> 신뢰할 수 없음
    ]
    df = df.drop(cols, axis=1)
    return df

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

def 해동된배아수(train, test, cap_value=7):
    train['해동된 배아 수'] = train['해동된 배아 수'].clip(upper=cap_value)
    test['해동된 배아 수'] = test['해동된 배아 수'].clip(upper=cap_value)
    return train, test

def 횟수_to_int(df_train, df_val):
    for col in [col for col in df_train.columns if '횟수' in col]:
        df_train[col] = df_train[col].replace({'6회 이상': '6회'})
        df_val[col] = df_val[col].replace({'6회 이상': '6회'})

        df_train[col] = df_train[col].str[0].astype(int)
        df_val[col] = df_val[col].str[0].astype(int)

    return df_train, df_val

def encoding(train, test, seed=42):
    categorical_columns = [
        '환자 시술 당시 나이', 
        '총 생성 배아 수', 
        '저장된 배아 수', 
        '해동된 배아 수', 
        '채취된 신선 난자 수', 
        '수정 시도된 난자 수', 
        '난자 출처', 
        '정자 출처', 
        '난자 기증자 나이', 
        '정자 기증자 나이', 
        '시술유형_통합'
        ]    
    train[categorical_columns] = train[categorical_columns].astype(str)
    test[categorical_columns] = test[categorical_columns].astype(str)

    train[categorical_columns] = train[categorical_columns].astype('category')
    test[categorical_columns] = test[categorical_columns].astype('category')

    return train, test

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

def cb_all_process(train, val):
    # train, val = drop_columns(train), drop_columns(val)
    
    # 범주형 변수 결측값 전처리 새로 추가
    이전_총_임신_성공_횟수(train, val)
    독립범주로보기(train, val)
    
    # 연속형 변수 결측값 전처리 새로 추가
    train, val = numeric_nan(train, val)
    
    train, val = 시술유형(train, val)

    train, val = 횟수_to_int(train, val)

    train, val = encoding(train, val)
    

    return train, val
