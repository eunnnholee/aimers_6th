# Stacking(pytorch_stacking.ipynb)
```
                ┌─────────────────────────────┐
                │         Input Data          │
                └─────────────────────────────┘
                          ↓
┌─────────────────────────────┬─────────────────────────────┬─────────────────────────────┬─────────────────────────────┐
│      Base Model 1           │     Base Model 2            │     Base Model 3            │     Base Model 4            │
│  CategoryEmbeddingModel     │     FTTransformer            │     TabNet                  │     GANDALF                 │
└─────────────────────────────┴─────────────────────────────┴─────────────────────────────┴─────────────────────────────┘
           ↓                           ↓                          ↓                          ↓
     [예측값 1]           +      [예측값 2]          +     [예측값 3]          +       [예측값 4]
                             ↓────────────────────────────────────────────────────────────────────────↓
                                     ▶▶▶ Concatenate (예측값을 한 벡터로 합침)
                                                        ↓
                                      ┌────────────────────────┐
                                      │      Meta Head         │
                                      │    (LinearHead / MLP)  │
                                      └────────────────────────┘
                                                        ↓
                                               최종 예측값 (확률)

```


# Embedding+LGBM(categorical_embedding.ipynb)

```
[ Raw Train ] → 전처리 → TabularModel(CategoricalEmbeddingModel) 학습
                           ↓
                    [ 학습된 Embedding Layer ]
                           ↓
   CategoricalEmbeddingTransformer 생성
                           ↓
     fit_transform(train), transform(test)
                           ↓
       [임베딩된 범주형] + [연속형] → ML 모델 학습

- k-fold 내부에 "lgbm 전처리, TabularModel 학습, CategoricalEmbeddingTransformer(embedding 변환)"이 포함되어 있습니다.
```
