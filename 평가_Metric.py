import numpy as np
from sklearn import metrics

def f1_score(true_prob, pred_prob):
    true_binary = (np.array(true_prob) > 0.5).astype(int)
    pred_binary = (np.array(pred_prob) > 0.5).astype(int)
    return metrics.f1_score(true_binary, pred_binary)

def weighted_brier_score(true_prob, pred_prob, alpha=4):
    weights = 1 + alpha * true_prob + np.abs(0.5 - true_prob) ** 2
    brier = np.sum(weights * (true_prob - pred_prob) ** 2) / np.sum(weights)
    adjusted_brier = max(0, 1 - brier)
    return adjusted_brier

def competition_metric(true_prob, pred_prob):
    true_prob = np.array(true_prob)
    pred_prob = np.array(pred_prob)

    if true_prob.shape != pred_prob.shape:
        raise ValueError("예측값과 정답값의 shape이 일치하지 않습니다.")
    if np.isnan(pred_prob).any():
        raise ValueError("예측값에 NaN이 포함되어 있습니다.")
    if not ((0 <= pred_prob) & (pred_prob <= 1)).all():
        raise ValueError("예측값이 0~1 범위를 벗어났습니다.")
    if not np.isfinite(pred_prob).all():
        raise ValueError("예측값에 inf 또는 -inf가 포함되어 있습니다.")

    wbs = weighted_brier_score(true_prob, pred_prob)
    f1 = f1_score(true_prob, pred_prob)
    score = 0.5 * wbs + 0.5 * f1
    return score

'''
# true 값도 soft label (예: 확률) 형태
true = np.array([0.95, 0.05, 0.85, 0.2, 0.7])
pred = np.array([0.9, 0.1, 0.8, 0.3, 0.6])

score = competition_metric(true, pred)
print("LB Score:", score)
'''
