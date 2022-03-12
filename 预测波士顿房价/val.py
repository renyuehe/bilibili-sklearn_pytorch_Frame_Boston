'''
模型评估
'''

from sklearn import metrics
import numpy as np

def reg_calculate(target, prediction, features=None):
    '''
    评估指标:
    mae:平均绝对值误差(也是L2)
    mse:均方误差(也是L2)
    rmse:军方根误差
    rmsle: 均方根对数误差

    r2: R2得分
    evs: 可解释方差
    '''
    prediction[prediction < 0] = 0

    # 平均绝对值误差
    mae = metrics.mean_absolute_error(target, prediction)

    # 均方误差
    mse = metrics.mean_squared_error(target, prediction)

    # 均方根误差
    rmse = np.sqrt(mse)

    # 均方根对数误差
    rmsle = np.sqrt(metrics.mean_squared_log_error(target, prediction))

    # r2 得分
    r2 = metrics.r2_score(target, prediction)

    # 可解释方差
    evs = metrics.explained_variance_score(target, prediction)

    return mse, r2, evs

