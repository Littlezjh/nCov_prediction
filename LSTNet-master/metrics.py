import numpy as np


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def MAPE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true)


def RSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2)) / normal_std(y_true)


def NRMSE(y_true, y_pred):
    """ Normalized RMSE"""
    t1 = np.sum((y_pred - y_true) **2) / np.size(y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    return np.sqrt(t1) / t2


def ND(y_true, y_pred):
    """ Normalized deviation"""
    t1 = np.sum(abs(y_pred-y_true)) / np.size(y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    return t1 / t2


def SMAPE(y_true, y_pred):
    s = 0
    for a, b in zip(y_pred, y_true):
        if abs(a) + abs(b) == 0:
            s += 0
        else:
            s += 2 * abs(a-b) / (abs(a) + abs(b))
    return s / np.size(y_true)

def calculate_errors(y_true, y_pred):
    errors = {}
    y_pred[y_pred < 0] = 0
    errors['RMSE'] = RMSE(y_true, y_pred)
    errors['RSE'] = RSE(y_true, y_pred)
    errors['MAE'] = MAE(y_true, y_pred)
    errors['SMAPE'] = SMAPE(y_true, y_pred)
    errors['NRMSE'] = NRMSE(y_true, y_pred)
    errors['ND'] = ND(y_true, y_pred)
    return errors