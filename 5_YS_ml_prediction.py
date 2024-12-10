"""
Yield strength of HEA modeling and prediction
"""
import os

from scipy.stats import pearsonr

from util.base_function import get_chemical_formula
from util.descriptor.magpie import get_magpie_features
from util.eval import cal_reg_metric
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import re
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv("./data/2_YS_magpie_feature.csv")
    print(dataset.shape)
    dataset.drop_duplicates(keep='first', inplace=True)
    print(dataset.shape)
    Q1 = dataset['YS'].quantile(0.25)
    Q3 = dataset['YS'].quantile(0.75)
    IQR = Q3 - Q1
    # 保留小于极端大的值
    dataset = dataset[dataset['YS'] <= (Q3 + (1.5 * IQR))]
    # 保留大于极端小的值
    dataset = dataset[dataset['YS'] >= (Q1 - (1.5 * IQR))]
    # print(dataset.iloc[228])
    Y_col = 'YS'
    best_features_zyj = ['10-norm', 'MagpieData mean MendeleevNumber', 'MagpieData range MeltingT', 'MagpieData avg_dev Electronegativity',
                     'MagpieData mean NdValence', 'MagpieData minimum GSvolume_pa', 'Radii local mismatch']
    alloy_feature = pd.read_csv('./data/2_YS_alloy_feature.csv')
    alloy_feature = alloy_feature.drop(['formula', 'YS'], axis=1)

    # magpie features
    # ml_dataset = dataset[best_features_zyj + [Y_col]].dropna()
    # X = ml_dataset[best_features_zyj]

    # alloy features
    ml_dataset = pd.concat([alloy_feature, dataset[Y_col]], axis=1).dropna()
    X = ml_dataset[alloy_feature.columns]

    Y = ml_dataset[Y_col]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=98)
    # model_final = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
    #                               max_depth=6, num_leaves=7, learning_rate=0.099,
    #                               n_estimators=420, feature_fraction=0.5, min_data_in_leaf=20,
    #                               metric='rmse', random_state=100)
    model_final = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', learning_rate=0.15)
    model_final.fit(X_train, Y_train)
    y_test_predict = model_final.predict(X_test)
    y_train_predict = model_final.predict(X_train)
    y_predict = model_final.predict(X)
    cvscore = cross_val_score(model_final, X_train, Y_train, cv=10, scoring='r2')
    evaluation_matrix = cal_reg_metric(Y_test, y_test_predict)
    evaluation_matrix_train = cal_reg_metric(Y_train, y_train_predict)
    y_cv_predict = cross_val_predict(model_final, X_train, Y_train, cv=10)
    R2 = r2_score(Y_train, y_cv_predict)
    RMSE = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    print(RMSE, R2)
    print(evaluation_matrix_train)

    lim_max = max(max(y_test_predict), max(Y_test), max(Y_train), max(y_train_predict)) * 1.02
    lim_min = min(min(y_test_predict), min(Y_test), min(Y_train), min(y_train_predict)) * 0.98

    plt.figure(figsize=(7, 5))
    plt.rcParams['font.sans-serif'] = ['Arial']  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()  # 获取坐标轴对象
    plt.scatter(Y_test, y_test_predict, color='red', alpha=0.4, label='test')
    plt.scatter(Y_train, y_train_predict, color='blue', alpha=0.4, label='train')
    plt.plot([lim_min, lim_max], [lim_min, lim_max], color='blue')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.xlabel("Measured(hardness)", fontsize=12, fontweight='bold')
    plt.ylabel("Predicted(hardness)", fontsize=12, fontweight='bold')
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    r2 = evaluation_matrix["R2"]
    mae = evaluation_matrix["MAE"]
    R = evaluation_matrix["R"]
    plt.text(0.05, 0.75, f"$R^2={r2:.3f}$\n$MAE={mae:.3f}$\n$R={R:.3f}$", transform=ax.transAxes)
    plt.legend()
    # plt.savefig(f'./figures/HEA_YS_reg.png', bbox_inches='tight')
    plt.show()