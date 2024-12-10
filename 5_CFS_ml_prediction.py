"""
compressive fracture strain of HEA modeling and prediction
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
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

def Linear_SVR(C=1.0, gamma=0.1, epsilon=1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("model", SVR(kernel="linear", C=C, gamma=gamma, epsilon=epsilon))
    ])


def RBF_SVR(C=1.0, gamma=1.0, epsilon=1.0):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon))
    ])


if __name__ == '__main__':
    dataset = pd.read_csv('./data/2_CFS_magpie_feature.csv')
    # dataset = pd.read_excel(r"D:\上海大学学习\毕设相关零碎材料收纳\Feature_CFS_bishe.xlsx", sheet_name=0)
    dataset.drop_duplicates(keep='first', inplace=True)
    print(dataset.shape)
    Q1 = dataset['CFS'].quantile(0.25)
    Q3 = dataset['CFS'].quantile(0.75)
    IQR = Q3 - Q1
    # 保留小于极端大的值
    dataset = dataset[dataset['CFS'] <= (Q3 + (1.5 * IQR))]
    # 保留大于极端小的值
    dataset = dataset[dataset['CFS'] >= (Q1 - (1.5 * IQR))]
    # print(dataset.iloc[228])
    Y_col = 'CFS'
    best_features_myy = ['MagpieData range MendeleevNumber', 'MagpieData avg_dev MendeleevNumber',
                'MagpieData avg_dev MeltingT', 'MagpieData mean NValence',
                 'MagpieData mean NsUnfilled', 'MagpieData maximum NUnfilled',
                'MagpieData maximum GSvolume_pa', 'MagpieData avg_dev GSvolume_pa',
                'MagpieData minimum SpaceGroupNumber', 'MagpieData mode SpaceGroupNumber',
                'Lambda entropy', 'Electronegativity local mismatch']  # from myy

    best_features_zyj = ['0-norm', 'MagpieData mean MendeleevNumber', 'MagpieData range Row',
                   'MagpieData avg_dev CovalentRadius', 'MagpieData avg_dev Electronegativity', 'MagpieData mode NsValence',
                   'MagpieData mean NValence', 'MagpieData mean NsUnfilled','MagpieData avg_dev NpUnfilled',
                   'MagpieData avg_dev GSvolume_pa', 'MagpieData mode SpaceGroupNumber']

    alloy_feature = pd.read_csv('./data/2_CFS_alloy_feature.csv')
    alloy_feature = alloy_feature.drop(['formula', 'CFS'], axis=1)
    # print(alloy_feature.head())

    # alloy features
    # ml_dataset = pd.concat([alloy_feature, dataset[Y_col]], axis=1).dropna()
    # X = ml_dataset[alloy_feature.columns]

    # magpie features
    ml_dataset = dataset[best_features_zyj + [Y_col]].dropna()
    X = ml_dataset[best_features_zyj]

    Y = ml_dataset[Y_col]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=250)
    model_final = RBF_SVR(C=100, gamma=0.13, epsilon=1.3)
    # model_final = RandomForestRegressor()
    model_final.fit(X_train, Y_train)
    y_test_predict = model_final.predict(X_test)
    y_train_predict = model_final.predict(X_train)
    y_predict = model_final.predict(X)
    cvscore = cross_val_score(model_final, X, Y, cv=50, scoring='r2')
    evaluation_matrix = cal_reg_metric(Y_test, y_test_predict)
    evaluation_matrix_train = cal_reg_metric(Y_train, y_train_predict)
    y_cv_predict = cross_val_predict(model_final, X, Y, cv=50)
    R2 = r2_score(Y, y_cv_predict)
    print(cvscore, R2)
    print(evaluation_matrix_train)

    lim_max = max(max(y_test_predict), max(Y_test), max(Y_train), max(y_train_predict)) * 1.02
    lim_min = min(min(y_test_predict), min(Y_test), min(Y_train), min(y_train_predict)) * 0.98

    # plt.figure(figsize=(7, 5))
    # plt.rcParams['font.sans-serif'] = ['Arial']  # 设置字体
    # plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    # plt.grid(linestyle="--")  # 设置背景网格线为虚线
    # ax = plt.gca()  # 获取坐标轴对象
    # plt.plot([lim_min, lim_max], [lim_min, lim_max], color='blue')
    # plt.scatter(Y, y_cv_predict, color='red', alpha=0.4)
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    # plt.xlabel("Measured", fontsize=12, fontweight='bold')
    # plt.ylabel("Predicted", fontsize=12, fontweight='bold')
    # plt.show()
    # plt.clf()

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
    # 为每个点标上序号
    # print(Y)
    for i, (x, y) in enumerate(zip(Y, y_predict)):
        plt.text(x, y, f'{i}', fontsize=8, ha='right', va='bottom', color='black')
    plt.text(0.05, 0.75, f"$R^2={r2:.3f}$\n$MAE={mae:.3f}$\n$R={R:.3f}$", transform=ax.transAxes)
    plt.legend()
    # plt.savefig(f'./figures/HEA_CFS_reg.png', bbox_inches='tight')
    plt.show()