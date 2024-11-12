"""
hardness of HEA modeling and prediction
"""
import os
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
    # 1. 读取高熵合金硬度数据集
    data_path = "./data/"
    dataset = pd.read_csv(os.path.join(data_path, "370composition.csv"))
    print("rows", dataset.shape[0])
    print("col", list(dataset.columns))
    Y_col = 'HV'
    elements_columns = list(dataset.columns[3:])
    print(elements_columns)
    dataset.head()
    chemical_formula_list = get_chemical_formula(dataset[elements_columns])
    df_chemistry_formula = pd.DataFrame({"formula": chemical_formula_list, "target": dataset[Y_col]})
    print(df_chemistry_formula.head())
    df_chemistry_formula.to_csv("./data/formula_hardness.csv", index=False)
    # if 特征已经计算过，不需重复计算
    feature_file_path = os.path.join(data_path, "magpie_feature_hardness.csv")
    if os.path.exists("./data/magpie_feature_hardness.csv"):
        df_magpie = pd.read_csv(feature_file_path)
    else:
        df_magpie = get_magpie_features("formula_hardness.csv", data_path="./data/")
        df_magpie.to_csv(feature_file_path, index=False)
        print(f"save features to {feature_file_path}")
    print(df_magpie)
    dataset_all = pd.concat([dataset, df_magpie], axis=1)
    # features = elements_columns + conditions_features + magpie_features
    valence_features = ['avg s valence electrons', 'avg p valence electrons', 'avg d valence electrons',
                        'avg f valence electrons']
    features = ['MagpieData avg_dev AtomicWeight', 'MagpieData avg_dev Column',
                'MagpieData avg_dev GSvolume_pa'] + valence_features
    print("len(features)", len(features))
    # ML 建模和评估
    ml_dataset = dataset_all[features + [Y_col]].dropna()
    ml_dataset.head()
    from sklearn.model_selection import train_test_split

    X = ml_dataset[features]
    Y = ml_dataset[Y_col]
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
    model_final = RBF_SVR(C=1642, epsilon=13.6, gamma=0.44)
    model_final.fit(X_train, Y_train)
    y_predict = model_final.predict(X_test)
    y_predict = y_predict
    y_train_predict = model_final.predict(X_train)
    y_true = Y_test
    evaluation_matrix = cal_reg_metric(y_true, y_predict)

    lim_max = max(max(y_predict), max(y_true), max(Y_train), max(y_train_predict)) * 1.02
    lim_min = min(min(y_predict), min(y_true), min(Y_train), min(y_train_predict)) * 0.98

    plt.figure(figsize=(7, 5), dpi=400)
    plt.rcParams['font.sans-serif'] = ['Arial']  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()  # 获取坐标轴对象
    plt.scatter(y_true, y_predict, color='red', alpha=0.4, label='test')
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
    plt.text(0.05, 0.75, f"$R^2={r2:.3f}$\n$MAE={mae:.3f}$\n", transform=ax.transAxes)
    plt.legend()
    plt.savefig(f'./figures/HEA_hardness_reg.png', bbox_inches='tight')

    # 预测硬度 (已计算好matminer特征, 计算过程在2_feature_calculation）
    dataset_predict = pd.read_csv(os.path.join(data_path, "2_oxidation_magpie_feature.csv"))
    y_predict = model_final.predict(dataset_predict[features])
    dataset_predict["hardness_predict"] = y_predict
    dataset_predict.to_csv("./data/2_oxidation_hardness_predict.csv", index=False)