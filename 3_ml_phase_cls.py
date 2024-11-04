"""
phase prediction for HEA
"""
import seaborn as sns
import pickle
from scipy import stats
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import os
import scipy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

val_config = {}


def read_data():
    dataset = pd.read_csv("./data/1_phase_ml_dataset.csv")
    Y_col = 'Phase'
    features = list(dataset.columns[:-1])
    return dataset, Y_col, features


def read_magpie_data():
    """
    使用化学式特征
    :return:
    """
    dataset = pd.read_csv("./data/2_phase_magpie_feature.csv")
    Y_col = 'Phase'
    features = list(dataset.columns[2:])
    return dataset, Y_col, features


def read_alloy_data():
    """
    使用23个特征
    :return:
    """
    dataset = pd.read_csv("./data/2_phase_alloy_feature.csv")
    Y_col = 'Phase'
    features = list(dataset.columns)
    return dataset, Y_col, features


if __name__ == '__main__':
    # use different feature subset

    # dataset, Y_col, features = read_data()        # use element%
    dataset, Y_col, features = read_magpie_data()   # use magpie features
    # dataset, Y_col, features = read_alloy_data()  # use alloy features

    print(f"features:{features}")
    ml_dataset = dataset
    # ml
    X = ml_dataset[features]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    Y = pd.read_csv("./data/1_phase_ml_dataset.csv")[Y_col]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=0)

    print(X_train.shape, Y_train.shape)
    X_train = pd.DataFrame(X_train, columns=features)
    columns_with_nulls = X_train.columns[X_train.isnull().any()]
    print(columns_with_nulls)
    feature_selection = SelectKBest(f_classif, k=len(features)).fit(X_train, Y_train)

    feature_scores = feature_selection.scores_
    print('feature_scores:', feature_scores)
    indices = np.argsort(feature_scores)[::-1]
    val_config['feature_num'] = len(features)
    best_features = list(X_train.columns.values[indices[0:val_config['feature_num']]])
    print("best_features", best_features)
    X_train = feature_selection.transform(X_train)
    X_test = feature_selection.transform(X_test)
    sc = MinMaxScaler()
    alg_dict = {
        # "LogisticRegression": LogisticRegression(),
        # "KNeighbors": KNeighborsClassifier(),
        # "DecisionTree": DecisionTreeClassifier(),
        # "RandomForest": RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=4, n_jobs=4),
        "ExtraTrees": ExtraTreesClassifier(
            **{'max_depth': 5, 'min_samples_split': 6, 'n_estimators': 133}),
        # "GradientBoosting": GradientBoostingClassifier(),
        # "AdaBoost": AdaBoostClassifier(),
        # "SVC": SVC(),
    }
    best_model = None
    best_score = 0
    best_y_predict = None
    # 选择测试集最优算法（数据较多时 可以不使用交叉验证）
    for alg_name in alg_dict.keys():
        model = alg_dict[alg_name]
        model.fit(X_train, Y_train)
        y_predict = model.predict(X_test)
        score = accuracy_score(Y_test, y_predict)
        print(f"{alg_name} {score}")
        if score > best_score:
            best_model = model
            best_score = score
            best_y_predict = y_predict.copy()
    # save the best model
    print(f"best accuracy_score {best_score} best model {best_model}")

    # dataset['Y_predict'] = best_y_predict
    class_labels = list(set(Y_test))
    cm = confusion_matrix(Y_test, best_y_predict, labels=class_labels)
    print(best_y_predict)
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(f'./figures/confusion_matrix.png', bbox_inches='tight')

    # use best model to predict
    model_final = best_model
    model_final.fit(X, Y)
    dataset_predict = pd.read_csv("./data/2_oxidation_magpie_feature.csv")
    y_predict = model_final.predict(scaler.transform(dataset_predict[features]))
    dataset_predict.insert(0, "phase_predict", y_predict)
    dataset_predict.to_csv("./data/3_oxidation_phase_predict.csv", index=False)
