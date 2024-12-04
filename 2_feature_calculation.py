"""
calculate formulas and descriptors
"""
import os

import pandas as pd

from util.alloys_features import formula_to_features
from util.descriptor.magpie import get_magpie_features
from util.base_function import get_chemical_formula

if __name__ == '__main__':
    # region
    dataset_name = "phase"
    dataset = pd.read_csv(os.path.join("./data/", "1_phase_ml_dataset.csv"))
    element_feature = dataset.columns[:-1]
    # print(element_feature)
    formulas = get_chemical_formula(dataset[element_feature])
    # print(formulas)
    df = pd.DataFrame({"formula": formulas})
    df.to_csv("./data/formula.csv", index=False)
    # generate magpie features to ｛data_path｝/
    skip_magpie = True
    if not skip_magpie:  # not true则跳过计算
        df_magpie = get_magpie_features("formula.csv", data_path="./data/")
        print(df_magpie.columns)
        df_magpie.to_csv(f"./data/2_{dataset_name}_magpie_feature.csv", index=False)
    # endregion
    alloy_feature = formula_to_features(df['formula'])
    print(alloy_feature.columns)
    print(alloy_feature.shape)
    from util.preprocessing import preprocessing_dataset
    alloy_feature.fillna(0, inplace=True)
    alloy_feature.to_csv(f"./data/2_{dataset_name}_alloy_feature.csv", index=False)

    # 计算 氧化增重数据集的 物理化学 features
    dataset_name = "oxidation"
    dataset = pd.read_csv(os.path.join("./data/", "2_oxidation_df_composition_dataset.csv"))
    element_feature = dataset.columns
    formulas = get_chemical_formula(dataset[element_feature])
    dataset["formula"] = formulas

    # 计算氧化增重数据集（计算斜率后）的物理化学features
    dataset_name = "oxidation_slope"
    dataset = pd.read_csv(os.path.join("./data/", "oxidation_slope.csv"))
    # 合金特征计算
    alloy_feature = formula_to_features(dataset['formula'])
    alloy_feature.fillna(0, inplace=True)
    alloy_feature.to_csv(f"./data/2_{dataset_name}_alloy_feature.csv", index=False)

    # 计算硬度数据集的物理化学features
    dataset_name = "Hardness"
    dataset = pd.read_csv(os.path.join("./data/", "370composition.csv"))
    dataset['formula'] = get_chemical_formula(dataset.iloc[:, 3:])
    print(dataset['formula'])
    dataset.to_csv(os.path.join("./data/", "370composition.csv"))
    # 合金特征计算
    alloy_feature = formula_to_features(dataset['formula'])
    alloy_feature.fillna(0, inplace=True)
    alloy_feature.to_csv(f"./data/2_{dataset_name}_alloy_feature.csv", index=False)

    # magpie feature
    df = dataset['formula']
    df.to_csv("./data/formula_slope.csv", index=False)
    # generate magpie features to ｛data_path｝/ magpie特征计算
    skip_magpie = False
    if not skip_magpie:  # not false不跳过计算
        df_magpie = get_magpie_features("formula_slope.csv", data_path="./data/")
        print(df_magpie.columns)
        df_magpie.to_csv(f"./data/2_{dataset_name}_magpie_feature.csv", index=False)

