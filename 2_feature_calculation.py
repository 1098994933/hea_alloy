"""
calculate formulas and descripor
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
    print(element_feature)
    formulas = get_chemical_formula(dataset[element_feature])
    print(formulas)
    df = pd.DataFrame({"formula": formulas})
    df.to_csv("./data/formula.csv", index=False)
    # generate magpie features to ｛data_path｝/
    skip_magpie = True
    if not skip_magpie:  # 跳过计算
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
    alloy_feature = formula_to_features(dataset['formula'])
    alloy_feature.fillna(0, inplace=True)
    alloy_feature.to_csv(f"./data/2_{dataset_name}_alloy_feature.csv", index=False)
    # magpie feature
    df = pd.DataFrame({"formula": formulas})
    df.to_csv("./data/formula.csv", index=False)
    # generate magpie features to ｛data_path｝/
    skip_magpie = False
    if not skip_magpie:  # 是否跳过计算
        df_magpie = get_magpie_features("formula.csv", data_path="./data/")
        print(df_magpie.columns)
        df_magpie.to_csv(f"./data/2_{dataset_name}_magpie_feature.csv", index=False)