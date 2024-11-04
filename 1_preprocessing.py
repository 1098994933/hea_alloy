"""
data preprocessing
"""
import matplotlib.pyplot as plt
import pandas as pd
from util.base_function import get_chemical_formula


def set_group_for_oxidation():
    """
    set group number for each alloy with same composition
    """
    dataset = pd.read_csv("./data/1_oxidation_ml_dataset.csv")
    ele_col = list(dataset.columns[:-3])
    print("elements", ele_col)
    # 计算化学式
    formulas = get_chemical_formula(dataset[ele_col])
    dataset["formula"] = formulas
    print(formulas)
    df = pd.DataFrame({"formula": list(set(formulas))})
    df["Group"] = list(df.index)
    print(df.head(30))
    df_all = pd.merge(dataset, df, on="formula")
    print(df_all.head(30))
    df_all.to_csv("./data/formula_group.csv", index=False)
    print("set_group_for_oxidation done")


if __name__ == '__main__':
    set_group_for_oxidation()

