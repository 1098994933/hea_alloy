"""
data preprocessing
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from util.base_function import get_chemical_formula


def set_group_for_oxidation():
    """
    set group number for each alloy with same composition
    """
    dataset = pd.read_csv("./data/1_oxidation_ml_dataset.csv")
    ele_col = list(dataset.columns[:-3])  # 元素列表
    print("elements", ele_col)
    # 计算化学式
    formulas = get_chemical_formula(dataset[ele_col])
    dataset["formula"] = formulas  # 添加成分列
    print(formulas)
    df = pd.DataFrame({"formula": list(set(formulas))})
    df["Group"] = list(df.index)  # 加索引
    print(df.head(30))
    df_all = pd.merge(dataset, df, on="formula")  # 把索引合到dataset里
    print(df_all.head(30))
    df_all.to_csv("./data/formula_group.csv", index=False)
    print("set_group_for_oxidation done")

def caculate_oxidation_slope():
    """
    calculate oxidation slope for each group
    """
    df = pd.read_csv("./data/1_oxidation_ml_dataset_modified.csv")
    ele_col = list(df.columns[:-3])  # 元素列表
    formulas = get_chemical_formula(df[ele_col])
    df["formula"] = formulas
    # print(df)
    # df.to_csv("./data/1_oxidation_ml_dataset_modified.csv", index=False)
    grouped = df.groupby(by=["formula"])
    formula = list(set(formulas))
    slope = []
    tem = []
    for i in formula:
        group_a = grouped.get_group(i)
        model = LinearRegression()
        Y = group_a["weight"].to_frame()
        X = group_a["Exposure"].to_frame()
        model.fit(X, Y)
        slope.append(float(model.coef_[0][0]))
        tem.append(group_a["Temperature"].iloc[0])
    print(formula, tem)
    print(f"slopes: {slope}")
    df_slope = pd.DataFrame({"formula": formula, "slope": slope, "temperature": tem})
    # df_slope.to_csv("./data/oxidation_slope.csv", index=False)


if __name__ == '__main__':
    # set_group_for_oxidation()
    caculate_oxidation_slope()
