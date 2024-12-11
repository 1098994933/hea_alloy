import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    CFS_model = joblib.load("./model/CFS_model.pkl")
    YS_model = joblib.load("./model/YS_model.pkl")
    HV_model = joblib.load("./model/hardness_model.pkl")
    dataset_virtual_samples_magpie = pd.read_csv("./data/2_virtual_samples_magpie_feature.csv")
    dataset_virtual_samples_alloy = pd.read_csv("./data/2_virtual_samples_alloy_feature.csv")
    ml_dataset = pd.concat([dataset_virtual_samples_magpie, dataset_virtual_samples_alloy.iloc[:, :-1]], axis=1)
    ml_dataset = ml_dataset.drop(['composition_obj'], axis=1)

    # TODO 后续优化
    CFS_features = ['0-norm', 'MagpieData mean MendeleevNumber', 'MagpieData range Row',
                   'MagpieData avg_dev CovalentRadius', 'MagpieData avg_dev Electronegativity', 'MagpieData mode NsValence',
                   'MagpieData mean NValence', 'MagpieData mean NsUnfilled','MagpieData avg_dev NpUnfilled',
                   'MagpieData avg_dev GSvolume_pa', 'MagpieData mode SpaceGroupNumber']
    HV_features = ['MagpieData avg_dev AtomicWeight', 'MagpieData avg_dev Column', 'MagpieData avg_dev GSvolume_pa',
                   'avg s valence electrons', 'avg p valence electrons', 'avg d valence electrons',
                   'avg f valence electrons', 'Melting temperature']
    YS_features = dataset_virtual_samples_alloy.iloc[:, :-1].columns

    scaler = StandardScaler()

    # CFS_prediction
    X_CFS = ml_dataset[CFS_features]
    CFS_prediction = CFS_model.predict(X_CFS)
    # YS_prediction
    X_YS = dataset_virtual_samples_alloy[YS_features]
    YS_prediction = YS_model.predict(X_YS)
    # HV_prediction
    X_HV = ml_dataset[HV_features]
    HV_prediction = HV_model.predict(X_HV)

    result = pd.DataFrame({"formula":dataset_virtual_samples_alloy['formula'], "CFS_prediction": CFS_prediction,
                           "YS_prediction": YS_prediction, "HV_prediction": HV_prediction})
    print(result.head(5))
    result.to_csv("./data/3_virtual_samples_prediction.csv", index=False)
