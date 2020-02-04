from lstm import *
from configuration import *
from data_preparation_modify import DataPreparation, DataPreparationDate, is_out_of_time_window


def scaling_by_col(df):
    result = df.copy()
    for feature_name in df.columns:
        if (feature_name != 'fluor') and (feature_name in FEATURES):
            result[feature_name] = (df[feature_name] - df[feature_name].mean()) / (df[feature_name].std())
    return result


def data_generator():
    all_df = DataPreparation().df

    X, y = [], []
    for current_date in DATE:
        all_df = scaling_by_col(all_df)
        dpd = DataPreparationDate(current_date, all_df)
        for i, target in enumerate(dpd.label_list):
            # use DataPreparation to determine fluor intensity
            features_df = dpd.peek_data(target=target, smoothing=False, ndarray_type=False)
            features_df = features_df[FEATURES]

            f = features_df['fluor'].values
            if is_out_of_time_window(f) or (features_df.shape[0] != TIMELENGTH):
                pass
            else:
                if np.sum(f) > 0.0:
                    # positive
                    y.append(1.0)
                else:
                    # negative
                    y.append(0.0)
                assert features_df.shape == (TIMELENGTH, len(FEATURES))
                X.append(features_df.drop(columns=['fluor']).values)

    X = np.array(X)
    y = np.array(y)
    print("X:", X.shape)
    print("y:", y.shape)
    return X, y

