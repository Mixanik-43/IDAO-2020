import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
root_dir = '..'
sys.path.append(root_dir)
import utils
from LinearAlignment.LinearAlignment import LinearAlignment

def epoch_to_int(df):
    df['epoch'] = pd.to_datetime(df['epoch']).astype(np.int64)

def track_1_sub30514611(train_path, test_path, out_path):
    train_data = pd.read_csv(train_path, index_col='id')
    epoch_to_int(train_data)
    test_data = pd.read_csv(test_path, index_col='id')
    epoch_to_int(test_data)

    features_list=('x', 'y', 'z', 'Vx', 'Vy', 'Vz')
    result_df = []
    alignment_model = LinearAlignment()

    satellites_list = test_data['sat_id'].unique()
    for sat_id in tqdm(satellites_list):
        # print(sat_id)
        pred = pd.DataFrame([])
        train_sat_data = utils.get_satellite_data(train_data, sat_id, reset_index=False)
        test_sat_data = utils.get_satellite_data(test_data, sat_id, reset_index=False)
        full_sat_data = pd.concat([train_sat_data, test_sat_data], sort=False).reset_index(drop=True)
        n_train = len(train_sat_data)
        try:
            full_sat_data = utils.remove_time_jumps_fast(full_sat_data)
        #         filtered_sat_data = utils.remove_time_jumps(sat_data)
        except KeyError as e:
            print(f'jump removal failed for satellite {sat_id}:\t{type(e).__name__} {e}')
            continue
        #     sat_data = sat_data.join(utils.remove_time_jumps(sat_data))
        train_sat_data = full_sat_data[:n_train]

        # try:
        for feature_name in features_list:
            alignment_model.fit(t=train_sat_data['epoch'].values,
                                x=train_sat_data[f'{feature_name}_sim'].values,
                                gt=train_sat_data[feature_name].values)
            pred[feature_name] = alignment_model.predict(t=full_sat_data['epoch'].values,
                                                         x=full_sat_data[f'{feature_name}_sim'].values)
        # except Exception as e:
        #     print(f'linear alignment failed for satellite {sat_id}:\t{type(e).__name__} {e}')
        #     continue

        # start = max(0, n_train - 100)
        # end = n_train + 100
        # start  =0
        # end = -1
        # plt.plot(full_sat_data[f'{feature_name}_sim'].values[start:end], label='sim')
        # plt.plot(full_sat_data[f'{feature_name}'].values[start:end], label='gt')
        # plt.plot(pred[f'{feature_name}'].values[start:end], label='pred')
        # plt.legend()
        pred = pred[n_train:]
        pred.index = test_sat_data.index
        # # plt.plot(pred['x'])
        # # plt.plot(test_sat_data['x_sim'])
        # plt.show()
        result_df.append(pred)
        # assert np.all(pred.index == train_sat_data.index)
    result_df = pd.concat(result_df, sort=False)
    # print(result_df.head())
    # print(test_data.head())
    result_df.to_csv(out_path, index_label='id')

if __name__ == '__main__':
    track_1_sub30514611('data/train.csv', 'data/Track 1/test.csv', 'data/Track 1/submissions/linear_alignment_with_rescale.csv')
    #
    #
    #     sat_sim_smape = utils.smape(sat_data.loc[n_train:, [f'{f}_sim' for f in features_list]].values,
    #                                 sat_data.loc[n_train:, features_list].values)
    #     sat_new_smape = utils.smape(pred.loc[n_train:, features_list].values,
    #                                 sat_data.loc[n_train:, features_list].values)
    #     result_df.append([sat_id, sat_sim_smape, sat_new_smape])
    # result_df = pd.DataFrame(result_df, columns=['sat_id', 'sat_simulation_smape', 'sat_new_smape'])