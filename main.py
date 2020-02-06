import pandas as pd
import numpy as np

test = pd.read_csv("test.csv")

def remove_time_jumps_fast(data, features_list=
                           ('x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim'),
                           threshold = 0.000001):
    #time_threshold 0.00003 sufficient for test and train
    #time_threshold 0.00002 will throw errors
    '''
        removes time jumps in the simulation for a single satellite
        for train and test data, sufficient to set time_threshold at default
        s_data = satellite data
        the features are replaced by the correction
        note that threshold here is not the same as in remove_time_jumps
        '''
    epoch_ind = data.columns.get_loc('epoch')
    data['t'] = ((pd.to_datetime(data['epoch']) - pd.to_datetime(data.iloc[0,epoch_ind])) /
                 np.timedelta64(1, 'D')).astype(float)
    data['dt'] = data['t'].diff(1)
                 
    index_for_correction = data[data['dt'] < threshold].index
                 #print(index_for_correction)
    if list(index_for_correction): #if non empty
        for feature in features_list:
            for i in index_for_correction:
                j = data.index.get_loc(i)
                data = insert_previous_and_shift(data,feature,j)
    return data

def insert_previous_and_shift(df,col_name,ind):
    '''
        input a data frame (df), column name (col_name), and index (ind)
        insert previous value of df[col_name] at index and shift the rest
        of df[col_name] from ind by +1;
        This is used for remove_time_jumps_fast
        '''
    shifted_series = df[col_name].shift(1)
    df[col_name].iloc[ind] = df[col_name].iloc[ind-1]
    df[col_name].iloc[ind+1:] = shifted_series.iloc[ind+1:]
    return df


def get_satellite_data(data, sat_id):
    '''
        returns all data for particular satellite by id
        '''
    return data[data['sat_id'] == sat_id]


remove_jumps = pd.DataFrame([])
for sat_id in test['sat_id'].unique():
    sat_data = get_satellite_data(test, sat_id)
    sat_data = remove_time_jumps_fast(sat_data)
    remove_jumps = remove_jumps.append(sat_data)

# just sending simulated values as the answer
submission = remove_jumps[["id", "x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]]
submission.columns = ["id", "x", "y", "z", "Vx", "Vy", "Vz"]
submission.to_csv("submission.csv", index=False)
