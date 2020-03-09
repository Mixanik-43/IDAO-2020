#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# Load the libraries

import numpy as np
import pandas as pd
from tqdm import tqdm
import utils
import NonlinearAlignment as na
# In[ ]:


# Load the data 

np.random.seed(50)
coord_cols = ['x', 'y', 'z']
speed_cols = ['Vx', 'Vy', 'Vz']
state_cols = coord_cols + speed_cols

print('Loading data...')
train_data = pd.read_csv('data/train.csv', index_col='id')
train_data['epoch'] = pd.to_datetime(train_data['epoch']).values.astype(float)

test_data = pd.read_csv('data/test.csv', index_col='id')
test_data['epoch'] = pd.to_datetime(test_data['epoch']).values.astype(float)

data = pd.concat([train_data, test_data], sort=False).reset_index(drop=True)
print('Data loaded.')

# In[ ]:


all_predictions_shiftzero_kp = {}

print('Begin modeling. Expected to take 7-10 hours.')
for sat_id in tqdm(test_data['sat_id'].unique()):
    try:
        sat_data = utils.get_satellite_data(data, sat_id).reset_index(drop=True)
        sat_data = utils.remove_time_jumps_fast(sat_data)
        
        train_t = utils.get_satellite_data(train_data, sat_id)['epoch']
        test_t = utils.get_satellite_data(test_data, sat_id)['epoch']
        pred_dfs = []
        sparse_pred_dfs = []
        # running sine_alignment for different lattices:
        # different alphas and anchor features
        for anchor in state_cols:
            for alpha in np.linspace(0, 1, 100)[1:]:
                pred_df = na.sine_alignment(sat_data, sat_id, na.ShiftZeroKeypointsGenerator(anchor, alpha), train_t.max())
                sparse_pred_dfs.append(pred_df)
        sparse_pred = pd.concat(sparse_pred_dfs).sort_values('epoch').reset_index(drop=True)
        dense_pred = na.sparse_pred_to_dense(sparse_pred, test_t, dt=8)
        all_predictions_shiftzero_kp[sat_id] = dense_pred
    except Exception as e:
        print(e)
        
print('Satellite 481 with broken simulation...')
# satellite 481 has broken simulation.
sat_data = utils.get_satellite_data(data, 481).reset_index(drop=True)
sat_data = utils.remove_time_jumps_fast(sat_data)
train_t = utils.get_satellite_data(train_data, sat_id)['epoch']
all_predictions_shiftzero_kp[481] = na.sine_alignment(sat_data,481, na.ShiftZeroKeypointsGenerator('x', 0), train_t.max())
print('Modeling complete...')

# "problem satellites", based on smape, identified by rerunning sine alingment on first 75% of January,
# predicting next 25% of January
problem_sat = sorted([1,587, 372,  37, 473, 523, 514, 253, 481,
                      35, 515, 162, 277, 244, 443, 572, 362,
                      550,  26, 310, 252, 517, 127, 396, 391,
                      471, 333, 28,  54, 6, 502, 316, 225,
                      82, 309, 268, 470, 456, 460, 510,516,
                      22, 194, 511, 544, 438, 435, 486,  52, 548, 528, 310])

print('Smoothing for 52 "problem satellites"...')
# smoothing with median() rolling window(N=3)
for sat_id in problem_sat:
    
    sat_data = all_predictions_shiftzero_kp[sat_id].copy()
    for feature_name in state_cols :
        indx = sat_data.index        
        sat_data.loc[indx,feature_name] = sat_data.loc[indx,feature_name].rolling(window=3).median().shift(-1).values
        all_predictions_shiftzero_kp[sat_id].loc[indx[1:-2], :]= sat_data.loc[indx[1:-2],:].values    



df_shiftzero_kp = pd.concat([all_predictions_shiftzero_kp[k]
                            for k in sorted(all_predictions_shiftzero_kp.keys())])[state_cols]
df_shiftzero_kp.index = test_data.index
# saving submission csv
df_shiftzero_kp.to_csv('data/submissions/shiftzero_kp.csv', index_label='id')









