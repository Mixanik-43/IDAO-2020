# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import utils

G = 6.6743e-11 # gravity constant
M = 5.972e+24  # Earth mass

# coord = np.array([x, y, z])
# speed = np.array([Vx, Vy, Vz])
# sat_state = np.array([x, y, z, Vx, Vy, Vz])
def Earth_gravity_model(sat_state, dt):
    
    coord = sat_state[:3]
    speed = sat_state[3:]
    r = np.linalg.norm(coord)       # distance to satellite
    a_abs = (G * M) / (r ** 2)      # acceleration absolutevalue
    a = -(coord / r) * a_abs
    result_speed = speed + a * dt
    result_coord = coord + (speed + result_speed) / 2 * dt
    return np.concatenate([result_coord, result_speed])


# iterating Earth_gravity_model or other function with same interface
def iterative_trajectory_modelling(model, start_state, t_simulation, dt):
    current_state = start_state
    for step in range(int(t_simulation / dt)):
        current_state = model(sat_state=current_state, dt=dt)
    current_state = model(current_state, dt=t_simulation % dt)
    return current_state

# predicting past states
def inverse_iterative_trajectory_modelling(model, start_state, t_simulation, dt=1):
    start_state[3:] *= -1
    end_state = iterative_trajectory_modelling(model, start_state, t_simulation, dt)
    end_state[3:] *= -1
    return end_state


def time_state(row):
    time = row['epoch']
    state = row[state_cols].values
    return time, state
    
def predict_segment(begin_row, end_row, t_new, dt):
    pred = defaultdict(lambda: {})
    if begin_row is not None:
        current_t, current_state = begin_t, begin_state = time_state(begin_row)
        for t in t_new:
            if t >= current_t:
                current_state = iterative_trajectory_modelling(
                    Earth_gravity_model, current_state * 1000, (t - current_t) / 10 ** 9,
                    dt=dt) / 1000
                current_t = t
                pred[current_t]['forward'] = {'pred': current_state, 'sim_duration': current_t - begin_t}
    if end_row is not None:
        current_t, current_state = end_t, end_state = time_state(end_row)
        for t in t_new[::-1]:
            if t <= current_t:
                current_state = inverse_iterative_trajectory_modelling(
                    Earth_gravity_model, current_state * 1000, (current_t - t) / 10 ** 9,
                    dt=dt) / 1000
                current_t = t
                pred[current_t]['backward'] = {'pred': current_state, 'sim_duration': end_t - current_t}
                
    segment_df = []
    for t in sorted(pred.keys()):
        assert len(pred[t]) > 0
        t_pred = np.zeros(6)
        t_weights_sum = 0
        
        for simulation, simulation_res in pred[t].items():
            assert simulation_res['sim_duration'] >= 0
            weight = 1 / (simulation_res['sim_duration'] + 1)
            t_pred += simulation_res['pred'] * weight
            t_weights_sum += weight
            
        segment_df.append(np.concatenate([[t], t_pred / t_weights_sum]))
    return segment_df

def sparse_pred_to_dense(sparse_sat_data, t_new, dt):
    
    
    result = []
    first_row = sparse_sat_data.iloc[0]
    result.extend(predict_segment(None, first_row, t_new[t_new < first_row['epoch']], dt=dt))
    
#     t_new_id_min = t_new_id_max = 0
    for row_id in range(len(sparse_sat_data) - 1):
        begin_row = sparse_sat_data.iloc[row_id]
        end_row = sparse_sat_data.iloc[row_id + 1]
        segment = t_new[(t_new >= begin_row['epoch']) & (t_new < end_row['epoch'])]
        result.extend(predict_segment(begin_row, end_row, segment, dt=dt))
    result.extend(predict_segment(end_row, None, t_new[t_new >= end_row['epoch']], dt=dt))
    return pd.DataFrame(result, columns=['t',] + state_cols)

# basic function used in ZeroKeypointsGenerator
# finding t: x(t)=0.
def get_key_points(t, x):
    spl = InterpolatedUnivariateSpline(t, x)
    roots = spl.roots()
    key_points = roots[1::2]
    if len(key_points) < 3:
        return key_points, np.zeros_like(key_points)
    
    outlier_scores = np.abs((key_points[2:] + key_points[:-2] - 2 * key_points[1:-1]) /
                             key_points[1:-1])
    np.pad(np.abs((key_points[2:] + key_points[:-2] - 2 * key_points[1:-1]) /
                             key_points[1:-1]), (2, 2), mode = 'constant',constant_values=1)
    
    threshold = 3 * np.percentile(outlier_scores, 75) - 2 * np.percentile(outlier_scores, 25)
    outliers = (np.convolve(np.pad(outlier_scores > threshold, (2, 2),mode = 'constant',constant_values=1), [1, 1, 1]) == 3)[2:-2]
    return key_points, outliers

def linear_params(t, x):
    model_func = lambda params, t: (params[0] * t + params[1])
    a = (x[-1] - x[0]) / (t[-1] - t[0])
    b = x[0] - a * t[0]
    init_params = (a, b)
    return model_func, init_params

# sinusoid + linear
def sinusoid_plus_linear_params(t, x):
    model_func = lambda params, t: (params[0] *
                                    np.sin(params[1] * t + params[2]) +
                                    params[3] + t * params[4])
    init_params = (np.std(x), 1/(t[-1] - t[0]), 0, np.mean(x), (x[-1] - x[0]) / (t[-1] - t[0]))
    return model_func, init_params

def pick_model_function(t, x):
#     print(len(t))
    if len(t) >= 10:
        return sinusoid_plus_linear_params(t, x)
    else:
        return linear_params(t, x)

def fit_curve(t, x, model_func, init_params):
    def optimize_func(params):
        return model_func(params, t) - x

    ls_params = leastsq(optimize_func, init_params)[0]
    return lambda x: model_func(ls_params, x)

class ZeroKeypointsGenerator:
    def __init__(self, anchor_feature):
        self.anchor_feature = anchor_feature
    
    def get_sim_keypoints(self, sat_data):
        return get_key_points(sat_data['epoch'], sat_data[self.anchor_feature + '_sim'])
    
    def get_gt_keypoints(self, sat_data):
        return get_key_points(sat_data['epoch'], sat_data[self.anchor_feature])
    
# generating a lattice of keypoints
class ShiftZeroKeypointsGenerator:
    def __init__(self, anchor_feature, alpha=1):
        self.anchor_feature = anchor_feature
        self.alpha = alpha
    
    def get_sim_keypoints(self, sat_data):
        kp, outliers = get_key_points(sat_data['epoch'], sat_data[self.anchor_feature + '_sim'])
        outliers = outliers[1:] | outliers[:-1]
        period = kp[1:] - kp[:-1]
        return kp[:-1] + period * self.alpha, outliers
    
    def get_gt_keypoints(self, sat_data):
        kp, outliers = get_key_points(sat_data['epoch'], sat_data[self.anchor_feature])
        outliers = outliers[1:] | outliers[:-1]
        period = kp[1:] - kp[:-1]
        return kp[:-1] + period * self.alpha, outliers
    
    
def sine_alignment(sat_id, kp_generator, train_t_max):
    sat_data = utils.get_satellite_data(data, sat_id).reset_index(drop=True)
    sat_data = utils.remove_time_jumps_fast(sat_data)
    
    train_sat_data = sat_data[sat_data['epoch'] <= train_t_max]
    
    all_sim_kp, all_sim_kp_outliers = kp_generator.get_sim_keypoints(sat_data)
    # broken simulation handling
    if sat_id == 481:
        pred = sat_data[sat_data['epoch'] > train_t_max][['epoch'] + [c + '_sim' for c in state_cols]]
        pred.columns =  ['t'] + state_cols
        return pred
        
    train_gt_kp, train_gt_kp_outliers = kp_generator.get_gt_keypoints(train_sat_data)
    
    stretch_data = all_sim_kp[:len(train_gt_kp)], train_gt_kp
    if len(train_gt_kp) >= 5:
        use_kp = ~(all_sim_kp_outliers[:len(train_gt_kp)] | train_gt_kp_outliers)
        stretch_data = (stretch_data[0][use_kp], stretch_data[1][use_kp])
    time_stretch_function = fit_curve(*stretch_data,
                                      *pick_model_function(all_sim_kp[:len(train_gt_kp)], train_gt_kp))

    
    keypoints = time_stretch_function(all_sim_kp)
    train_keypoints = keypoints[keypoints < train_t_max]
    test_keypoints = keypoints[len(train_keypoints):]
    
    
    
    sim_stretched_t = time_stretch_function(sat_data['epoch'])
    train_sim_stretched_t = sim_stretched_t[:len(train_sat_data)]
    
    pred = []
    gt = []
    for feature in state_cols:
        sim_feature = feature + '_sim'

        
        # values of simulation at all key points
        all_kp_sim_feature = utils.resample(t=sim_stretched_t.values,
                                              x=sat_data[sim_feature].values,
                                              t_new=keypoints)
        # values of simulation at train key points
        train_kp_sim_feature = all_kp_sim_feature[:len(train_keypoints)]
        
        # ground truth values at train key points
        train_kp_gt_feature = utils.resample(t=train_sat_data['epoch'],
                                             x=train_sat_data[feature],
                                             t_new=train_keypoints)

        # difference between train and ground truth at train keypoints
        train_diff = train_kp_gt_feature - train_kp_sim_feature
#         kp_diff_func = lambda x: np.ones_like(x) * np.mean(train_diff)
        kp_diff_func = fit_curve(train_keypoints, train_diff,
                                  *linear_params(train_keypoints, train_diff))
        pred_kp_diff = kp_diff_func(test_keypoints)


        pred.append(pred_kp_diff + all_kp_sim_feature[len(train_keypoints):])
    pred_df = pd.DataFrame(np.array(pred), index=state_cols).T
    pred_df['epoch'] = test_keypoints
    return pred_df
def sine_alignment_part(sat_id, kp_generator, train_t_max,train_t_min = 0):
    '''
    Args:
    sat_id
    kp_generator = instance of ShiftZeroKeypointsGenerator class
    train_t_max = max time to train on
    
    
    '''
    sat_data = utils.get_satellite_data(data, sat_id).reset_index(drop=True)
    sat_data = utils.remove_time_jumps_fast(sat_data)
    
    train_sat_data = sat_data[sat_data['epoch'] <= train_t_max]
    m = int(train_t_min*train_sat_data.shape[0])
    train_sat_data = train_sat_data[m:]
    
    all_sim_kp, all_sim_kp_outliers = kp_generator.get_sim_keypoints(sat_data)
    # broken simulation handling
    if sat_id == 481:
        pred = sat_data[sat_data['epoch'] > train_t_max][['epoch'] + [c + '_sim' for c in state_cols]]
        pred.columns =  ['t'] + state_cols
        return pred
        
    train_gt_kp, train_gt_kp_outliers = kp_generator.get_gt_keypoints(train_sat_data)
    
    stretch_data = all_sim_kp[:len(train_gt_kp)], train_gt_kp #sim, followed by train  keypts
    if len(train_gt_kp) >= 5:
        use_kp = ~(all_sim_kp_outliers[:len(train_gt_kp)] | train_gt_kp_outliers)
        stretch_data = (stretch_data[0][use_kp], stretch_data[1][use_kp])
    time_stretch_function = fit_curve(*stretch_data,
                                      *pick_model_function(all_sim_kp[:len(train_gt_kp)], train_gt_kp)) # ln=False
    
    keypoints = time_stretch_function(all_sim_kp)
    train_keypoints = keypoints[keypoints < train_t_max]
    test_keypoints = keypoints[len(train_keypoints):]
    
    sim_stretched_t = time_stretch_function(sat_data['epoch'])
    train_sim_stretched_t = sim_stretched_t[:len(train_sat_data)]
    
    pred = []
    gt = []
    for feature in state_cols:
        sim_feature = feature + '_sim'

        
        # values of simulation at all key points
        all_kp_sim_feature = utils.resample(t=sim_stretched_t.values,
                                              x=sat_data[sim_feature].values,
                                              t_new=keypoints)
        # values of simulation at train key points
        train_kp_sim_feature = all_kp_sim_feature[:len(train_keypoints)]
        
        # ground truth values at train key points
        train_kp_gt_feature = utils.resample(t=train_sat_data['epoch'],
                                             x=train_sat_data[feature],
                                             t_new=train_keypoints)

        # difference between train and ground truth at train keypoints
        train_diff = train_kp_gt_feature - train_kp_sim_feature
#         kp_diff_func = lambda x: np.ones_like(x) * np.mean(train_diff)
        kp_diff_func = fit_curve(train_keypoints, train_diff,
                                  *linear_params(train_keypoints, train_diff))
        pred_kp_diff = kp_diff_func(test_keypoints)


        pred.append(pred_kp_diff + all_kp_sim_feature[len(train_keypoints):])
    pred_df = pd.DataFrame(np.array(pred), index=state_cols).T
    pred_df['epoch'] = test_keypoints
    return pred_df
