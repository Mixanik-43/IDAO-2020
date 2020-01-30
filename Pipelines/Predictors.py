import numpy as np
import pandas as pd
from tqdm import tqdm
import utils
from LinearAlignment.LinearAlignment import LinearAlignment
from Pipelines.BasePredictors import SingleSatellitePredictor


class Track1_sub30514611(SingleSatellitePredictor):
    def __init__(self):
        super(self, Track1_sub30514611).__init__()
        self.features_list = ('x', 'y', 'z', 'Vx', 'Vy', 'Vz')

    def fit_satellite(self, sat_df):
        self.preprocess_data(sat_df)
        for feature_name in self.features_list:
        alignment_model = LinearAlignment()


    def predict_satellite(self, sat_df, sat_model):
        self.preprocess_data(sat_df)
        sat_pred = pd.DataFrame([])
        for feature_name in self.features_list:
        alignment_model = LinearAlignment()


    @staticmethod
    def preprocess_data(df):
        df['epoch'] = pd.to_datetime(df['epoch']).astype(np.int64)



