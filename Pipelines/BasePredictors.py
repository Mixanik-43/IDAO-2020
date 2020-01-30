import pandas as pd
import utils


class SingleSatellitePredictor:
    def fit(self, train_csv_path):
        data = pd.read_csv(train_csv_path)
        satellites_list = data['sat_id'].unique()
        self.sat_models_params = {}
        for sat_id in satellites_list:
            sat_data = utils.get_satellite_data(data, sat_id)
            self.sat_models_params[sat_id] = self.fit_satellite(sat_data)


    def predict(self, test_csv_path):
        sat_preds = []
        data = pd.read_csv(test_csv_path)
        satellites_list = data['sat_id'].unique()
        for sat_id in satellites_list:
            sat_data = utils.get_satellite_data(data, sat_id)
            sat_preds.append(self.predict_satellite(sat_data, self.sat_models[sat_id]))
        return pd.concat(sat_preds)

    def fit_satellite(self, sat_df):
        raise NotImplementedError

    def predict_satellite(self, sat_df, sat_model):
        raise NotImplementedError

    def fit_predict(self, train_csv_path, test_csv_path):
        self.fit(train_csv_path)
        return  self.predict_satellite(test_csv_path)
