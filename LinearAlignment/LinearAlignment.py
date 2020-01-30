import utils


# transforms data array x = (x_{t_0}, x_{t_1}, ..., x_{t_n})
# assume t_0 = 0, then
# x_t -> x_{a * t} + b * t
class LinearAlignment:
    def __init__(self):
        self.a = None
        self.b = None

    # t -- time array
    # x -- simulation data array
    # gt -- ground truth data array
    def fit(self, t, x, gt):
        x_ = utils.resample(t, x)
        t_, gt_ = utils.resample(t, gt, return_t=True)

        self.a = self._eval_speed_up_coef(t=t_, x=x_, gt=gt_)
        self.sim_trend_coef, self.gt_trend_coef, self.init_center, self.init_amplitude, self.scale_coef = self._eval_trend_coef(t=t_, x=x_, gt=gt_)
        # print(self.b)

    def predict(self, t, x):
        assert self.a is not None, 'please fit the model first'
        # assert self.b is not None, 'please fit the model first'

        modified_t = self._apply_speed_up_coef(t)
        modified_x = x
        modified_x = utils.resample(modified_t, modified_x, t_new=t)  # resample to original t
        modified_x = self._apply_trend_coef(t=modified_t, x=modified_x)
        return modified_x

    # estimates a
    @staticmethod
    def _eval_speed_up_coef(t, x, gt):
        peaks_x = utils.get_peaks(x)
        peaks_gt = utils.get_peaks(gt)
        common_peaks_count = min(len(peaks_x), len(peaks_gt))
        speed_up_coef = (t[peaks_gt[common_peaks_count - 1]] - t[peaks_gt[0]]) / (
                t[peaks_x[common_peaks_count - 1]] - t[peaks_x[0]])
        return speed_up_coef

    # estimates b
    @staticmethod
    def _eval_trend_coef(t, x, gt):
        # return 1
        peaks = utils.get_peaks(x)
        if len(peaks) < 3:
            return 0
        first_cycle = x[peaks[0]: peaks[1]]
        last_cycle = x[peaks[-2]: peaks[-1]]
        trend_coef_max = ((last_cycle.max() - first_cycle.max()) / (t[peaks[-1]] - t[peaks[1]]))
        trend_coef_min = ((last_cycle.min() - first_cycle.min()) / (t[peaks[-1]] - t[peaks[1]]))
        scale_coef_sim = (((last_cycle.max() - last_cycle.min()) - (first_cycle.max() - first_cycle.min())) /
                         (t[peaks[-1]] - t[peaks[1]]))
        sim_trend_coef = (trend_coef_max + trend_coef_min) / 2

        gt_peaks = utils.get_peaks(gt)
        first_cycle = gt[gt_peaks[0]: gt_peaks[1]]
        last_cycle = gt[gt_peaks[-2]: gt_peaks[-1]]
        gt_trend_coef_max = ((last_cycle.max() - first_cycle.max()) / (t[peaks[-1]] - t[peaks[1]]))
        gt_trend_coef_min = ((last_cycle.min() - first_cycle.min()) / (t[peaks[-1]] - t[peaks[1]]))
        gt_trend_coef = (gt_trend_coef_max + gt_trend_coef_min) / 2
        # print('last cycle', last_cycle.max(), last_cycle.min())

        trend_coef = (gt_trend_coef_min + gt_trend_coef_max - trend_coef_min - trend_coef_max) / 2
        init_center = (first_cycle.min() + first_cycle.max()) / 2
        init_amplitude = (first_cycle.max() - first_cycle.min())
        # print('scale', (last_cycle.max(), last_cycle.min(), last_cycle.max() - last_cycle.min()), (first_cycle.max(), first_cycle.min(), first_cycle.max() - first_cycle.min()))
        scale_coef_gt = (((last_cycle.max() - last_cycle.min()) - (first_cycle.max() - first_cycle.min())) /
                      (t[gt_peaks[-1]] - t[gt_peaks[1]]))
        # print('scale coef', scale_coef_sim * (t[peaks[-1]] - t[peaks[1]]), scale_coef_gt * (t[peaks[-1]] - t[peaks[1]]))
        return sim_trend_coef, gt_trend_coef, init_center, init_amplitude, scale_coef_gt - scale_coef_sim

    def _apply_speed_up_coef(self, t):
        return t[0] + (t - t[0]) * self.a

    def _apply_trend_coef(self, t, x):
        dt = t - t[0]
        return (x - self.init_center - self.sim_trend_coef * dt) / self.init_amplitude * (self.init_amplitude + self.scale_coef * (t - t[0])) + self.init_center + self.gt_trend_coef * dt
        # return (x - self.init_center + self.b * (t - t[0])) * (1 + self.scale_coef * (t - t[0])) + self.init_center
