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
        self.b = self._eval_trend_coef(t=t_, x=x_)

    def predict(self, t, x):
        assert self.a is not None, 'please fit the model first'
        assert self.b is not None, 'please fit the model first'

        modified_t = self._apply_speed_up_coef(t)
        modified_x = x
        # modified_x = self._apply_trend_coef(t=modified_t, x=x)
        modified_x = utils.resample(modified_t, modified_x, t_new=t)  # resample to original t
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
    def _eval_trend_coef(t, x):
        peaks = utils.get_peaks(x)
        if len(peaks) < 3:
            return 1
        first_cycle = x[peaks[0]: peaks[1]]
        last_cycle = x[peaks[-2]: peaks[-1]]
        trend_coef = (((last_cycle.max() - last_cycle.min()) -
                 (first_cycle.max() - first_cycle.min())) /
                (t[peaks[-1]] - t[peaks[1]]))
        return trend_coef

    def _apply_speed_up_coef(self, t):
        return t[0] + (t - t[0]) * self.a

    def _apply_trend_coef(self, t, x):
        return x + t[0] + self.b * (t - t[0])
