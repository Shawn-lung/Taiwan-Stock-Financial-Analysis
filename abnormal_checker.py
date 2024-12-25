import numpy as np

class AbnormalMetricChecker:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def detect_abnormal_value(self, base_value, historical_values, var_name):

        if not historical_values:
            return f"{var_name}: 無法偵測 (缺少歷史資料)", None

        mean_hist = np.mean(historical_values)
        if mean_hist == 0:
            return f"{var_name}: 無法偵測 (歷史平均 = 0)", None

        # ratio > 1 => 基期高於平均; ratio < 1 => 基期低於平均
        ratio = base_value / mean_hist

        # 根據 threshold 判定
        upper_bound = 1 + self.threshold  # 高於這個倍數→異常高
        lower_bound = 1 - self.threshold  # 低於這個倍數→異常低

        if ratio > upper_bound:
            status = f"{var_name}: 異常偏高 (基期約為歷史平均的 {ratio:.2f} 倍)"
        elif ratio < lower_bound:
            status = f"{var_name}: 異常偏低 (基期約為歷史平均的 {ratio:.2f} 倍)"
        else:
            status = f"{var_name}: 正常範圍 (基期約為歷史平均的 {ratio:.2f} 倍)"

        return status, ratio

    def detect_abnormal_base_year_metrics(self, base_metrics, hist_metrics):

        results = {}
        for var_name, base_value in base_metrics.items():
            historical_values = hist_metrics.get(var_name, [])
            status, ratio = self.detect_abnormal_value(base_value, historical_values, var_name)
            results[var_name] = (status, ratio)
        return results
