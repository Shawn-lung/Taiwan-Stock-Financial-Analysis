from Wacc import WACCModel
import yfinance as yf
import pandas as pd

# 從 abnormal_checker.py import 您的檢測器
from abnormal_checker import AbnormalMetricChecker


class DCFModel:
    def __init__(
        self,
        stock_code,
        forecast_years=5,
        perpetual_growth_rate=0.025,
        manual_growth_rates=None
    ):


        self.stock_code = stock_code
        self.forecast_years = forecast_years
        self.perpetual_growth_rate = perpetual_growth_rate


        self.wacc_model = WACCModel(stock_code)
        self.stock = self.wacc_model.stock


        self.income_stmt = self.stock.financials  # 損益表
        self.cash_flow = self.stock.cashflow      # 現金流量表
        self.balance_sheet = self.stock.balance_sheet  # 資產負債表


        self.latest_year_income = None
        if not self.income_stmt.empty:
            self.income_stmt = self.income_stmt.sort_index(axis=1, ascending=True)
            self.latest_year_income = self.income_stmt.columns[-1]

        self.latest_year_cf = None
        if not self.cash_flow.empty:
            self.cash_flow = self.cash_flow.sort_index(axis=1, ascending=True)
            self.latest_year_cf = self.cash_flow.columns[-1]

        self.latest_year_bs = None
        if not self.balance_sheet.empty:
            self.balance_sheet = self.balance_sheet.sort_index(axis=1, ascending=True)
            self.latest_year_bs = self.balance_sheet.columns[-1]


        self.current_revenue = self.get_latest_revenue()
        self.operating_income = self.get_latest_operating_income()
        self.depreciation = self.get_latest_depreciation()
        self.capex = self.get_latest_capex()
        self.tax_rate = self.get_tax_rate_estimate()
        self.net_debt = self.get_net_debt()

        self.current_working_capital = self.get_latest_working_capital()


        self.shares_outstanding = self.stock.info.get("sharesOutstanding", 1) or 1
        self.wacc = self.wacc_model.calculate_wacc()

        self.manual_growth_rates = manual_growth_rates  # 若使用者有手動帶入，就儲存起來


        self.base_year_metrics = {
            "Revenue": self.current_revenue,
            "OperatingIncome": self.operating_income,
            "Depreciation": self.depreciation,
            "CAPEX": self.capex,
            "WorkingCapital": self.current_working_capital
        }

        # 抓「歷史數據」(多年度)
        self.hist_metrics = self.prepare_historical_metrics()

        # 執行「基期年度異常檢測」
        self.abnormal_threshold = 3  # 超過3個指標異常，就視為「基期太多異常」
        self.abnormal_count = 0      # 計算異常項目個數
        self.too_many_abnormal = False

        self.check_base_year_anomalies()


    def get_latest_revenue(self):
        if self.latest_year_income is None:
            return 0
        try:
            for possible_key in ["Total Revenue", "Revenue"]:
                if possible_key in self.income_stmt.index:
                    val = self.income_stmt.loc[possible_key, self.latest_year_income]
                    if pd.notna(val):
                        return val
            return 0
        except:
            return 0

    def get_latest_operating_income(self):
        if self.latest_year_income is None:
            return 0
        try:
            if "Operating Income" in self.income_stmt.index:
                val = self.income_stmt.loc["Operating Income", self.latest_year_income]
                return val
            return 0
        except:
            return 0

    def get_latest_depreciation(self):
        if self.latest_year_cf is None:
            return 0
        try:
            for possible_key in ["Depreciation", "Depreciation & Amortization"]:
                if possible_key in self.cash_flow.index:
                    val = self.cash_flow.loc[possible_key, self.latest_year_cf]
                    return abs(val)
            return 0
        except:
            return 0

    def get_latest_capex(self):
        if self.latest_year_cf is None:
            return 0
        try:
            if "Capital Expenditure" in self.cash_flow.index:
                val = self.cash_flow.loc["Capital Expenditure", self.latest_year_cf]
                return abs(val)
            return 0
        except:
            return 0

    def get_tax_rate_estimate(self):
        if self.latest_year_income is None:
            return 0.20
        try:
            tax_expense_keys = ["Income Tax Expense", "Tax Provision"]
            pretax_keys = ["Income Before Tax", "Pretax Income"]

            tax_val = None
            pretax_val = None
            for tk in tax_expense_keys:
                if tk in self.income_stmt.index:
                    tax_val = self.income_stmt.loc[tk, self.latest_year_income]
                    break
            for pk in pretax_keys:
                if pk in self.income_stmt.index:
                    pretax_val = self.income_stmt.loc[pk, self.latest_year_income]
                    break

            if pd.notna(tax_val) and pd.notna(pretax_val) and pretax_val != 0:
                est_rate = abs(tax_val / pretax_val)
                if 0 < est_rate < 0.5:
                    return est_rate
            return 0.20
        except:
            return 0.20

    def get_net_debt(self):
        if self.latest_year_bs is None:
            return 0
        try:
            td = 0
            cash_sti = 0
            if "Total Debt" in self.balance_sheet.index:
                tmp = self.balance_sheet.loc["Total Debt", self.latest_year_bs]
                if pd.notna(tmp):
                    td = tmp

            # 假設"Cash" / "Cash Equivalents"等
            possible_cash_keys = [
                "Cash And Cash Equivalents",
                "Cash",
                "Cash Cash Equivalents And Short Term Investments"
            ]
            for ck in possible_cash_keys:
                if ck in self.balance_sheet.index:
                    c_val = self.balance_sheet.loc[ck, self.latest_year_bs]
                    if pd.notna(c_val):
                        cash_sti += c_val
                        break
            net_debt = td - cash_sti
            print(net_debt)
            return net_debt
        except:
            return 0

    def get_latest_working_capital(self):
        """
        WorkingCapital = Current Assets - Current Liabilities
        若要簡化, 就抓 'Current Assets' / 'Current Liabilities' (無 'Total')
        """
        if self.latest_year_bs is None:
            return 0
        try:
            curr_assets = 0
            curr_liab = 0

            if "Current Assets" in self.balance_sheet.index:
                ca_tmp = self.balance_sheet.loc["Current Assets", self.latest_year_bs]
                if pd.notna(ca_tmp):
                    curr_assets = ca_tmp

            if "Current Liabilities" in self.balance_sheet.index:
                cl_tmp = self.balance_sheet.loc["Current Liabilities", self.latest_year_bs]
                if pd.notna(cl_tmp):
                    curr_liab = cl_tmp

            return curr_assets - curr_liab
        except:
            return 0


    def prepare_historical_metrics(self):
        """
        回傳 dict of list:
          {
            "Revenue": [...],
            "OperatingIncome": [...],
            "Depreciation": [...],
            "CAPEX": [...],
            "WorkingCapital": [...]
          }
        """
        hist_data = {
            "Revenue": [],
            "OperatingIncome": [],
            "Depreciation": [],
            "CAPEX": [],
            "WorkingCapital": []
        }
        # columns sorted
        if self.income_stmt.empty:
            return hist_data

        all_cols = sorted(self.income_stmt.columns)
        if len(all_cols) > 1:
            past_cols = all_cols[:-1]  # 除去最新
        else:
            past_cols = []

        for col in past_cols:
            # Revenue
            rev_val = 0
            for possible_key in ["Total Revenue", "Revenue"]:
                if possible_key in self.income_stmt.index:
                    tmp = self.income_stmt.loc[possible_key, col]
                    if pd.notna(tmp):
                        rev_val = tmp
                        break

            # OperatingIncome
            op_val = 0
            if "Operating Income" in self.income_stmt.index:
                tmp = self.income_stmt.loc["Operating Income", col]
                if pd.notna(tmp):
                    op_val = tmp

            # Dep
            dep_val = 0
            if not self.cash_flow.empty and col in self.cash_flow.columns:
                for dk in ["Depreciation", "Depreciation & Amortization"]:
                    if dk in self.cash_flow.index:
                        d_ = self.cash_flow.loc[dk, col]
                        if pd.notna(d_):
                            dep_val = abs(d_)
                            break

            # CAPEX
            capex_val = 0
            if not self.cash_flow.empty and col in self.cash_flow.columns:
                if "Capital Expenditure" in self.cash_flow.index:
                    c_ = self.cash_flow.loc["Capital Expenditure", col]
                    if pd.notna(c_):
                        capex_val = abs(c_)

            # WC
            wc_val = 0
            if not self.balance_sheet.empty and col in self.balance_sheet.columns:
                ca_ = 0
                cl_ = 0
                if "Current Assets" in self.balance_sheet.index:
                    ca_tmp = self.balance_sheet.loc["Current Assets", col]
                    if pd.notna(ca_tmp):
                        ca_ = ca_tmp
                if "Current Liabilities" in self.balance_sheet.index:
                    cl_tmp = self.balance_sheet.loc["Current Liabilities", col]
                    if pd.notna(cl_tmp):
                        cl_ = cl_tmp
                wc_val = ca_ - cl_

            hist_data["Revenue"].append(rev_val)
            hist_data["OperatingIncome"].append(op_val)
            hist_data["Depreciation"].append(dep_val)
            hist_data["CAPEX"].append(capex_val)
            hist_data["WorkingCapital"].append(wc_val)

        return hist_data

    def check_base_year_anomalies(self, threshold=0.5):
        """
        利用 AbnormalMetricChecker 進行基期 vs 歷史平均的檢測
        若異常指標 >= self.abnormal_threshold (例: 3)，則 self.too_many_abnormal = True
        """
        checker = AbnormalMetricChecker(threshold=threshold)
        results = checker.detect_abnormal_base_year_metrics(self.base_year_metrics, self.hist_metrics)

        print("\n=== [基期年度異常檢測] ===")
        self.abnormal_count = 0
        for var_name, (status, ratio) in results.items():
            print(status)
            # 如果 status 裡包含 "異常偏高" 或 "異常偏低" 就計為一次「異常」
            if ("異常偏高" in status) or ("異常偏低" in status):
                self.abnormal_count += 1
        print("================================\n")

        if self.abnormal_count >= self.abnormal_threshold:
            self.too_many_abnormal = True

    def get_base_fcf(self):
        """
        基期 FCF = OperatingIncome * (1 - tax_rate) + depreciation - capex
        """
        ebit_after_tax = self.operating_income * (1 - self.tax_rate)
        fcf = ebit_after_tax + self.depreciation - self.capex
        return fcf

    def forecast_fcf_list(self):
        """
        若 manual_growth_rates 有值 => 用之
        否則 => 用 estimate_historical_revenue_growth() or 5% 預設
        """
        base_fcf = self.get_base_fcf()
        if base_fcf is None:
            return []

        # 手動
        if self.manual_growth_rates is not None and len(self.manual_growth_rates) == self.forecast_years:
            fcf_list = []
            fcf_t = base_fcf
            for g in self.manual_growth_rates:
                fcf_list.append(fcf_t)
                fcf_t = fcf_t * (1 + g)
            return fcf_list

        # 自動
        avg_revenue_growth = self.estimate_historical_revenue_growth()
        if avg_revenue_growth is None:
            avg_revenue_growth = 0.05
        print(f"Auto forecast revenue growth = {avg_revenue_growth:.2%}")

        fcf_list = []
        fcf_t = base_fcf
        for i in range(self.forecast_years):
            fcf_list.append(fcf_t)
            fcf_t = fcf_t * (1 + avg_revenue_growth)
        return fcf_list

    def estimate_historical_revenue_growth(self, years=3):
        if self.income_stmt.empty:
            return None
        cols_sorted = sorted(self.income_stmt.columns)
        if len(cols_sorted) < years + 1:
            return None

        rev_key = None
        for k in ["Total Revenue", "Revenue"]:
            if k in self.income_stmt.index:
                rev_key = k
                break
        if rev_key is None:
            return None

        older_col = cols_sorted[-(years+1)]
        newer_col = cols_sorted[-1]
        older_rev = self.income_stmt.loc[rev_key, older_col]
        newer_rev = self.income_stmt.loc[rev_key, newer_col]

        if older_rev <= 0 or pd.isna(older_rev) or pd.isna(newer_rev):
            return None

        cagr = (newer_rev / older_rev) ** (1/years) - 1
        return cagr if cagr > -1 else None

    def calculate_intrinsic_value(self):
        """
        1) 若 self.too_many_abnormal => 跳過
        2) 取得 N 年 FCF => 貼現
        3) 終值 => 貼現
        4) 扣除 net_debt => Equity Value
        """
        # 若基期過多異常 => 跳過
        if self.too_many_abnormal:
            print("基期年有過多異常指標，可能影響預測結果")
            #return None

        if self.wacc is None:
            print("Error: WACC is None, cannot proceed DCF.")
            return None

        if self.wacc <= self.perpetual_growth_rate:
            print("Error: WACC <= perpetual growth rate, model invalid.")
            return None

        fcf_list = self.forecast_fcf_list()
        if not fcf_list:
            print("No FCF data to forecast.")
            return None

        npv_stage_1 = 0.0
        for i, fcf_t in enumerate(fcf_list, start=1):
            npv_stage_1 += fcf_t / ((1 + self.wacc) ** i)

        last_fcf = fcf_list[-1]
        fcf_next = last_fcf * (1 + self.perpetual_growth_rate)
        terminal_value = fcf_next / (self.wacc - self.perpetual_growth_rate)
        discounted_tv = terminal_value / ((1 + self.wacc) ** len(fcf_list))

        enterprise_value = npv_stage_1 + discounted_tv
        equity_value = enterprise_value - self.net_debt

        print("----- DCF Calculation -----")
        print(f"Stage 1 (NPV of FCF): {npv_stage_1:,.2f}")
        print(f"Terminal Value (undiscounted): {terminal_value:,.2f}")
        print(f"Discounted Terminal Value: {discounted_tv:,.2f}")
        print(f"Enterprise Value: {enterprise_value:,.2f}")
        print(f"Equity Value: {equity_value:,.2f}")
        return equity_value

    def calculate_stock_price(self):
        equity_value = self.calculate_intrinsic_value()
        if equity_value is None:
            return None
        if not self.shares_outstanding or self.shares_outstanding <= 0:
            print("Error: invalid shares_outstanding.")
            return None

        fair_price = equity_value / self.shares_outstanding
        print(f"Fair Price = {fair_price:.2f} per share")
        return fair_price



if __name__ == "__main__":
    print("=== [Manual Growth DCF] ===")
    manual_rates = [0.2, 0.15, 0.1, 0.05, 0.05]  # 5年
    dcf = DCFModel(
        "8069.TWO",
        forecast_years=5,
        perpetual_growth_rate=0.025,
        manual_growth_rates=manual_rates
    )
    estimated_price = dcf.calculate_stock_price()
    print(f"Estimated Price (Manual) = {estimated_price}")
