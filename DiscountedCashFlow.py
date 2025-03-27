import yfinance as yf
import pandas as pd
import numpy as np

import Wacc  # 假設您有自己的 Wacc.WACCModel

class DCFModel:
    def __init__(
        self,
        stock_code,
        forecast_years=5,
        perpetual_growth_rate=0.025,
        # (A) 營收、CAPEX、WC：原本就有的三組可手動 yoy 因子
        manual_growth_rates=None,
        manual_capex_factors=None,
        manual_wc_factors=None,
        # (B) 下面新增三組可手動 yoy 因子
        manual_depr_factors=None,     # 折舊 yoy 調整 (相對前一年)
        manual_opincome_factors=None, # 營業利益 yoy 調整 (相對前一年)
        manual_tax_factors=None,       # 稅率 yoy 調整 (相對前一年)
        manual_mvfirm_sales_ratio=None
    ):
        """
        :param stock_code: 股票代號 (e.g. "2330.TW")
        :param forecast_years: 預測年數 (Stage 1)
        :param perpetual_growth_rate: 永續成長率 (用於計算Terminal Value)

        :param manual_growth_rates: list, e.g. [0.2, 0.1, 0.1, 0.05, 0.05]
               => 每年(相對前一年)的「營收」成長率
        :param manual_capex_factors: list, e.g. [0.2, 0.1, 0, -0.05, -0.05]
               => 每年(相對前一年) CAPEX 的增減比例
        :param manual_wc_factors: list, e.g. [0.1, 0.05, 0, 0, 0]
               => 每年(相對前一年) Working Capital 的增減比例

        :param manual_depr_factors: list, e.g. [0.0, 0.05, 0.05, 0, -0.1]
               => 每年(相對前一年) 「折舊」增減比例
        :param manual_opincome_factors: list, e.g. [0.25, 0.15, 0.1, 0.08, 0.05]
               => 若要單獨指定「營業利益」yoy (相對前一年)
               => 若不指定，會自動沿用 manual_growth_rates 去算營業利益
        :param manual_tax_factors: list, e.g. [0.0, 0.0, 0.05, -0.1, 0.0]
               => 每年(相對前一年) 稅率增減比例 (非絕對值，而是相對前一年±%)
               => 注意要避免變成 < 0% 或 > 100%
        """

        self.stock_code = stock_code
        self.forecast_years = forecast_years
        self.perpetual_growth_rate = perpetual_growth_rate

        # 原本三組 yoy 因子
        self.manual_growth_rates = manual_growth_rates
        self.manual_capex_factors = manual_capex_factors
        self.manual_wc_factors = manual_wc_factors

        # 新增三組 yoy 因子
        self.manual_depr_factors = manual_depr_factors
        self.manual_opincome_factors = manual_opincome_factors
        self.manual_tax_factors = manual_tax_factors

        self.manual_mvfirm_sales_ratio = manual_mvfirm_sales_ratio
        # yfinance 抓取財報
        self.stock = None
        self.income_stmt = None
        self.cash_flow = None
        self.balance_sheet = None
        self.latest_year_income = None
        self.latest_year_cf = None
        self.latest_year_bs = None

        # 基期數值
        self.current_revenue = 0
        self.operating_income = 0
        self.depreciation = 0
        self.capex = 0
        self.tax_rate = 0.20
        self.net_debt = 0
        self.current_working_capital = 0

        self.shares_outstanding = 1
        self.wacc = None

        # 歷史數據檢測用 (若有需要)
        self.base_year_metrics = {}
        self.hist_metrics = {}

        # 初始化
        self.initialize_model()


    # -------------------------
    # (1) 初始化流程
    # -------------------------
    def initialize_model(self):
        self.stock = yf.Ticker(self.stock_code)

        # 報表
        self.income_stmt = self.stock.financials
        self.cash_flow = self.stock.cashflow
        self.balance_sheet = self.stock.balance_sheet

        # sort columns & get latest
        if not self.income_stmt.empty:
            self.income_stmt = self.income_stmt.sort_index(axis=1, ascending=True)
            self.latest_year_income = self.income_stmt.columns[-1]

        if not self.cash_flow.empty:
            self.cash_flow = self.cash_flow.sort_index(axis=1, ascending=True)
            self.latest_year_cf = self.cash_flow.columns[-1]

        if not self.balance_sheet.empty:
            self.balance_sheet = self.balance_sheet.sort_index(axis=1, ascending=True)
            self.latest_year_bs = self.balance_sheet.columns[-1]

        # 抓基期 (最新年度)
        self.current_revenue = self.get_latest_revenue()
        self.operating_income = self.get_latest_operating_income()
        self.depreciation = self.get_latest_depreciation()
        self.capex = self.get_latest_capex()
        self.tax_rate = self.get_tax_rate_estimate()
        self.net_debt = self.get_net_debt()
        self.current_working_capital = self.get_latest_working_capital()

        # 股數
        so = self.stock.info.get("sharesOutstanding", 1)
        if so is None or so <= 0:
            so = 1
        self.shares_outstanding = so

        # 計算 WACC
        self.wacc_for_stock= Wacc.WACCModel(self.stock_code)
        self.wacc = self.wacc_for_stock.calculate_wacc()

        # 可自行做基期異常檢測 (若有需要)
        self.base_year_metrics = {
            "Revenue": self.current_revenue,
            "OperatingIncome": self.operating_income,
            "Depreciation": self.depreciation,
            "CAPEX": self.capex,
            "WorkingCapital": self.current_working_capital
        }
        self.hist_metrics = self.prepare_historical_metrics()


    # -------------------------
    # (2) 報表抓基期 / Tax
    # -------------------------
    def get_latest_revenue(self):
        if self.latest_year_income is None:
            return 0
        try:
            for k in ["Total Revenue", "Revenue"]:
                if k in self.income_stmt.index:
                    val = self.income_stmt.loc[k, self.latest_year_income]
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
            for k in ["Depreciation", "Depreciation & Amortization"]:
                if k in self.cash_flow.index:
                    d_ = self.cash_flow.loc[k, self.latest_year_cf]
                    return abs(d_)
            return 0
        except:
            return 0

    def get_latest_capex(self):
        if self.latest_year_cf is None:
            return 0
        try:
            if "Capital Expenditure" in self.cash_flow.index:
                c_ = self.cash_flow.loc["Capital Expenditure", self.latest_year_cf]
                return abs(c_)
            return 0
        except:
            return 0

    def get_tax_rate_estimate(self):
        if self.latest_year_income is None:
            return 0.20
        try:
            tax_keys = ["Income Tax Expense", "Tax Provision"]
            pretax_keys = ["Income Before Tax", "Pretax Income"]
            tv = None
            pv = None
            for tk in tax_keys:
                if tk in self.income_stmt.index:
                    tv_ = self.income_stmt.loc[tk, self.latest_year_income]
                    if pd.notna(tv_):
                        tv = tv_
                        break
            for pk in pretax_keys:
                if pk in self.income_stmt.index:
                    pv_ = self.income_stmt.loc[pk, self.latest_year_income]
                    if pd.notna(pv_):
                        pv = pv_
                        break
            if pd.notna(tv) and pd.notna(pv) and pv != 0:
                r_ = abs(tv/pv)
                if 0 < r_ < 0.5:  # 合理範圍
                    return r_
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
                td_val = self.balance_sheet.loc["Total Debt", self.latest_year_bs]
                if pd.notna(td_val):
                    td = td_val

            possible_cashes = [
                "Cash And Cash Equivalents",
                "Cash",
                "Cash Cash Equivalents And Short Term Investments"
            ]
            for ck in possible_cashes:
                if ck in self.balance_sheet.index:
                    c_ = self.balance_sheet.loc[ck, self.latest_year_bs]
                    if pd.notna(c_):
                        cash_sti += c_
                        break

            return td - cash_sti
        except:
            return 0

    def get_latest_working_capital(self):
        if self.latest_year_bs is None:
            return 0
        try:
            ca_ = 0
            cl_ = 0
            if "Current Assets" in self.balance_sheet.index:
                c_ = self.balance_sheet.loc["Current Assets", self.latest_year_bs]
                if pd.notna(c_):
                    ca_ = c_
            if "Current Liabilities" in self.balance_sheet.index:
                c2_ = self.balance_sheet.loc["Current Liabilities", self.latest_year_bs]
                if pd.notna(c2_):
                    cl_ = c2_
            return ca_ - cl_
        except:
            return 0

    # -------------------------
    # (3) 準備歷史數據 (若要做異常檢測)
    # -------------------------
    def prepare_historical_metrics(self):
        result = {
            "Revenue": [],
            "OperatingIncome": [],
            "Depreciation": [],
            "CAPEX": [],
            "WorkingCapital": []
        }
        if self.income_stmt.empty:
            return result

        cols = sorted(self.income_stmt.columns)
        if len(cols) > 1:
            past_cols = cols[:-1]
        else:
            past_cols = []

        for col in past_cols:
            rev = 0
            for rkk in ["Total Revenue", "Revenue"]:
                if rkk in self.income_stmt.index:
                    tmp = self.income_stmt.loc[rkk, col]
                    if pd.notna(tmp):
                        rev = tmp
                        break

            op = 0
            if "Operating Income" in self.income_stmt.index:
                tmpop = self.income_stmt.loc["Operating Income", col]
                if pd.notna(tmpop):
                    op = tmpop

            dep = 0
            if not self.cash_flow.empty and (col in self.cash_flow.columns):
                for dkk in ["Depreciation", "Depreciation & Amortization"]:
                    if dkk in self.cash_flow.index:
                        d_ = self.cash_flow.loc[dkk, col]
                        if pd.notna(d_):
                            dep = abs(d_)
                            break

            cap = 0
            if not self.cash_flow.empty and (col in self.cash_flow.columns):
                if "Capital Expenditure" in self.cash_flow.index:
                    c_ = self.cash_flow.loc["Capital Expenditure", col]
                    if pd.notna(c_):
                        cap = abs(c_)

            wc = 0
            if not self.balance_sheet.empty and (col in self.balance_sheet.columns):
                ca_ = 0
                cl_ = 0
                if "Current Assets" in self.balance_sheet.index:
                    catmp = self.balance_sheet.loc["Current Assets", col]
                    if pd.notna(catmp):
                        ca_ = catmp
                if "Current Liabilities" in self.balance_sheet.index:
                    cltmp = self.balance_sheet.loc["Current Liabilities", col]
                    if pd.notna(cltmp):
                        cl_ = cltmp
                wc = ca_ - cl_

            result["Revenue"].append(rev)
            result["OperatingIncome"].append(op)
            result["Depreciation"].append(dep)
            result["CAPEX"].append(cap)
            result["WorkingCapital"].append(wc)

        return result


    # -------------------------
    # (4) 一個小工具：估算近 3 年營收 CAGR
    # -------------------------
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
        if not rev_key:
            return None

        older_col = cols_sorted[-(years+1)]
        newer_col = cols_sorted[-1]
        older_rev = self.income_stmt.loc[rev_key, older_col]
        newer_rev = self.income_stmt.loc[rev_key, newer_col]
        if pd.isna(older_rev) or pd.isna(newer_rev) or older_rev <= 0:
            return None

        cagr = (newer_rev / older_rev) ** (1/years) - 1
        if cagr < -1:
            return None
        return cagr


    # -------------------------
    # (5) 多年度預測 FCF (改為 yoy & 可手動調整 折舊/營業利益/稅率)
    # -------------------------
    def forecast_fcf_list(self):
        """
        逐年計算(相對前一年 yoy)：
          revenue_i   = revenue_{i-1}   * (1 + growth_rates[i])
          (option1) operating_income_i = operating_income_{i-1} * (1 + manual_opincome_factors[i])
              or 如果沒給 manual_opincome_factors => 同 revenue 的 yoy
          capex_i     = capex_{i-1}    * (1 + manual_capex_factors[i])
          wc_i        = wc_{i-1}       * (1 + manual_wc_factors[i])
          depreciation_i = depreciation_{i-1} * (1 + manual_depr_factors[i])
              or 如果沒給 => 維持基期折舊不變

          tax_rate_i  = tax_rate_{i-1} * (1 + manual_tax_factors[i]) 
              or 如果沒給 => 維持初始的 tax_rate

          FCF_i = operating_income_i*(1 - tax_rate_i) + depreciation_i - capex_i - ΔWC
        """

        n = self.forecast_years

        # (A) 各種 yoy 因子的 fallback
        wc_factors    = [0]*n if not self.manual_wc_factors else self.manual_wc_factors
        tax_factors   = [0]*n if not self.manual_tax_factors else self.manual_tax_factors

        # (B) Revenue growth
        if self.manual_growth_rates and len(self.manual_growth_rates) == n:
            growth_rates = self.manual_growth_rates
        else:
            auto_g = self.estimate_historical_revenue_growth()
            if auto_g is None:
                auto_g = 0.05
            growth_rates = [auto_g]*n

        # (C) Operating Income growth
        # 如果有 manual_opincome_factors 就用它，否則就同步 revenue 的 yoy
        if self.manual_opincome_factors and len(self.manual_opincome_factors) == n:
            oi_factors = self.manual_opincome_factors
        else:
            # 同步 revenue
            oi_factors = growth_rates
        if self.manual_capex_factors and len(self.manual_capex_factors) == n:
            capex_factors = self.manual_capex_factors
        else:
            # 同步 revenue
            capex_factors = growth_rates
        
        if self.manual_depr_factors and len(self.manual_depr_factors) == n:
            depr_factors = self.manual_depr_factors
        else:
            # 同步 revenue
            depr_factors = growth_rates

            
        # (D) 第 0 年(基期)
        rev_prev  = float(self.current_revenue)
        op_prev   = float(self.operating_income)
        capex_prev= float(self.capex)
        wc_prev   = float(self.current_working_capital)
        depr_prev = float(self.depreciation)
        tax_prev  = float(self.tax_rate)

        fcf_list = []
        print(oi_factors,capex_factors,wc_factors,depr_factors)
        # (E) 逐年做 yoy
        for i in range(n):
            rev_i   = rev_prev   * (1 + growth_rates[i])
            op_i    = op_prev    * (1 + oi_factors[i])
            capex_i = capex_prev * (1 + capex_factors[i])
            wc_i    = wc_prev    * (1 + wc_factors[i])
            depr_i  = depr_prev  * (1 + depr_factors[i])

            # tax rate 也可能 yoy
            tax_rate_i = tax_prev * (1 + tax_factors[i])
            # 防呆：不讓它超過 100% 或小於 0%
            if tax_rate_i < 0:
                tax_rate_i = 0
            elif tax_rate_i > 1:
                tax_rate_i = 1

            # 計算 FCF_i
            ebit_after_tax = op_i * (1 - tax_rate_i)
            delta_wc = wc_i - wc_prev
            fcf_i = ebit_after_tax + depr_i - capex_i - delta_wc
            fcf_list.append(fcf_i)

            # update => 作為下一輪基準
            rev_prev   = rev_i
            op_prev    = op_i
            capex_prev = capex_i
            wc_prev    = wc_i
            depr_prev  = depr_i
            tax_prev   = tax_rate_i
        self.rev_last = rev_i

        return fcf_list


    # -------------------------
    # (6) 計算 DCF
    # -------------------------
    def calculate_intrinsic_value(self):
        """
        1) 先計算前 forecast_years 的 FCF 並折現
        2) 算最後一年之後的永續成長 (Terminal Value) 並折現
        3) Enterprise Value = 上述兩者相加
        4) Equity Value = EV - NetDebt
        """
        # 防呆：WACC <= g
        if (not self.wacc) or (self.wacc <= self.perpetual_growth_rate):
            print("WACC <= perpetual growth => invalid.")
            return None

        fcf_list = self.forecast_fcf_list()
        if not fcf_list:
            print("No FCF data.")
            return None

        # 計算 Stage 1
        npv_stage_1 = 0.0
        for i, fcf_i in enumerate(fcf_list, start=1):
            npv_stage_1 += fcf_i / ((1 + self.wacc)**i)

        # 算 Terminal Value
        last_fcf = fcf_list[-1]
        fcf_next = last_fcf * (1 + self.perpetual_growth_rate)
        terminal_value = fcf_next / (self.wacc - self.perpetual_growth_rate)

        discounted_tv = terminal_value / ((1 + self.wacc)**len(fcf_list))

        enterprise_value = npv_stage_1 + discounted_tv
        equity_value = enterprise_value - self.net_debt
        if self.manual_mvfirm_sales_ratio:
            self.mvfirm_sales = self.rev_last * self.manual_mvfirm_sales_ratio
        else:
            mvfirm_sales_times = self.wacc_for_stock.get_latest_stock_price()*self.shares_outstanding/float(self.current_revenue)
            print(mvfirm_sales_times)
            self.mvfirm_sales = self.rev_last*mvfirm_sales_times

        discounted_tv_with_mvfirm = self.mvfirm_sales / ((1 + self.wacc)**(len(fcf_list)-1))
        self.equity_value_with_mvfirm = discounted_tv_with_mvfirm - self.net_debt
        # Debug print
        print("----- DCF Calculation -----")
        print(f"Stage 1 (NPV of FCF): {npv_stage_1:,.2f}")
        print(f"Terminal Value: {terminal_value:,.2f}")
        print(f"Discounted TV: {discounted_tv:,.2f}")
        print(f"Enterprise Value: {enterprise_value:,.2f}")
        print(f"Equity Value: {equity_value:,.2f}")

        return equity_value

    def calculate_stock_price(self):
        eqv = self.calculate_intrinsic_value()
        if eqv is None:
            return None
        if self.shares_outstanding <= 0:
            print("Invalid shares_outstanding.")
            return None
        fair_price_with_mvfirm = self.equity_value_with_mvfirm / self.shares_outstanding
        fair_price = eqv / self.shares_outstanding
        print(f"Fair Price= {fair_price:.2f} per share")
        print(f"Fair Price with MV/Firm Sales: {fair_price_with_mvfirm:.2f} per share")
        return fair_price


# -------------------------
# (7) 範例執行
# -------------------------
if __name__ == "__main__":
    # 假設我們要 5 年的預測
    # yoy: e.g. manual_growth_rates=[0.2,0.1,0.1,0.05,0.05]
    # yoy: CAPEX => [0.2,0.1, 0, -0.05, -0.05]
    # yoy: WC => [0.1,0.05,0,0,0]
    # yoy: Depreciation => [0, 0.05, 0.05, 0, -0.1]  (前一年 ±%)
    # yoy: OperatingIncome => [0.25, 0.15, 0.1, 0.08, 0.05] (若想自行指定)
    # yoy: Tax Rate => [0, 0, 0, 0.05, -0.1] => 表示後面幾年稅率變動
    dcf = DCFModel(
        "2330.TW",
        forecast_years=5,
        perpetual_growth_rate=0.025,
        manual_growth_rates=[0.2, 0.1, 0.1, 0.05, 0.05],
        manual_capex_factors=[0.2, 0.1, 0, -0.05, -0.05],
        manual_wc_factors=[0.1, 0.05, 0, 0, 0],
        manual_depr_factors=[0, 0.05, 0.05, 0, -0.1],
        #manual_opincome_factors=[0.25, 0.15, 0.1, 0.08, 0.05],
        manual_tax_factors=[0, 0, 0, 0.05, -0.1]
    )

    price = dcf.calculate_stock_price()
    print(f"Estimated Price= {price}")
