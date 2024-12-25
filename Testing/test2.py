from Wacc import WACCModel
import yfinance as yf
import pandas as pd

class DCFModel:
    def __init__(self, stock_code, forecast_years=5, perpetual_growth_rate=0.025, manual_growth_rates=None):
        """
        :param stock_code: 股票代號 (e.g. "2330.TW")
        :param forecast_years: 要預測的年數 (例如5年)
        :param perpetual_growth_rate: 永續增長率 (計算Terminal Value用)
        :param manual_growth_rates: 如果有手動輸入的多年度成長率 [0.1, 0.08, ...]，沒有則 None
        """
        self.stock_code = stock_code
        self.forecast_years = forecast_years
        self.perpetual_growth_rate = perpetual_growth_rate
        
        # 1) 初始化 WACCModel
        self.wacc_model = WACCModel(stock_code)
        self.stock = self.wacc_model.stock

        # 抓取 Income Statement, Cash Flow, Balance Sheet (省略細節與錯誤處理)
        self.income_stmt = self.stock.financials
        self.cash_flow = self.stock.cashflow
        self.balance_sheet = self.stock.balance_sheet

        # 2) 找到最新年度欄位 (若有資料)
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

        # 3) 抓財務數據 (收入, OP Income, 折舊, CAPEX, 稅率, NetDebt)
        self.current_revenue = self.get_latest_revenue()
        self.operating_income = self.get_latest_operating_income()
        self.depreciation = self.get_latest_depreciation()
        self.capex = self.get_latest_capex()
        self.tax_rate = self.get_tax_rate_estimate()
        self.net_debt = self.get_net_debt()

        # 4) 股數、WACC
        self.shares_outstanding = self.stock.info.get("sharesOutstanding", 1) or 1
        self.wacc = self.wacc_model.calculate_wacc()

        # 5) 手動/自動成長率
        self.manual_growth_rates = manual_growth_rates  # 若使用者有手動帶入，就儲存起來

    ######################################################
    # 從 yfinance 報表抓取所需財務數據 (示範以下數個方法) #
    ######################################################
    def get_latest_revenue(self):
        """
        從 income statement 中抓 "Total Revenue" 最新年度數字
        """
        if self.latest_year_income is None:
            return 0
        try:
            # 有時索引名稱是 "Total Revenue" 或 "Revenue"
            for possible_key in ["Total Revenue", "Revenue"]:
                if possible_key in self.income_stmt.index:
                    val = self.income_stmt.loc[possible_key, self.latest_year_income]
                    if pd.notna(val):
                        return val
            print("Warning: 'Total Revenue' or 'Revenue' not found. Return 0.")
            return 0
        except Exception as e:
            print(f"Error getting latest revenue: {e}")
            return 0

    def get_latest_operating_income(self):
        """
        從 income statement 中抓 "Operating Income" 最新年度
        通常用於替代 EBIT(Operating profit)
        """
        if self.latest_year_income is None:
            return 0
        try:
            if "Operating Income" in self.income_stmt.index:
                val = self.income_stmt.loc["Operating Income", self.latest_year_income]
                return val
            else:
                print("Warning: 'Operating Income' not found. Using 0.")
                return 0
        except Exception as e:
            print(f"Error getting operating income: {e}")
            return 0

    def get_latest_depreciation(self):
        """
        從現金流量表中抓 "Depreciation" (或 "Depreciation & Amortization")
        """
        if self.latest_year_cf is None:
            return 0
        try:
            for possible_key in ["Depreciation", "Depreciation & Amortization"]:
                if possible_key in self.cash_flow.index:
                    val = self.cash_flow.loc[possible_key, self.latest_year_cf]
                    # 通常為負值(會計列式)，所以取絕對值
                    return abs(val)
            print("Warning: Depreciation item not found. Return 0.")
            return 0
        except Exception as e:
            print(f"Error getting depreciation: {e}")
            return 0

    def get_latest_capex(self):
        """
        從現金流量表中抓 "Capital Expenditures"
        通常為負值(現金流出)，取絕對值
        """
        if self.latest_year_cf is None:
            return 0
        try:
            if "Capital Expenditure" in self.cash_flow.index:
                val = self.cash_flow.loc["Capital Expenditure", self.latest_year_cf]
                return abs(val)
            print("Warning: 'Capital Expenditures' not found. Return 0.")
            return 0
        except Exception as e:
            print(f"Error getting capex: {e}")
            return 0

    def get_tax_rate_estimate(self):
        """
        嘗試估計稅率 = (Income Tax Expense) / (Pretax Income)
        或者可用 "Tax Provision" / "Pretax Income" (視yfinance欄位而定)
        """
        try:
            if self.latest_year_income is None:
                return 0.20  # 給一個預設值
            tax_expense_keys = ["Income Tax Expense", "Tax Provision"]
            pretax_income_keys = ["Income Before Tax", "Pretax Income"]
            tax_expense_val = None
            pretax_income_val = None

            for tk in tax_expense_keys:
                if tk in self.income_stmt.index:
                    tax_expense_val = self.income_stmt.loc[tk, self.latest_year_income]
                    break
            for pk in pretax_income_keys:
                if pk in self.income_stmt.index:
                    pretax_income_val = self.income_stmt.loc[pk, self.latest_year_income]
                    break

            if pd.notna(tax_expense_val) and pd.notna(pretax_income_val) and pretax_income_val != 0:
                est_tax_rate = abs(tax_expense_val / pretax_income_val)
                # 一般不會 > 1，也不會 < 0
                if 0 < est_tax_rate < 0.5:
                    return est_tax_rate
            # 若抓不到，就給預設值
            return 0.20
        except Exception as e:
            print(f"Error getting tax rate: {e}")
            return 0.20

    def get_net_debt(self):
        """
        嘗試計算淨負債 = Total Debt - (Cash + Short Term Investments)
        若抓不到就回傳 0
        """
        if self.latest_year_bs is None:
            return 0
        try:
            total_debt = 0
            cash_and_sti = 0

            # 1) 取 Total Debt
            if "Total Debt" in self.balance_sheet.index:
                td_val = self.balance_sheet.loc["Total Debt", self.latest_year_bs]
                if pd.notna(td_val):
                    total_debt = td_val

            # 2) 取 Cash 或 Cash And Cash Equivalents And Short Term Investments
            #    依您的實務情況調整
            possible_cash_keys = [
                "Cash And Cash Equivalents",
                "Cash",
                "Cash Cash Equivalents And Short Term Investments"
            ]
            for ck in possible_cash_keys:
                if ck in self.balance_sheet.index:
                    c_val = self.balance_sheet.loc[ck, self.latest_year_bs]
                    if pd.notna(c_val):
                        cash_and_sti += c_val
                        break

            net_debt = total_debt - cash_and_sti
            print (net_debt)
            return net_debt
        except Exception as e:
            print(f"Error calculating net debt: {e}")
            return 0

    ####################################
    # (2) 計算 FCF，並做 5 年簡易預測   #
    ####################################
    def get_base_fcf(self):
        """依 (OperatingIncome * (1 - 稅率) + 折舊 - CAPEX) 計算當年基期 FCF"""
        ebit_after_tax = self.operating_income * (1 - self.tax_rate)
        fcf = ebit_after_tax + self.depreciation - self.capex
        return fcf

    def forecast_fcf_list(self):
        """
        如果使用者有手動輸入多年度成長率，就用那個
        否則就自動用 estimate_historical_revenue_growth() or 預設 5%
        """
        base_fcf = self.get_base_fcf()
        if base_fcf is None:
            return []

        # 若使用者有傳入 manual_growth_rates，就用它
        if self.manual_growth_rates is not None and len(self.manual_growth_rates) == self.forecast_years:
            # 逐年應用使用者指定的成長率
            fcf_list = []
            fcf_t = base_fcf
            for g in self.manual_growth_rates:
                fcf_list.append(fcf_t)
                fcf_t = fcf_t * (1 + g)
            return fcf_list
        else:
            # 否則自動計算平均營收成長率 or 預設 5%
            avg_revenue_growth = self.estimate_historical_revenue_growth() 
            if avg_revenue_growth is None:
                avg_revenue_growth = 0.05  # 預設

            print(f"Auto forecast revenue growth = {avg_revenue_growth:.2%}")

            fcf_list = []
            fcf_t = base_fcf
            for i in range(self.forecast_years):
                fcf_list.append(fcf_t)
                fcf_t = fcf_t * (1 + avg_revenue_growth)
            return fcf_list

    def estimate_historical_revenue_growth(self, years=3):
        """
        (可選) 看最近N年的 Revenue，計算平均年化成長率
        """
        if self.income_stmt.empty:
            return None

        # columns 通常從舊到新 or 反之, 先排序
        cols_sorted = sorted(self.income_stmt.columns)
        # 如果 columns 不足 years+1，就無法算
        if len(cols_sorted) < years + 1:
            return None

        # 拿最舊跟最新做近似 (或分段逐年算)
        rev_key = None
        for possible_key in ["Total Revenue", "Revenue"]:
            if possible_key in self.income_stmt.index:
                rev_key = possible_key
                break
        if rev_key is None:
            return None

        older_col = cols_sorted[-(years+1)]  # N+1 年前
        newer_col = cols_sorted[-1]          # 最新年

        older_rev = self.income_stmt.loc[rev_key, older_col]
        newer_rev = self.income_stmt.loc[rev_key, newer_col]
        if older_rev <= 0 or pd.isna(older_rev) or pd.isna(newer_rev):
            return None

        # (newer / older)^(1/years) - 1
        cagr = (newer_rev / older_rev) ** (1 / years) - 1
        return cagr if cagr > -1 else None

    ###########################################
    # (3) 計算最終 DCF 的內在價值 & 股價       #
    ###########################################
    def calculate_intrinsic_value(self):
        """
        1) 取得未來 N 年的 FCF & 貼現
        2) 終值 (terminal value) = FCF_{n+1} / (WACC - g)
        3) 總和得到企業價值(Enterprise Value)
        4) 扣除 Net Debt -> 股東價值 (Equity Value)
        """
        if self.wacc is None:
            print("Error: WACC is None, cannot proceed DCF.")
            return None

        if self.wacc <= self.perpetual_growth_rate:
            print("Error: WACC <= perpetual growth rate, model invalid.")
            return None

        # 1) 預測 FCF
        fcf_list = self.forecast_fcf_list()
        if not fcf_list:
            print("No FCF data to forecast.")
            return None

        # 2) 計算前 N 年的現值
        npv_stage_1 = 0.0
        for i, fcf_t in enumerate(fcf_list, start=1):
            npv_stage_1 += fcf_t / ((1 + self.wacc) ** i)

        # 3) 第 N+1 年 FCF
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
        """
        1) compute intrinsic value
        2) divide by shares outstanding
        """
        equity_value = self.calculate_intrinsic_value()
        if equity_value is None:
            return None
        if not self.shares_outstanding or self.shares_outstanding <= 0:
            print("Error: invalid shares_outstanding.")
            return None

        fair_price = equity_value / self.shares_outstanding
        print(f"Fair Price = {fair_price:.2f} per share")
        return fair_price


##############################################
# (範例) 主程式: 執行DCFModel進行估值
##############################################
if __name__ == "__main__":
    # 讓使用者決定要不要手動輸入 growth rates
    user_input = input("Manual input growth rate? (yes/no): ").strip().lower()
    if user_input == "yes":
        growth_str = input("Enter growth rates (comma-separated), e.g. 0.1,0.08,0.05: ")
        try:
            manual_rates = [float(x.strip()) for x in growth_str.split(",")]
        except:
            print("Invalid input. Use auto forecast instead.")
            manual_rates = None
    else:
        manual_rates = None

    dcf = DCFModel("4763.TW", forecast_years=5, perpetual_growth_rate=0.025, manual_growth_rates=manual_rates)
    estimated_price = dcf.calculate_stock_price()
    print(f"Estimated Price: {estimated_price}")