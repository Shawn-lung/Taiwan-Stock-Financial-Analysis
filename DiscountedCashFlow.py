import Wacc


class DCFModel:
    def __init__(self, stock_code, growth_rates=None, perpetual_growth_rate=0.025):
        """
        初始化 DCF 模型
        :param stock_code: 股票代号
        :param growth_rates: 前 5 年的成长率列表（如 [0.1, 0.09, 0.08, 0.07, 0.06]）
        :param perpetual_growth_rate: 永续增长率 (默认 2.5%)
        """
        self.stock_code = stock_code
        self.wacc_model = Wacc.WACCModel(stock_code)
        self.perpetual_growth_rate = perpetual_growth_rate  # 永续增长率
        self.total_liability = None  # 初始化总负债
        self.shares_outstanding = self.wacc_model.stock.info.get("sharesOutstanding", None)  # 获取发行股数

        # 如果未提供增长率，默认使用永续增长率
        if growth_rates is None:
            self.growth_rates = [perpetual_growth_rate] * 5
        else:
            self.growth_rates = growth_rates  # 使用提供的增长率列表

    def get_fcf(self):
        """
        使用公式计算自由现金流 (Free Cash Flow, FCF)
        FCF = EBIT * (1 - 税率) + 折旧 - 营运资金变化 - 资本支出
        """
        try:
            # 提取财务报表数据
            income_statement = self.wacc_model.annual_financials
            balance_sheet = self.wacc_model.stock.balance_sheet
            cash_flow_statement = self.wacc_model.stock.cashflow
            print(cash_flow_statement)
            # 提取 EBIT
            ebit = income_statement.loc["EBIT"].iloc[0]
            print(f"EBIT: {ebit:.2f}")

            # 提取税率
            tax_rate = self.wacc_model.get_tax_data()
            if tax_rate is None:
                print("Tax rate data is missing.")
                return None
            print(f"Tax Rate: {tax_rate:.2%}")

            # 提取折旧
            depreciation = cash_flow_statement.loc["Depreciation"].iloc[0]
            print(f"Depreciation: {depreciation:.2f}")

            # 计算营运资金变化 (Net Working Capital Investment)
            current_assets = balance_sheet.loc["Current Assets"]
            current_liabilities = balance_sheet.loc["Current Liabilities"]
            if len(current_assets) > 1 and len(current_liabilities) > 1:
                nwc_investment = (current_assets.iloc[0] - current_liabilities.iloc[0]) - \
                                 (current_assets.iloc[1] - current_liabilities.iloc[1])
            else:
                nwc_investment = 0  # 如果数据缺失，则假设营运资本变化为 0
            print(f"Net Working Capital Investment: {nwc_investment:.2f}")

            # 提取资本支出 (CAPEX)
            capex = cash_flow_statement.loc["Capital Expenditure"].iloc[0]
            print(f"Capital Expenditures (CAPEX): {capex:.2f}")

            # 获取总负债
            self.total_liability = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]

            # 计算自由现金流
            fcf = cash_flow_statement.loc['Free Cash Flow'].iloc[0]
            print(f"Free Cash Flow (FCF): {fcf:.2f}")
            return fcf
        except Exception as e:
            print(f"Error calculating Free Cash Flow (FCF): {e}")
            return None

    def calculate_intrinsic_value(self):
        """
        分阶段 DCF 模型：前 5 年增长 + 永续增长
        """
        try:
            # 获取最新自由现金流
            fcf = self.get_fcf()
            if fcf is None:
                print("FCF data is missing. Cannot calculate intrinsic value.")
                return None

            # 计算 WACC
            wacc = self.wacc_model.calculate_wacc()
            if wacc is None:
                print("WACC is missing. Cannot calculate intrinsic value.")
                return None

            # 确保 WACC > 永续增长率
            if wacc <= self.perpetual_growth_rate:
                print("WACC must be greater than the perpetual growth rate to use the DCF model.")
                return None

            # 第一阶段：计算前 5 年的现值
            npv_stage_1 = 0
            for t in range(1, 6):  # 前 5 年
                growth_rate = self.growth_rates[t - 1]  # 获取每年的增长率
                fcf_t = fcf * ((1 + growth_rate) ** t)  # FCF 按增长率增长
                discounted_fcf = fcf_t / ((1 + wacc) ** t)  # 折现
                npv_stage_1 += discounted_fcf
                print(f"Year {t}: Growth Rate = {growth_rate:.2%}, FCF = {fcf_t:.2f}, Discounted FCF = {discounted_fcf:.2f}")

            # 第二阶段：计算终值 (Terminal Value)
            fcf_6 = fcf * ((1 + self.growth_rates[-1]) ** 5) * (1 + self.perpetual_growth_rate)
            terminal_value = fcf_6 / (wacc - self.perpetual_growth_rate)
            discounted_terminal_value = terminal_value / ((1 + wacc) ** 5)  # 第 5 年末折现
            print(f"Terminal Value: {terminal_value:.2f}, Discounted Terminal Value: {discounted_terminal_value:.2f}")

            # 合并阶段
            intrinsic_value = npv_stage_1 + discounted_terminal_value
            print(f"Intrinsic Value of {self.stock_code}: {intrinsic_value:.2f}")
            return intrinsic_value
        except Exception as e:
            print(f"Error calculating intrinsic value: {e}")
            return None

    def calculate_stock_price(self):
        """
        根据公司价值和股权计算每股价格
        """
        try:
            intrinsic_value = self.calculate_intrinsic_value()
            if intrinsic_value is None or self.total_liability is None or self.shares_outstanding is None:
                print("Missing data to calculate stock price.")
                return None

            equity_value = intrinsic_value - self.total_liability
            stock_price = equity_value / self.shares_outstanding
            print(f"Calculated Stock Price: {stock_price:.2f}")
            return stock_price
        except Exception as e:
            print(f"Error calculating stock price: {e}")
            return None


if __name__ == "__main__":
    # 提供每年增长率 [0.1, 0.09, 0.08, 0.07, 0.06]，若不提供则默认使用永续增长率
    growth_rates = [0.2, 0.15, 0.1, 0.1, 0.08]
    dcf = DCFModel("2330.TW", growth_rates=growth_rates, perpetual_growth_rate=0.025)
    dcf.calculate_stock_price()
