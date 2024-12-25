import Wacc


class DCFModel:
    def __init__(self, stock_code, perpetual_growth_rate=0.025):
        """
        初始化 DCF 模型
        :param stock_code: 股票代号
        :param perpetual_growth_rate: 永续增长率 (默认 2.5%)
        """
        self.stock_code = stock_code
        self.wacc_model = Wacc.WACCModel(stock_code)
        self.perpetual_growth_rate = perpetual_growth_rate  # 永续增长率

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
            print(cash_flow_statement.index)
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
            nwc_investment = current_assets.iloc[0] - current_liabilities[0] - current_assets.iloc[1] + current_liabilities[1]
            print(f"Net Working Capital Investment: {nwc_investment:.2f}")

            # 提取资本支出 (CAPEX)
            capex = balance_sheet.loc["Gross PPE"].iloc[0] - balance_sheet.loc["Gross PPE"].iloc[1]
            print(f"Capital Expenditures (CAPEX): {capex:.2f}")

            # 计算自由现金流
            fcf = ebit * (1 - tax_rate) + depreciation - nwc_investment - capex
            self.total_liability = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
            print(f"Free Cash Flow (FCF): {fcf:.2f}")
            return fcf
        except Exception as e:
            print(f"Error calculating Free Cash Flow (FCF): {e}")
            return None

    def calculate_intrinsic_value(self):
        """
        使用永续增长法计算公司的内在价值
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

            # 计算公司的内在价值
            intrinsic_value = fcf * (1 + self.perpetual_growth_rate) / (wacc - self.perpetual_growth_rate)
            print(f"Intrinsic Value of {self.stock_code}: {intrinsic_value:.2f}")
            return intrinsic_value
        except Exception as e:
            print(f"Error calculating intrinsic value: {e}")
            return None
    def calculate_stock_price(self):
        intrinsic_value = self.calculate_intrinsic_value()
        stock_price = (intrinsic_value-self.total_liability)/self.wacc_model.share_outstanding
        return stock_price

if __name__ == "__main__":
    dcf = DCFModel("2330.TW")  # 使用 8044.TWO 作为测试股票代码
    print(dcf.calculate_stock_price())
