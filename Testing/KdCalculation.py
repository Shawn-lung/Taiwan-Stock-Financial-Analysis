import pandas as pd
import yfinance as yf


class CostofDebtModel:
    def __init__(self, stock_code):
        self.stock = yf.Ticker(stock_code)
        self.stock_code = stock_code

    def get_valid_annual_data(self, balance_sheet, financials):
        """
        獲取最近有數據的年度，確保 Total Debt 和 Interest Expense 來自同一年
        """
        try:
            total_debt_series = balance_sheet.loc["Total Debt"] if "Total Debt" in balance_sheet.index else None
            interest_expense_series = financials.loc["Interest Expense"] if "Interest Expense" in financials.index else None

            if total_debt_series is None or interest_expense_series is None:
                print(f"Missing 'Total Debt' or 'Interest Expense' data in balance sheet or financials.")
                return None, None

            # 查找最近有數據的一年
            for year in total_debt_series.index:
                total_debt = total_debt_series[year]
                interest_expense = interest_expense_series[year]

                if pd.notna(total_debt) and pd.notna(interest_expense) and total_debt != 0 and interest_expense != 0:
                    print(f"Using data from {year}: Total Debt = {total_debt}, Interest Expense = {interest_expense}")
                    return total_debt, interest_expense

            print(f"No valid data found for both Total Debt and Interest Expense.")
            return None, None

        except Exception as e:
            print(f"Error processing annual data: {e}")
            return None, None

    def get_tax_data(self, financials):
        """
        計算最近有效年度的稅率
        稅率公式: Tax Rate = Tax Provision / Pretax Income
        """
        try:
            tax_provision_series = financials.loc["Tax Provision"] if "Tax Provision" in financials.index else None
            pretax_income_series = financials.loc["Pretax Income"] if "Pretax Income" in financials.index else None

            if tax_provision_series is None or pretax_income_series is None:
                print(f"Missing 'Tax Provision' or 'Pretax Income' data in financials.")
                return None

            # 查找最近有數據的一年
            for year in tax_provision_series.index:
                tax_provision = tax_provision_series[year]
                pretax_income = pretax_income_series[year]

                if pd.notna(tax_provision) and pd.notna(pretax_income) and pretax_income != 0:
                    tax_rate = tax_provision / pretax_income
                    print(f"Using data from {year}: Tax Provision = {tax_provision}, Pretax Income = {pretax_income}, Tax Rate = {tax_rate:.2%}")
                    return tax_rate

            print(f"No valid data found for Tax Provision and Pretax Income.")
            return None

        except Exception as e:
            print(f"Error calculating tax data: {e}")
            return None

    def get_cost_of_debt(self):
        """
        計算稅後 Cost of Debt
        """
        try:
            # 抓取年度數據
            annual_balance_sheet = self.stock.balance_sheet  # 年報資產負債表
            annual_financials = self.stock.financials        # 年報損益表

            # 獲取負債相關數據
            total_debt, interest_expense = self.get_valid_annual_data(annual_balance_sheet, annual_financials)

            # 獲取稅率
            tax_rate = self.get_tax_data(annual_financials)

            if total_debt is not None and interest_expense is not None:
                # 計算稅前 Cost of Debt
                cost_of_debt = interest_expense / total_debt
                print(f"Total Debt (Latest Valid Year): {total_debt}")
                print(f"Interest Expense (Latest Valid Year): {interest_expense}")
                print(f"Cost of Debt (K_D): {cost_of_debt:.2%}")

                # 如果有稅率，計算稅後 Cost of Debt
                if tax_rate is not None:
                    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
                    print(f"Tax Rate: {tax_rate:.2%}")
                    print(f"After-Tax Cost of Debt (K_D): {after_tax_cost_of_debt:.2%}")
                    return after_tax_cost_of_debt
                else:
                    print(f"Missing tax rate. Returning pre-tax cost of debt.")
                    return cost_of_debt
            else:
                print(f"Missing data to calculate cost of debt for {self.stock_code}.")
                return None
        except Exception as e:
            print(f"Error fetching debt data for {self.stock_code}: {e}")
            return None


if __name__ == "__main__":
    cod = CostofDebtModel("2345.TW")  # 替換為目標公司代碼
    cod.get_cost_of_debt()
