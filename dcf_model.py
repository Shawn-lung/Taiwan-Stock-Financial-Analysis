# dcf_model.py
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import Wacc  # Make sure you have a Wacc module with WACCModel implemented

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DCFModel:
    """
    A Discounted Cash Flow (DCF) Model for intrinsic stock valuation.

    Attributes:
        stock_code (str): Ticker symbol (e.g., "2330.TW").
        forecast_years (int): Number of years to forecast.
        perpetual_growth_rate (float): Growth rate used for terminal value calculation.
        manual_growth_rates (list): Annual revenue growth rates.
        manual_capex_factors (list): Annual CAPEX adjustment factors.
        manual_wc_factors (list): Annual working capital adjustment factors.
        manual_depr_factors (list): Annual depreciation adjustment factors.
        manual_opincome_factors (list): Annual operating income adjustment factors.
        manual_tax_factors (list): Annual tax rate adjustment factors.
    """

    def __init__(
        self,
        stock_code,
        forecast_years=5,
        perpetual_growth_rate=0.025,
        manual_growth_rates=None,
        manual_capex_factors=None,
        manual_wc_factors=None,
        manual_depr_factors=None,
        manual_opincome_factors=None,
        manual_tax_factors=None
    ):
        self.stock_code = stock_code
        self.forecast_years = forecast_years
        self.perpetual_growth_rate = perpetual_growth_rate
        self.manual_growth_rates = manual_growth_rates
        self.manual_capex_factors = manual_capex_factors
        self.manual_wc_factors = manual_wc_factors
        self.manual_depr_factors = manual_depr_factors
        self.manual_opincome_factors = manual_opincome_factors
        self.manual_tax_factors = manual_tax_factors

        # Initialize financial data placeholders
        self.stock = None
        self.income_stmt = None
        self.cash_flow = None
        self.balance_sheet = None
        self.latest_year_income = None
        self.latest_year_cf = None
        self.latest_year_bs = None

        # Base metrics
        self.current_revenue = 0.0
        self.operating_income = 0.0
        self.depreciation = 0.0
        self.capex = 0.0
        self.tax_rate = 0.20
        self.net_debt = 0.0
        self.current_working_capital = 0.0
        self.shares_outstanding = 1

        self.wacc = None
        self.base_year_metrics = {}
        self.hist_metrics = {}

        # Initialize financial data and compute base values
        self.initialize_model()

    def initialize_model(self):
        """Fetch financial data from yfinance and initialize base metrics."""
        try:
            self.stock = yf.Ticker(self.stock_code)
            self.income_stmt = self.stock.financials
            self.cash_flow = self.stock.cashflow
            self.balance_sheet = self.stock.balance_sheet

            # Sort columns by date and select the latest data
            if not self.income_stmt.empty:
                self.income_stmt = self.income_stmt.sort_index(axis=1, ascending=True)
                self.latest_year_income = self.income_stmt.columns[-1]
            if not self.cash_flow.empty:
                self.cash_flow = self.cash_flow.sort_index(axis=1, ascending=True)
                self.latest_year_cf = self.cash_flow.columns[-1]
            if not self.balance_sheet.empty:
                self.balance_sheet = self.balance_sheet.sort_index(axis=1, ascending=True)
                self.latest_year_bs = self.balance_sheet.columns[-1]

            # Get base values from the latest financial statements
            self.current_revenue = self.get_latest_revenue()
            self.operating_income = self.get_latest_operating_income()
            self.depreciation = self.get_latest_depreciation()
            self.capex = self.get_latest_capex()
            self.tax_rate = self.get_tax_rate_estimate()
            self.net_debt = self.get_net_debt()
            self.current_working_capital = self.get_latest_working_capital()

            # Get shares outstanding
            so = self.stock.info.get("sharesOutstanding", 1)
            self.shares_outstanding = so if so and so > 0 else 1

            # Try to calculate WACC using the external Wacc module
            try:
                self.wacc = Wacc.WACCModel(self.stock_code).calculate_wacc()
                if self.wacc is None or self.wacc <= 0:
                    # Fallback to a default WACC estimate if calculation fails
                    logger.warning(f"WACC calculation failed for {self.stock_code}. Using default WACC.")
                    self.wacc = self.estimate_default_wacc()
            except Exception as e:
                logger.warning(f"Error calculating WACC: {e}. Using default WACC.")
                self.wacc = self.estimate_default_wacc()

            # Store base year metrics for potential anomaly detection
            self.base_year_metrics = {
                "Revenue": self.current_revenue,
                "OperatingIncome": self.operating_income,
                "Depreciation": self.depreciation,
                "CAPEX": self.capex,
                "WorkingCapital": self.current_working_capital
            }
            self.hist_metrics = self.prepare_historical_metrics()

            logger.info("Initialization complete for stock: %s", self.stock_code)
        except Exception as e:
            logger.error("Error during initialization: %s", e)

    def estimate_default_wacc(self):
        """Estimate a default WACC when the calculation fails."""
        try:
            # Get country code from ticker to determine default rates
            country_code = self.stock_code.split('.')[-1] if '.' in self.stock_code else 'US'
            
            # Default WACC estimates by country/region
            default_rates = {
                'TW': 0.085,   # Taiwan
                'TWO': 0.085,  # Taiwan OTC
                'HK': 0.09,    # Hong Kong
                'CN': 0.10,    # China
                'US': 0.08,    # United States
                'UK': 0.075,   # United Kingdom
                'JP': 0.07,    # Japan
                'KR': 0.09,    # South Korea
                'SG': 0.08,    # Singapore
            }
            
            # Get base rate from country, or use 8.5% as ultimate default
            base_rate = default_rates.get(country_code, 0.085)
            
            # Adjust based on industry if available (simplified)
            try:
                industry = self.stock.info.get('industry', '')
                industry_adjustments = {
                    'Technology': 0.01,       # Higher for tech
                    'Software': 0.015,        # Higher for software
                    'Healthcare': 0.005,      # Slightly higher for healthcare
                    'Utilities': -0.02,       # Lower for utilities
                    'Energy': 0.01,           # Higher for energy
                    'Consumer Defensive': -0.01   # Lower for consumer defensive
                }
                adjustment = 0
                for ind_keyword, adj in industry_adjustments.items():
                    if ind_keyword.lower() in industry.lower():
                        adjustment += adj
                
                wacc = base_rate + adjustment
                logger.info(f"Using default WACC of {wacc:.2%} for {self.stock_code} (base: {base_rate:.2%}, industry adj: {adjustment:.2%})")
                return max(wacc, self.perpetual_growth_rate + 0.03)  # Ensure minimum spread
                
            except:
                logger.info(f"Using default WACC of {base_rate:.2%} for {self.stock_code}")
                return max(base_rate, self.perpetual_growth_rate + 0.03)  # Ensure minimum spread
                
        except Exception as e:
            logger.error(f"Error in default WACC estimation: {e}")
            # Absolute fallback - 8.5% WACC
            return max(0.085, self.perpetual_growth_rate + 0.03)

    def get_latest_revenue(self):
        """Extract the latest revenue value from the income statement."""
        if self.latest_year_income is None:
            return 0.0
        try:
            for key in ["Total Revenue", "Revenue"]:
                if key in self.income_stmt.index:
                    val = self.income_stmt.loc[key, self.latest_year_income]
                    if pd.notna(val):
                        return float(val)
            return 0.0
        except Exception as e:
            logger.error("Error fetching latest revenue: %s", e)
            return 0.0

    def get_latest_operating_income(self):
        """Extract the latest operating income from the income statement."""
        if self.latest_year_income is None:
            return 0.0
        try:
            if "Operating Income" in self.income_stmt.index:
                val = self.income_stmt.loc["Operating Income", self.latest_year_income]
                return float(val) if pd.notna(val) else 0.0
            return 0.0
        except Exception as e:
            logger.error("Error fetching operating income: %s", e)
            return 0.0

    def get_latest_depreciation(self):
        """Extract the latest depreciation value from the cash flow statement."""
        if self.latest_year_cf is None:
            return 0.0
        try:
            for key in ["Depreciation", "Depreciation & Amortization"]:
                if key in self.cash_flow.index:
                    val = self.cash_flow.loc[key, self.latest_year_cf]
                    if pd.notna(val):
                        return abs(float(val))
            return 0.0
        except Exception as e:
            logger.error("Error fetching depreciation: %s", e)
            return 0.0

    def get_latest_capex(self):
        """Extract the latest CAPEX from the cash flow statement."""
        if self.latest_year_cf is None:
            return 0.0
        try:
            if "Capital Expenditure" in self.cash_flow.index:
                val = self.cash_flow.loc["Capital Expenditure", self.latest_year_cf]
                if pd.notna(val):
                    return abs(float(val))
            return 0.0
        except Exception as e:
            logger.error("Error fetching CAPEX: %s", e)
            return 0.0

    def get_tax_rate_estimate(self):
        """Estimate the tax rate based on income statement data."""
        if self.latest_year_income is None:
            return 0.20
        try:
            tax_keys = ["Income Tax Expense", "Tax Provision"]
            pretax_keys = ["Income Before Tax", "Pretax Income"]
            tax_val, pretax_val = None, None
            for tk in tax_keys:
                if tk in self.income_stmt.index:
                    val = self.income_stmt.loc[tk, self.latest_year_income]
                    if pd.notna(val):
                        tax_val = float(val)
                        break
            for pk in pretax_keys:
                if pk in self.income_stmt.index:
                    val = self.income_stmt.loc[pk, self.latest_year_income]
                    if pd.notna(val):
                        pretax_val = float(val)
                        break
            if tax_val is not None and pretax_val and pretax_val != 0:
                rate = abs(tax_val / pretax_val)
                return rate if 0 < rate < 0.5 else 0.20
            return 0.20
        except Exception as e:
            logger.error("Error estimating tax rate: %s", e)
            return 0.20

    def get_net_debt(self):
        """Calculate net debt from the balance sheet."""
        if self.latest_year_bs is None:
            return 0.0
        try:
            total_debt = 0.0
            cash_val = 0.0
            
            # Get total debt - ensure positive value regardless of accounting convention
            if "Total Debt" in self.balance_sheet.index:
                td = self.balance_sheet.loc["Total Debt", self.latest_year_bs]
                if pd.notna(td):
                    total_debt = abs(float(td))  # Use absolute value to ensure positive
                    logger.info(f"Found Total Debt: {total_debt:,.0f}")
            
            # If no direct total debt, try to find long-term debt
            if total_debt == 0:
                for debt_key in ["Long Term Debt", "LongTermDebt", "LongTermBorrowings"]:
                    if debt_key in self.balance_sheet.index:
                        ltd = self.balance_sheet.loc[debt_key, self.latest_year_bs]
                        if pd.notna(ltd):
                            total_debt += abs(float(ltd))
                            logger.info(f"Found Long Term Debt: {abs(float(ltd)):,.0f}")
                            break
            
            # Add short-term debt/current portion if available
            for st_key in ["Short Term Debt", "CurrentPortionLongTermDebt", "CurrentDebt"]:
                if st_key in self.balance_sheet.index:
                    std = self.balance_sheet.loc[st_key, self.latest_year_bs]
                    if pd.notna(std):
                        total_debt += abs(float(std))
                        logger.info(f"Found Short Term Debt: {abs(float(std)):,.0f}")
            
            # Get cash and cash equivalents
            for key in ["Cash And Cash Equivalents", "Cash", "Cash Cash Equivalents And Short Term Investments"]:
                if key in self.balance_sheet.index:
                    c_val = self.balance_sheet.loc[key, self.latest_year_bs]
                    if pd.notna(c_val):
                        cash_val = abs(float(c_val))  # Use absolute value to ensure positive
                        logger.info(f"Found Cash: {cash_val:,.0f}")
                        break
            
            # Net debt calculation and handling of negative net debt
            net_debt = total_debt - cash_val
            logger.info(f"Calculated Net Debt: {net_debt:,.0f} (Total Debt: {total_debt:,.0f} - Cash: {cash_val:,.0f})")
            
            # Store cash separately for enterprise to equity value calculation
            self.cash = cash_val  # Initialize self.cash attribute here
            self.total_debt = total_debt  # Also store total_debt for potential future use
            
            return net_debt
        except Exception as e:
            logger.error(f"Error calculating net debt: {e}")
            self.cash = 0.0  # Ensure cash is always initialized
            self.total_debt = 0.0
            return 0.0

    def get_latest_working_capital(self):
        """Calculate the latest working capital from the balance sheet."""
        if self.latest_year_bs is None:
            return 0.0
        try:
            current_assets = 0.0
            current_liabilities = 0.0
            if "Current Assets" in self.balance_sheet.index:
                ca = self.balance_sheet.loc["Current Assets", self.latest_year_bs]
                if pd.notna(ca):
                    current_assets = float(ca)
            if "Current Liabilities" in self.balance_sheet.index:
                cl = self.balance_sheet.loc["Current Liabilities", self.latest_year_bs]
                if pd.notna(cl):
                    current_liabilities = float(cl)
            return current_assets - current_liabilities
        except Exception as e:
            logger.error("Error calculating working capital: %s", e)
            return 0.0

    def prepare_historical_metrics(self):
        """Prepare historical financial metrics for analysis."""
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
        past_cols = cols[:-1] if len(cols) > 1 else []
        for col in past_cols:
            rev = 0.0
            for key in ["Total Revenue", "Revenue"]:
                if key in self.income_stmt.index:
                    tmp = self.income_stmt.loc[key, col]
                    if pd.notna(tmp):
                        rev = float(tmp)
                        break
            op = 0.0
            if "Operating Income" in self.income_stmt.index:
                tmpop = self.income_stmt.loc["Operating Income", col]
                if pd.notna(tmpop):
                    op = float(tmpop)
            dep = 0.0
            if not self.cash_flow.empty and col in self.cash_flow.columns:
                for key in ["Depreciation", "Depreciation & Amortization"]:
                    if key in self.cash_flow.index:
                        d_val = self.cash_flow.loc[key, col]
                        if pd.notna(d_val):
                            dep = abs(float(d_val))
                            break
            cap = 0.0
            if not self.cash_flow.empty and col in self.cash_flow.columns:
                if "Capital Expenditure" in self.cash_flow.index:
                    c_val = self.cash_flow.loc["Capital Expenditure", col]
                    if pd.notna(c_val):
                        cap = abs(float(c_val))
            wc = 0.0
            if not self.balance_sheet.empty and col in self.balance_sheet.columns:
                ca = 0.0
                cl = 0.0
                if "Current Assets" in self.balance_sheet.index:
                    ca_val = self.balance_sheet.loc["Current Assets", col]
                    if pd.notna(ca_val):
                        ca = float(ca_val)
                if "Current Liabilities" in self.balance_sheet.index:
                    cl_val = self.balance_sheet.loc["Current Liabilities", col]
                    if pd.notna(cl_val):
                        cl = float(cl_val)
                wc = ca - cl
            result["Revenue"].append(rev)
            result["OperatingIncome"].append(op)
            result["Depreciation"].append(dep)
            result["CAPEX"].append(cap)
            result["WorkingCapital"].append(wc)
        return result

    def estimate_historical_revenue_growth(self, years=3):
        """Estimate the Compound Annual Growth Rate (CAGR) for revenue over a specified period."""
        if self.income_stmt.empty:
            return None
        cols_sorted = sorted(self.income_stmt.columns)
        if len(cols_sorted) < years + 1:
            return None
        rev_key = None
        for key in ["Total Revenue", "Revenue"]:
            if key in self.income_stmt.index:
                rev_key = key
                break
        if not rev_key:
            return None
        older_col = cols_sorted[-(years + 1)]
        newer_col = cols_sorted[-1]
        older_rev = self.income_stmt.loc[rev_key, older_col]
        newer_rev = self.income_stmt.loc[rev_key, newer_col]
        if pd.isna(older_rev) or pd.isna(newer_rev) or older_rev <= 0:
            return None
        cagr = (newer_rev / older_rev) ** (1 / years) - 1
        return cagr if cagr >= -1 else None

    def forecast_fcf_list(self):
        """Forecast Free Cash Flow with ratio-based predictions."""
        try:
            fcf_list = []
            revenue = float(self.current_revenue)
            operating_margin = self.operating_income / revenue if self.operating_income and revenue else 0.15
            
            # Calculate base values and ratios
            base_capex_ratio = self.capex / revenue if self.capex and revenue else 0.1
            base_depr_ratio = self.depreciation / revenue if self.depreciation and revenue else 0.05
            base_wc_ratio = self.current_working_capital / revenue if self.current_working_capital and revenue else 0.05
            
            # Log base metrics
            logger.info(f"Base metrics for {self.stock_code}:")
            logger.info(f"Current Revenue: {revenue:,.0f}")
            logger.info(f"Operating Margin: {operating_margin:.2%}")
            logger.info(f"Base CAPEX/Revenue: {base_capex_ratio:.2%}")
            logger.info(f"Base Depreciation/Revenue: {base_depr_ratio:.2%}")
            logger.info(f"Base WC/Revenue: {base_wc_ratio:.2%}")
            
            # Get growth rates and factors (or use defaults)
            growth_rates = self.manual_growth_rates or [0.05] * self.forecast_years
            capex_ratios = self.manual_capex_factors or [base_capex_ratio] * self.forecast_years
            depr_ratios = self.manual_depr_factors or [base_depr_ratio] * self.forecast_years
            wc_ratios = self.manual_wc_factors or [base_wc_ratio] * self.forecast_years
            tax_rates = self.manual_tax_factors or [self.tax_rate] * self.forecast_years
            
            # Ensure we have opinion factors (or create empty list)
            op_factors = self.manual_opincome_factors or [0] * self.forecast_years
            
            # Keep track of working capital for change calculation
            current_wc = self.current_working_capital
            
            # Apply sanity checks to ratios
            # Cap CAPEX ratios at reasonable levels
            capex_ratios = [min(ratio, 0.25) for ratio in capex_ratios]
            # Ensure WC ratios are reasonable
            wc_ratios = [min(max(ratio, -0.05), 0.15) for ratio in wc_ratios]
            
            # Gradually reduce growth rates for long-term sustainability
            for i in range(1, len(growth_rates)):
                if growth_rates[i] > 0.2 and i >= 2:
                    growth_rates[i] = growth_rates[i] * 0.85  # Dampen high growth rates in later years
            
            # First calculate revenues for all years upfront
            revenues = [revenue]
            for i in range(self.forecast_years):
                revenues.append(revenues[-1] * (1 + growth_rates[i]))
            
            # Calculate a more gradual working capital transition
            # This prevents massive WC changes in year 1
            target_wc_percent = wc_ratios[0]
            current_wc_percent = current_wc / revenue
            
            # If large WC adjustment needed, spread it over multiple years
            wc_adjustment_years = 3 if abs(current_wc_percent - target_wc_percent) > 0.1 else 1
            
            for i in range(self.forecast_years):
                # Apply growth to revenue
                new_revenue = revenues[i+1]
                
                # Use factors as direct ratios
                capex_ratio = capex_ratios[i]
                depr_ratio = depr_ratios[i]
                tax_rate = tax_rates[i]
                
                # Calculate working capital with gradual transition
                if i < wc_adjustment_years:
                    # Gradually move toward target WC
                    adjusted_wc_percent = current_wc_percent - (i+1) * ((current_wc_percent - target_wc_percent) / wc_adjustment_years)
                else:
                    adjusted_wc_percent = wc_ratios[i]
                
                # Calculate absolute values
                capex = new_revenue * capex_ratio
                depr = new_revenue * depr_ratio
                new_wc = new_revenue * adjusted_wc_percent
                delta_wc = new_wc - current_wc  # Change in working capital
                
                # Apply factors to operating income
                new_margin = operating_margin * (1 + (op_factors[i] if i < len(op_factors) else 0))
                op_income = new_revenue * new_margin
                
                # Calculate FCF
                nopat = op_income * (1 - tax_rate)
                fcf = nopat + depr - capex - delta_wc
                
                # Log detailed projections
                logger.info(f"Year {i+1} Projections:")
                logger.info(f"Revenue: {new_revenue:,.0f} (Growth: {growth_rates[i]:.2%})")
                logger.info(f"CAPEX: {capex:,.0f} (Ratio: {capex_ratio:.2%})")
                logger.info(f"Depreciation: {depr:,.0f} (Ratio: {depr_ratio:.2%})")
                logger.info(f"Working Capital: {new_wc:,.0f} (Ratio: {adjusted_wc_percent:.2%}, Change: {delta_wc:,.0f})")
                logger.info(f"Operating Income: {op_income:,.0f} (Margin: {new_margin:.2%})")
                logger.info(f"NOPAT: {nopat:,.0f} (Tax Rate: {tax_rate:.2%})")
                logger.info(f"FCF: {fcf:,.0f} (FCF Margin: {fcf/new_revenue:.2%})")
                
                fcf_list.append(float(fcf))
                
                # Update for next iteration
                revenue = new_revenue
                current_wc = new_wc
                
                # If FCF is negative, apply sanity check for next year
                if fcf < 0 and i < self.forecast_years - 1:
                    # Apply correction - reduce capex ratio for next year
                    if i+1 < len(capex_ratios):
                        capex_ratios[i+1] = max(depr_ratios[i+1], capex_ratios[i+1] * 0.9)
            
            return fcf_list
        
        except Exception as e:
            logger.error(f"Error in FCF forecast: {e}", exc_info=True)
            # Return a simple estimate as fallback
            if self.operating_income > 0:
                return [self.operating_income * 0.75] * self.forecast_years
            return [1000000] * self.forecast_years  # Generic fallback

    def calculate_intrinsic_value(self):
        """Calculate intrinsic value with improved error handling and sanity checks."""
        try:
            if self.wacc is None:
                # If WACC is still None after initialization, use default
                logger.warning("WACC is None. Using default WACC for valuation.")
                self.wacc = self.estimate_default_wacc()
                
            if self.wacc <= self.perpetual_growth_rate:
                # Force minimum spread if WACC is too close to growth rate
                self.wacc = max(self.wacc, self.perpetual_growth_rate + 0.03)
                logger.info(f"Adjusted WACC to {self.wacc:.2%} to maintain spread from growth rate")

            fcf_list = self.forecast_fcf_list()
            if not fcf_list:
                return None
                
            # Handle negative FCFs - either ignore them or cap them at 0 for terminal value purposes
            positive_fcfs = [max(0, fcf) for fcf in fcf_list if fcf > 0]
            if not positive_fcfs:
                logger.warning("No positive FCFs in forecast period. Using operating income as base.")
                if self.operating_income > 0:
                    positive_fcfs = [self.operating_income * 0.5]
                else:
                    return 0  # Can't reasonably value with no positive cashflows

            # Calculate present value of FCFs
            npv_stage_1 = sum(
                fcf / (1 + self.wacc) ** (i+1)
                for i, fcf in enumerate(fcf_list)
            )

            # Calculate terminal value using most reliable forecast as base
            # For terminal value calculation, use the average of positive FCFs or last year's FCF if it's positive
            if fcf_list[-1] > 0:
                normalized_fcf = fcf_list[-1]  # Use last year if positive
            elif fcf_list[-2] > 0:
                normalized_fcf = fcf_list[-2]  # Use second-to-last year if positive
            elif positive_fcfs:
                normalized_fcf = sum(positive_fcfs) / len(positive_fcfs)  # Average of positive FCFs
            else:
                # Fall back to operating income-based estimate if no good FCF values
                normalized_fcf = self.operating_income * 0.7
                logger.warning(f"Using operating income-based FCF estimate for terminal value: {normalized_fcf:,.0f}")

            # Ensure terminal growth rate is appropriate for high-growth companies
            # For companies with high historical growth, use a higher terminal rate
            terminal_growth = self.perpetual_growth_rate
            
            # Ensure minimum spread between WACC and terminal growth
            if self.wacc - terminal_growth < 0.03:
                terminal_growth = self.wacc - 0.03
                logger.info(f"Adjusted terminal growth to {terminal_growth:.2%} to maintain spread from WACC")
            
            terminal_value = normalized_fcf * (1 + terminal_growth) / (self.wacc - terminal_growth)
            
            # For high growth companies, use a higher terminal multiple cap
            # Get estimated current year's EBITDA
            ebitda = self.operating_income + self.depreciation
            
            # Adjust multiple cap based on growth profile
            high_growth_threshold = 0.15  # Consider high growth if first year growth > 15%
            
            # Determine appropriate cap for terminal value multiple
            if fcf_list and len(fcf_list) > 0 and normalized_fcf > 0:
                implied_tv_multiple = terminal_value / normalized_fcf
                
                # Use higher multiple cap for high growth companies
                if self.manual_growth_rates and self.manual_growth_rates[0] > high_growth_threshold:
                    max_multiple = 30  # Higher cap for high growth companies
                else:
                    max_multiple = 20  # Standard cap
                
                if implied_tv_multiple > max_multiple:
                    terminal_value = normalized_fcf * max_multiple
                    logger.info(f"Capped terminal value multiple from {implied_tv_multiple:.1f}x to {max_multiple:.1f}x")
            
            # Discount terminal value
            discounted_tv = terminal_value / (1 + self.wacc) ** len(fcf_list)

            enterprise_value = npv_stage_1 + discounted_tv
            
            # Apply reasonable EV/EBITDA multiple check
            if self.operating_income > 0:
                ebitda = self.operating_income + self.depreciation
                implied_ev_ebitda = enterprise_value / ebitda
                
                # Use different caps based on growth profile
                if self.manual_growth_rates and self.manual_growth_rates[0] > high_growth_threshold:
                    max_ev_multiple = 35  # Higher cap for high growth companies
                else:
                    max_ev_multiple = 25  # Standard cap
                    
                if implied_ev_ebitda > max_ev_multiple:
                    enterprise_value = ebitda * max_ev_multiple
                    logger.info(f"Capped EV/EBITDA from {implied_ev_ebitda:.1f}x to {max_ev_multiple:.1f}x")
                
                # Adjust floor for high-growth companies
                min_ev_multiple = 8 if (self.manual_growth_rates and self.manual_growth_rates[0] > high_growth_threshold) else 5
                if implied_ev_ebitda < 0 or implied_ev_ebitda < min_ev_multiple:
                    enterprise_value = ebitda * min_ev_multiple
                    logger.info(f"Applied floor EV/EBITDA of {min_ev_multiple:.1f}x (was {implied_ev_ebitda:.1f}x)")

            # Calculate equity value considering net debt correctly
            # If net_debt is negative (more cash than debt), it will add to enterprise value
            equity_value = enterprise_value - self.net_debt
            
            # Log calculations with better explanations
            logger.info(f"Base Revenue: {self.current_revenue:,.2f}")
            logger.info(f"NPV of FCF: {npv_stage_1:,.2f}")
            logger.info(f"Normalized FCF for Terminal Value: {normalized_fcf:,.2f}")
            logger.info(f"Terminal Value Multiple: {implied_tv_multiple:.1f}x")
            if self.operating_income > 0:
                logger.info(f"EV/EBITDA Multiple: {(enterprise_value/ebitda):.1f}x")
            logger.info(f"Enterprise Value: {enterprise_value:,.2f}")
            logger.info(f"Net Debt: {self.net_debt:,.2f}")
            logger.info(f"Cash: {self.cash:,.2f}")
            logger.info(f"Equity Value: {equity_value:,.2f} (EV - Net Debt)")
            
            return equity_value

        except Exception as e:
            logger.error(f"Error calculating intrinsic value: {e}", exc_info=True)
            return None

    def calculate_stock_price(self):
        """
        Calculate the fair stock price by dividing the intrinsic equity value by the number of shares outstanding.
        """
        eq_value = self.calculate_intrinsic_value()
        if eq_value is None:
            return None
        if self.shares_outstanding <= 0:
            logger.error("Invalid shares outstanding.")
            return None
        fair_price = eq_value / self.shares_outstanding
        logger.info("Calculated Fair Price: %.2f per share", fair_price)
        return fair_price

    def perform_sensitivity_analysis(self, wacc_range=0.02, growth_range=0.01):
        """
        Perform sensitivity analysis on WACC and perpetual growth rate.
        
        Args:
            wacc_range (float): Range to vary WACC (+/-)
            growth_range (float): Range to vary perpetual growth rate (+/-)
        
        Returns:
            dict: Matrix of stock prices under different scenarios
        """
        base_wacc = self.wacc
        base_growth = self.perpetual_growth_rate
        
        wacc_values = np.linspace(base_wacc - wacc_range, base_wacc + wacc_range, 5)
        growth_values = np.linspace(base_growth - growth_range, base_growth + growth_range, 5)
        
        sensitivity_matrix = {}
        
        for w in wacc_values:
            row = {}
            self.wacc = w
            for g in growth_values:
                self.perpetual_growth_rate = g
                price = self.calculate_stock_price()
                row[f"{g:.1%}"] = price if price is not None else 0
            sensitivity_matrix[f"{w:.1%}"] = row
        
        self.wacc = base_wacc
        self.perpetual_growth_rate = base_growth
        
        return sensitivity_matrix

    def perform_comprehensive_sensitivity(self, variation=0.2):
        """
        Perform comprehensive sensitivity analysis on all major inputs.
        
        Args:
            variation (float): Percentage variation to test (+/-)
        
        Returns:
            dict: Impact of each factor on stock price
        """
        base_price = self.calculate_stock_price()
        if base_price is None:
            return None
            
        results = {}
        
        # Test revenue growth sensitivity
        original_growth = self.manual_growth_rates[:]
        high_growth = [g * (1 + variation) for g in original_growth]
        low_growth = [g * (1 - variation) for g in original_growth]
        
        self.manual_growth_rates = high_growth
        high_price = self.calculate_stock_price()
        self.manual_growth_rates = low_growth
        low_price = self.calculate_stock_price()
        self.manual_growth_rates = original_growth
        
        results['Revenue Growth'] = {
            'high': (high_price - base_price) / base_price,
            'low': (low_price - base_price) / base_price
        }
        
        # Test operating margins through operating income factors
        if self.operating_income != 0:
            base_op = self.operating_income
            self.operating_income = base_op * (1 + variation)
            high_price = self.calculate_stock_price()
            self.operating_income = base_op * (1 - variation)
            low_price = self.calculate_stock_price()
            self.operating_income = base_op
            
            results['Operating Margin'] = {
                'high': (high_price - base_price) / base_price,
                'low': (low_price - base_price) / base_price
            }
        
        # Test CAPEX sensitivity
        original_capex = self.manual_capex_factors[:]
        high_capex = [c * (1 + variation) for c in original_capex]
        low_capex = [c * (1 - variation) for c in original_capex]
        
        self.manual_capex_factors = high_capex
        high_price = self.calculate_stock_price()
        self.manual_capex_factors = low_capex
        low_price = self.calculate_stock_price()
        self.manual_capex_factors = original_capex
        
        results['CAPEX'] = {
            'high': (high_price - base_price) / base_price,
            'low': (low_price - base_price) / base_price
        }
        
        # Test Working Capital sensitivity
        original_wc = self.manual_wc_factors[:]
        high_wc = [w * (1 + variation) for w in original_wc]
        low_wc = [w * (1 - variation) for w in original_wc]
        
        self.manual_wc_factors = high_wc
        high_price = self.calculate_stock_price()
        self.manual_wc_factors = low_wc
        low_price = self.calculate_stock_price()
        self.manual_wc_factors = original_wc
        
        results['Working Capital'] = {
            'high': (high_price - base_price) / base_price,
            'low': (low_price - base_price) / base_price
        }

        return results

    def get_financial_data(self):
        """Get financial data for the stock that can be used by other models.
        
        Returns:
            dict: Dictionary containing key financial metrics
        """
        try:
            # Compile financial data into a dictionary
            financial_data = {
                'stock_code': self.stock_code,
                'revenue': self.current_revenue,
                'operating_income': self.operating_income,
                'depreciation': self.depreciation,
                'capex': self.capex,
                'tax_rate': self.tax_rate,
                'net_debt': self.net_debt,
                'working_capital': self.current_working_capital,
                'cash': getattr(self, 'cash', 0.0),  # Safely get cash attribute
                'total_debt': getattr(self, 'total_debt', 0.0),  # Safely get total_debt attribute
                'wacc': self.wacc
            }
            
            # Add historical financial data if available
            if hasattr(self, 'hist_metrics') and self.hist_metrics:
                for key, values in self.hist_metrics.items():
                    if values:  # Only include non-empty lists
                        financial_data[f'historical_{key.lower()}'] = values
            
            # Add stock market data if available
            if self.stock and hasattr(self.stock, 'info'):
                info = self.stock.info
                
                # Extract key market data
                for field in ['marketCap', 'beta', 'dividendYield', 'forwardPE', 'trailingPE']:
                    if field in info and info[field] is not None:
                        financial_data[field] = info[field]
            
            return financial_data
        except Exception as e:
            logger.error(f"Error getting financial data: {e}")
            # Return minimal data when error occurs
            return {
                'stock_code': self.stock_code,
                'revenue': self.current_revenue or 0.0,
                'operating_income': self.operating_income or 0.0
            }

if __name__ == "__main__":
    dcf = DCFModel(
        "2330.TW",
        forecast_years=5,
        perpetual_growth_rate=0.025,
        manual_growth_rates=[0.2, 0.1, 0.1, 0.05, 0.05],
        manual_capex_factors=[0.2, 0.1, 0, -0.05, -0.05],
        manual_wc_factors=[0.1, 0.05, 0, 0, 0],
        manual_depr_factors=[0, 0.05, 0.05, 0, -0.1],
        manual_tax_factors=[0, 0, 0, 0.05, -0.1]
    )
    price = dcf.calculate_stock_price()
    if price is not None:
        print(f"Estimated Stock Price: {price:.2f}")
    else:
        print("Could not calculate stock price.")
