# test_dcf_model.py
import pytest
from dcf_model import DCFModel

# For testing, we override the WACC model to return a fixed value.
class DummyWACCModel:
    def __init__(self, stock_code):
        self.stock_code = stock_code
    def calculate_wacc(self):
        return 0.08  # Fixed WACC for testing purposes

# Monkey-patch the Wacc.WACCModel in the dcf_model module
import Wacc
Wacc.WACCModel = DummyWACCModel

def test_forecast_fcf_length():
    """Test that forecast_fcf_list returns the correct number of forecast years."""
    dcf = DCFModel(
        stock_code="2330.TW",
        forecast_years=5,
        perpetual_growth_rate=0.025,
        manual_growth_rates=[0.1, 0.1, 0.1, 0.1, 0.1],
        manual_capex_factors=[0.0, 0.0, 0.0, 0.0, 0.0],
        manual_wc_factors=[0.0, 0.0, 0.0, 0.0, 0.0],
        manual_depr_factors=[0.0, 0.0, 0.0, 0.0, 0.0],
        manual_tax_factors=[0.0, 0.0, 0.0, 0.0, 0.0]
    )
    fcf_list = dcf.forecast_fcf_list()
    assert len(fcf_list) == 5, "Forecast FCF list should have 5 entries."

def test_stock_price_calculation():
    """Test that the calculated stock price is a positive float."""
    dcf = DCFModel(
        stock_code="2330.TW",
        forecast_years=3,
        perpetual_growth_rate=0.025,
        manual_growth_rates=[0.1, 0.1, 0.1],
        manual_capex_factors=[0.0, 0.0, 0.0],
        manual_wc_factors=[0.0, 0.0, 0.0],
        manual_depr_factors=[0.0, 0.0, 0.0],
        manual_tax_factors=[0.0, 0.0, 0.0]
    )
    price = dcf.calculate_stock_price()
    assert isinstance(price, float), "Stock price should be a float."
    assert price > 0, "Stock price should be positive."

if __name__ == "__main__":
    pytest.main()