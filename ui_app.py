from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout, QFormLayout, QGridLayout
)
from PyQt5.QtCore import Qt
from DiscountedCashFlow import DCFModel  # 引入您的 DCF 計算邏輯


class StockApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Stock Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main Layout
        self.layout = QVBoxLayout()

        # Inputs Section
        self.input_layout = QFormLayout()

        self.stock_label = QLabel("Stock Code:")
        self.stock_input = QLineEdit()
        self.input_layout.addRow(self.stock_label, self.stock_input)

        self.forecast_label = QLabel("Forecast Years:")
        self.forecast_input = QLineEdit()
        self.forecast_input.textChanged.connect(self.update_dynamic_inputs)
        self.input_layout.addRow(self.forecast_label, self.forecast_input)

        self.growth_label = QLabel("Perpetual Growth Rate:")
        self.growth_input = QLineEdit()
        self.input_layout.addRow(self.growth_label, self.growth_input)

        self.dynamic_inputs = QWidget()
        self.dynamic_inputs_layout = QVBoxLayout()
        self.dynamic_inputs.setLayout(self.dynamic_inputs_layout)
        self.input_layout.addRow(self.dynamic_inputs)

        self.layout.addLayout(self.input_layout)

        # Buttons Section
        self.button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_inputs)
        self.button_layout.addWidget(self.run_button)
        self.button_layout.addWidget(self.clear_button)
        self.layout.addLayout(self.button_layout)

        # Results Section
        self.output_label = QLabel("Results:")
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_label)
        self.layout.addWidget(self.output_text)

        self.central_widget.setLayout(self.layout)

        # Store dynamic inputs
        self.dynamic_input_fields = {}

    def update_dynamic_inputs(self):
        """Dynamically update the input fields based on the number of forecast years."""
        # 清空目前的動態輸入框
        while self.dynamic_inputs_layout.count():
            item = self.dynamic_inputs_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        try:
            forecast_years = int(self.forecast_input.text())
        except ValueError:
            return  # 如果輸入不是整數，直接返回

        # 創建每個 manual_*_factors 的輸入區域
        self.dynamic_input_fields = {}
        manual_factors = [
            "manual_growth_rates",
            "manual_capex_factors",
            "manual_wc_factors",
            "manual_depr_factors",
            "manual_opincome_factors",
            "manual_tax_factors"
        ]

        for factor in manual_factors:
            # 為每個 factor 創建標籤和輸入區域
            factor_label = QLabel(f"{factor}:")
            self.dynamic_inputs_layout.addWidget(factor_label)

            factor_input_layout = QHBoxLayout()  # 使用水平佈局來放每年的輸入框
            factor_fields = []

            for year in range(forecast_years):
                input_field = QLineEdit()
                input_field.setPlaceholderText(f"Year {year + 1}")
                factor_input_layout.addWidget(input_field)
                factor_fields.append(input_field)

            self.dynamic_inputs_layout.addLayout(factor_input_layout)
            self.dynamic_input_fields[factor] = factor_fields


    def run_analysis(self):
        # Fetch input data
        stock_code = self.stock_input.text()
        forecast_years = self.forecast_input.text()
        perpetual_growth_rate = self.growth_input.text()

        # Validate inputs
        try:
            forecast_years = int(forecast_years)
            perpetual_growth_rate = float(perpetual_growth_rate)
        except ValueError:
            self.output_text.setText("Invalid input. Please provide numeric values for forecast years and growth rate.")
            return

        # Parse dynamic inputs
        manual_factors_data = {}
        try:
            for factor, input_fields in self.dynamic_input_fields.items():
                # 如果輸入框有值，將其轉換為 float，否則跳過
                values = [float(field.text()) for field in input_fields if field.text()]
                if values:  # 如果有輸入值，才加入到 manual_factors_data
                    if len(values) != forecast_years:
                        raise ValueError(f"{factor} must have {forecast_years} values if provided.")
                    manual_factors_data[factor] = values
        except ValueError as e:
            self.output_text.setText(str(e))
            return


        # Run DCFModel
        try:
            dcf = DCFModel(
                stock_code=stock_code,
                forecast_years=forecast_years,
                perpetual_growth_rate=perpetual_growth_rate,
                **manual_factors_data
            )
            price = dcf.calculate_stock_price()
            fair_price_with_mvfirm = dcf.equity_value_with_mvfirm / dcf.shares_outstanding
            self.output_text.setText(
                f"Estimated Stock Price: {price:.2f}\n"
                f"Fair Price with MV/Firm Sales: {fair_price_with_mvfirm:.2f}"
            )
        except Exception as e:
            self.output_text.setText(f"Error: {e}")

    def clear_inputs(self):
        self.stock_input.clear()
        self.forecast_input.clear()
        self.growth_input.clear()
        for input_fields in self.dynamic_input_fields.values():
            for field in input_fields:
                field.clear()
        self.output_text.clear()
