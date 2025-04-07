import sys
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, 
                             QTabWidget, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
                             QTextEdit, QMessageBox, QProgressBar, QFileDialog, QAction)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

# Import your existing finance models
from dcf_integrator import IntegratedValuationModel
from dcf_model import DCFModel
from ml_predictor import GrowthPredictor
from deep_learning import DeepFinancialForecaster

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatplotlibCanvas(FigureCanvas):
    """Canvas for embedding Matplotlib figures in PyQt5."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)

class ValuationWorker(QThread):
    """Worker thread for running valuation models without freezing the UI."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    
    def __init__(self, ticker, industry=None, forecast_years=5, 
                 perpetual_growth=0.025, use_ml=True, use_dl=True, use_industry=True):
        super().__init__()
        self.ticker = ticker
        self.industry = industry
        self.forecast_years = forecast_years
        self.perpetual_growth = perpetual_growth
        self.use_ml = use_ml
        self.use_dl = use_dl
        self.use_industry = use_industry
        
    def run(self):
        try:
            self.progress.emit(10, "Initializing models...")
            model = IntegratedValuationModel(
                use_ml=self.use_ml,
                use_dl=self.use_dl,
                use_industry=self.use_industry
            )
            
            self.progress.emit(20, "Running standard DCF model...")
            result = model.run_valuation(
                ticker=self.ticker,
                industry=self.industry if self.industry != "Auto-detect" else None
            )
            
            self.finished.emit(result)
            
        except Exception as e:
            logger.error(f"Valuation error: {str(e)}")
            self.error.emit(f"Error in valuation: {str(e)}")

class SensitivityAnalysisWorker(QThread):
    """Worker thread for running sensitivity analysis."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    
    def __init__(self, ticker, base_valuation):
        super().__init__()
        self.ticker = ticker
        self.base_valuation = base_valuation
        
    def run(self):
        try:
            self.progress.emit(10, "Initializing sensitivity analysis...")
            # Create base DCF model
            dcf = DCFModel(stock_code=self.ticker)
            
            # Factors to analyze
            factors = ['growth_rates', 'wacc', 'perpetual_growth_rate']
            results = {}
            
            total_runs = len(factors) * 2  # high and low for each factor
            current_run = 0
            
            for factor in factors:
                self.progress.emit(20 + (current_run / total_runs) * 70, f"Analyzing {factor}...")
                
                # Store original value
                original_value = getattr(dcf, factor) if hasattr(dcf, factor) else None
                
                if factor == 'growth_rates':
                    # For growth rates, adjust the first year's growth rate
                    if dcf.manual_growth_rates:
                        original = dcf.manual_growth_rates[0]
                        # High case: +20%
                        dcf.manual_growth_rates[0] = original * 1.2
                        high_val = dcf.calculate_stock_price()
                        current_run += 1
                        
                        # Low case: -20%
                        dcf.manual_growth_rates[0] = original * 0.8
                        low_val = dcf.calculate_stock_price()
                        current_run += 1
                        
                        # Restore original
                        dcf.manual_growth_rates[0] = original
                    else:
                        high_val = low_val = None
                elif factor == 'wacc':
                    # High case: +20%
                    dcf.wacc = dcf.wacc * 1.2
                    high_val = dcf.calculate_stock_price()
                    current_run += 1
                    
                    # Low case: -20%
                    dcf.wacc = dcf.wacc * 0.8 / 1.2  # Adjusting from the high case
                    low_val = dcf.calculate_stock_price()
                    current_run += 1
                    
                    # Restore original
                    dcf.wacc = original_value
                elif factor == 'perpetual_growth_rate':
                    # High case: +20%
                    dcf.perpetual_growth_rate = min(dcf.wacc - 0.01, dcf.perpetual_growth_rate * 1.2)
                    high_val = dcf.calculate_stock_price()
                    current_run += 1
                    
                    # Low case: -20%
                    dcf.perpetual_growth_rate = max(0.01, dcf.perpetual_growth_rate * 0.8 / 1.2)
                    low_val = dcf.calculate_stock_price()
                    current_run += 1
                    
                    # Restore original
                    dcf.perpetual_growth_rate = original_value
                
                # Calculate impact
                if high_val and low_val and self.base_valuation:
                    high_impact = (high_val - self.base_valuation) / self.base_valuation
                    low_impact = (low_val - self.base_valuation) / self.base_valuation
                    results[factor] = {'high': high_impact, 'low': low_impact, 
                                      'high_val': high_val, 'low_val': low_val}
            
            self.progress.emit(95, "Completing analysis...")
            self.finished.emit(results)
            
        except Exception as e:
            logger.error(f"Sensitivity analysis error: {str(e)}")
            self.error.emit(f"Error in sensitivity analysis: {str(e)}")

class FinancialMetricsTable(QTableWidget):
    """Table for displaying financial metrics."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(["Metric", "Year -4", "Year -3", "Year -2", "Year -1"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        
    def populate_data(self, financial_data):
        if financial_data is None or not hasattr(financial_data, 'index'):
            self.setRowCount(0)
            return
            
        metrics = [
            'Total Revenue', 'Operating Income', 'Net Income', 
            'Operating Margin', 'Net Margin', 'ROE',
            'Total Assets', 'Total Debt', 'Total Equity'
        ]
        
        available_metrics = [m for m in metrics if m in financial_data.index]
        self.setRowCount(len(available_metrics))
        
        # Determine the number of years of data available
        if hasattr(financial_data, 'columns'):
            years = min(4, len(financial_data.columns))
        else:
            years = 0
            
        # Fill the table
        for i, metric in enumerate(available_metrics):
            self.setItem(i, 0, QTableWidgetItem(metric))
            
            for y in range(years):
                col = financial_data.columns[-(y+1)]
                value = financial_data.loc[metric, col]
                
                # Format the value appropriately
                if 'Margin' in metric or 'ROE' in metric:
                    formatted_value = f"{value:.1%}" if isinstance(value, (int, float)) else str(value)
                else:
                    formatted_value = f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)
                    
                self.setItem(i, years-y, QTableWidgetItem(formatted_value))

class MainWindow(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Taiwan Stock Financial Analysis System")
        self.setGeometry(100, 100, 1280, 800)
        
        # Initialize instance variables
        self.current_ticker = None
        self.valuation_results = None
        self.financial_data = None
        self.sensitivity_results = None
        
        # Set up the menu bar
        self.setup_menu()
        
        # Set up the main UI layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create the UI components
        self.create_input_panel()
        self.create_results_tabs()
        
        # Connect signals
        self.connect_signals()
    
    def setup_menu(self):
        """Set up the application menu."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Save report action
        save_action = QAction('Save Report', self)
        save_action.triggered.connect(self.save_report)
        file_menu.addAction(save_action)
        
        # Export data action
        export_action = QAction('Export Data', self)
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        # About action
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_input_panel(self):
        """Create the input panel for valuation parameters."""
        input_group = QGroupBox("Valuation Parameters")
        input_layout = QFormLayout()
        
        # Stock ticker input
        self.ticker_input = QLineEdit("2330.TW")
        input_layout.addRow("Stock Code:", self.ticker_input)
        
        # Industry selection
        self.industry_combo = QComboBox()
        self.industry_combo.addItem("Auto-detect")
        industries = [
            "Semiconductors", "Electronics", "Banking", "Telecommunications", 
            "Financial Services", "Computer Hardware", "Food & Beverage", "Retail", 
            "Healthcare", "Utilities", "Materials", "Electronics Manufacturing",
            "Other"
        ]
        self.industry_combo.addItems(industries)
        input_layout.addRow("Industry:", self.industry_combo)
        
        # Forecast parameters
        self.forecast_years = QSpinBox()
        self.forecast_years.setRange(1, 15)
        self.forecast_years.setValue(5)
        input_layout.addRow("Forecast Years:", self.forecast_years)
        
        self.perpetual_growth = QDoubleSpinBox()
        self.perpetual_growth.setRange(0.005, 0.05)
        self.perpetual_growth.setSingleStep(0.001)
        self.perpetual_growth.setValue(0.025)
        self.perpetual_growth.setDecimals(3)
        input_layout.addRow("Perpetual Growth Rate:", self.perpetual_growth)
        
        # Model options
        options_layout = QHBoxLayout()
        
        self.use_ml_check = QCheckBox("Use ML model")
        self.use_ml_check.setChecked(True)
        options_layout.addWidget(self.use_ml_check)
        
        self.use_dl_check = QCheckBox("Use Deep Learning")
        self.use_dl_check.setChecked(True)
        options_layout.addWidget(self.use_dl_check)
        
        self.use_industry_check = QCheckBox("Apply Industry Adjustments")
        self.use_industry_check.setChecked(True)
        options_layout.addWidget(self.use_industry_check)
        
        input_layout.addRow("Model Options:", options_layout)
        
        # Calculate button and progress bar
        button_layout = QHBoxLayout()
        
        self.calculate_button = QPushButton("Calculate Intrinsic Value")
        self.calculate_button.setMinimumHeight(40)
        button_layout.addWidget(self.calculate_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v")
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        button_layout.addWidget(self.progress_bar)
        
        input_layout.addRow("", button_layout)
        
        input_group.setLayout(input_layout)
        self.main_layout.addWidget(input_group)
    
    def create_results_tabs(self):
        """Create tabs for different result views."""
        self.results_tabs = QTabWidget()
        
        # Summary tab
        self.summary_tab = QWidget()
        self.create_summary_tab()
        self.results_tabs.addTab(self.summary_tab, "Valuation Summary")
        
        # Financial metrics tab
        self.financials_tab = QWidget()
        self.create_financials_tab()
        self.results_tabs.addTab(self.financials_tab, "Financial Data")
        
        # Growth predictions tab
        self.growth_tab = QWidget()
        self.create_growth_tab()
        self.results_tabs.addTab(self.growth_tab, "Growth Predictions")
        
        # Sensitivity analysis tab
        self.sensitivity_tab = QWidget()
        self.create_sensitivity_tab()
        self.results_tabs.addTab(self.sensitivity_tab, "Sensitivity Analysis")
        
        # Detailed valuation tab
        self.details_tab = QWidget()
        self.create_details_tab()
        self.results_tabs.addTab(self.details_tab, "Detailed Valuation")
        
        self.main_layout.addWidget(self.results_tabs)
    
    def create_summary_tab(self):
        """Create the summary tab content."""
        layout = QVBoxLayout(self.summary_tab)
        
        # Title label
        title = QLabel("Valuation Summary")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Create a horizontal layout for side-by-side valuation results and chart
        h_layout = QHBoxLayout()
        
        # Valuation results table on the left
        self.valuation_table = QTableWidget()
        self.valuation_table.setColumnCount(2)
        self.valuation_table.setHorizontalHeaderLabels(["Model", "Intrinsic Value"])
        self.valuation_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.valuation_table.verticalHeader().setVisible(False)
        h_layout.addWidget(self.valuation_table, 1)
        
        # Valuation chart on the right
        self.valuation_chart = MatplotlibCanvas(width=5, height=4)
        h_layout.addWidget(self.valuation_chart, 1)
        
        layout.addLayout(h_layout)
        
        # Industry adjustment details section
        adj_group = QGroupBox("Industry Adjustment Details")
        adj_layout = QVBoxLayout()
        
        self.industry_info = QLabel("Industry: Not detected yet")
        adj_layout.addWidget(self.industry_info)
        
        self.adjustment_table = QTableWidget()
        self.adjustment_table.setColumnCount(4)
        self.adjustment_table.setHorizontalHeaderLabels([
            "Model", "Base Value", "Adjusted Value", "Adjustment Factor"
        ])
        self.adjustment_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.adjustment_table.verticalHeader().setVisible(False)
        adj_layout.addWidget(self.adjustment_table)
        
        adj_group.setLayout(adj_layout)
        layout.addWidget(adj_group)
    
    def create_financials_tab(self):
        """Create the financial data tab content."""
        layout = QVBoxLayout(self.financials_tab)
        
        # Title and description
        title = QLabel("Historical Financial Data")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        description = QLabel("Key financial metrics for the past years:")
        layout.addWidget(description)
        
        # Financial metrics table
        self.financial_metrics_table = FinancialMetricsTable()
        layout.addWidget(self.financial_metrics_table)
        
        # Financial charts section
        charts_group = QGroupBox("Financial Trends")
        charts_layout = QVBoxLayout()
        
        self.financial_charts = MatplotlibCanvas(width=8, height=4)
        charts_layout.addWidget(self.financial_charts)
        
        charts_group.setLayout(charts_layout)
        layout.addWidget(charts_group)
    
    def create_growth_tab(self):
        """Create the growth predictions tab content."""
        layout = QVBoxLayout(self.growth_tab)
        
        # Title and description
        title = QLabel("Growth Rate Predictions")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Horizontal layout for predictions and chart
        h_layout = QHBoxLayout()
        
        # Predictions table on the left
        self.growth_table = QTableWidget()
        self.growth_table.setColumnCount(6)
        self.growth_table.setHorizontalHeaderLabels(["Factor", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"])
        self.growth_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.growth_table.verticalHeader().setVisible(False)
        h_layout.addWidget(self.growth_table)
        
        # Growth chart on the right
        self.growth_chart = MatplotlibCanvas(width=5, height=4)
        h_layout.addWidget(self.growth_chart)
        
        layout.addLayout(h_layout)
        
        # Machine Learning vs Deep Learning comparison
        ml_dl_group = QGroupBox("ML vs. Deep Learning Comparison")
        ml_dl_layout = QVBoxLayout()
        
        self.ml_dl_chart = MatplotlibCanvas(width=8, height=4)
        ml_dl_layout.addWidget(self.ml_dl_chart)
        
        ml_dl_group.setLayout(ml_dl_layout)
        layout.addWidget(ml_dl_group)
    
    def create_sensitivity_tab(self):
        """Create the sensitivity analysis tab content."""
        layout = QVBoxLayout(self.sensitivity_tab)
        
        # Title and description
        title = QLabel("Sensitivity Analysis")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        description = QLabel("Analysis of how changes to key factors affect the valuation:")
        layout.addWidget(description)
        
        # Run analysis button
        self.run_sensitivity_button = QPushButton("Run Sensitivity Analysis")
        layout.addWidget(self.run_sensitivity_button)
        
        # Progress bar for sensitivity analysis
        self.sensitivity_progress = QProgressBar()
        self.sensitivity_progress.setTextVisible(True)
        self.sensitivity_progress.setFormat("%p% - %v")
        self.sensitivity_progress.setValue(0)
        layout.addWidget(self.sensitivity_progress)
        
        # Horizontal layout for table and chart
        h_layout = QHBoxLayout()
        
        # Sensitivity results table on the left
        self.sensitivity_table = QTableWidget()
        self.sensitivity_table.setColumnCount(3)
        self.sensitivity_table.setHorizontalHeaderLabels(["Factor", "+20% Impact", "-20% Impact"])
        self.sensitivity_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.sensitivity_table.verticalHeader().setVisible(False)
        h_layout.addWidget(self.sensitivity_table)
        
        # Sensitivity chart on the right
        self.sensitivity_chart = MatplotlibCanvas(width=5, height=4)
        h_layout.addWidget(self.sensitivity_chart)
        
        layout.addLayout(h_layout)
    
    def create_details_tab(self):
        """Create the detailed valuation tab content."""
        layout = QVBoxLayout(self.details_tab)
        
        # Title
        title = QLabel("Detailed Valuation Results")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Detailed valuation information as text
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        layout.addWidget(self.details_text)
    
    def connect_signals(self):
        """Connect UI signals to their handlers."""
        self.calculate_button.clicked.connect(self.run_valuation)
        self.run_sensitivity_button.clicked.connect(self.run_sensitivity_analysis)
    
    def run_valuation(self):
        """Start the valuation process in a separate thread."""
        # Validate inputs
        ticker = self.ticker_input.text().strip()
        if not ticker:
            QMessageBox.warning(self, "Input Error", "Please enter a valid stock code.")
            return
        
        # Get selected industry
        industry = self.industry_combo.currentText()
        if industry == "Auto-detect":
            industry = None
        
        # Get other parameters
        forecast_years = self.forecast_years.value()
        perpetual_growth = self.perpetual_growth.value()
        use_ml = self.use_ml_check.isChecked()
        use_dl = self.use_dl_check.isChecked()
        use_industry = self.use_industry_check.isChecked()
        
        # Store the current ticker
        self.current_ticker = ticker
        
        # Disable the calculate button while processing
        self.calculate_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Create and start the worker thread
        self.worker = ValuationWorker(
            ticker=ticker,
            industry=industry,
            forecast_years=forecast_years,
            perpetual_growth=perpetual_growth,
            use_ml=use_ml,
            use_dl=use_dl,
            use_industry=use_industry
        )
        
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.handle_valuation_results)
        self.worker.error.connect(self.handle_valuation_error)
        self.worker.start()
    
    def run_sensitivity_analysis(self):
        """Start the sensitivity analysis in a separate thread."""
        if not self.valuation_results or 'models' not in self.valuation_results:
            QMessageBox.warning(self, "No Data", "Run a valuation first before performing sensitivity analysis.")
            return
        
        # Get the base valuation (standard DCF)
        base_valuation = self.valuation_results['models'].get('standard_dcf')
        if not base_valuation:
            QMessageBox.warning(self, "No Data", "Standard DCF valuation result is missing.")
            return
        
        # Disable the button while processing
        self.run_sensitivity_button.setEnabled(False)
        self.sensitivity_progress.setValue(0)
        
        # Create and start the worker thread
        self.sensitivity_worker = SensitivityAnalysisWorker(
            ticker=self.current_ticker,
            base_valuation=base_valuation
        )
        
        self.sensitivity_worker.progress.connect(self.update_sensitivity_progress)
        self.sensitivity_worker.finished.connect(self.handle_sensitivity_results)
        self.sensitivity_worker.error.connect(self.handle_sensitivity_error)
        self.sensitivity_worker.start()
    
    def update_progress(self, value, message):
        """Update the progress bar during valuation."""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}% - {message}")
    
    def update_sensitivity_progress(self, value, message):
        """Update the progress bar during sensitivity analysis."""
        self.sensitivity_progress.setValue(value)
        self.sensitivity_progress.setFormat(f"{value}% - {message}")
    
    def handle_valuation_results(self, results):
        """Process and display the valuation results."""
        self.valuation_results = results
        self.calculate_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("100% - Complete")
        
        # Update the UI with the results
        self.update_summary_tab()
        self.update_financials_tab()
        self.update_growth_tab()
        self.update_details_tab()
        
        # Show the results tab
        self.results_tabs.setCurrentIndex(0)
        
        # Display a success message
        QMessageBox.information(self, "Success", f"Valuation completed for {results['ticker']}")
    
    def handle_valuation_error(self, error_message):
        """Handle errors from the valuation process."""
        self.calculate_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Error")
        
        # Display error message
        QMessageBox.critical(self, "Valuation Error", error_message)
    
    def handle_sensitivity_results(self, results):
        """Process and display the sensitivity analysis results."""
        self.sensitivity_results = results
        self.run_sensitivity_button.setEnabled(True)
        self.sensitivity_progress.setValue(100)
        self.sensitivity_progress.setFormat("100% - Complete")
        
        # Update the sensitivity tab
        self.update_sensitivity_tab()
    
    def handle_sensitivity_error(self, error_message):
        """Handle errors from the sensitivity analysis process."""
        self.run_sensitivity_button.setEnabled(True)
        self.sensitivity_progress.setValue(0)
        self.sensitivity_progress.setFormat("Error")
        
        # Display error message
        QMessageBox.critical(self, "Sensitivity Analysis Error", error_message)
    
    def update_summary_tab(self):
        """Update the summary tab with valuation results."""
        if not self.valuation_results:
            return
            
        # Update valuation table
        self.valuation_table.setRowCount(0)
        if 'models' in self.valuation_results:
            models = self.valuation_results['models']
            self.valuation_table.setRowCount(len(models))
            
            for i, (model_name, price) in enumerate(models.items()):
                # Format model name
                display_name = model_name.replace('_', ' ').title()
                self.valuation_table.setItem(i, 0, QTableWidgetItem(display_name))
                
                # Format price
                formatted_price = f"{price:,.2f}" if price is not None else "N/A"
                self.valuation_table.setItem(i, 1, QTableWidgetItem(formatted_price))
        
        # Update industry information
        if 'detected_industry' in self.valuation_results:
            industry = self.valuation_results['detected_industry']
            self.industry_info.setText(f"Industry: {industry}")
        
        # Update adjustment table
        self.adjustment_table.setRowCount(0)
        adjustment_keys = [k for k in self.valuation_results if k.endswith('_industry_adjusted')]
        
        if adjustment_keys:
            self.adjustment_table.setRowCount(len(adjustment_keys))
            
            for i, key in enumerate(adjustment_keys):
                adj = self.valuation_results[key]
                model_name = key.replace('_industry_adjusted', '').replace('_', ' ').title()
                
                self.adjustment_table.setItem(i, 0, QTableWidgetItem(model_name))
                self.adjustment_table.setItem(i, 1, QTableWidgetItem(f"{adj['base_valuation']:,.2f}"))
                self.adjustment_table.setItem(i, 2, QTableWidgetItem(f"{adj['adjusted_valuation']:,.2f}"))
                self.adjustment_table.setItem(i, 3, QTableWidgetItem(f"{adj['total_adjustment']:.2f}x"))
        
        # Create valuation comparison chart
        if 'models' in self.valuation_results:
            models = self.valuation_results['models']
            self.valuation_chart.axes.clear()
            
            model_names = [name.replace('_', ' ').title() for name in models.keys()]
            prices = [price if price is not None else 0 for price in models.values()]
            
            # Create bar chart
            bars = self.valuation_chart.axes.bar(model_names, prices, color=['blue', 'green', 'purple'])
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                self.valuation_chart.axes.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:,.2f}',
                    ha='center', va='bottom', rotation=0
                )
            
            self.valuation_chart.axes.set_title('Valuation by Model')
            self.valuation_chart.axes.set_ylabel('Intrinsic Value')
            plt.tight_layout()
            self.valuation_chart.draw()
    
    def update_financials_tab(self):
        """Update the financials tab with data."""
        if not self.current_ticker:
            return
            
        # Get financial data using the standard DCF model
        try:
            dcf = DCFModel(stock_code=self.current_ticker)
            self.financial_data = dcf.get_financial_data()
            
            # Update the metrics table
            self.financial_metrics_table.populate_data(self.financial_data)
            
            # Create financial trends chart
            if self.financial_data is not None and hasattr(self.financial_data, 'index'):
                self.financial_charts.axes.clear()
                
                # Plot revenue trend
                if 'Total Revenue' in self.financial_data.index:
                    revenue = self.financial_data.loc['Total Revenue']
                    self.financial_charts.axes.plot(
                        revenue.index, revenue.values, 'b-o', 
                        label='Revenue'
                    )
                    
                # Plot net income trend on the same chart
                if 'Net Income' in self.financial_data.index:
                    net_income = self.financial_data.loc['Net Income']
                    self.financial_charts.axes.plot(
                        net_income.index, net_income.values, 'g-s', 
                        label='Net Income'
                    )
                
                self.financial_charts.axes.set_title('Revenue and Net Income Trends')
                self.financial_charts.axes.legend()
                self.financial_charts.axes.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                self.financial_charts.draw()
                
        except Exception as e:
            logger.error(f"Error updating financial data: {str(e)}")
    
    def update_growth_tab(self):
        """Update the growth predictions tab with data."""
        if not self.valuation_results:
            return
        
        ml_predictions = self.valuation_results.get('ml_predictions', {})
        dl_predictions = self.valuation_results.get('dl_predictions', [])
        
        # Update growth rates table
        self.growth_table.setRowCount(0)
        if ml_predictions:
            # Define factors to display
            factors = [
                ('Growth Rates', ml_predictions.get('growth_rates', [])),
                ('CAPEX Factors', ml_predictions.get('capex_factors', [])),
                ('Working Capital Factors', ml_predictions.get('wc_factors', [])),
                ('Depreciation Factors', ml_predictions.get('depr_factors', [])),
                ('Tax Factors', ml_predictions.get('tax_factors', []))
            ]
            
            # Add DL growth rates if available
            if dl_predictions:
                factors.append(('DL Growth Rates', dl_predictions))
                
            # Fill the table
            self.growth_table.setRowCount(len(factors))
            for i, (factor_name, values) in enumerate(factors):
                self.growth_table.setItem(i, 0, QTableWidgetItem(factor_name))
                
                for j, value in enumerate(values[:5]):  # Ensure we only display up to 5 years
                    if factor_name.lower().endswith('rates'):
                        formatted_value = f"{value:.1%}" if value is not None else "N/A"
                    else:
                        formatted_value = f"{value:.2f}" if value is not None else "N/A"
                    self.growth_table.setItem(i, j+1, QTableWidgetItem(formatted_value))
        
        # Create growth chart
        if ml_predictions.get('growth_rates'):
            self.growth_chart.axes.clear()
            
            years = list(range(1, len(ml_predictions['growth_rates'])+1))
            
            # Plot ML growth rates
            self.growth_chart.axes.plot(
                years, 
                [g*100 for g in ml_predictions['growth_rates']], 
                'b-o', 
                label='ML Growth'
            )
            
            # Add DL growth rates if available
            if dl_predictions:
                self.growth_chart.axes.plot(
                    years[:len(dl_predictions)], 
                    [g*100 for g in dl_predictions], 
                    'r-s', 
                    label='DL Growth'
                )
            
            # Add terminal growth line
            perpetual = self.perpetual_growth.value() * 100
            self.growth_chart.axes.axhline(
                y=perpetual, 
                color='g', 
                linestyle='--', 
                label=f'Terminal ({perpetual:.1f}%)'
            )
            
            self.growth_chart.axes.set_title('Predicted Growth Rates')
            self.growth_chart.axes.set_xlabel('Year')
            self.growth_chart.axes.set_ylabel('Growth Rate (%)')
            self.growth_chart.axes.set_xticks(years)
            self.growth_chart.axes.legend()
            self.growth_chart.axes.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            self.growth_chart.draw()
            
            # Create ML vs DL comparison chart if both are available
            if dl_predictions and ml_predictions.get('growth_rates'):
                self.ml_dl_chart.axes.clear()
                
                # Create width for bars
                bar_width = 0.35
                index = np.arange(len(years[:len(dl_predictions)]))
                
                # Plot ML and DL side by side
                self.ml_dl_chart.axes.bar(
                    index - bar_width/2, 
                    [g*100 for g in ml_predictions['growth_rates'][:len(dl_predictions)]], 
                    bar_width, 
                    label='ML Model'
                )
                
                self.ml_dl_chart.axes.bar(
                    index + bar_width/2, 
                    [g*100 for g in dl_predictions], 
                    bar_width, 
                    label='Deep Learning'
                )
                
                self.ml_dl_chart.axes.set_title('ML vs Deep Learning Growth Predictions')
                self.ml_dl_chart.axes.set_xlabel('Year')
                self.ml_dl_chart.axes.set_ylabel('Growth Rate (%)')
                self.ml_dl_chart.axes.set_xticks(index)
                self.ml_dl_chart.axes.set_xticklabels([f'Year {y}' for y in range(1, len(dl_predictions)+1)])
                self.ml_dl_chart.axes.legend()
                self.ml_dl_chart.axes.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                self.ml_dl_chart.draw()
    
    def update_sensitivity_tab(self):
        """Update the sensitivity analysis tab with results."""
        if not self.sensitivity_results:
            return
        
        # Update the table
        self.sensitivity_table.setRowCount(len(self.sensitivity_results))
        
        for i, (factor, impact) in enumerate(self.sensitivity_results.items()):
            # Format the factor name for display
            if factor == 'growth_rates':
                display_name = 'Growth Rate'
            elif factor == 'wacc':
                display_name = 'WACC'
            elif factor == 'perpetual_growth_rate':
                display_name = 'Terminal Growth'
            else:
                display_name = factor.replace('_', ' ').title()
                
            self.sensitivity_table.setItem(i, 0, QTableWidgetItem(display_name))
            
            # Format the impact values as percentages
            high_impact = impact.get('high', 0) * 100
            low_impact = impact.get('low', 0) * 100
            
            self.sensitivity_table.setItem(i, 1, QTableWidgetItem(f"{high_impact:+.1f}%"))
            self.sensitivity_table.setItem(i, 2, QTableWidgetItem(f"{low_impact:+.1f}%"))
        
        # Create sensitivity chart
        self.sensitivity_chart.axes.clear()
        
        # Extract factors and impact values
        factors = []
        high_impacts = []
        low_impacts = []
        
        for factor, impact in self.sensitivity_results.items():
            if factor == 'growth_rates':
                display_name = 'Growth Rate'
            elif factor == 'wacc':
                display_name = 'WACC'
            elif factor == 'perpetual_growth_rate':
                display_name = 'Terminal Growth'
            else:
                display_name = factor.replace('_', ' ').title()
                
            factors.append(display_name)
            high_impacts.append(impact.get('high', 0) * 100)
            low_impacts.append(impact.get('low', 0) * 100)
        
        # Create grouped bar chart
        x = np.arange(len(factors))
        width = 0.35
        
        self.sensitivity_chart.axes.bar(x - width/2, high_impacts, width, label='+20%', color='green', alpha=0.6)
        self.sensitivity_chart.axes.bar(x + width/2, low_impacts, width, label='-20%', color='red', alpha=0.6)
        
        self.sensitivity_chart.axes.set_ylabel('Impact on Stock Price (%)')
        self.sensitivity_chart.axes.set_title('Sensitivity Analysis')
        self.sensitivity_chart.axes.set_xticks(x)
        self.sensitivity_chart.axes.set_xticklabels(factors, rotation=45)
        self.sensitivity_chart.axes.legend()
        
        plt.tight_layout()
        self.sensitivity_chart.draw()
    
    def update_details_tab(self):
        """Update the detailed valuation text with results."""
        if not self.valuation_results:
            return
            
        details = [f"Detailed Valuation Results for {self.valuation_results['ticker']}"]
        details.append("=" * 50)
        details.append("")
        
        # Basic information
        if 'detected_industry' in self.valuation_results:
            details.append(f"Detected Industry: {self.valuation_results['detected_industry']}")
        details.append("")
        
        # Base valuations
        details.append("Base Valuations:")
        details.append("-" * 30)
        if 'models' in self.valuation_results:
            for model, price in self.valuation_results['models'].items():
                display_name = model.replace('_', ' ').title()
                details.append(f"{display_name}: {price:,.2f}")
        details.append("")
        
        # ML predictions if available
        if 'ml_predictions' in self.valuation_results:
            ml_pred = self.valuation_results['ml_predictions']
            details.append("ML Growth Predictions:")
            details.append("-" * 30)
            
            for factor, values in ml_pred.items():
                display_name = factor.replace('_', ' ').title()
                formatted_values = []
                
                for val in values:
                    if 'growth' in factor or 'rates' in factor:
                        formatted_values.append(f"{val:.1%}")
                    else:
                        formatted_values.append(f"{val:.2f}")
                
                details.append(f"{display_name}: {', '.join(formatted_values)}")
            details.append("")
        
        # DL predictions if available
        if 'dl_predictions' in self.valuation_results:
            dl_pred = self.valuation_results['dl_predictions']
            details.append("Deep Learning Growth Predictions:")
            details.append("-" * 30)
            
            formatted_values = [f"{val:.1%}" for val in dl_pred]
            details.append(f"Growth Rates: {', '.join(formatted_values)}")
            details.append("")
        
        # Industry adjustments if available
        industry_adj_keys = [k for k in self.valuation_results if k.endswith('_industry_adjusted')]
        if industry_adj_keys:
            details.append("Industry Adjustments:")
            details.append("-" * 30)
            
            for key in industry_adj_keys:
                adj = self.valuation_results[key]
                model_name = key.replace('_industry_adjusted', '').replace('_', ' ').title()
                details.append(f"{model_name}:")
                details.append(f"  Base Valuation: {adj['base_valuation']:,.2f}")
                details.append(f"  Adjusted Valuation: {adj['adjusted_valuation']:,.2f}")
                details.append(f"  Adjustment Factor: {adj['total_adjustment']:.2f}x")
                
                if 'industry_factor' in adj:
                    details.append(f"  Industry Factor: {adj['industry_factor']:.2f}x")
                    
                if 'return_factor' in adj:
                    details.append(f"  Return Factor: {adj['return_factor']:.2f}x")
                    
                if 'expected_return' in adj and adj['expected_return'] is not None:
                    details.append(f"  Expected 6-Month Return: {adj['expected_return']:.1%}")
                    
                details.append("")
        
        # Join and display the details
        self.details_text.setText("\n".join(details))
    
    def save_report(self):
        """Save the valuation report to a file."""
        if not self.valuation_results:
            QMessageBox.warning(self, "No Data", "Run a valuation first before saving a report.")
            return
            
        # Get filename from user
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", f"{self.current_ticker}_valuation_report.txt", "Text Files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    # Get the detailed report text
                    report_text = self.details_text.toPlainText()
                    f.write(report_text)
                    
                QMessageBox.information(self, "Success", f"Report saved to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save report: {str(e)}")
    
    def export_data(self):
        """Export valuation data to a CSV file."""
        if not self.valuation_results:
            QMessageBox.warning(self, "No Data", "Run a valuation first before exporting data.")
            return
            
        # Get filename from user
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", f"{self.current_ticker}_valuation_data.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                data = {
                    'Ticker': self.current_ticker,
                    'Industry': self.valuation_results.get('detected_industry', 'Unknown')
                }
                
                # Add model valuations
                if 'models' in self.valuation_results:
                    for model, price in self.valuation_results['models'].items():
                        display_name = model.replace('_', ' ').title()
                        data[display_name] = price
                
                # Add adjusted valuations
                for key in [k for k in self.valuation_results if k.endswith('_industry_adjusted')]:
                    adj = self.valuation_results[key]
                    model_name = key.replace('_industry_adjusted', '').replace('_', ' ').title()
                    data[f"{model_name} Adjusted"] = adj['adjusted_valuation']
                    data[f"{model_name} Adjustment Factor"] = adj['total_adjustment']
                
                # Add ML predictions
                if 'ml_predictions' in self.valuation_results:
                    ml_pred = self.valuation_results['ml_predictions']
                    for factor, values in ml_pred.items():
                        display_name = factor.replace('_', ' ').title()
                        for i, val in enumerate(values):
                            data[f"{display_name} Year {i+1}"] = val
                
                # Add DL predictions
                if 'dl_predictions' in self.valuation_results:
                    dl_pred = self.valuation_results['dl_predictions']
                    for i, val in enumerate(dl_pred):
                        data[f"DL Growth Year {i+1}"] = val
                
                # Create DataFrame and save to CSV
                df = pd.DataFrame([data])
                df.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Success", f"Data exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
        Taiwan Stock Financial Analysis System
        
        Version 1.0
        
        A comprehensive system for financial data collection, analysis, 
        and stock valuation with a focus on Taiwan stock market.
        
        Features:
        - DCF (Discounted Cash Flow) valuation
        - Machine Learning growth prediction
        - Deep Learning financial forecasting
        - Industry-specific valuation adjustments
        - Sensitivity analysis
        
        © 2025 All rights reserved.
        """
        
        QMessageBox.about(self, "About", about_text)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()