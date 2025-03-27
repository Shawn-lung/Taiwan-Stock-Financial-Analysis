import sys
from PyQt5.QtWidgets import QApplication
from ui_app import StockApp  # 引入 PyQt 界面

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockApp()
    window.show()
    sys.exit(app.exec_())
