from DiscountedCashFlow import DCFModel
import yfinance as yf

"""
def tickname(stock):

    檢查股票代碼是否為上市 (TW) 或上櫃 (TWO)，並驗證數據有效性。
    如果沒有 info（tick.info 為空）時，馬上拋出錯誤。

    # 嘗試 .TW
    try:
        tick = yf.Ticker(stock + '.TW')
        if not tick.info:
            raise ValueError(f"[.TW] No info found for '{stock}'.TW，可能是無效或已下市。")

        if 'regularMarketPrice' in tick.info and tick.info['regularMarketPrice'] is not None:
            print(f"Found valid stock code: {stock}.TW")
            return stock + '.TW'
        else:
            print(f"[.TW] No regularMarketPrice found for '{stock}'.TW")
    except Exception as e:
        print(f"Error with .TW: {e}")

    # 嘗試 .TWO
    try:
        tick = yf.Ticker(stock + '.TWO')
        if not tick.info:
            raise ValueError(f"[.TWO] No info found for '{stock}'.TWO，可能是無效或已下市。")

        if 'regularMarketPrice' in tick.info and tick.info['regularMarketPrice'] is not None:
            print(f"Found valid stock code: {stock}.TWO")
            return stock + '.TWO'
        else:
            print(f"[.TWO] No regularMarketPrice found for '{stock}'.TWO")
    except Exception as e:
        print(f"Error with .TWO: {e}")

    # 如果兩種都失敗
    print(f"Error: Unable to fetch data for stock code: {stock}. It might be delisted.")
    return None
"""


if __name__ == "__main__":
    # 用户输入股票代码
    stock_code = input("Enter stock code: ")

    # 检查并获取正确的股票代码

    # 用户选择是否手动输入增长率
    manual_input = input("Manual input growth rate? (yes/no): ").strip().lower() == 'yes'

    if manual_input:
        # 手动输入增长率
        growth_rates = input("Enter growth rates separated by commas: ").split(',')
        try:
            growth_rates = list(map(float, growth_rates))  # 转换为浮点数
        except ValueError:
            print("Invalid growth rates. Please enter valid numbers separated by commas.")
            exit()

        dcf = DCFModel(stock_code, growth_rates=growth_rates, perpetual_growth_rate=0.025)
    else:
        # 使用默认永续增长率
        dcf = DCFModel(stock_code, perpetual_growth_rate=0.025)

    # 计算股票价格
    try:
        stock_price = dcf.calculate_stock_price()
        if stock_price is not None:
            print(f"Estimated Stock Price: {stock_price:.2f}")
        else:
            print("Stock price calculation failed.")
    except Exception as e:
        print(f"Error calculating stock price: {e}")
