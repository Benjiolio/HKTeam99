import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import hashlib
import hmac
import urllib.parse
from datetime import datetime, timedelta
import json

# ========== API配置 ==========
# 🔑 请在这里填入你的Roostoo API信息
API_KEY = "PvXrtqLPiu7DqiVyC6aCAAoE0kgRtJdXeoC7wLn0OIOf5qIKrb58GbATFctkMWn0"  # 替换为你的API Key
SECRET_KEY = "94WfpKd5PHng5u2ySWvZW0URKxZofI5rON3MJ0CURKgz4gKj1vxI8HZmvugrOt4U"  # 替换为你的Secret Key
BASE_URL = "https://mock-api.roostoo.com"  # Roostoo API基础地址

# ========== 交易对配置 ==========
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'BNBUSDT']

# ========== API工具函数 ==========
def generate_signature(params, secret_key):
    """
    根据Roostoo API要求生成HMAC SHA256签名
    """
    # 参数按key排序后连接成字符串
    sorted_params = sorted(params.items())
    query_string = '&'.join([f"{key}={urllib.parse.quote(str(value))}" for key, value in sorted_params])
    
    # 使用HMAC SHA256生成签名
    signature = hmac.new(
        secret_key.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return signature

def get_timestamp():
    """获取13位毫秒时间戳"""
    return str(int(time.time() * 1000))

def get_account_balance():
    """获取账户余额信息"""
    try:
        timestamp = get_timestamp()
        
        params = {
            'timestamp': timestamp
        }
        
        signature = generate_signature(params, SECRET_KEY)
        
        headers = {
            'RST-API-KEY': API_KEY,
            'MSG-SIGNATURE': signature,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = requests.get(f"{BASE_URL}/v3/balance", headers=headers, params=params, timeout=10)
        data = response.json()
        
        if data.get('Success'):
            print("✅ 账户余额获取成功")
            return data.get('Wallet', {})
        else:
            print(f"❌ 获取余额失败: {data.get('ErrMsg', '未知错误')}")
            return None
            
    except Exception as e:
        print(f"❌ 获取余额时发生异常: {e}")
        return None

def get_realtime_ticker(symbol):
    """获取单个交易对的实时行情"""
    try:
        timestamp = get_timestamp()
        
        params = {
            'pair': symbol,
            'timestamp': timestamp
        }
        
        response = requests.get(f"{BASE_URL}/v3/ticker", params=params, timeout=10)
        data = response.json()
        
        if data.get('Success'):
            ticker_data = data.get('Data', {}).get(symbol, {})
            return {
                'symbol': symbol,
                'last_price': float(ticker_data.get('LastPrice', 0)),
                'volume': float(ticker_data.get('Volume', 0)),
                'timestamp': datetime.now()
            }
        else:
            print(f"❌ 获取{symbol}行情失败: {data.get('ErrMsg', '未知错误')}")
            return None
            
    except Exception as e:
        print(f"❌ 获取{symbol}行情时发生异常: {e}")
        return None

def get_all_tickers():
    """获取所有交易对的实时行情"""
    tickers = {}
    for symbol in SYMBOLS:
        ticker_data = get_realtime_ticker(symbol)
        if ticker_data:
            tickers[symbol] = ticker_data
        time.sleep(0.1)  # 避免请求过于频繁
    return tickers

def place_order(symbol, side, quantity, order_type="MARKET"):
    """下订单"""
    try:
        timestamp = get_timestamp()
        
        params = {
            'pair': symbol,
            'side': side.upper(),  # BUY 或 SELL
            'quantity': float(quantity),
            'type': order_type.upper(),
            'timestamp': timestamp
        }
        
        signature = generate_signature(params, SECRET_KEY)
        
        headers = {
            'RST-API-KEY': API_KEY,
            'MSG-SIGNATURE': signature,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = requests.post(f"{BASE_URL}/v3/place_order", headers=headers, data=params, timeout=10)
        data = response.json()
        
        if data.get('Success'):
            print(f"✅ 订单提交成功: {side} {quantity} {symbol}")
            return True
        else:
            print(f"❌ 订单提交失败: {data.get('ErrMsg', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"❌ 下单时发生异常: {e}")
        return False

def get_kline_data(symbol, interval='5m', limit=100):
    """获取K线数据用于动量计算"""
    try:
        timestamp = get_timestamp()
        
        params = {
            'pair': symbol,
            'interval': interval,
            'limit': limit,
            'timestamp': timestamp
        }
        
        response = requests.get(f"{BASE_URL}/v3/klines", params=params, timeout=10)
        data = response.json()
        
        if data.get('Success'):
            klines = data.get('Data', {}).get(symbol, [])
            # 转换为DataFrame
            df_data = []
            for kline in klines:
                df_data.append({
                    'open_time': datetime.fromtimestamp(kline[0] / 1000),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'symbol': symbol
                })
            return pd.DataFrame(df_data)
        else:
            print(f"❌ 获取{symbol}K线数据失败: {data.get('ErrMsg', '未知错误')}")
            return None
            
    except Exception as e:
        print(f"❌ 获取K线数据时发生异常: {e}")
        return None

# ========== 市场轮动策略类 ==========
class MarketRotationStrategy:
    """
    基于Roostoo API的市场轮动策略
    核心思想：定期选择动量最强的币种进行投资
    """
    def __init__(self, initial_cash=10000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # 持仓数量 {symbol: quantity}
        self.portfolio_value_history = []
        self.trade_history = []
        
        # ========== 策略核心参数 ==========
        self.rebalance_hours = 24          # 调仓频率：每24小时
        self.top_n = 2                     # 持有前N个币种
        self.momentum_periods = [7, 14, 30]  # 动量计算周期（天）
        
        # ========== 用于标注买卖点的数据 ==========
        self.buy_points = {symbol: [] for symbol in SYMBOLS}
        self.sell_points = {symbol: [] for symbol in SYMBOLS}
        
        # ========== 数据缓存 ==========
        self.price_history = {symbol: [] for symbol in SYMBOLS}
        self.last_rebalance = None
        
        print("🎯 市场轮动策略初始化完成")

    def calculate_momentum_score(self, df):
        """
        计算动量得分 - 策略的核心大脑
        不只是看谁涨得多，还要看谁涨得稳
        """
        if len(df) < max(self.momentum_periods):
            return 0
        
        momentum_scores = []
        
        for period in self.momentum_periods:
            if len(df) >= period:
                # 计算周期收益率
                period_return = (df['close'].iloc[-1] / df['close'].iloc[-period] - 1)
                
                # 计算周期波动率（风险）
                recent_returns = df['close'].pct_change().tail(period)
                volatility = recent_returns.std()
                
                # 风险调整收益：收益/波动率
                if volatility > 0:
                    risk_adjusted_return = period_return / volatility
                else:
                    risk_adjusted_return = period_return
                
                momentum_scores.append(risk_adjusted_return)
        
        # 综合多个时间维度的得分
        return np.mean(momentum_scores) if momentum_scores else 0

    def get_current_prices(self):
        """获取所有交易对的当前价格"""
        tickers = get_all_tickers()
        current_prices = {}
        for symbol, ticker in tickers.items():
            if ticker:
                current_prices[symbol] = ticker['last_price']
                # 更新价格历史
                self.price_history[symbol].append({
                    'timestamp': ticker['timestamp'],
                    'price': ticker['last_price']
                })
        return current_prices

    def calculate_portfolio_value(self, current_prices):
        """计算当前投资组合总价值"""
        total_value = self.cash
        for symbol, quantity in self.positions.items():
            if quantity > 0 and symbol in current_prices:
                total_value += quantity * current_prices[symbol]
        return total_value

    def execute_rebalance(self):
        """执行调仓操作"""
        print(f"\n🔄 开始调仓操作 - {datetime.now()}")
        
        # 1. 获取所有币种的K线数据并计算动量得分
        momentum_scores = {}
        for symbol in SYMBOLS:
            df = get_kline_data(symbol, interval='1d', limit=50)  # 获取50天日线数据
            if df is not None and len(df) > 0:
                score = self.calculate_momentum_score(df)
                momentum_scores[symbol] = score
                print(f"   📊 {symbol}: 动量得分 = {score:.4f}")
            else:
                momentum_scores[symbol] = 0
                print(f"   ⚠️  {symbol}: 无法计算动量得分")
        
        # 2. 选择动量最强的top_n个币种
        top_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_n]
        selected_symbols = [s[0] for s in top_symbols]
        
        print(f"   🏆 选中币种: {selected_symbols}")
        
        # 3. 获取当前价格
        current_prices = self.get_current_prices()
        if not current_prices:
            print("   ❌ 无法获取当前价格，调仓中止")
            return
        
        # 4. 卖出不在top_n中的持仓
        symbols_to_sell = []
        for symbol in list(self.positions.keys()):
            if (self.positions[symbol] > 0 and 
                symbol not in selected_symbols and
                symbol in current_prices):
                symbols_to_sell.append(symbol)
        
        for symbol in symbols_to_sell:
            current_price = current_prices[symbol]
            quantity = self.positions[symbol]
            
            # 执行卖出订单
            if place_order(symbol, 'SELL', quantity):
                sell_value = quantity * current_price * 0.999  # 考虑手续费
                self.cash += sell_value
                
                # 记录卖出点
                self.sell_points[symbol].append((datetime.now(), current_price))
                
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': current_price,
                    'value': sell_value,
                    'reason': '调出轮动组合'
                })
                
                print(f"   🔴 卖出 {symbol}: {quantity:.4f} 单位 @ ${current_price:.2f}")
                self.positions[symbol] = 0
        
        # 5. 买入选中的币种（等权重分配）
        if selected_symbols and self.cash > 10:  # 至少10美元才交易
            cash_per_symbol = self.cash / len(selected_symbols)
            
            for symbol in selected_symbols:
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    
                    # 如果已经有持仓，跳过
                    if self.positions.get(symbol, 0) > 0:
                        continue
                    
                    quantity = cash_per_symbol / current_price
                    
                    # 执行买入订单
                    if place_order(symbol, 'BUY', quantity):
                        self.positions[symbol] = quantity
                        self.cash -= cash_per_symbol * 0.999  # 考虑手续费
                        
                        # 记录买入点
                        self.buy_points[symbol].append((datetime.now(), current_price))
                        
                        self.trade_history.append({
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': quantity,
                            'price': current_price,
                            'value': cash_per_symbol,
                            'reason': f'动量得分: {momentum_scores[symbol]:.4f}'
                        })
                        
                        print(f"   🟢 买入 {symbol}: {quantity:.4f} 单位 @ ${current_price:.2f} (得分: {momentum_scores[symbol]:.4f})")
        
        self.last_rebalance = datetime.now()
        print("   ✅ 调仓操作完成")

    def run_live_strategy(self, run_duration_hours=24):
        """
        运行实时策略
        run_duration_hours: 策略运行时长（小时）
        """
        print(f"🚀 启动实时市场轮动策略")
        print(f"⏰ 运行时长: {run_duration_hours} 小时")
        print(f"📊 监控币种: {SYMBOLS}")
        print(f"🔄 调仓频率: 每 {self.rebalance_hours} 小时")
        print(f"🎯 持仓数量: 前 {self.top_n} 个币种")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=run_duration_hours)
        
        # 初始调仓
        self.execute_rebalance()
        
        while datetime.now() < end_time:
            current_time = datetime.now()
            
            # 检查是否到达调仓时间
            if (self.last_rebalance is None or 
                (current_time - self.last_rebalance).total_seconds() >= self.rebalance_hours * 3600):
                
                self.execute_rebalance()
            
            # 记录投资组合价值
            current_prices = self.get_current_prices()
            if current_prices:
                portfolio_value = self.calculate_portfolio_value(current_prices)
                self.portfolio_value_history.append({
                    'timestamp': current_time,
                    'portfolio_value': portfolio_value
                })
                
                print(f"📈 当前组合价值: ${portfolio_value:.2f} | 现金: ${self.cash:.2f} | 时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 等待5分钟再检查
            time.sleep(300)  # 5分钟
        
        print(f"\n✅ 策略运行完成")
        self.print_final_report()

    def print_final_report(self):
        """打印最终报告"""
        if not self.portfolio_value_history:
            return
            
        portfolio_df = pd.DataFrame(self.portfolio_value_history)
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        
        print(f"\n" + "="*60)
        print("📊 策略最终报告")
        print("="*60)
        print(f"💰 初始资金: ${self.initial_cash:,.2f}")
        print(f"💰 最终价值: ${final_value:,.2f}")
        print(f"📈 总收益率: {total_return:.2f}%")
        print(f"🔢 总交易次数: {len(self.trade_history)}")
        
        # 买卖统计
        buy_trades = [t for t in self.trade_history if t['action'] == 'BUY']
        sell_trades = [t for t in self.trade_history if t['action'] == 'SELL']
        
        print(f"🟢 买入交易: {len(buy_trades)} 次")
        print(f"🔴 卖出交易: {len(sell_trades)} 次")
        
        print(f"\n📦 最终持仓:")
        print(f"  现金: ${self.cash:.2f}")
        current_prices = self.get_current_prices()
        for symbol in SYMBOLS:
            if self.positions.get(symbol, 0) > 0 and symbol in current_prices:
                value = self.positions[symbol] * current_prices[symbol]
                print(f"  {symbol}: {self.positions[symbol]:.6f} 单位, 价值: ${value:.2f}")
        
        # 绘制图表
        self.plot_performance(portfolio_df)

    def plot_performance(self, portfolio_df):
        """绘制策略表现图表"""
        if len(portfolio_df) < 2:
            print("⚠️ 数据不足，无法绘制图表")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 图表1：投资组合价值
        ax1.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], 
                linewidth=2, color='blue', label='组合价值')
        ax1.axhline(y=self.initial_cash, color='red', linestyle='--', alpha=0.7, label='初始资金')
        ax1.set_title('🎯 市场轮动策略 - 实时表现', fontsize=14, fontweight='bold')
        ax1.set_ylabel('组合价值 (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 标注调仓点
        rebalance_times = []
        rebalance_values = []
        for trade in self.trade_history:
            if trade['action'] == 'BUY':
                rebalance_times.append(trade['timestamp'])
                # 找到最近的组合价值
                time_diff = [(t - trade['timestamp']).total_seconds() for t in portfolio_df['timestamp']]
                closest_idx = np.argmin(np.abs(time_diff))
                rebalance_values.append(portfolio_df['portfolio_value'].iloc[closest_idx])
        
        if rebalance_times and rebalance_values:
            ax1.scatter(rebalance_times, rebalance_values, color='orange', 
                       s=50, zorder=5, label='调仓点', alpha=0.7)
            ax1.legend()
        
        # 图表2：价格走势
        ax2.set_title('📈 币种价格走势', fontsize=14, fontweight='bold')
        ax2.set_ylabel('相对价格')
        ax2.set_xlabel('时间')
        
        # 由于实时数据难以标准化，这里显示最近的价格变化
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, symbol in enumerate(SYMBOLS):
            if self.price_history[symbol]:
                prices = [p['price'] for p in self.price_history[symbol]]
                times = [p['timestamp'] for p in self.price_history[symbol]]
                if len(prices) > 1:
                    # 标准化到起始点
                    normalized_prices = [p / prices[0] * 100 for p in prices]
                    ax2.plot(times, normalized_prices, 
                            label=symbol, linewidth=1.5, color=colors[i % len(colors)])
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ========== 主程序 ==========
def main():
    """主程序"""
    print("🚀 Roostoo API 市场轮动策略")
    print("="*50)
    
    # 检查API配置
    if API_KEY == "your_api_key_here" or SECRET_KEY == "your_secret_key_here":
        print("❌ 请先配置你的API密钥和Secret Key")
        print("📍 修改代码中的 API_KEY 和 SECRET_KEY 变量")
        return
    
    # 测试API连接
    print("🔗 测试API连接...")
    balance = get_account_balance()
    if balance is None:
        print("❌ API连接失败，请检查网络和API密钥")
        return
    
    print("✅ API连接成功")
    
    # 获取初始资金
    initial_cash = balance.get('USD', {}).get('Free', 10000)
    print(f"💰 初始资金: ${initial_cash:.2f}")
    
    # 创建并运行策略
    strategy = MarketRotationStrategy(initial_cash=initial_cash)
    
    # 运行策略24小时
    strategy.run_live_strategy(run_duration_hours=24)

if __name__ == "__main__":
    main()