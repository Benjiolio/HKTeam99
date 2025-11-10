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
import os
import logging

# ========== 配置日志 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== API配置 ==========
# 从环境变量获取API密钥（AWS部署推荐）
API_KEY = os.getenv('API_KEY', "PvXrtqLPiu7DqiVyC6aCAAoE0kgRtJdXeoC7wLn0OIOf5qIKrb58GbATFctkMWn0")
SECRET_KEY = os.getenv('SECRET_KEY', "94WfpKd5PHng5u2ySWvZW0URKxZofI5rON3MJ0CURKgz4gKj1vxI8HZmvugrOt4U")
BASE_URL = "https://mock-api.roostoo.com"

# ========== 交易对配置 ==========
# Roostoo交易对格式（根据文档调整）
SYMBOLS = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'BNB/USD']

# ========== API工具函数 ==========
def generate_signature(params, secret_key):
    """根据Roostoo API要求生成HMAC SHA256签名"""
    try:
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
    except Exception as e:
        logger.error(f"生成签名失败: {e}")
        return None

def get_timestamp():
    """获取13位毫秒时间戳"""
    return str(int(time.time() * 1000))

def get_account_balance():
    """获取账户余额信息"""
    try:
        timestamp = get_timestamp()
        
        params = {'timestamp': timestamp}
        signature = generate_signature(params, SECRET_KEY)
        
        if not signature:
            return None
            
        headers = {
            'RST-API-KEY': API_KEY,
            'MSG-SIGNATURE': signature,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = requests.get(f"{BASE_URL}/v3/balance", headers=headers, params=params, timeout=10)
        data = response.json()
        
        if data.get('Success'):
            logger.info("✅ 账户余额获取成功")
            return data.get('Wallet', {})
        else:
            logger.error(f"❌ 获取余额失败: {data.get('ErrMsg', '未知错误')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ 获取余额时发生异常: {e}")
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
            logger.warning(f"⚠️ 获取{symbol}行情失败: {data.get('ErrMsg', '未知错误')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ 获取{symbol}行情时发生异常: {e}")
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
            'side': side.upper(),
            'quantity': float(quantity),
            'type': order_type.upper(),
            'timestamp': timestamp
        }
        
        signature = generate_signature(params, SECRET_KEY)
        
        if not signature:
            return False
            
        headers = {
            'RST-API-KEY': API_KEY,
            'MSG-SIGNATURE': signature,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = requests.post(f"{BASE_URL}/v3/place_order", headers=headers, data=params, timeout=10)
        data = response.json()
        
        if data.get('Success'):
            logger.info(f"✅ 订单提交成功: {side} {quantity} {symbol}")
            return True
        else:
            logger.error(f"❌ 订单提交失败: {data.get('ErrMsg', '未知错误')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 下单时发生异常: {e}")
        return False

def get_kline_data(symbol, interval='5m', limit=100):
    """修复后的K线数据获取 - 添加签名"""
    try:
        timestamp = get_timestamp()
        
        params = {
            'pair': symbol,
            'interval': interval,
            'limit': limit,
            'timestamp': timestamp
        }
        
        # 🔑 修复：添加签名和请求头
        signature = generate_signature(params, SECRET_KEY)
        
        if not signature:
            return None
            
        headers = {
            'RST-API-KEY': API_KEY,
            'MSG-SIGNATURE': signature,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        # 使用headers发送请求
        response = requests.get(f"{BASE_URL}/v3/klines", headers=headers, params=params, timeout=10)
        data = response.json()
        
        logger.info(f"K线接口响应: {data.get('Success', False)} - {data.get('ErrMsg', 'No Error')}")
        
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
            logger.info(f"✅ 成功获取{symbol}K线数据: {len(df_data)}条")
            return pd.DataFrame(df_data)
        else:
            logger.error(f"❌ 获取{symbol}K线数据失败: {data.get('ErrMsg', '未知错误')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ 获取K线数据时发生异常: {e}")
        return None

# ========== 市场轮动策略类 ==========
class MarketRotationStrategy:
    """基于Roostoo API的市场轮动策略"""
    
    def __init__(self, initial_cash=10000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.portfolio_value_history = []
        self.trade_history = []
        
        # ========== 策略核心参数 ==========
        self.rebalance_hours = 6           # 调仓频率：每6小时（避免高频交易）
        self.top_n = 3                     # 持有前N个币种
        self.momentum_periods = [3, 7, 14] # 动量计算周期（天）- 缩短以适应比赛
        self.min_trade_amount = 50         # 最小交易金额
        
        # ========== 数据记录 ==========
        self.buy_points = {symbol: [] for symbol in SYMBOLS}
        self.sell_points = {symbol: [] for symbol in SYMBOLS}
        self.price_history = {symbol: [] for symbol in SYMBOLS}
        self.last_rebalance = None
        
        logger.info("🎯 市场轮动策略初始化完成")

    def calculate_momentum_score(self, df):
        """计算动量得分"""
        if df is None or len(df) < max(self.momentum_periods):
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
        
        return np.mean(momentum_scores) if momentum_scores else 0

    def calculate_risk_metrics(self):
        """计算风险调整指标（Sharpe, Sortino, Calmar）- 比赛评分关键"""
        if len(self.portfolio_value_history) < 2:
            return None
        
        try:
            portfolio_df = pd.DataFrame(self.portfolio_value_history)
            portfolio_df = portfolio_df.sort_values('timestamp')
            portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change().dropna()
            
            if len(portfolio_df['returns']) < 2:
                return None
            
            # 计算基本指标
            mean_return = portfolio_df['returns'].mean()
            total_return = (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0] - 1)
            
            # Sharpe Ratio (总风险调整)
            std_dev = portfolio_df['returns'].std()
            sharpe = mean_return / std_dev if std_dev != 0 else 0
            
            # Sortino Ratio (下行风险调整) - 权重0.4
            downside_returns = portfolio_df[portfolio_df['returns'] < 0]['returns']
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino = mean_return / downside_std if downside_std != 0 else 0
            
            # Calmar Ratio (最大回撤调整) - 权重0.3
            portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cummax']) / portfolio_df['cummax']
            max_drawdown = portfolio_df['drawdown'].min()
            calmar = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # 综合得分（二等奖评分标准）
            composite_score = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'max_drawdown': max_drawdown,
                'composite_score': composite_score
            }
        except Exception as e:
            logger.error(f"计算风险指标时出错: {e}")
            return None

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
        logger.info(f"🔄 开始调仓操作 - {datetime.now()}")
        
        # 1. 获取所有币种的K线数据并计算动量得分
        momentum_scores = {}
        for symbol in SYMBOLS:
            df = get_kline_data(symbol, interval='1d', limit=50)
            if df is not None and len(df) > 0:
                score = self.calculate_momentum_score(df)
                momentum_scores[symbol] = score
                logger.info(f"   📊 {symbol}: 动量得分 = {score:.4f}")
            else:
                momentum_scores[symbol] = 0
                logger.warning(f"   ⚠️  {symbol}: 无法计算动量得分")
        
        # 2. 选择动量最强的top_n个币种
        top_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_n]
        selected_symbols = [s[0] for s in top_symbols]
        
        logger.info(f"   🏆 选中币种: {selected_symbols}")
        
        # 3. 获取当前价格
        current_prices = self.get_current_prices()
        if not current_prices:
            logger.error("   ❌ 无法获取当前价格，调仓中止")
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
                sell_value = quantity * current_price * 0.999  # 考虑0.1%手续费
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
                
                logger.info(f"   🔴 卖出 {symbol}: {quantity:.4f} 单位 @ ${current_price:.2f}")
                self.positions[symbol] = 0
        
        # 5. 买入选中的币种（等权重分配）
        if selected_symbols and self.cash > self.min_trade_amount:
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
                        self.cash -= cash_per_symbol * 0.999  # 考虑0.1%手续费
                        
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
                        
                        logger.info(f"   🟢 买入 {symbol}: {quantity:.4f} 单位 @ ${current_price:.2f} (得分: {momentum_scores[symbol]:.4f})")
        
        self.last_rebalance = datetime.now()
        logger.info("   ✅ 调仓操作完成")
        
        # 监控性能指标
        self.monitor_performance()

    def monitor_performance(self):
        """实时监控策略表现"""
        metrics = self.calculate_risk_metrics()
        if metrics:
            logger.info(f"\n📊 实时表现监控:")
            logger.info(f"   总收益率: {metrics['total_return']*100:.2f}%")
            logger.info(f"   Sharpe比率: {metrics['sharpe_ratio']:.4f}")
            logger.info(f"   Sortino比率: {metrics['sortino_ratio']:.4f}") 
            logger.info(f"   Calmar比率: {metrics['calmar_ratio']:.4f}")
            logger.info(f"   最大回撤: {metrics['max_drawdown']*100:.2f}%")
            logger.info(f"   综合得分: {metrics['composite_score']:.4f}")

    def run_live_strategy(self, run_duration_hours=336):
        """
        运行实时策略
        run_duration_hours: 策略运行时长（小时）- 默认14天
        """
        logger.info(f"🚀 启动实时市场轮动策略")
        logger.info(f"⏰ 运行时长: {run_duration_hours} 小时")
        logger.info(f"📊 监控币种: {SYMBOLS}")
        logger.info(f"🔄 调仓频率: 每 {self.rebalance_hours} 小时")
        logger.info(f"🎯 持仓数量: 前 {self.top_n} 个币种")
        
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
                
                logger.info(f"📈 当前组合价值: ${portfolio_value:.2f} | 现金: ${self.cash:.2f} | 时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 等待5分钟再检查
            time.sleep(300)
        
        logger.info(f"\n✅ 策略运行完成")
        self.print_final_report()

    def print_final_report(self):
        """打印最终报告"""
        if not self.portfolio_value_history:
            logger.warning("⚠️ 无投资组合历史数据")
            return
            
        portfolio_df = pd.DataFrame(self.portfolio_value_history)
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        
        # 计算风险指标
        metrics = self.calculate_risk_metrics()
        
        print(f"\n" + "="*60)
        print("📊 策略最终报告")
        print("="*60)
        print(f"💰 初始资金: ${self.initial_cash:,.2f}")
        print(f"💰 最终价值: ${final_value:,.2f}")
        print(f"📈 总收益率: {total_return:.2f}%")
        print(f"🔢 总交易次数: {len(self.trade_history)}")
        
        if metrics:
            print(f"\n🏆 风险调整指标（比赛评分）:")
            print(f"   Sharpe比率: {metrics['sharpe_ratio']:.4f} (权重: 0.3)")
            print(f"   Sortino比率: {metrics['sortino_ratio']:.4f} (权重: 0.4)")
            print(f"   Calmar比率: {metrics['calmar_ratio']:.4f} (权重: 0.3)")
            print(f"   最大回撤: {metrics['max_drawdown']*100:.2f}%")
            print(f"   综合得分: {metrics['composite_score']:.4f}")
        
        # 买卖统计
        buy_trades = [t for t in self.trade_history if t['action'] == 'BUY']
        sell_trades = [t for t in self.trade_history if t['action'] == 'SELL']
        
        print(f"\n📝 交易统计:")
        print(f"   🟢 买入交易: {len(buy_trades)} 次")
        print(f"   🔴 卖出交易: {len(sell_trades)} 次")
        
        print(f"\n📦 最终持仓:")
        print(f"   现金: ${self.cash:.2f}")
        current_prices = self.get_current_prices()
        for symbol in SYMBOLS:
            if self.positions.get(symbol, 0) > 0 and symbol in current_prices:
                value = self.positions[symbol] * current_prices[symbol]
                print(f"   {symbol}: {self.positions[symbol]:.6f} 单位, 价值: ${value:.2f}")
        
        # 绘制图表
        self.plot_performance(portfolio_df)

    def plot_performance(self, portfolio_df):
        """绘制策略表现图表"""
        if len(portfolio_df) < 2:
            print("⚠️ 数据不足，无法绘制图表")
            return
            
        try:
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
            plt.savefig('strategy_performance.png')  # 保存图片用于报告
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制图表时出错: {e}")

# ========== 主程序 ==========
def main():
    """主程序"""
    print("🚀 Roostoo Hackathon - 市场轮动策略")
    print("="*50)
    
    # 检查API配置
    if API_KEY == "your_api_key_here" or SECRET_KEY == "your_secret_key_here":
        print("❌ 请先配置你的API密钥和Secret Key")
        print("📍 设置环境变量或修改代码中的API_KEY和SECRET_KEY")
        return
    
    # 测试API连接
    print("🔗 测试API连接...")
    balance = get_account_balance()
    if balance is None:
        print("❌ API连接失败，请检查:")
        print("   1. API密钥是否正确")
        print("   2. 网络连接是否正常") 
        print("   3. 交易对格式是否正确")
        return
    
    print("✅ API连接成功")
    
    # 获取初始资金
    initial_cash = balance.get('USD', {}).get('Free', 10000)
    print(f"💰 初始资金: ${initial_cash:.2f}")
    
    # 创建策略实例
    strategy = MarketRotationStrategy(initial_cash=initial_cash)
    
    # 比赛期间持续运行（14天）
    print("\n🎯 开始正式比赛运行...")
    strategy.run_live_strategy(run_duration_hours=336)  # 14天

if __name__ == "__main__":
    main()