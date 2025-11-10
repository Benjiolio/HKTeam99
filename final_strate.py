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
API_KEY = "PvXrtqLPiu7DqiVyC6aCAAoE0kgRtJdXeoC7wLn0OIOf5qIKrb58GbATFctkMWn0"
SECRET_KEY = "94WfpKd5PHng5u2ySWvZW0URKxZofI5rON3MJ0CURKgz4gKj1vxI8HZmvugrOt4U"
BASE_URL = "https://mock-api.roostoo.com"

# ========== 交易对配置 ==========
# 修正交易对格式 - 根据Roostoo文档使用正确格式
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'BNBUSDT']

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
            'symbol': symbol,  # 修正参数名
            'timestamp': timestamp
        }
        
        response = requests.get(f"{BASE_URL}/v3/ticker", params=params, timeout=10)
        
        # 调试信息：打印原始响应
        logger.debug(f"Ticker响应状态: {response.status_code}")
        logger.debug(f"Ticker响应内容: {response.text[:200]}...")
        
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
    """下订单 - 修正数量精度问题"""
    try:
        timestamp = get_timestamp()
        
        # 修正数量精度 - 根据交易对调整精度
        quantity = self.adjust_quantity_precision(symbol, float(quantity))
        
        params = {
            'symbol': symbol,  # 修正参数名
            'side': side.upper(),
            'quantity': quantity,
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
        
        response = requests.post(f"{BASE_URL}/v3/order", headers=headers, data=params, timeout=10)
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
    """修复K线数据获取 - 修正API端点"""
    try:
        timestamp = get_timestamp()
        
        params = {
            'symbol': symbol,  # 修正参数名
            'interval': interval,
            'limit': limit,
            'timestamp': timestamp
        }
        
        # 尝试不同的API端点
        endpoints = [
            f"{BASE_URL}/v3/klines",
            f"{BASE_URL}/api/v3/klines",  # 常见格式
            f"{BASE_URL}/v3/market/kline"  # 备选端点
        ]
        
        signature = generate_signature(params, SECRET_KEY)
        
        if not signature:
            return None
            
        headers = {
            'RST-API-KEY': API_KEY,
            'MSG-SIGNATURE': signature,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        # 尝试多个端点
        for endpoint in endpoints:
            try:
                logger.info(f"尝试K线端点: {endpoint}")
                response = requests.get(endpoint, headers=headers, params=params, timeout=10)
                
                # 调试信息
                logger.debug(f"K线响应状态: {response.status_code}")
                logger.debug(f"K线响应内容: {response.text[:500]}...")
                
                # 检查响应内容
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('Success'):
                        klines = data.get('Data', [])
                        if not klines:
                            klines = data  # 有些API直接返回数组
                        
                        # 转换为DataFrame
                        df_data = []
                        for kline in klines:
                            # 处理不同的K线格式
                            if isinstance(kline, list) and len(kline) >= 6:
                                df_data.append({
                                    'open_time': datetime.fromtimestamp(kline[0] / 1000),
                                    'open': float(kline[1]),
                                    'high': float(kline[2]),
                                    'low': float(kline[3]),
                                    'close': float(kline[4]),
                                    'volume': float(kline[5]),
                                    'symbol': symbol
                                })
                            elif isinstance(kline, dict):
                                df_data.append({
                                    'open_time': datetime.fromtimestamp(kline.get('openTime', 0) / 1000),
                                    'open': float(kline.get('open', 0)),
                                    'high': float(kline.get('high', 0)),
                                    'low': float(kline.get('low', 0)),
                                    'close': float(kline.get('close', 0)),
                                    'volume': float(kline.get('volume', 0)),
                                    'symbol': symbol
                                })
                        
                        if df_data:
                            logger.info(f"✅ 成功获取{symbol}K线数据: {len(df_data)}条")
                            return pd.DataFrame(df_data)
                    
                    else:
                        logger.warning(f"端点 {endpoint} 返回失败: {data.get('ErrMsg', '未知错误')}")
                
            except Exception as e:
                logger.warning(f"端点 {endpoint} 失败: {e}")
                continue
        
        logger.error(f"❌ 所有K线端点都失败了")
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
        self.rebalance_hours = 6
        self.top_n = 3
        self.momentum_periods = [3, 7, 14]
        self.min_trade_amount = 50
        
        # 数量精度配置（根据交易对调整）
        self.quantity_precision = {
            'BTCUSDT': 6,
            'ETHUSDT': 4,
            'ADAUSDT': 0,
            'DOTUSDT': 2,
            'BNBUSDT': 3
        }
        
        # ========== 数据记录 ==========
        self.buy_points = {symbol: [] for symbol in SYMBOLS}
        self.sell_points = {symbol: [] for symbol in SYMBOLS}
        self.price_history = {symbol: [] for symbol in SYMBOLS}
        self.last_rebalance = None
        
        logger.info("🎯 市场轮动策略初始化完成")

    def adjust_quantity_precision(self, symbol, quantity):
        """调整数量精度以避免step size错误"""
        precision = self.quantity_precision.get(symbol, 4)
        return round(quantity, precision)

    def calculate_momentum_score(self, df):
        """计算动量得分 - 添加回退逻辑"""
        if df is None or len(df) < 2:
            return 0
        
        try:
            # 确保有足够数据
            available_periods = []
            for period in self.momentum_periods:
                if len(df) >= period:
                    available_periods.append(period)
            
            if not available_periods:
                # 如果没有足够数据，使用可用数据计算
                if len(df) >= 2:
                    simple_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
                    volatility = df['close'].pct_change().std()
                    if volatility > 0:
                        return simple_return / volatility
                    return simple_return
                return 0
            
            momentum_scores = []
            for period in available_periods:
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
            
        except Exception as e:
            logger.error(f"计算动量得分时出错: {e}")
            return 0

    def calculate_risk_metrics(self):
        """计算风险调整指标"""
        if len(self.portfolio_value_history) < 2:
            return None
        
        try:
            portfolio_df = pd.DataFrame(self.portfolio_value_history)
            portfolio_df = portfolio_df.sort_values('timestamp')
            portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change().dropna()
            
            if len(portfolio_df['returns']) < 2:
                return None
            
            mean_return = portfolio_df['returns'].mean()
            total_return = (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0] - 1)
            
            # Sharpe Ratio
            std_dev = portfolio_df['returns'].std()
            sharpe = mean_return / std_dev if std_dev != 0 else 0
            
            # Sortino Ratio
            downside_returns = portfolio_df[portfolio_df['returns'] < 0]['returns']
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino = mean_return / downside_std if downside_std != 0 else 0
            
            # Calmar Ratio
            portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cummax']) / portfolio_df['cummax']
            max_drawdown = portfolio_df['drawdown'].min()
            calmar = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # 综合得分
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
        """获取所有交易对的当前价格 - 添加回退逻辑"""
        tickers = get_all_tickers()
        current_prices = {}
        
        # 如果API失败，使用模拟数据继续运行
        if not tickers:
            logger.warning("⚠️ 无法获取实时价格，使用模拟数据继续运行")
            # 生成模拟价格（基于初始假设）
            base_prices = {'BTCUSDT': 45000, 'ETHUSDT': 3000, 'ADAUSDT': 0.45, 'DOTUSDT': 6.5, 'BNBUSDT': 350}
            for symbol in SYMBOLS:
                # 添加一些随机波动
                change = np.random.normal(0, 0.01)
                current_prices[symbol] = base_prices.get(symbol, 100) * (1 + change)
                self.price_history[symbol].append({
                    'timestamp': datetime.now(),
                    'price': current_prices[symbol]
                })
        else:
            for symbol, ticker in tickers.items():
                if ticker:
                    current_prices[symbol] = ticker['last_price']
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
        """执行调仓操作 - 添加容错机制"""
        logger.info(f"🔄 开始调仓操作 - {datetime.now()}")
        
        # 1. 获取K线数据并计算动量得分
        momentum_scores = {}
        for symbol in SYMBOLS:
            # 尝试获取K线数据，如果失败使用简单方法
            df = get_kline_data(symbol, interval='1h', limit=50)  # 改为小时线，数据量更合适
            
            if df is not None and len(df) > 1:
                score = self.calculate_momentum_score(df)
                momentum_scores[symbol] = score
                logger.info(f"   📊 {symbol}: 动量得分 = {score:.4f}")
            else:
                # 如果K线数据获取失败，使用价格历史计算简单动量
                if self.price_history.get(symbol):
                    prices = [p['price'] for p in self.price_history[symbol][-10:]]  # 最近10个价格
                    if len(prices) >= 2:
                        returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
                        if returns:
                            mean_return = np.mean(returns)
                            volatility = np.std(returns) if len(returns) > 1 else 0.01
                            score = mean_return / volatility if volatility > 0 else mean_return
                            momentum_scores[symbol] = score
                            logger.info(f"   📊 {symbol}: 备用动量得分 = {score:.4f}")
                            continue
                
                # 如果所有方法都失败，使用随机得分
                momentum_scores[symbol] = np.random.normal(0, 0.1)
                logger.info(f"   📊 {symbol}: 随机动量得分 = {momentum_scores[symbol]:.4f}")
        
        # 2. 选择动量最强的币种
        top_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_n]
        selected_symbols = [s[0] for s in top_symbols]
        
        logger.info(f"   🏆 选中币种: {selected_symbols}")
        
        # 3. 获取当前价格
        current_prices = self.get_current_prices()
        if not current_prices:
            logger.error("   ❌ 无法获取当前价格，调仓中止")
            return
        
        # 4. 卖出不在选中列表的持仓
        symbols_to_sell = []
        for symbol in list(self.positions.keys()):
            if (self.positions[symbol] > 0 and 
                symbol not in selected_symbols and
                symbol in current_prices):
                symbols_to_sell.append(symbol)
        
        for symbol in symbols_to_sell:
            current_price = current_prices[symbol]
            quantity = self.positions[symbol]
            
            # 调整数量精度
            adjusted_quantity = self.adjust_quantity_precision(symbol, quantity)
            
            # 执行卖出订单
            if place_order(symbol, 'SELL', adjusted_quantity):
                sell_value = adjusted_quantity * current_price * 0.999
                self.cash += sell_value
                
                self.sell_points[symbol].append((datetime.now(), current_price))
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': adjusted_quantity,
                    'price': current_price,
                    'value': sell_value,
                    'reason': '调出轮动组合'
                })
                
                logger.info(f"   🔴 卖出 {symbol}: {adjusted_quantity:.6f} 单位 @ ${current_price:.2f}")
                self.positions[symbol] = 0
        
        # 5. 买入选中的币种
        if selected_symbols and self.cash > self.min_trade_amount:
            cash_per_symbol = self.cash / len(selected_symbols)
            
            for symbol in selected_symbols:
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    
                    # 跳过已有持仓
                    if self.positions.get(symbol, 0) > 0:
                        continue
                    
                    quantity = cash_per_symbol / current_price
                    adjusted_quantity = self.adjust_quantity_precision(symbol, quantity)
                    
                    # 确保数量大于0
                    if adjusted_quantity <= 0:
                        continue
                    
                    # 执行买入订单
                    if place_order(symbol, 'BUY', adjusted_quantity):
                        self.positions[symbol] = adjusted_quantity
                        self.cash -= cash_per_symbol * 0.999
                        
                        self.buy_points[symbol].append((datetime.now(), current_price))
                        self.trade_history.append({
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': adjusted_quantity,
                            'price': current_price,
                            'value': cash_per_symbol,
                            'reason': f'动量得分: {momentum_scores[symbol]:.4f}'
                        })
                        
                        logger.info(f"   🟢 买入 {symbol}: {adjusted_quantity:.6f} 单位 @ ${current_price:.2f}")
        
        self.last_rebalance = datetime.now()
        logger.info("   ✅ 调仓操作完成")
        
        # 监控性能
        self.monitor_performance()

    def monitor_performance(self):
        """实时监控策略表现"""
        metrics = self.calculate_risk_metrics()
        if metrics:
            logger.info(f"📊 实时表现:")
            logger.info(f"   总收益率: {metrics['total_return']*100:.2f}%")
            logger.info(f"   Sharpe: {metrics['sharpe_ratio']:.4f}")
            logger.info(f"   Sortino: {metrics['sortino_ratio']:.4f}") 
            logger.info(f"   Calmar: {metrics['calmar_ratio']:.4f}")
            logger.info(f"   最大回撤: {metrics['max_drawdown']*100:.2f}%")
            logger.info(f"   综合得分: {metrics['composite_score']:.4f}")

    def run_live_strategy(self, run_duration_hours=24):
        """运行实时策略"""
        logger.info(f"🚀 启动市场轮动策略")
        logger.info(f"⏰ 运行时长: {run_duration_hours}小时")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=run_duration_hours)
        
        # 初始调仓
        self.execute_rebalance()
        
        cycle_count = 0
        while datetime.now() < end_time:
            current_time = datetime.now()
            cycle_count += 1
            
            # 每6小时调仓一次
            if (self.last_rebalance is None or 
                (current_time - self.last_rebalance).total_seconds() >= self.rebalance_hours * 3600):
                
                logger.info(f"\n🔄 第{cycle_count}次调仓周期")
                self.execute_rebalance()
            
            # 记录组合价值
            current_prices = self.get_current_prices()
            if current_prices:
                portfolio_value = self.calculate_portfolio_value(current_prices)
                self.portfolio_value_history.append({
                    'timestamp': current_time,
                    'portfolio_value': portfolio_value
                })
                
                if cycle_count % 12 == 0:  # 每小时记录一次
                    logger.info(f"📈 组合价值: ${portfolio_value:.2f} | 现金: ${self.cash:.2f}")
            
            # 等待5分钟
            time.sleep(300)
        
        logger.info(f"\n✅ 策略运行完成")
        self.print_final_report()

    def print_final_report(self):
        """打印最终报告"""
        if not self.portfolio_value_history:
            logger.warning("⚠️ 无投资组合数据")
            return
            
        portfolio_df = pd.DataFrame(self.portfolio_value_history)
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        
        metrics = self.calculate_risk_metrics()
        
        print(f"\n" + "="*60)
        print("📊 最终报告")
        print("="*60)
        print(f"💰 初始资金: ${self.initial_cash:,.2f}")
        print(f"💰 最终价值: ${final_value:,.2f}")
        print(f"📈 总收益率: {total_return:.2f}%")
        
        if metrics:
            print(f"\n🏆 风险指标:")
            print(f"   Sharpe: {metrics['sharpe_ratio']:.4f}")
            print(f"   Sortino: {metrics['sortino_ratio']:.4f}")
            print(f"   Calmar: {metrics['calmar_ratio']:.4f}")
            print(f"   最大回撤: {metrics['max_drawdown']*100:.2f}%")
            print(f"   综合得分: {metrics['composite_score']:.4f}")
        
        print(f"\n📝 交易次数: {len(self.trade_history)}")
        print(f"📦 最终持仓:")
        print(f"   现金: ${self.cash:.2f}")
        current_prices = self.get_current_prices()
        for symbol in SYMBOLS:
            if self.positions.get(symbol, 0) > 0:
                value = self.positions[symbol] * current_prices.get(symbol, 0)
                print(f"   {symbol}: {self.positions[symbol]:.6f} 单位")

# ========== 主程序 ==========
def main():
    """主程序 - 简化版本用于测试"""
    print("🚀 Roostoo比赛策略 - 修复版本")
    print("="*50)
    
    # 测试API连接
    print("🔗 测试API连接...")
    balance = get_account_balance()
    if balance is None:
        print("❌ API连接失败，使用模拟模式运行")
        # 继续运行但使用模拟数据
    
    print("✅ 策略就绪")
    
    # 创建策略实例
    strategy = MarketRotationStrategy(initial_cash=10000)
    
    # 先运行24小时测试
    print("\n🎯 开始24小时测试运行...")
    strategy.run_live_strategy(run_duration_hours=24)

if __name__ == "__main__":
    main()