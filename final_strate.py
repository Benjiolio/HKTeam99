import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import hmac
import hashlib
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
# 根据文档使用正确的交易对格式
SYMBOLS = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'BNB/USD']

# ========== API工具函数 ==========
def get_timestamp():
    """获取13位毫秒时间戳"""
    return str(int(time.time() * 1000))

def generate_signature(params):
    """
    根据Roostoo API文档生成HMAC SHA256签名
    严格按照文档要求：参数排序后连接，使用secret_key作为HMAC密钥
    """
    try:
        # 参数按key排序后连接成字符串
        sorted_params = sorted(params.items())
        query_string = '&'.join([f"{key}={urllib.parse.quote(str(value))}" for key, value in sorted_params])
        
        logger.debug(f"签名原始字符串: {query_string}")
        
        # 使用HMAC SHA256生成签名
        signature = hmac.new(
            SECRET_KEY.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        logger.debug(f"生成签名: {signature}")
        return signature
        
    except Exception as e:
        logger.error(f"生成签名失败: {e}")
        return None

def get_signed_headers(params):
    """生成签名请求头"""
    timestamp = get_timestamp()
    params['timestamp'] = timestamp
    
    signature = generate_signature(params)
    if not signature:
        return None, None
    
    headers = {
        'RST-API-KEY': API_KEY,
        'MSG-SIGNATURE': signature,
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    return headers, params

def get_exchange_info():
    """获取交易所信息 - 用于获取交易对精度"""
    try:
        url = f"{BASE_URL}/v3/exchangeInfo"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get('IsRunning', False):
            logger.info("✅ 成功获取交易所信息")
            return data.get('TradePairs', {})
        else:
            logger.error("❌ 获取交易所信息失败")
            return None
            
    except Exception as e:
        logger.error(f"❌ 获取交易所信息时发生异常: {e}")
        return None

def get_account_balance():
    """获取账户余额信息"""
    try:
        headers, params = get_signed_headers({})
        if not headers:
            return None
            
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

def get_realtime_ticker(pair):
    """获取单个交易对的实时行情"""
    try:
        params = {'pair': pair}
        headers, params = get_signed_headers(params)
        if not headers:
            return None
            
        response = requests.get(f"{BASE_URL}/v3/ticker", headers=headers, params=params, timeout=10)
        data = response.json()
        
        if data.get('Success'):
            ticker_data = data.get('Data', {}).get(pair, {})
            return {
                'pair': pair,
                'last_price': float(ticker_data.get('LastPrice', 0)),
                'change': float(ticker_data.get('Change', 0)),
                'volume': float(ticker_data.get('UnitTradeValue', 0)),
                'timestamp': datetime.now()
            }
        else:
            logger.warning(f"⚠️ 获取{pair}行情失败: {data.get('ErrMsg', '未知错误')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ 获取{pair}行情时发生异常: {e}")
        return None

def get_all_tickers():
    """获取所有交易对的实时行情"""
    tickers = {}
    for pair in SYMBOLS:
        ticker_data = get_realtime_ticker(pair)
        if ticker_data:
            tickers[pair] = ticker_data
        time.sleep(0.1)  # 避免请求过于频繁
    return tickers

def place_order(pair, side, quantity, order_type="MARKET", price=None):
    """下订单 - 严格按照文档格式"""
    try:
        # 构建参数
        params = {
            'pair': pair,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': str(quantity)
        }
        
        # LIMIT订单需要价格参数
        if order_type.upper() == "LIMIT" and price is not None:
            params['price'] = str(price)
        
        headers, params = get_signed_headers(params)
        if not headers:
            return False
            
        # 使用data参数发送POST请求，按照文档要求
        sorted_params = sorted(params.items())
        query_string = '&'.join([f"{key}={urllib.parse.quote(str(value))}" for key, value in sorted_params])
        
        response = requests.post(f"{BASE_URL}/v3/place_order", headers=headers, data=query_string, timeout=10)
        data = response.json()
        
        if data.get('Success'):
            order_detail = data.get('OrderDetail', {})
            logger.info(f"✅ 订单提交成功: {side} {quantity} {pair} - 状态: {order_detail.get('Status', 'UNKNOWN')}")
            return True
        else:
            logger.error(f"❌ 订单提交失败: {data.get('ErrMsg', '未知错误')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 下单时发生异常: {e}")
        return False

def get_market_data(pair, days=30):
    """
    获取市场数据用于动量计算
    由于文档中没有K线接口，我们使用ticker数据模拟历史数据
    """
    try:
        # 获取当前ticker数据
        ticker = get_realtime_ticker(pair)
        if not ticker:
            return None
        
        # 模拟生成历史数据（在实际比赛中可能需要使用Horus数据或其他数据源）
        base_price = ticker['last_price']
        dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]
        
        # 生成模拟价格数据（带随机波动）
        prices = [base_price]
        for i in range(1, days):
            change = np.random.normal(0, 0.02)  # 2%的日波动
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # 创建DataFrame
        df_data = []
        for i, date in enumerate(dates):
            df_data.append({
                'date': date,
                'open': prices[i] * (1 + np.random.normal(0, 0.005)),
                'high': prices[i] * (1 + abs(np.random.normal(0, 0.01))),
                'low': prices[i] * (1 - abs(np.random.normal(0, 0.01))),
                'close': prices[i],
                'volume': np.random.normal(1000000, 200000),
                'pair': pair
            })
        
        return pd.DataFrame(df_data)
        
    except Exception as e:
        logger.error(f"❌ 获取市场数据时发生异常: {e}")
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
        self.momentum_periods = [7, 14, 30]
        self.min_trade_amount = 10
        
        # ========== 交易对精度信息 ==========
        self.exchange_info = None
        self.load_exchange_info()
        
        # ========== 数据记录 ==========
        self.buy_points = {pair: [] for pair in SYMBOLS}
        self.sell_points = {pair: [] for pair in SYMBOLS}
        self.price_history = {pair: [] for pair in SYMBOLS}
        self.last_rebalance = None
        self.initial_prices = {}  # 用于盈亏计算
        
        logger.info("🎯 市场轮动策略初始化完成")

    def load_exchange_info(self):
        """加载交易所信息，获取交易对精度"""
        self.exchange_info = get_exchange_info()
        if self.exchange_info:
            logger.info("✅ 已加载交易对精度信息")
        else:
            logger.warning("⚠️ 无法获取交易对精度信息，使用默认值")

    def get_amount_precision(self, pair):
        """获取交易对的数量精度"""
        if self.exchange_info and pair in self.exchange_info:
            return self.exchange_info[pair].get('AmountPrecision', 4)
        return 4  # 默认精度

    def adjust_quantity_precision(self, pair, quantity):
        """调整数量精度"""
        precision = self.get_amount_precision(pair)
        return round(quantity, precision)

    def calculate_momentum_score(self, df):
        """计算动量得分"""
        if df is None or len(df) < max(self.momentum_periods):
            return 0
        
        try:
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
            
            # Sortino Ratio (下行风险调整)
            downside_returns = portfolio_df[portfolio_df['returns'] < 0]['returns']
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino = mean_return / downside_std if downside_std != 0 else 0
            
            # Calmar Ratio (最大回撤调整)
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
        
        for pair, ticker in tickers.items():
            if ticker:
                current_prices[pair] = ticker['last_price']
                self.price_history[pair].append({
                    'timestamp': ticker['timestamp'],
                    'price': ticker['last_price']
                })
                
                # 记录初始价格用于盈亏计算
                if pair not in self.initial_prices:
                    self.initial_prices[pair] = ticker['last_price']
        
        return current_prices

    def calculate_portfolio_value(self, current_prices):
        """计算当前投资组合总价值"""
        total_value = self.cash
        for pair, quantity in self.positions.items():
            if quantity > 0 and pair in current_prices:
                total_value += quantity * current_prices[pair]
        return total_value

    def display_portfolio_status(self):
        """每10秒显示最新持仓信息"""
        current_time = datetime.now()
        
        print(f"\n🔄 持仓更新 - {current_time.strftime('%H:%M:%S')}")
        print("=" * 60)
        
        # 获取当前价格
        current_prices = self.get_current_prices()
        if not current_prices:
            print("❌ 无法获取价格数据")
            return
        
        # 计算持仓盈亏
        total_position_value = 0
        total_pnl = 0
        
        print(f"💰 现金: ${self.cash:.2f}")
        print("-" * 60)
        print(f"{'币种':<10} {'持仓量':<12} {'当前价':<10} {'市值':<12} {'盈亏($)':<12} {'盈亏(%)':<10}")
        print("-" * 60)
        
        for pair in SYMBOLS:
            if pair in current_prices and pair in self.positions:
                price = current_prices[pair]
                quantity = self.positions[pair]
                value = quantity * price
                total_position_value += value
                
                # 计算盈亏
                if pair in self.initial_prices:
                    initial_price = self.initial_prices[pair]
                    pnl = (price - initial_price) * quantity
                    pnl_percent = (price - initial_price) / initial_price * 100
                    total_pnl += pnl
                else:
                    pnl = 0
                    pnl_percent = 0
                
                # 颜色标记盈亏
                pnl_color = "🟢" if pnl >= 0 else "🔴"
                pnl_percent_color = "🟢" if pnl_percent >= 0 else "🔴"
                
                print(f"{pair:<10} {quantity:<12.6f} ${price:<9.2f} ${value:<11.2f} "
                      f"{pnl_color} ${pnl:<9.2f} {pnl_percent_color} {pnl_percent:<8.2f}%")
            elif pair in current_prices:
                # 显示无持仓的币种价格
                price = current_prices[pair]
                print(f"{pair:<10} {'0':<12} ${price:<9.2f} {'$0':<11} {'-':<12} {'-':<10}")
        
        # 计算总投资组合
        total_portfolio_value = self.cash + total_position_value
        total_return = (total_portfolio_value - self.initial_cash) / self.initial_cash * 100
        
        print("-" * 60)
        print(f"📊 持仓总值: ${total_position_value:.2f}")
        print(f"💵 组合总值: ${total_portfolio_value:.2f}")
        print(f"📈 总盈亏: ${total_pnl:.2f} ({total_return:+.2f}%)")
        print("=" * 60)

    def execute_rebalance(self):
        """执行调仓操作"""
        logger.info(f"🔄 开始调仓操作 - {datetime.now()}")
        
        # 1. 获取市场数据并计算动量得分
        momentum_scores = {}
        for pair in SYMBOLS:
            df = get_market_data(pair, days=30)
            if df is not None and len(df) > 0:
                score = self.calculate_momentum_score(df)
                momentum_scores[pair] = score
                logger.info(f"   📊 {pair}: 动量得分 = {score:.4f}")
            else:
                # 如果无法获取数据，使用ticker的变化率作为简单动量
                ticker = get_realtime_ticker(pair)
                if ticker:
                    momentum_scores[pair] = ticker['change']
                    logger.info(f"   📊 {pair}: 使用变化率作为动量 = {ticker['change']:.4f}")
                else:
                    momentum_scores[pair] = 0
                    logger.warning(f"   ⚠️  {pair}: 无法计算动量得分")
        
        # 2. 选择动量最强的top_n个币种
        top_pairs = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_n]
        selected_pairs = [s[0] for s in top_pairs]
        
        logger.info(f"   🏆 选中币种: {selected_pairs}")
        
        # 3. 获取当前价格
        current_prices = self.get_current_prices()
        if not current_prices:
            logger.error("   ❌ 无法获取当前价格，调仓中止")
            return
        
        # 4. 卖出不在选中列表的持仓
        pairs_to_sell = []
        for pair in list(self.positions.keys()):
            if (self.positions[pair] > 0 and 
                pair not in selected_pairs and
                pair in current_prices):
                pairs_to_sell.append(pair)
        
        for pair in pairs_to_sell:
            current_price = current_prices[pair]
            quantity = self.positions[pair]
            
            # 调整数量精度
            adjusted_quantity = self.adjust_quantity_precision(pair, quantity)
            
            # 执行卖出订单
            if place_order(pair, 'SELL', adjusted_quantity):
                # 计算实际交易价值（考虑手续费）
                sell_value = adjusted_quantity * current_price * 0.999
                self.cash += sell_value
                
                self.sell_points[pair].append((datetime.now(), current_price))
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'pair': pair,
                    'action': 'SELL',
                    'quantity': adjusted_quantity,
                    'price': current_price,
                    'value': sell_value,
                    'reason': '调出轮动组合'
                })
                
                logger.info(f"   🔴 卖出 {pair}: {adjusted_quantity:.6f} 单位 @ ${current_price:.2f}")
                self.positions[pair] = 0
        
        # 5. 买入选中的币种
        if selected_pairs and self.cash > self.min_trade_amount:
            cash_per_pair = self.cash / len(selected_pairs)
            
            for pair in selected_pairs:
                if pair in current_prices:
                    current_price = current_prices[pair]
                    
                    # 跳过已有持仓
                    if self.positions.get(pair, 0) > 0:
                        continue
                    
                    quantity = cash_per_pair / current_price
                    adjusted_quantity = self.adjust_quantity_precision(pair, quantity)
                    
                    # 确保数量大于0
                    if adjusted_quantity <= 0:
                        continue
                    
                    # 执行买入订单
                    if place_order(pair, 'BUY', adjusted_quantity):
                        self.positions[pair] = adjusted_quantity
                        self.cash -= cash_per_pair * 0.999  # 考虑手续费
                        
                        self.buy_points[pair].append((datetime.now(), current_price))
                        self.trade_history.append({
                            'timestamp': datetime.now(),
                            'pair': pair,
                            'action': 'BUY',
                            'quantity': adjusted_quantity,
                            'price': current_price,
                            'value': cash_per_pair,
                            'reason': f'动量得分: {momentum_scores[pair]:.4f}'
                        })
                        
                        logger.info(f"   🟢 买入 {pair}: {adjusted_quantity:.6f} 单位 @ ${current_price:.2f}")
        
        self.last_rebalance = datetime.now()
        logger.info("   ✅ 调仓操作完成")
        
        # 监控性能
        self.monitor_performance()

    def monitor_performance(self):
        """实时监控策略表现"""
        metrics = self.calculate_risk_metrics()
        if metrics:
            logger.info(f"📊 实时表现监控:")
            logger.info(f"   总收益率: {metrics['total_return']*100:.2f}%")
            logger.info(f"   Sharpe比率: {metrics['sharpe_ratio']:.4f}")
            logger.info(f"   Sortino比率: {metrics['sortino_ratio']:.4f}") 
            logger.info(f"   Calmar比率: {metrics['calmar_ratio']:.4f}")
            logger.info(f"   最大回撤: {metrics['max_drawdown']*100:.2f}%")
            logger.info(f"   综合得分: {metrics['composite_score']:.4f}")

    def run_live_strategy(self, run_duration_hours=24):
        """运行实时策略"""
        logger.info(f"🚀 启动实时市场轮动策略")
        logger.info(f"⏰ 运行时长: {run_duration_hours} 小时")
        logger.info(f"📊 监控币种: {SYMBOLS}")
        logger.info(f"🔄 调仓频率: 每 {self.rebalance_hours} 小时")
        logger.info(f"🎯 持仓数量: 前 {self.top_n} 个币种")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=run_duration_hours)
        
        # 初始调仓
        self.execute_rebalance()
        
        cycle_count = 0
        status_count = 0
        while datetime.now() < end_time:
            current_time = datetime.now()
            cycle_count += 1
            status_count += 1
            
            # 每10秒显示一次持仓状态
            if status_count % 2 == 0:  # 每2个循环（10秒）显示一次
                self.display_portfolio_status()
            
            # 检查是否到达调仓时间
            if (self.last_rebalance is None or 
                (current_time - self.last_rebalance).total_seconds() >= self.rebalance_hours * 3600):
                
                logger.info(f"\n🔄 第{cycle_count}次调仓周期")
                self.execute_rebalance()
            
            # 记录投资组合价值
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
            time.sleep(5)  # 改为5秒以便更频繁地显示状态
        
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
        for pair in SYMBOLS:
            if self.positions.get(pair, 0) > 0 and pair in current_prices:
                value = self.positions[pair] * current_prices[pair]
                print(f"   {pair}: {self.positions[pair]:.6f} 单位, 价值: ${value:.2f}")

# ========== 主程序 ==========
def main():
    """主程序"""
    print("🚀 Roostoo Hackathon - 市场轮动策略")
    print("="*50)
    
    # 检查API配置
    if not API_KEY or not SECRET_KEY:
        print("❌ 请配置API密钥和Secret Key")
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
    
    # 创建策略实例
    strategy = MarketRotationStrategy(initial_cash=initial_cash)
    
    # 先运行24小时测试
    print("\n🎯 开始24小时测试运行...")
    strategy.run_live_strategy(run_duration_hours=24)

if __name__ == "__main__":
    main()