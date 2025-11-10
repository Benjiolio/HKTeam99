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
SECRET_KEY = "94WfpKd5PHng5u2ySWvZW0URKxZxofI5rON3MJ0CURKgz4gKj1vxI8HZmvugrOt4U"
BASE_URL = "https://mock-api.roostoo.com"

# ========== 交易对配置 ==========
SYMBOLS = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'BNB/USD']

# ========== API工具函数 ==========
def get_timestamp():
    """获取13位毫秒时间戳"""
    return str(int(time.time() * 1000))

def generate_signature(params):
    """
    根据Roostoo API文档生成HMAC SHA256签名
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

def get_open_orders(pair=None):
    """获取当前挂单"""
    try:
        params = {}
        if pair:
            params['pair'] = pair
            
        headers, params = get_signed_headers(params)
        if not headers:
            return None
            
        response = requests.get(f"{BASE_URL}/v3/open_orders", headers=headers, params=params, timeout=10)
        data = response.json()
        
        if data.get('Success'):
            return data.get('Orders', [])
        else:
            logger.warning(f"⚠️ 获取挂单失败: {data.get('ErrMsg', '未知错误')}")
            return []
            
    except Exception as e:
        logger.error(f"❌ 获取挂单时发生异常: {e}")
        return []

# ========== 持仓监控类 ==========
class PortfolioMonitor:
    """持仓监控类 - 每10秒更新持仓情况"""
    
    def __init__(self, initial_cash=10000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.portfolio_value_history = []
        self.update_count = 0
        
        # 数据记录
        self.price_history = {pair: [] for pair in SYMBOLS}
        
        # 加载交易所信息
        self.exchange_info = get_exchange_info()
        if self.exchange_info:
            logger.info("✅ 已加载交易对精度信息")
        else:
            logger.warning("⚠️ 无法获取交易对精度信息，使用默认值")
        
        logger.info("📊 持仓监控器初始化完成")

    def get_current_prices(self):
        """获取所有交易对的当前价格"""
        tickers = get_all_tickers()
        current_prices = {}
        
        for pair, ticker in tickers.items():
            if ticker:
                current_prices[pair] = ticker['last_price']
                # 记录价格历史（只保留最近100条）
                self.price_history[pair].append({
                    'timestamp': ticker['timestamp'],
                    'price': ticker['last_price']
                })
                if len(self.price_history[pair]) > 100:
                    self.price_history[pair].pop(0)
        
        return current_prices

    def update_portfolio_status(self):
        """更新持仓状态"""
        self.update_count += 1
        current_time = datetime.now()
        
        logger.info(f"\n🔄 第{self.update_count}次持仓更新 - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 获取当前价格
        current_prices = self.get_current_prices()
        if not current_prices:
            logger.error("❌ 无法获取当前价格，跳过本次更新")
            return
        
        # 2. 获取账户余额和持仓（模拟或实际API）
        account_balance = get_account_balance()
        if account_balance:
            # 更新现金余额（这里需要根据实际API响应结构调整）
            usd_balance = account_balance.get('USD', {}).get('Free', self.cash)
            self.cash = usd_balance
            logger.info(f"💰 现金余额: ${self.cash:.2f}")
        
        # 3. 计算持仓价值
        total_position_value = 0
        position_details = []
        
        for pair in SYMBOLS:
            if pair in current_prices:
                price = current_prices[pair]
                
                # 模拟持仓数据（实际中应从API获取）
                # 这里使用简单的模拟持仓逻辑
                if pair not in self.positions:
                    # 初始分配一些持仓用于演示
                    if pair == 'BTC/USD':
                        self.positions[pair] = 0.1
                    elif pair == 'ETH/USD':
                        self.positions[pair] = 1.0
                    else:
                        self.positions[pair] = 100.0
                
                position_value = self.positions[pair] * price
                total_position_value += position_value
                
                # 计算涨跌幅（如果有历史价格）
                change_percent = 0
                if len(self.price_history[pair]) > 1:
                    prev_price = self.price_history[pair][-2]['price'] if len(self.price_history[pair]) > 1 else price
                    change_percent = (price - prev_price) / prev_price * 100
                
                position_details.append({
                    'pair': pair,
                    'quantity': self.positions[pair],
                    'price': price,
                    'value': position_value,
                    'change': change_percent
                })
                
                logger.info(f"   {pair}: {self.positions[pair]:.6f} × ${price:.2f} = ${position_value:.2f} ({change_percent:+.2f}%)")
        
        # 4. 计算总投资组合价值
        total_portfolio_value = self.cash + total_position_value
        total_return = (total_portfolio_value - self.initial_cash) / self.initial_cash * 100
        
        logger.info(f"📊 持仓总价值: ${total_position_value:.2f}")
        logger.info(f"💵 投资组合总值: ${total_portfolio_value:.2f}")
        logger.info(f"📈 总收益率: {total_return:+.2f}%")
        
        # 5. 记录投资组合价值历史
        self.portfolio_value_history.append({
            'timestamp': current_time,
            'portfolio_value': total_portfolio_value,
            'cash': self.cash,
            'positions_value': total_position_value
        })
        
        # 6. 检查挂单情况
        open_orders = get_open_orders()
        if open_orders:
            logger.info(f"📋 当前挂单数量: {len(open_orders)}")
            for order in open_orders[:3]:  # 只显示前3个挂单
                logger.info(f"   - {order.get('Side', 'UNKNOWN')} {order.get('Quantity', 0)} {order.get('Pair', 'UNKNOWN')} @ ${order.get('Price', 0):.2f}")
        else:
            logger.info("📋 当前无挂单")
        
        # 7. 定期生成简要报告（每10次更新）
        if self.update_count % 10 == 0:
            self.generate_summary_report()

    def generate_summary_report(self):
        """生成简要报告"""
        if len(self.portfolio_value_history) < 2:
            return
        
        logger.info(f"\n{'='*50}")
        logger.info("📈 持仓监控简要报告")
        logger.info(f"{'='*50}")
        
        # 计算期间收益
        start_value = self.portfolio_value_history[0]['portfolio_value']
        current_value = self.portfolio_value_history[-1]['portfolio_value']
        period_return = (current_value - start_value) / start_value * 100
        
        logger.info(f"监控开始时间: {self.portfolio_value_history[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"当前时间: {self.portfolio_value_history[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"期间收益率: {period_return:+.2f}%")
        logger.info(f"总更新次数: {self.update_count}")
        logger.info(f"{'='*50}")

    def run_monitoring(self, run_duration_hours=24):
        """运行持仓监控"""
        logger.info(f"🚀 启动持仓监控系统")
        logger.info(f"⏰ 运行时长: {run_duration_hours} 小时")
        logger.info(f"📊 监控币种: {SYMBOLS}")
        logger.info(f"🔄 更新频率: 每 10 秒")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=run_duration_hours)
        
        # 初始更新
        self.update_portfolio_status()
        
        while datetime.now() < end_time:
            try:
                # 等待10秒
                time.sleep(10)
                
                # 执行持仓更新
                self.update_portfolio_status()
                
            except KeyboardInterrupt:
                logger.info("⏹️ 用户中断监控")
                break
            except Exception as e:
                logger.error(f"❌ 监控过程中发生异常: {e}")
                # 继续运行，不要因为一次异常而停止
        
        logger.info(f"\n✅ 持仓监控完成")
        self.print_final_report()

    def print_final_report(self):
        """打印最终报告"""
        if not self.portfolio_value_history:
            logger.warning("⚠️ 无投资组合历史数据")
            return
            
        final_record = self.portfolio_value_history[-1]
        final_value = final_record['portfolio_value']
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        
        print(f"\n" + "="*60)
        print("📊 持仓监控最终报告")
        print("="*60)
        print(f"💰 初始资金: ${self.initial_cash:,.2f}")
        print(f"💰 最终价值: ${final_value:,.2f}")
        print(f"📈 总收益率: {total_return:+.2f}%")
        print(f"🔢 总更新次数: {self.update_count}")
        print(f"⏰ 运行时长: {len(self.portfolio_value_history) * 10 / 3600:.2f} 小时")
        
        print(f"\n📦 最终持仓详情:")
        print(f"   现金: ${self.cash:.2f}")
        
        current_prices = self.get_current_prices()
        for pair in SYMBOLS:
            if self.positions.get(pair, 0) > 0 and pair in current_prices:
                value = self.positions[pair] * current_prices[pair]
                print(f"   {pair}: {self.positions[pair]:.6f} 单位, 价值: ${value:.2f}")

# ========== 主程序 ==========
def main():
    """主程序"""
    print("🚀 Roostoo Hackathon - 持仓监控系统")
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
    
    # 创建监控实例
    monitor = PortfolioMonitor(initial_cash=initial_cash)
    
    # 运行监控
    print("\n🎯 开始持仓监控...")
    monitor.run_monitoring(run_duration_hours=24)

if __name__ == "__main__":
    main()