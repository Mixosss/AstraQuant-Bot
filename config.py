import logging
import os
import sys
from okx_engine import OKXExecutionEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TRADING_MODE = os.getenv('TRADING_MODE', 'REAL').upper()

ACCOUNTS = [
    {
        "name": "AstraQuant",
        "api_key": os.getenv('OKX_API_KEY_1', ''),
        "api_secret": os.getenv('OKX_API_SECRET_1', ''),
        "passphrase": os.getenv('OKX_API_PASSPHRASE_1', ''),
        "prompt_mode": "aggressive",
    }
]

def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _normalize_proxy_url(proxy_url):
    if not proxy_url or not str(proxy_url).strip():
        return None
    normalized = str(proxy_url).strip()
    if normalized.startswith('sock5://'):
        normalized = 'socks5://' + normalized[len('sock5://'):]
    return normalized


PROXY_URL = _normalize_proxy_url(os.getenv('PROXY_URL', ''))
OKX_USE_PROXY = _env_bool('OKX_USE_PROXY', False)
AI_USE_PROXY = _env_bool('AI_USE_PROXY', False)
OKX_PROXIES = {'http': PROXY_URL, 'https': PROXY_URL} if (PROXY_URL and OKX_USE_PROXY) else None
AI_PROXIES = {'http': PROXY_URL, 'https': PROXY_URL} if (PROXY_URL and AI_USE_PROXY) else None
API_TIMEOUT = int(os.getenv('API_TIMEOUT', 60))

AI_API_KEY = os.getenv('AI_API_KEY', '')
AI_API_URL = os.getenv('AI_API_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions')
AI_MODEL = os.getenv('AI_MODEL', 'deepseek-v3.1')
AI_MAX_RETRIES = int(os.getenv('AI_MAX_RETRIES', 3))
AI_CACHE_TTL = int(os.getenv('AI_CACHE_TTL', 3600))

ENABLE_QUANT_SCORING = str(os.getenv('ENABLE_QUANT_SCORING', 'True')).lower() == 'true'
MIN_OPEN_SCORE = float(os.getenv('MIN_OPEN_SCORE', 4.5))
MAX_OPEN_POSITIONS = 7

MAINSTREAM = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
STABLE_MAJORS = ['BTCUSDT', 'ETHUSDT']
OPPORTUNITY_MAJORS = ['SOLUSDT', 'XRPUSDT']
MID_TIER = []
MEME_SHIT = ['RIVERUSDT']
SYMBOLS = MID_TIER + MAINSTREAM

SYMBOL_CONFIG = {
    'MAINSTREAM': {
        'leverage': 15,
        'margin_ratio': 0.1,
        'risk_level': 2,
        'description': '主流强势币',
    },
    'MID_TIER': {
        'leverage': 10,
        'margin_ratio': 0.1,
        'risk_level': 6,
        'description': '中波段品种',
    },
    'MEME_SHIT': {
        'leverage': 5,
        'margin_ratio': 0.1,
        'risk_level': 7,
        'description': '高波动防守组',
    },
}

FUNDS_MODE = os.getenv('FUNDS_MODE', 'SHARED').upper()
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 10000.0))
FEE_RATE = float(os.getenv('FEE_RATE', 0.0004))
SLIPPAGE_RATE = float(os.getenv('SLIPPAGE_RATE', 0.0005))
RR_INCLUDE_FEES = str(os.getenv('RR_INCLUDE_FEES', 'True')).lower() == 'true'
RR_INCLUDE_SLIPPAGE = str(os.getenv('RR_INCLUDE_SLIPPAGE', 'True')).lower() == 'true'
RR_INCLUDE_FUNDING = str(os.getenv('RR_INCLUDE_FUNDING', 'True')).lower() == 'true'
FUNDING_HOLD_HOURS = float(os.getenv('FUNDING_HOLD_HOURS', 8.0))
MIN_NET_RR_HARD_FLOOR = float(os.getenv('MIN_NET_RR_HARD_FLOOR', 0.7))
SYMBOL_TRADING_GROUPS = {
    'btc_stable': ['BTCUSDT'],
    'eth_defensive': ['ETHUSDT'],
    'major_opportunity': OPPORTUNITY_MAJORS,
    'sol_opportunity': ['SOLUSDT'],
    'xrp_defensive': ['XRPUSDT'],
}
SYMBOL_TRADING_PARAMS = {
    'btc_stable': {
        'min_net_rr_floor': float(os.getenv('STABLE_MAJOR_MIN_NET_RR_FLOOR', 0.5)),
        'same_side_ai_confidence': float(os.getenv('STABLE_MAJOR_SAME_SIDE_AI_CONFIDENCE', 65)),
        'reverse_ai_confidence': float(os.getenv('STABLE_MAJOR_REVERSE_AI_CONFIDENCE', 74)),
        'position_scaler': float(os.getenv('STABLE_MAJOR_POSITION_SCALER', 0.9)),
        'm15_strong_signal_buffer': float(os.getenv('STABLE_MAJOR_M15_STRONG_SIGNAL_BUFFER', 0.8)),
        'm15_reduced_scaler': float(os.getenv('STABLE_MAJOR_M15_REDUCED_SCALER', 0.75)),
    },
    'eth_defensive': {
        'min_net_rr_floor': float(os.getenv('ETHUSDT_MIN_NET_RR_FLOOR', 0.6)),
        'same_side_ai_confidence': float(os.getenv('ETHUSDT_SAME_SIDE_AI_CONFIDENCE', 60)),
        'reverse_ai_confidence': float(os.getenv('ETHUSDT_REVERSE_AI_CONFIDENCE', 74)),
        'position_scaler': float(os.getenv('ETHUSDT_POSITION_SCALER', 0.6)),
        'm15_strong_signal_buffer': float(os.getenv('ETHUSDT_M15_STRONG_SIGNAL_BUFFER', 0.8)),
        'm15_reduced_scaler': float(os.getenv('ETHUSDT_M15_REDUCED_SCALER', 0.75)),
    },
    'major_opportunity': {
        'min_net_rr_floor': float(os.getenv('OPPORTUNITY_MAJOR_MIN_NET_RR_FLOOR', 0.45)),
        'same_side_ai_confidence': float(os.getenv('OPPORTUNITY_MAJOR_SAME_SIDE_AI_CONFIDENCE', 60)),
        'reverse_ai_confidence': float(os.getenv('OPPORTUNITY_MAJOR_REVERSE_AI_CONFIDENCE', 72)),
        'position_scaler': float(os.getenv('OPPORTUNITY_MAJOR_POSITION_SCALER', 0.75)),
        'm15_strong_signal_buffer': float(os.getenv('OPPORTUNITY_MAJOR_M15_STRONG_SIGNAL_BUFFER', 0.6)),
        'm15_reduced_scaler': float(os.getenv('OPPORTUNITY_MAJOR_M15_REDUCED_SCALER', 0.65)),
    },
    'sol_opportunity': {
        'min_net_rr_floor': float(os.getenv('SOLUSDT_MIN_NET_RR_FLOOR', 0.45)),
        'same_side_ai_confidence': float(os.getenv('SOLUSDT_SAME_SIDE_AI_CONFIDENCE', 60)),
        'reverse_ai_confidence': float(os.getenv('SOLUSDT_REVERSE_AI_CONFIDENCE', 72)),
        'position_scaler': float(os.getenv('SOLUSDT_POSITION_SCALER', 0.6)),
        'm15_strong_signal_buffer': float(os.getenv('SOLUSDT_M15_STRONG_SIGNAL_BUFFER', 0.7)),
        'm15_reduced_scaler': float(os.getenv('SOLUSDT_M15_REDUCED_SCALER', 0.6)),
    },
    'xrp_defensive': {
        'min_net_rr_floor': float(os.getenv('XRPUSDT_MIN_NET_RR_FLOOR', 0.5)),
        'same_side_ai_confidence': float(os.getenv('XRPUSDT_SAME_SIDE_AI_CONFIDENCE', 60)),
        'reverse_ai_confidence': float(os.getenv('XRPUSDT_REVERSE_AI_CONFIDENCE', 76)),
        'position_scaler': float(os.getenv('XRPUSDT_POSITION_SCALER', 0.5)),
        'm15_strong_signal_buffer': float(os.getenv('XRPUSDT_M15_STRONG_SIGNAL_BUFFER', 0.9)),
        'm15_reduced_scaler': float(os.getenv('XRPUSDT_M15_REDUCED_SCALER', 0.55)),
    },
}
PORTFOLIO_RISK_CLUSTERS = {
    'core': ['BTCUSDT', 'ETHUSDT'],
    'opportunity': ['SOLUSDT', 'XRPUSDT'],
}
MAX_ACCOUNT_EXPOSURE_RATIO = float(os.getenv('MAX_ACCOUNT_EXPOSURE_RATIO', 0.8))
MAX_SYMBOL_EXPOSURE_RATIO = float(os.getenv('MAX_SYMBOL_EXPOSURE_RATIO', 0.35))
MAX_DIRECTION_EXPOSURE_RATIO = float(os.getenv('MAX_DIRECTION_EXPOSURE_RATIO', 0.65))
MAX_CLUSTER_EXPOSURE_RATIO = float(os.getenv('MAX_CLUSTER_EXPOSURE_RATIO', 0.55))
MAX_CLUSTER_CONCENTRATION_RATIO = float(os.getenv('MAX_CLUSTER_CONCENTRATION_RATIO', 0.35))
MIN_OPEN_NOTIONAL_USDT = float(os.getenv('MIN_OPEN_NOTIONAL_USDT', 30.0))
MIN_OPEN_MARGIN_RATIO = float(os.getenv('MIN_OPEN_MARGIN_RATIO', 0.02))


MIN_OPEN_FLOOR_NET_RR = float(os.getenv('MIN_OPEN_FLOOR_NET_RR', 1.5))
SYMBOL_REENTRY_COOLDOWN_MINUTES = float(os.getenv('SYMBOL_REENTRY_COOLDOWN_MINUTES', 45))
SYMBOL_REENTRY_OVERRIDE_AI_CONFIDENCE = float(os.getenv('SYMBOL_REENTRY_OVERRIDE_AI_CONFIDENCE', 75))
SYMBOL_REENTRY_OVERRIDE_NET_RR = float(os.getenv('SYMBOL_REENTRY_OVERRIDE_NET_RR', 1.5))
SL_BUFFER_LONG = 0.995
SL_BUFFER_SHORT = 1.005
TAKE_PROFIT_RR = 2.0
SCAN_INTERVAL = 113
ENABLE_WEB_DASHBOARD = str(os.getenv('ENABLE_WEB_DASHBOARD', 'True')).lower() == 'true'
WEB_DASHBOARD_HOST = os.getenv('WEB_DASHBOARD_HOST', '127.0.0.1')
WEB_DASHBOARD_PORT = int(os.getenv('WEB_DASHBOARD_PORT', 5000))
AI_CACHE_TTL = 900
POSITION_AI_NORMAL_TTL = int(os.getenv('POSITION_AI_NORMAL_TTL', 3600))
POSITION_AI_RISK_TTL = int(os.getenv('POSITION_AI_RISK_TTL', 900))
POSITION_AI_CONFIRMED_RISK_TTL = int(os.getenv('POSITION_AI_CONFIRMED_RISK_TTL', 300))
POSITION_AI_ADVERSE_PNL_PCT = float(os.getenv('POSITION_AI_ADVERSE_PNL_PCT', 3.0))
POSITION_AI_PROFIT_PROTECT_MIN_PCT = float(os.getenv('POSITION_AI_PROFIT_PROTECT_MIN_PCT', 0.5))
POSITION_AI_PROFIT_DRAWDOWN_RATIO = float(os.getenv('POSITION_AI_PROFIT_DRAWDOWN_RATIO', 0.5))
POSITION_AI_STOP_BREACH_BUFFER = float(os.getenv('POSITION_AI_STOP_BREACH_BUFFER', 0.002))
ENABLE_POSITION_AI_EXECUTION = str(os.getenv('ENABLE_POSITION_AI_EXECUTION', 'False')).lower() == 'true'
POSITION_AI_BREAKEVEN_CONFIDENCE = float(os.getenv('POSITION_AI_BREAKEVEN_CONFIDENCE', 75))
POSITION_AI_REDUCE_CONFIDENCE = float(os.getenv('POSITION_AI_REDUCE_CONFIDENCE', 80))
POSITION_AI_EXIT_CONFIDENCE = float(os.getenv('POSITION_AI_EXIT_CONFIDENCE', 90))
POSITION_AI_REDUCE_RATIO = float(os.getenv('POSITION_AI_REDUCE_RATIO', 0.3))
REQUIRE_CLOSE_CONFIRMATION = str(os.getenv('REQUIRE_CLOSE_CONFIRMATION', 'False')).lower() == 'true'
CLOSE_CONFIRMATION_EXPIRE_BARS = int(os.getenv('CLOSE_CONFIRMATION_EXPIRE_BARS', 2))
CLOSE_TO_SUPPORT_RESISTANCE_RATIO = float(os.getenv('CLOSE_TO_SUPPORT_RESISTANCE_RATIO', 0.003))
CLOSE_TO_EMA200_RATIO = float(os.getenv('CLOSE_TO_EMA200_RATIO', 0.005))
REQUIRE_15M_CONFIRMATION = str(os.getenv('REQUIRE_15M_CONFIRMATION', 'True')).lower() == 'true'
M15_CONFIRM_EXPIRE_BARS = int(os.getenv('M15_CONFIRM_EXPIRE_BARS', 5))
M15_CONFIRM_STRONG_SIGNAL_BUFFER = float(os.getenv('M15_CONFIRM_STRONG_SIGNAL_BUFFER', 1.0))
M15_CONFIRM_REDUCED_SCALER = float(os.getenv('M15_CONFIRM_REDUCED_SCALER', 0.7))
LOSS_TRACK_WINDOW = int(os.getenv('LOSS_TRACK_WINDOW', 10))
ENABLE_LOSS_COOLDOWN = str(os.getenv('ENABLE_LOSS_COOLDOWN', 'False')).lower() == 'true'
LOSS_COOLDOWN_TRIGGER = int(os.getenv('LOSS_COOLDOWN_TRIGGER', 2))
LOSS_COOLDOWN_HOURS = float(os.getenv('LOSS_COOLDOWN_HOURS', 4))
DINGTALK_WEBHOOK = os.getenv('DINGTALK_WEBHOOK', '')
FEISHU_WEBHOOK = os.getenv('FEISHU_WEBHOOK', '')


def _validate_config():
    if TRADING_MODE in ['REAL', 'TESTNET']:
        valid_accounts = [
            acc for acc in ACCOUNTS
            if acc.get('api_key') and acc.get('api_secret') and acc.get('passphrase')
        ]
        if not valid_accounts:
            logger.error('当前为实盘/测试网模式，但 ACCOUNTS 中没有完整的 OKX key/secret/passphrase。')
            sys.exit(1)
        logger.info(f'账户校验通过：已加载 {len(valid_accounts)} 个 OKX 账户。')

    if not AI_API_KEY:
        logger.warning('未配置 AI_API_KEY，系统将退回纯算法执行。')
    if not DINGTALK_WEBHOOK and not FEISHU_WEBHOOK:
        logger.warning('未配置 DINGTALK_WEBHOOK 或 FEISHU_WEBHOOK，通知只会输出到本地日志。')

    proxy_scheme = (PROXY_URL.split('://', 1)[0].lower() if PROXY_URL and '://' in PROXY_URL else ('custom' if PROXY_URL else 'disabled'))
    logger.info(f'代理状态 | OKX: {"ON" if OKX_USE_PROXY and PROXY_URL else "OFF"} | AI: {"ON" if AI_USE_PROXY and PROXY_URL else "OFF"} | 地址: {PROXY_URL or "未配置"} | 协议: {proxy_scheme}')
    if PROXY_URL and proxy_scheme.startswith('socks'):
        logger.info('已检测到 SOCKS 代理地址，请确认本机已安装 requests 的 socks 支持（如 PySocks）。')


_validate_config()


