import json
import logging
import os
import re
import time
from datetime import datetime

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)


class ToolKit:
    @staticmethod
    def _notifications_silenced():
        return bool(getattr(config, 'BACKTEST_SILENT_NOTIFICATIONS', False))

    @staticmethod
    def _display_market_regime(value):
        labels = {
            'strong_trend': '强趋势',
            'weak_trend': '弱趋势',
            'high_vol_range': '高波动震荡',
            'low_vol_chop': '低波动震荡',
            'event_volatility': '事件波动',
            'event_driven': '事件驱动',
            'low_liquidity': '低流动性',
        }
        key = str(value or '').strip()
        return labels.get(key, key or '未分类')

    @staticmethod
    def _display_strategy_bucket(value):
        labels = {
            'trend_continuation': '趋势延续',
            'pullback_continuation': '趋势回调延续',
            'mean_reversion': '均值回归',
            'reversal_probe': '反转试探',
        }
        key = str(value or '').strip()
        return labels.get(key, key or '未分类')

    @staticmethod
    def _rr_quality(net_rr, symbol=''):
        try:
            rr_value = float(net_rr or 0.0)
        except Exception:
            rr_value = 0.0
        if rr_value < 0.5:
            if symbol == 'BTCUSDT':
                return '偏低，BTC不允许低RR高分放行'
            return '偏低，仅适合特殊小仓或过滤'
        if rr_value < 1.0:
            return '一般，适合谨慎仓位'
        if rr_value < 1.5:
            return '尚可，可按普通仓位评估'
        return '健康，其他条件同步通过时可正常开仓'

    @staticmethod
    def _build_trade_logic_summary(result):
        symbol = result.get('symbol', '')
        market_regime = ToolKit._display_market_regime(result.get('market_regime'))
        strategy_bucket = ToolKit._display_strategy_bucket(result.get('strategy_bucket'))
        ai_entry_label = str(result.get('ai_entry_label') or '').strip() or '未标记'
        net_rr = result.get('net_expected_rr', result.get('net_rr', 0.0))
        rr_quality = ToolKit._rr_quality(net_rr, symbol=symbol)
        reasons = []
        if '谨慎' in ai_entry_label or '边缘' in ai_entry_label:
            reasons.append('AI谨慎')
        try:
            if float(net_rr or 0.0) < 1.0:
                reasons.append('RR偏低')
        except Exception:
            pass
        raw_regime = str(result.get('market_regime') or '')
        if raw_regime in {'high_vol_range', 'event_volatility', 'low_liquidity'}:
            reasons.append('波动/流动性风险较高')
        if float(result.get('m15_entry_multiplier') or 1.0) < 1.0:
            reasons.append('15M确认缩仓')
        position_reason = ' + '.join(reasons) if reasons else '信号质量与风控约束正常'
        risk_notes = []
        if raw_regime == 'high_vol_range':
            risk_notes.append('高波动震荡中可能出现二次扫损')
        elif raw_regime == 'low_vol_chop':
            risk_notes.append('低波动震荡中空间不足，手续费磨损较明显')
        elif raw_regime in {'event_volatility', 'event_driven'}:
            risk_notes.append('事件波动下技术信号稳定性下降')
        elif raw_regime == 'strong_trend' and result.get('strategy_bucket') in {'reversal_probe', 'mean_reversion'}:
            risk_notes.append('强趋势中逆势试探失败率较高')
        if float(net_rr or 0.0) < 1.0:
            risk_notes.append('净RR未达到健康区间')
        risk_text = '；'.join(risk_notes) if risk_notes else '主要关注止损执行与信号延续性'
        return (
            f"**交易逻辑摘要**\n"
            f"- **交易类型**: {market_regime} / {strategy_bucket}\n"
            f"- **开仓性质**: {ai_entry_label}\n"
            f"- **RR评价**: 净RR {float(net_rr or 0.0):.2f}，{rr_quality}\n"
            f"- **仓位原因**: {position_reason}\n"
            f"- **主要风险**: {risk_text}\n\n"
        )

    @staticmethod
    def format_dingtalk_card(result, mode, enable_ai):
        symbol = result.get('symbol', 'UNKNOWN')
        base_symbol = symbol.replace('USDT', '')
        price = result.get('price', 0.0)
        signal_text = result.get('signal', '未知信号')
        current_time = datetime.now().strftime('%H:%M:%S')
        is_scoring_enabled = getattr(config, 'ENABLE_QUANT_SCORING', False)

        ai_act = result.get('ai_action', '未启用')
        if ai_act == 'LONG':
            ai_act_display = '<font color="#28a745">LONG</font>'
        elif ai_act == 'SHORT':
            ai_act_display = '<font color="#dc3545">SHORT</font>'
        else:
            ai_act_display = ai_act

        ai_header = f" | AI判定: **{ai_act_display}**" if enable_ai else " | 纯算法执行"
        ai_reason = result.get('ai_reason', '算法直接执行')
        ai_insight = (
            f"洞察: {ai_reason}\n"
            f"核心价位:\n"
            f"- **[1H]** 支撑 {result.get('support_1h', 'N/A')} | 阻力 {result.get('resistance_1h', 'N/A')}\n"
            f"- **[4H]** 支撑 {result.get('support_4h', 'N/A')} | 阻力 {result.get('resistance_4h', 'N/A')}\n"
            f"- **[1D]** 支撑 {result.get('support_1d', 'N/A')} | 阻力 {result.get('resistance_1d', 'N/A')}"
        )
        ai_entry_label = result.get('ai_entry_label')
        if ai_entry_label:
            ai_insight = f"AI执行级别: {ai_entry_label}\n" + ai_insight

        trade_exec_info = ""
        if result.get('is_trade_triggered'):
            side = result.get('trade_side')
            tp_val = result.get('trade_tp', 0.0)
            sl_val = result.get('trade_sl', 0.0)
            margin_val = result.get('trade_margin', 0.0)
            total_balance = result.get('current_balance', 0.0)
            actual_leverage = result.get('actual_leverage', 20)
            real_margin_pct = (margin_val / total_balance) * 100 if total_balance > 0 else 0.0
            real_effective_leverage = (margin_val * actual_leverage) / total_balance if total_balance > 0 else 0.0
            side_color = '<font color="#28a745">LONG</font>' if side == 'LONG' else '<font color="#dc3545">SHORT</font>'
            trade_exec_info = (
                f"\n\n**[{mode}] 执行开仓报告**\n"
                f"- **开仓方向**: {side_color} | **触发价**: **{price:.5f}**\n"
                f"- **止盈/止损**: 目标 {tp_val:.5f} | 防守 {sl_val:.5f}\n"
                f"- **资金分配**: 投入 {margin_val:.2f} U (占总仓 {real_margin_pct:.1f}%)\n"
                f"- **当前杠杆**: {actual_leverage}x\n"
                f"- **全仓净杠杆**: **{real_effective_leverage:.2f} x**"
            )

        if is_scoring_enabled:
            strategy_content = (
                f"**量化打分系统**\n"
                f"- **最终评分**: 多头 **{result.get('long_score', 'N/A')}** | 空头 **{result.get('short_score', 'N/A')}**\n"
                f"- **大级别趋势**: {str(result.get('trend_state', 'N/A')).upper()}\n"
                f"- **波动率缩放**: **{result.get('suggested_scaler', 1.0)} x**\n\n"
                f"**辅助技术面**\n"
                f"- **防守(ATR/CCI)**: ATR {result.get('atr', 0):.4f} | CCI {result.get('cci_info', 'N/A')}\n"
                f"- **动能(MACD)**: {result.get('macd_info', 'N/A')}\n\n"
            )
        else:
            strategy_content = (
                f"**经典技术面特征**\n"
                f"- **趋势(EMA)**: {result.get('ema_trend', 'N/A')}\n"
                f"- **动能(MACD)**: {result.get('macd_info', 'N/A')}\n"
                f"- **K线形态**: {result.get('naked_k', 'N/A')}\n"
                f"- **防守(CCI)**: {result.get('cci_info', 'N/A')}\n\n"
            )

        trade_logic_summary = ToolKit._build_trade_logic_summary(result) if result.get('is_trade_triggered') else ""
        card = (
            f"### ({base_symbol}) 执行报告 [{current_time}]\n"
            f"**{symbol} {price:.5f}**\n**{signal_text}**{ai_header}\n\n{trade_logic_summary}{ai_insight}{trade_exec_info}\n\n"
            f"{strategy_content}"
            f"**主力资金与相对量能**\n{result.get('money_flow', '暂无数据')}"
        )
        return card.strip()

    @staticmethod
    def _markdown_to_feishu(markdown_text: str) -> str:
        text = markdown_text.replace('<br>', '\n')
        text = re.sub(r'<font[^>]*>(.*?)</font>', r'\1', text, flags=re.IGNORECASE)
        text = text.replace('### ', '').replace('**', '')
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _build_feishu_card(text: str, title: str):
        content = ToolKit._markdown_to_feishu(text)
        is_open_notice = ('开仓' in title) or ('交易执行' in content)
        if is_open_notice:
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            summary = lines[0] if lines else title
            details = lines[1:]
            detail_text = '\n'.join(f'- {line.lstrip("- ")}' for line in details[:8]) if details else '暂无明细'
            return {
                'msg_type': 'interactive',
                'card': {
                    'config': {
                        'wide_screen_mode': True,
                        'enable_forward': True
                    },
                    'header': {
                        'template': 'green',
                        'title': {
                            'tag': 'plain_text',
                            'content': title
                        }
                    },
                    'elements': [
                        {
                            'tag': 'div',
                            'text': {
                                'tag': 'lark_md',
                                'content': f'**开仓摘要**\n{summary}'
                            }
                        },
                        {
                            'tag': 'hr'
                        },
                        {
                            'tag': 'div',
                            'text': {
                                'tag': 'lark_md',
                                'content': f'**执行明细**\n{detail_text}'
                            }
                        }
                    ]
                }
            }

        return {
            'msg_type': 'interactive',
            'card': {
                'config': {
                    'wide_screen_mode': True,
                    'enable_forward': True
                },
                'header': {
                    'template': 'blue',
                    'title': {
                        'tag': 'plain_text',
                        'content': title
                    }
                },
                'elements': [
                    {
                        'tag': 'markdown',
                        'content': content
                    }
                ]
            }
        }

    @staticmethod
    def format_position_ai_card(position_data, decision, state):
        symbol = position_data.get('symbol', 'UNKNOWN')
        base_symbol = symbol.replace('USDT', '')
        side = position_data.get('side', 'N/A')
        action = decision.get('action', '未知')
        confidence = float(decision.get('confidence', 0.0) or 0.0)
        risk_level = decision.get('risk_level', 'normal')
        reason = decision.get('reason', '无')
        confirmed_action = state.get('confirmed_action', '观察中')
        risk_streak = int(state.get('risk_streak', 0) or 0)
        hold_streak = int(state.get('hold_streak', 0) or 0)
        current_time = datetime.now().strftime('%H:%M:%S')

        return (
            f"### [{base_symbol}] 持仓后AI评估 [{current_time}]\n"
            f"- **账户**: {position_data.get('account_name', 'N/A')}\n"
            f"- **交易对**: {symbol}\n"
            f"- **订单ID**: {position_data.get('trade_id', 'N/A')}\n"
            f"- **方向**: {side}\n"
            f"- **开仓价 / 当前价**: {position_data.get('entry_price', 'N/A')} / {position_data.get('current_price', 'N/A')}\n"
            f"- **持仓时间**: {position_data.get('holding_minutes', 'N/A')} 分钟\n"
            f"- **浮动盈亏**: {position_data.get('unrealized_pnl', 'N/A')} U ({position_data.get('unrealized_pnl_pct', 'N/A')}%)\n\n"
            f"**AI原始建议**\n"
            f"- **动作**: {action}\n"
            f"- **置信度**: {confidence:.1f}\n"
            f"- **风险等级**: {risk_level}\n"
            f"- **理由**: {reason}\n\n"
            f"**系统确认状态**\n"
            f"- **状态**: {confirmed_action}\n"
            f"- **连续风险次数**: {risk_streak}\n"
            f"- **连续持有次数**: {hold_streak}\n"
            f"- **执行状态**: 否，仅记录，不自动交易"
        )

    @staticmethod
    def send_feishu_msg(text: str, title='量化交易通知'):
        if ToolKit._notifications_silenced():
            return
        webhook = getattr(config, 'FEISHU_WEBHOOK', None)
        if not webhook:
            return
        try:
            payload = ToolKit._build_feishu_card(text, title)
            response = requests.post(webhook, json=payload, proxies={'http': None, 'https': None}, timeout=5)
            if response.status_code >= 400:
                logger.error(f'飞书通知发送失败 {response.status_code}: {response.text[:300]}')
        except Exception as e:
            logger.error(f'飞书通知发送失败: {e}')

    @staticmethod
    def send_dingtalk_msg(markdown_text: str, title='量化交易通知'):
        if ToolKit._notifications_silenced():
            return
        webhook = getattr(config, 'DINGTALK_WEBHOOK', None)
        if webhook:
            try:
                payload = {'msgtype': 'markdown', 'markdown': {'title': title, 'text': markdown_text}}
                requests.post(webhook, json=payload, proxies={'http': None, 'https': None}, timeout=5)
            except Exception as e:
                logger.error(f'钉钉通知发送失败: {e}')

        ToolKit.send_feishu_msg(markdown_text, title=title)

    @staticmethod
    def save_virtual_trade_history(symbol, trade_id, open_time, side, leverage, entry_price, close_price, close_reason, net_pnl, current_balance):
        trade_record = {
            '平仓时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '模式': '模拟盘',
            '开仓时间': open_time,
            '订单ID': trade_id,
            '交易对': symbol,
            '方向': side,
            '杠杆': f'{leverage}x',
            '进场价格': entry_price,
            '出场价格': close_price,
            '平仓原因': close_reason,
            '盈亏': round(net_pnl, 2),
            '账户余额': current_balance,
        }
        try:
            pd.DataFrame([trade_record]).to_csv('virtual_trades_history.csv', mode='a', header=not os.path.isfile('virtual_trades_history.csv'), index=False, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f'写入平仓历史 CSV 失败: {e}')

    @staticmethod
    def log_trade_close(symbol, trade_id, open_time, side, notional, entry_price, close_price, close_reason, net_pnl, current_balance, enable_ai, leverage, market_regime='', strategy_bucket='', estimated_cost_u=None, actual_cost_u=None):
        if not enable_ai:
            return
        log_file = 'ai_decision_logs.csv'
        if not os.path.isfile(log_file):
            return
        try:
            try:
                df_dec = pd.read_csv(log_file, encoding='utf-8-sig')
            except Exception:
                df_dec = pd.read_csv(log_file)
            if '订单ID' not in df_dec.columns:
                return
            matched = df_dec['订单ID'].astype(str) == str(trade_id)
            if not matched.any():
                return

            close_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            holding_minutes = None
            try:
                open_dt = datetime.strptime(str(open_time), '%Y-%m-%d %H:%M:%S')
                close_dt = datetime.strptime(close_time_str, '%Y-%m-%d %H:%M:%S')
                holding_minutes = round((close_dt - open_dt).total_seconds() / 60.0, 2)
            except Exception:
                holding_minutes = None

            estimated_cost_value = None
            if estimated_cost_u is not None:
                try:
                    estimated_cost_value = float(estimated_cost_u)
                except Exception:
                    estimated_cost_value = None
            actual_cost_value = None
            if actual_cost_u is not None:
                try:
                    actual_cost_value = float(actual_cost_u)
                except Exception:
                    actual_cost_value = None
            cost_delta_u = round(-float(estimated_cost_value), 4) if estimated_cost_value is not None else None
            calibration_delta_u = round(float(actual_cost_value) - float(estimated_cost_value), 4) if actual_cost_value is not None and estimated_cost_value is not None else None

            required_columns = [
                '平仓时间', '市场状态', '策略桶', '持仓时长(分钟)', '实际持仓方向',
                '开平成本偏差(U)', '平仓价格', '盈亏', '余额', '预估成本(U)', '实际成本(U)', '成本校准偏差(U)'
            ]
            for column in required_columns:
                if column not in df_dec.columns:
                    df_dec[column] = None

            for column in ['平仓时间', '市场状态', '策略桶', '实际持仓方向']:
                df_dec[column] = df_dec[column].astype(object)

            df_dec.loc[matched, '平仓时间'] = close_time_str
            df_dec.loc[matched, '市场状态'] = market_regime
            df_dec.loc[matched, '策略桶'] = strategy_bucket
            df_dec.loc[matched, '持仓时长(分钟)'] = holding_minutes
            df_dec.loc[matched, '实际持仓方向'] = side
            df_dec.loc[matched, '开平成本偏差(U)'] = cost_delta_u
            df_dec.loc[matched, '平仓价格'] = close_price
            df_dec.loc[matched, '盈亏'] = round(net_pnl, 2)
            df_dec.loc[matched, '余额'] = current_balance
            df_dec.loc[matched, '预估成本(U)'] = estimated_cost_value
            df_dec.loc[matched, '实际成本(U)'] = actual_cost_value
            df_dec.loc[matched, '成本校准偏差(U)'] = calibration_delta_u

            temp_file = f'{log_file}.tmp'
            df_dec.to_csv(temp_file, index=False, encoding='utf-8-sig')
            os.replace(temp_file, log_file)
        except Exception as e:
            logger.error(f'回写 AI 平仓日志失败: {e}')

    @staticmethod
    def get_clean_raw_data(data):
        clean_data = {k: v for k, v in data.items() if not isinstance(v, (pd.DataFrame, pd.Series, list))}
        return str(clean_data)

    @staticmethod
    def get_expanded_raw_dict(data):
        expanded = {}
        for k, v in data.items():
            if isinstance(v, (pd.DataFrame, pd.Series, list)):
                continue
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    expanded[f'原始_{k}_{sub_k}'] = sub_v
            else:
                expanded[f'原始_{k}'] = v
        return expanded

    @staticmethod
    def get_strategy_data_str(data, signal_text):
        if getattr(config, 'ENABLE_QUANT_SCORING', False):
            ls = data.get('long_score', 0)
            ss = data.get('short_score', 0)
            scaler = data.get('suggested_scaler', 1.0)
            return f"现价:{data.get('price')} | 信号:{signal_text} | 得分(多{ls}/空{ss}) | 仓位系数:{scaler}x"
        return f"现价:{data.get('price')} | 信号:{signal_text} | CCI:{data.get('cci_info', 'N/A')} | MACD:{data.get('macd_info', 'N/A')}"

    @staticmethod
    def check_and_restart_proxy(last_restart_time):
        current_time = time.time()
        if current_time - last_restart_time > 300:
            logger.error('🌐 发现严重网络阻断！请检查您的本地代理通道(Clash/v2ray等)或网络连通性。')
            logger.info('⏳ 为保护系统稳定，引擎将静默休眠 15 秒等待网络自行恢复...')
            time.sleep(15)
            return current_time
        logger.warning('🌐 网络仍然超时，系统继续等待恢复...')
        time.sleep(5)
        return last_restart_time

    @staticmethod
    def save_active_trades(trades_dict, account_name='Default'):
        filename = f'active_trades_{account_name}.json'
        temp_filename = f'{filename}.tmp'
        try:
            with open(temp_filename, 'w', encoding='utf-8') as f:
                json.dump(trades_dict, f, indent=4, ensure_ascii=False)
            os.replace(temp_filename, filename)
        except Exception as e:
            logger.error(f'保存 {account_name} 的本地交易状态失败: {e}')
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception:
                    pass

    @staticmethod
    def load_active_trades(account_name='Default'):
        filename = f'active_trades_{account_name}.json'
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f'读取 {account_name} 的本地未完成交易失败: {e}')
        return {}

    @staticmethod
    def save_runtime_state(file_path, monitor, accounts):
        payload = {
            'recent_open_rejections': [],
            'accounts': {},
        }
        for item in list(getattr(monitor, 'recent_open_rejections', []) or []):
            payload['recent_open_rejections'].append({
                'time': item.get('time').strftime('%Y-%m-%d %H:%M:%S') if item.get('time') else None,
                'account_name': item.get('account_name'),
                'symbol': item.get('symbol'),
                'reason': item.get('reason'),
            })
        for acc in accounts or []:
            payload['accounts'][acc.name] = {
                'bucket_cooldowns': dict(getattr(acc, 'bucket_cooldowns', {}) or {}),
                'bucket_loss_streaks': dict(getattr(acc, 'bucket_loss_streaks', {}) or {}),
                'symbol_reentry_cooldowns': dict(getattr(acc, 'symbol_reentry_cooldowns', {}) or {}),
            }
        temp_file = f'{file_path}.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(temp_file, file_path)

    @staticmethod
    def load_runtime_state(file_path):
        if not os.path.exists(file_path):
            return {'recent_open_rejections': [], 'accounts': {}}
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


