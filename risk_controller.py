import logging
import time

from tool import ToolKit
import config
from pnl_utils import _get_realized_net_pnl, get_realized_pnl_record

logger = logging.getLogger(__name__)


class DynamicRiskController:
    def __init__(self, account):
        # account is a SubAccount instance.
        self.acc = account

    def _bucket_profile(self, strategy_bucket):
        bucket = str(strategy_bucket or '').lower()
        if bucket == 'reversal_probe':
            return {
                'close_ratio_1r': 0.6,
                'close_ratio_2r': 0.3,
                'phase2_sl_atr_mult': 0.3,
                'trail_mult': 1.0,
            }
        if bucket == 'mean_reversion':
            return {
                'close_ratio_1r': 0.5,
                'close_ratio_2r': 0.4,
                'phase2_sl_atr_mult': 0.5,
                'trail_mult': 1.5,
            }
        if bucket == 'trend_continuation':
            return {
                'close_ratio_1r': 0.5,
                'close_ratio_2r': 0.25,
                'phase2_sl_atr_mult': 0.5,
                'trail_mult': 2.0,
            }
        return {
            'close_ratio_1r': 0.5,
            'close_ratio_2r': 0.3,
            'phase2_sl_atr_mult': 0.5,
            'trail_mult': None,
        }

    def manage_positions(self, symbol, current_price, override_level=2, leverage=10):
        if override_level in [0, 5]:
            return

        is_real = self.acc.mode in ['TESTNET', 'REAL']

        # 1. Load current position context.
        pos_data = self._get_position_context(symbol, is_real)
        if not pos_data:
            return

        # 2. Load ATR from market data; fallback to 2% of price if missing.
        market_data = self.acc.analyzer.get_market_data(symbol)
        atr = market_data.get('atr') if market_data else None
        if not atr or atr <= 0:
            atr = current_price * 0.02

        # 3. Evaluate state machine.
        action = self._evaluate_state_machine(
            current_price=current_price,
            pos_data=pos_data,
            atr=atr,
            risk_level=override_level,
        )

        # 4. Execute dynamic risk actions.
        if action['close_ratio'] > 0 or action['update_sl']:
            self._execute_action(symbol, action, pos_data, is_real)

    # ==========================================
    # Core state machine
    # ==========================================
    def _evaluate_state_machine(self, current_price, pos_data, atr, risk_level):
        entry_price = pos_data['entry']
        side = pos_data['side']
        tp_phase = pos_data['tp_phase']
        current_sl = pos_data['sl']
        peak_price = pos_data['peak_price']
        bucket_profile = self._bucket_profile(pos_data.get('strategy_bucket'))

        # 1. Update absolute peak price.
        if (side == 'LONG' and current_price > peak_price) or \
           (side == 'SHORT' and current_price < peak_price):
            peak_price = current_price

        # 2. Compute the true R multiple.
        # If still in phase 0 and the initial stop exists, use the original risk distance.
        # Otherwise fall back to 1.5 ATR as a conservative baseline.
        if tp_phase == 0 and current_sl != 0:
            initial_risk_distance = abs(entry_price - current_sl)
        else:
            initial_risk_distance = atr * 1.5

        # Guard against division by zero.
        if initial_risk_distance <= 0:
            initial_risk_distance = current_price * 0.02

        move = (current_price - entry_price) * (1 if side == 'LONG' else -1)
        r_multiple = move / initial_risk_distance

        action = {
            'close_ratio': 0.0,
            'new_sl': current_sl,
            'new_phase': tp_phase,
            'update_sl': False,
            'phase_msg': '',
            'peak_price': peak_price,
            'current_price': current_price,
        }

        # Fee/slippage buffer to avoid a fake breakeven stop.
        fee_rate = getattr(config, 'FEE_RATE', 0.0004)
        fee_offset = entry_price * (fee_rate * 2 + 0.0002)
        breakeven_price = entry_price + fee_offset if side == 'LONG' else entry_price - fee_offset

        ratio_1r = float(bucket_profile.get('close_ratio_1r', getattr(config, 'CLOSE_RATIO_1R', 0.5)) or getattr(config, 'CLOSE_RATIO_1R', 0.5))
        ratio_2r = float(bucket_profile.get('close_ratio_2r', getattr(config, 'CLOSE_RATIO_2R', 0.3)) or getattr(config, 'CLOSE_RATIO_2R', 0.3))

        # 3. Phase transitions.
        if tp_phase == 0 and r_multiple >= 1.0:
            action.update({
                'close_ratio': ratio_1r,
                'new_sl': breakeven_price,
                'new_phase': 1,
                'update_sl': True,
                'phase_msg': f'1R 达标 (减仓 {ratio_1r*100}% + 推保本防守)',
            })

        elif tp_phase == 1 and r_multiple >= 2.0:
            sl_offset = atr * float(bucket_profile.get('phase2_sl_atr_mult', 0.5) or 0.5)
            action.update({
                'close_ratio': ratio_2r,
                'new_sl': entry_price + sl_offset if side == 'LONG' else entry_price - sl_offset,
                'new_phase': 2,
                'update_sl': True,
                'phase_msg': f'2R 达标 (减仓 {ratio_2r*100}% + 锁定利润防线)',
            })

        elif tp_phase >= 2:
            # ATR trailing-stop width mapping by strategy bucket first, then risk level.
            trail_mult = bucket_profile.get('trail_mult')
            if trail_mult is None:
                if risk_level >= 7:
                    trail_mult = 0.8
                elif risk_level == 6:
                    trail_mult = 1.0
                elif risk_level <= 4:
                    trail_mult = 2.0
                else:
                    trail_mult = 1.5

            trail_sl = peak_price - (float(trail_mult) * atr) if side == 'LONG' else peak_price + (float(trail_mult) * atr)

            # Only update if the new line moved by more than 0.2 ATR.
            if current_sl == 0 or abs(trail_sl - current_sl) > (0.2 * atr):
                if (side == 'LONG' and trail_sl > current_sl) or \
                   (side == 'SHORT' and (current_sl == 0 or trail_sl < current_sl)):
                    action.update({
                        'new_sl': trail_sl,
                        'update_sl': True,
                        'phase_msg': f'ATR 动态防线收紧至 {trail_sl:.4f}',
                    })

        return action

    # ==========================================
    # Execution layer
    # ==========================================
    def _execute_action(self, symbol, action, pos_data, is_real):
        side = pos_data['side']
        qty = float(pos_data['qty'])
        curr_price = action['current_price']

        # Keep local peak-price cache in sync.
        if is_real:
            self.acc.real_trade_meta[symbol]['peak_price'] = action['peak_price']
        else:
            self.acc.open_positions[symbol]['peak_price'] = action['peak_price']

        if is_real:
            # 1. Format quantities with exchange precision.
            target_qty = float(self.acc.engine.format_quantity(symbol, qty * action['close_ratio']))
            remaining_qty = float(self.acc.engine.format_quantity(symbol, qty - target_qty))

            # Strict min notional checks for both the closing slice and the remainder.
            min_qty = float(self.acc.engine.get_symbol_info(symbol).get('minQty', 0))
            min_effective_notional = 6.0
            target_notional = self.acc.engine.estimate_notional(symbol, target_qty, curr_price)
            remaining_notional = self.acc.engine.estimate_notional(symbol, remaining_qty, curr_price)

            if target_qty > 0 and target_notional < min_effective_notional:
                logger.warning(f"[{self.acc.name}] {symbol} 计划平仓价值不足 {min_effective_notional:.0f}U，直接转为全平。")
                target_qty = qty
                action['close_ratio'] = 1.0
                remaining_qty = 0.0
                target_notional = self.acc.engine.estimate_notional(symbol, target_qty, curr_price)
                remaining_notional = 0.0

            if remaining_qty > 0 and (remaining_qty < min_qty or remaining_notional < min_effective_notional):
                logger.warning(f"[{self.acc.name}] {symbol} 剩余仓位价值不足 {min_effective_notional:.0f}U，直接转为全平。")
                target_qty = qty
                action['close_ratio'] = 1.0
                remaining_qty = 0.0
                target_notional = self.acc.engine.estimate_notional(symbol, target_qty, curr_price)
                remaining_notional = 0.0

            safe_sl = float(self.acc.engine.format_price(symbol, action['new_sl'])) if action['new_sl'] > 0 else 0
            safe_tp = self.acc.real_trade_meta[symbol].get('tp')

            # A. Partial close / full close.
            if action['close_ratio'] > 0:
                logger.info(f"[{self.acc.name}] 实盘风控执行: {symbol} 触发 {action['phase_msg']}")

                try:
                    # 1. Execute market close on OKX.
                    close_order_id = self.acc.engine.execute_market_close(symbol, side, target_qty)

                    # 2. Write the realized PnL slice into trade history.
                    meta = self.acc.real_trade_meta.get(symbol, {})
                    realized_pnl = get_realized_pnl_record(
                        self.acc.engine,
                        symbol,
                        target_qty,
                        pos_data['entry'],
                        curr_price,
                        side,
                        order_id=close_order_id,
                    )
                    net_pnl = float(realized_pnl['net_pnl'])
                    if realized_pnl.get('avg_price'):
                        curr_price = float(realized_pnl['avg_price'])

                    spec = self.acc.monitor.get_symbol_spec(symbol)

                    self.acc._save_to_trade_history(
                        is_simulate=False,
                        symbol=symbol,
                        trade_id=meta.get('trade_id', f'RISK_{int(time.time())}'),
                        open_time=meta.get('open_time', '未知'),
                        side=side,
                        leverage=spec.get('leverage', 10),
                        entry=pos_data['entry'],
                        close_price=curr_price,
                        close_reason=action['phase_msg'],  # Record whether this was 1R/2R reduction or a full exit.
                        net_pnl=net_pnl,
                        balance=self.acc.get_base_capital(),
                    )

                    # 3. Cancel old protection orders and rebuild them for the remainder.
                    self.acc.engine.cancel_all_orders(symbol)
                    time.sleep(0.5)  # Give the exchange engine a moment.

                    if remaining_qty > 0:
                        is_sl_success = self._replace_protection_orders(symbol, side, safe_sl, safe_tp)

                        if not is_sl_success:
                            logger.error(f"[{self.acc.name}] {symbol} 新保护单重建失败，立即强平剩余仓位。")
                            self.acc.engine.execute_market_close(symbol, side, remaining_qty)
                            remaining_qty = 0.0
                            action['close_ratio'] = 1.0
                            action['phase_msg'] += ' (保护单失败，尾仓已紧急撤退)'

                    # 4. Sync local state.
                    if remaining_qty > 0:
                        self.acc.real_trade_meta[symbol]['sl'] = safe_sl
                        self.acc.real_trade_meta[symbol]['tp_phase'] = action['new_phase']
                        self.acc.real_positions[symbol]['qty'] = remaining_qty
                    else:
                        if symbol in self.acc.real_trade_meta:
                            del self.acc.real_trade_meta[symbol]
                        if symbol in self.acc.real_positions:
                            del self.acc.real_positions[symbol]

                    ToolKit.save_active_trades(self.acc.real_trade_meta, account_name=self.acc.name)

                    # 5. Send notification.
                    self._trigger_notifications(symbol, side, action, remaining_qty, safe_sl)

                except Exception as e:
                    logger.error(f"[{self.acc.name}] {symbol} 减仓及推保护过程发生异常: {e}")

            # B. Only move the protective stop line.
            elif action['update_sl']:
                self.acc.engine.cancel_all_orders(symbol)
                time.sleep(0.3)

                if self._replace_protection_orders(symbol, side, safe_sl, safe_tp):
                    self.acc.real_trade_meta[symbol]['sl'] = safe_sl
                    ToolKit.save_active_trades(self.acc.real_trade_meta, account_name=self.acc.name)
                    logger.info(f"[{self.acc.name}] {symbol} 追踪防线已移动至 {safe_sl}")

        else:
            # Simulated trading branch.
            pos = self.acc.open_positions[symbol]

            if action['close_ratio'] > 0:
                close_ratio = action['close_ratio']
                if pos['notional'] * (1 - close_ratio) < 6.0:
                    close_ratio = 1.0  # Simulated trading also respects the 6U rule.

                close_notional = pos['notional'] * close_ratio
                close_margin = pos['margin'] * close_ratio

                pnl_pct = (curr_price - pos['entry']) / pos['entry'] * (1 if side == 'LONG' else -1)
                net_pnl = (close_notional * pnl_pct) - (close_notional * getattr(config, 'FEE_RATE', 0.0004))

                # Roll funds back into the virtual balance pool.
                if self.acc.funds_mode == 'SHARED':
                    self.acc.virtual_balance += (close_margin + net_pnl)

                logger.info(f"[{self.acc.name}] 模拟风控触发: {symbol} {action['phase_msg']} | 盈亏: {net_pnl:+.2f} U")

                if close_ratio >= 1.0:
                    del self.acc.open_positions[symbol]
                else:
                    pos['notional'] -= close_notional
                    pos['margin'] -= close_margin
                    pos['qty'] = pos.get('qty', 1.0) * (1 - close_ratio)
                    pos['sl'] = action['new_sl']
                    pos['tp_phase'] = action['new_phase']

            elif action['update_sl']:
                pos['sl'] = action['new_sl']

    # ==========================================
    # Internal helpers
    # ==========================================
    def _replace_protection_orders(self, symbol, position_side, safe_sl, safe_tp):
        """Rebuild stop-loss and take-profit orders. Return False if SL placement fails."""
        sl_success = True

        try:
            qty = self.acc.real_positions.get(symbol, {}).get('qty', 0)
            sl_success = self.acc.engine.place_protection_orders(
                symbol,
                position_side,
                sl_price=safe_sl if safe_sl > 0 else None,
                tp_price=safe_tp if safe_tp and safe_tp > 0 else None,
                qty=qty,
            )
        except Exception as e:
            logger.error(f"[{self.acc.name}] {symbol} 新止损挂载失败: {e}")
            sl_success = False

        return sl_success

    def _get_position_context(self, symbol, is_real):
        try:
            if is_real:
                if symbol not in self.acc.real_positions or symbol not in self.acc.real_trade_meta:
                    return None
                pos = self.acc.real_positions[symbol]
                meta = self.acc.real_trade_meta[symbol]
                return {
                    'entry': pos['entry'],
                    'side': pos['side'],
                    'qty': pos['qty'],
                    'tp_phase': meta.get('tp_phase', 0),
                    'sl': meta.get('sl', 0),
                    'peak_price': meta.get('peak_price', pos['entry']),
                    'market_regime': meta.get('market_regime', ''),
                    'strategy_bucket': meta.get('strategy_bucket', ''),
                }
            else:
                if symbol not in self.acc.open_positions:
                    return None
                pos = self.acc.open_positions[symbol]
                return {
                    'entry': pos['entry'],
                    'side': pos['side'],
                    'qty': pos.get('qty', 1.0),
                    'tp_phase': pos.get('tp_phase', 0),
                    'sl': pos.get('sl', 0),
                    'peak_price': pos.get('peak_price', pos['entry']),
                }
        except KeyError:
            return None

    def _trigger_notifications(self, symbol, side, action, remaining_qty, safe_sl=None):
        msg = action['phase_msg']
        ratio = action['close_ratio']
        curr_price = action['current_price']

        info = (
            f"### {msg}\n"
            f"- **交易对**: {symbol} ({side})\n"
            f"- **触发价格**: {curr_price:.5f}\n"
            f"- **执行动作**: 减仓 {ratio*100}%\n"
            f"- **剩余仓位**: {remaining_qty}\n"
        )

        if safe_sl is not None and safe_sl > 0:
            info += f"- **新防线(SL)**: {safe_sl:.5f}\n"

        info += f"- **账户归属**: {self.acc.name}"

        ToolKit.send_dingtalk_msg(info, title=f"[{self.acc.name}] 动态风控触发")
