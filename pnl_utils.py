import config


def get_realized_pnl_record(engine, symbol, qty, entry_price, close_price, side, fee_rate=None, order_id=None):
    if order_id and hasattr(engine, 'get_order_realized_pnl'):
        realized = engine.get_order_realized_pnl(symbol, order_id)
        if realized and realized.get('net_pnl') is not None:
            return realized

    close_notional = float(engine.estimate_notional(symbol, qty, close_price))
    pnl_pct = (float(close_price) - float(entry_price)) / float(entry_price) * (1 if str(side).upper() == 'LONG' else -1)
    fee = close_notional * float(fee_rate if fee_rate is not None else getattr(config, 'FEE_RATE', 0.0004))
    return {'net_pnl': (close_notional * pnl_pct) - fee, 'gross_pnl': close_notional * pnl_pct, 'fee': -fee, 'avg_price': None}


def _get_realized_net_pnl(engine, symbol, qty, entry_price, close_price, side, fee_rate=None, order_id=None):
    return float(get_realized_pnl_record(engine, symbol, qty, entry_price, close_price, side, fee_rate=fee_rate, order_id=order_id)['net_pnl'])
