import base64
import hashlib
import hmac
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

import requests
from requests.exceptions import ConnectionError, ProxyError, ReadTimeout, SSLError, Timeout

logger = logging.getLogger(__name__)


class OKXExecutionEngine:
    def __init__(self, api_key, api_secret, passphrase, mode="REAL", proxies=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.mode = mode.upper()
        self.base_url = "https://www.okx.com"
        self.proxies = proxies or None
        self.timeout = 15
        self.symbol_rules = {}
        self.session = requests.Session()
        self.session.trust_env = False
        if self.proxies:
            self.session.proxies.update(self.proxies)
        self.session.headers.update({"Content-Type": "application/json"})
        self._load_exchange_info()

    def _timestamp(self):
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _sign(self, timestamp, method, path, body=""):
        payload = f"{timestamp}{method.upper()}{path}{body}"
        digest = hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode("utf-8")

    def _symbol_to_inst_id(self, symbol):
        if "-" in symbol:
            return symbol
        clean = symbol.upper().replace("-", "")
        if clean.endswith("USDT"):
            return f"{clean[:-4]}-USDT-SWAP"
        return symbol

    def _inst_id_to_symbol(self, inst_id):
        return inst_id.upper().replace("-USDT-SWAP", "USDT").replace("-", "")

    def _request(self, method, path, params=None, payload=None, private=False):
        body = ""
        headers = {}
        if payload:
            import json
            body = json.dumps(payload, separators=(",", ":"))

        request_path = path
        if params:
            query = requests.compat.urlencode(params)
            request_path = f"{path}?{query}"

        if private:
            timestamp = self._timestamp()
            headers.update({
                "OK-ACCESS-KEY": self.api_key,
                "OK-ACCESS-SIGN": self._sign(timestamp, method, request_path, body),
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": self.passphrase,
            })
            if self.mode == "TESTNET":
                headers["x-simulated-trading"] = "1"

        return self.session.request(
            method=method.upper(),
            url=f"{self.base_url}{path}",
            params=params,
            data=body or None,
            headers=headers,
            timeout=self.timeout,
        )

    def _safe_request(self, method, path, params=None, payload=None, private=False, max_retries=4):
        for attempt in range(max_retries):
            try:
                resp = self._request(method, path, params=params, payload=payload, private=private)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") == "0":
                    return data.get("data", [])
                logger.warning(f"OKX API异常 {path}: code={data.get('code')} msg={data.get('msg')}")
            except (Timeout, ConnectionError, ProxyError, ReadTimeout, SSLError) as e:
                logger.warning(f"OKX网络异常 {path}: {type(e).__name__}")
            except Exception as e:
                logger.warning(f"OKX请求失败 {path}: {e}")
            time.sleep(2 ** attempt)
        return None

    def _load_exchange_info(self):
        instruments = self._safe_request(
            "GET",
            "/api/v5/public/instruments",
            params={"instType": "SWAP"},
            private=False,
        )
        if not instruments:
            return

        for item in instruments:
            symbol = self._inst_id_to_symbol(item["instId"])
            self.symbol_rules[symbol] = {
                "instId": item["instId"],
                "price_tick": item.get("tickSz", "0.1"),
                "lot_size": item.get("lotSz", "1"),
                "min_qty": item.get("minSz", "1"),
                "max_qty": item.get("maxMktSz") or item.get("maxLmtSz") or "999999999",
                "ct_val": item.get("ctVal") or "1",
                "ct_mult": item.get("ctMult") or "1",
                "ct_val_ccy": item.get("ctValCcy") or "",
            }

    def get_symbol_info(self, symbol):
        rule = self.symbol_rules.get(symbol, {})
        return {"minQty": float(rule.get("min_qty", 0))}

    def _floor_to_step(self, value, step):
        value_dec = Decimal(str(value))
        step_dec = Decimal(str(step))
        if step_dec <= 0:
            return value_dec
        return (value_dec / step_dec).quantize(Decimal("1"), rounding=ROUND_DOWN) * step_dec

    def format_price(self, symbol, price):
        rule = self.symbol_rules.get(symbol)
        if not rule:
            return float(price)
        return float(self._floor_to_step(price, rule["price_tick"]))

    def format_quantity(self, symbol, qty):
        rule = self.symbol_rules.get(symbol)
        if not rule:
            return float(qty)
        qty_dec = self._floor_to_step(qty, rule["lot_size"])
        max_qty = Decimal(str(rule["max_qty"]))
        if qty_dec > max_qty:
            qty_dec = max_qty
        return float(qty_dec)

    def _contract_notional_usdt(self, symbol, price):
        rule = self.symbol_rules.get(symbol)
        if not rule:
            return float(price)
        ct_val = float(rule.get("ct_val") or 1.0)
        ct_mult = float(rule.get("ct_mult") or 1.0)
        ct_val_ccy = str(rule.get("ct_val_ccy") or "").upper()
        unit_value = ct_val * ct_mult
        if ct_val_ccy in {"USDT", "USD"}:
            return unit_value
        return unit_value * float(price)

    def contracts_from_notional(self, symbol, target_notional, price):
        contract_notional = self._contract_notional_usdt(symbol, price)
        if contract_notional <= 0:
            return 0.0
        return self.format_quantity(symbol, float(target_notional) / contract_notional)

    def estimate_notional(self, symbol, contracts, price):
        safe_contracts = self.format_quantity(symbol, contracts)
        if safe_contracts <= 0:
            return 0.0
        return safe_contracts * self._contract_notional_usdt(symbol, price)

    def sync_real_positions(self, target_symbols):
        positions = self._safe_request(
            "GET",
            "/api/v5/account/positions",
            params={"instType": "SWAP"},
            private=True,
        )
        if positions is None:
            return None

        target_set = set(target_symbols)
        real_positions = {}
        for pos in positions:
            symbol = self._inst_id_to_symbol(pos.get("instId", ""))
            if symbol not in target_set:
                continue

            pos_qty = float(pos.get("pos", 0) or 0)
            if pos_qty == 0:
                continue

            pos_side = str(pos.get("posSide", "")).lower()
            if pos_side == "long":
                side = "LONG"
            elif pos_side == "short":
                side = "SHORT"
            else:
                side = "LONG" if pos_qty > 0 else "SHORT"

            real_positions[symbol] = {
                "side": side,
                "entry": float(pos.get("avgPx", 0) or 0),
                "qty": abs(pos_qty),
                "unrealized_pnl": float(pos.get("upl", 0) or 0),
            }
        return real_positions

    def get_real_usdt_balance(self):
        balances = self._safe_request(
            "GET",
            "/api/v5/account/balance",
            params={"ccy": "USDT"},
            private=True,
        )
        if not balances:
            return None
        details = balances[0].get("details", [])
        for item in details:
            if item.get("ccy") == "USDT":
                return float(item.get("eq") or item.get("cashBal") or item.get("availBal") or 0)
        return 0.0

    def validate_api_credentials(self):
        if not self.api_key or not self.api_secret or not self.passphrase:
            return False, "请输入 OKX API Key / Secret / Passphrase。"

        result = self._safe_request(
            "GET",
            "/api/v5/account/balance",
            params={"ccy": "USDT"},
            private=True,
            max_retries=1,
        )
        if result is None:
            return False, "OKX 凭证校验失败，请检查 Key / Secret / Passphrase / 代理 / IP 白名单。"
        return True, "OKX 凭证校验通过。"

    def cancel_all_orders(self, symbol):
        inst_id = self._symbol_to_inst_id(symbol)

        normal_orders = self._safe_request(
            "GET",
            "/api/v5/trade/orders-pending",
            params={"instId": inst_id},
            private=True,
        ) or []
        for order in normal_orders:
            ord_id = order.get("ordId")
            if ord_id:
                self._safe_request(
                    "POST",
                    "/api/v5/trade/cancel-order",
                    payload={"instId": inst_id, "ordId": ord_id},
                    private=True,
                )

        algo_orders = self._safe_request(
            "GET",
            "/api/v5/trade/orders-algo-pending",
            params={"ordType": "conditional", "instId": inst_id},
            private=True,
        ) or []
        if algo_orders:
            cancel_list = [{"instId": inst_id, "algoId": o["algoId"]} for o in algo_orders if o.get("algoId")]
            if cancel_list:
                self._safe_request(
                    "POST",
                    "/api/v5/trade/cancel-algos",
                    payload=cancel_list,
                    private=True,
                )

    def _to_okx_pos_side(self, side):
        return "long" if str(side).upper() == "LONG" else "short"

    def execute_market_close(self, symbol, current_position_side, qty):
        if qty <= 0:
            return None
        safe_qty = self.format_quantity(symbol, qty)
        if safe_qty <= 0:
            return None

        inst_id = self._symbol_to_inst_id(symbol)
        side = "sell" if current_position_side == "LONG" else "buy"
        result = self._safe_request(
            "POST",
            "/api/v5/trade/order",
            payload={
                "instId": inst_id,
                "posSide": self._to_okx_pos_side(current_position_side),
                "tdMode": "cross",
                "side": side,
                "ordType": "market",
                "sz": str(safe_qty),
                "reduceOnly": "true",
            },
            private=True,
        )
        if result and result[0].get("ordId"):
            return result[0].get("ordId")
        return result is not None

    def get_order_realized_pnl(self, symbol, order_id):
        if not order_id or order_id is True:
            return None
        fills = self._safe_request(
            "GET",
            "/api/v5/trade/fills",
            params={"instId": self._symbol_to_inst_id(symbol), "ordId": str(order_id)},
            private=True,
        )
        if not fills:
            return None
        fill_pnl = sum(float(fill.get("fillPnl") or 0.0) for fill in fills)
        fee = sum(float(fill.get("fee") or 0.0) for fill in fills)
        prices = [float(fill.get("fillPx") or 0.0) for fill in fills if float(fill.get("fillPx") or 0.0) > 0]
        avg_price = sum(prices) / len(prices) if prices else None
        return {"net_pnl": fill_pnl + fee, "gross_pnl": fill_pnl, "fee": fee, "avg_price": avg_price}

    def _set_leverage(self, symbol, leverage):
        inst_id = self._symbol_to_inst_id(symbol)
        result = self._safe_request(
            "POST",
            "/api/v5/account/set-leverage",
            payload={
                "instId": inst_id,
                "lever": str(leverage),
                "mgnMode": "cross",
            },
            private=True,
        )
        return result is not None

    def place_protection_orders(self, symbol, position_side, sl_price=None, tp_price=None, qty=None):
        safe_qty = self.format_quantity(symbol, qty) if qty else None
        if safe_qty is None or safe_qty <= 0:
            return False if sl_price else True

        inst_id = self._symbol_to_inst_id(symbol)
        payload = {
            "instId": inst_id,
            "posSide": self._to_okx_pos_side(position_side),
            "tdMode": "cross",
            "side": "sell" if position_side == "LONG" else "buy",
            "ordType": "conditional",
            "sz": str(safe_qty),
        }
        if tp_price and tp_price > 0:
            payload["tpTriggerPx"] = str(self.format_price(symbol, tp_price))
            payload["tpOrdPx"] = "-1"
        if sl_price and sl_price > 0:
            payload["slTriggerPx"] = str(self.format_price(symbol, sl_price))
            payload["slOrdPx"] = "-1"

        if "tpTriggerPx" not in payload and "slTriggerPx" not in payload:
            return True

        result = self._safe_request(
            "POST",
            "/api/v5/trade/order-algo",
            payload=payload,
            private=True,
        )
        return result is not None

    def execute_open_with_sltp(self, symbol, side, qty, sl_price, tp_price, leverage):
        safe_qty = self.format_quantity(symbol, qty)
        if safe_qty <= 0:
            return False

        if not self._set_leverage(symbol, leverage):
            return False

        inst_id = self._symbol_to_inst_id(symbol)
        result = self._safe_request(
            "POST",
            "/api/v5/trade/order",
            payload={
                "instId": inst_id,
                "posSide": self._to_okx_pos_side(side),
                "tdMode": "cross",
                "side": "buy" if side == "LONG" else "sell",
                "ordType": "market",
                "sz": str(safe_qty),
            },
            private=True,
        )
        if result is None:
            return False

        if not self.place_protection_orders(symbol, side, sl_price=sl_price, tp_price=tp_price, qty=safe_qty):
            self.execute_market_close(symbol, side, safe_qty)
            return False
        return True
