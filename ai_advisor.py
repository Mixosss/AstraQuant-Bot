import json
import logging
import re
import time

import requests

import config

logger = logging.getLogger(__name__)

POSITION_AI_ACTIONS = {'继续持有', '减仓防守', '推保本线', '全部退出'}
POSITION_AI_RISK_LEVELS = {'low', 'normal', 'high'}


def _repair_mojibake(text):
    if not isinstance(text, str):
        return text
    if all(marker not in text for marker in ('Ã', 'Â', 'æ', 'ç', 'ï', 'å', 'ä', 'é')):
        return text
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _repair_json_strings(obj):
    if isinstance(obj, dict):
        return {k: _repair_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_repair_json_strings(v) for v in obj]
    if isinstance(obj, str):
        return _repair_mojibake(obj)
    return obj


class AITradingAdvisor:
    def __init__(self):
        self.api_key = getattr(config, 'AI_API_KEY', None)
        self.api_url = getattr(config, 'AI_API_URL', 'https://api.openai.com/v1/chat/completions')
        self.model = getattr(config, 'AI_MODEL', 'gpt-4o')
        self.max_retries = getattr(config, 'AI_MAX_RETRIES', 3)
        self.proxies = getattr(config, 'AI_PROXIES', None)
        self.timeout = getattr(config, 'API_TIMEOUT', 60)
        self.session = requests.Session()
        self.session.trust_env = False

    def _soften_aggressive_pass(self, market_data, ai_result):
        if str(ai_result.get('action', '')).upper() != 'PASS':
            return ai_result

        direction = str(market_data.get('direction', '') or '').upper()
        trend_state = str(market_data.get('trend_state', '') or '').lower()
        reason = str(ai_result.get('reason', '') or '')

        try:
            rr_value = float(market_data.get('realistic_rr', market_data.get('rr', market_data.get('current_rr'))))
        except (TypeError, ValueError):
            rr_value = None

        soft_keywords = ('盈亏比', 'RR', '等待确认', '观望', '方向不明', '震荡')
        hard_keywords = (
            '结构质量差', '结构不佳', '方向冲突', '趋势冲突', '缺乏共振', '数据缺失', '证据不足',
            '不宜立即入场', '不适合立即入场', '性价比不足', '性价比过差', '空间严重不足',
            '盈利空间不足', '做多空间严重不足', '做空空间严重不足', '追多风险高', '追空风险高',
            '追单风险高', '回归风险高'
        )
        same_trend = (direction == 'LONG' and trend_state == 'long') or (direction == 'SHORT' and trend_state == 'short')
        rr_too_low = rr_value is None or rr_value < 0.8
        reason_has_soft = any(k in reason for k in soft_keywords)
        reason_has_hard = any(k in reason for k in hard_keywords)

        if not same_trend or rr_too_low or reason_has_hard or not reason_has_soft:
            return ai_result

        if rr_value >= 1.5:
            leverage_factor = 1.0
        elif rr_value >= 1.3:
            leverage_factor = 0.85
        elif rr_value >= 1.0:
            leverage_factor = 0.7
        else:
            leverage_factor = 0.5

        softened = dict(ai_result)
        softened['action'] = direction
        softened['confidence'] = max(65.0, float(ai_result.get('confidence', 0) or 0))
        softened['leverage_factor'] = leverage_factor
        softened['reason'] = f"{reason}；顺大周期方向，按净RR分层{leverage_factor:.2f}x试仓。"[:100]
        return softened

    def _format_klines(self, klines):
        if not klines:
            return "暂无数据"

        parsed = []
        for k in klines[-2:]:
            if isinstance(k, list) and len(k) >= 6:
                try:
                    parsed.append({
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "vol": float(k[5]),
                    })
                    continue
                except (TypeError, ValueError):
                    pass

            if isinstance(k, dict):
                try:
                    parsed.append({
                        "open": float(k.get("open", 0)),
                        "high": float(k.get("high", 0)),
                        "low": float(k.get("low", 0)),
                        "close": float(k.get("close", 0)),
                        "vol": float(k.get("vol", k.get("volume", 0))),
                    })
                    continue
                except (TypeError, ValueError):
                    pass

            if isinstance(k, str):
                match = re.search(
                    r"O:(?P<open>-?\d+(?:\.\d+)?)\s+H:(?P<high>-?\d+(?:\.\d+)?)\s+L:(?P<low>-?\d+(?:\.\d+)?)\s+C:(?P<close>-?\d+(?:\.\d+)?)\s+V:(?P<vol>-?\d+(?:\.\d+)?)",
                    k,
                )
                if match:
                    try:
                        parsed.append({
                            "open": float(match.group("open")),
                            "high": float(match.group("high")),
                            "low": float(match.group("low")),
                            "close": float(match.group("close")),
                            "vol": float(match.group("vol")),
                        })
                        continue
                    except (TypeError, ValueError):
                        pass

        if not parsed:
            return "暂无数据"

        latest = parsed[-1]
        prev_close = parsed[-2]["close"] if len(parsed) > 1 else latest["open"]
        change_pct = ((latest["close"] - prev_close) / prev_close * 100) if prev_close else 0.0
        avg_vol = sum(item["vol"] for item in parsed) / len(parsed)
        return (
            f"最新收:{latest['close']:.4f} | 区间:{latest['low']:.4f}-{latest['high']:.4f} | "
            f"近2根涨跌:{change_pct:+.2f}% | 均量:{avg_vol:.2f}"
        )

    def _decode_response_json(self, response):
        try:
            return json.loads(response.content.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return response.json()

    def _extract_reply_text(self, body):
        if isinstance(body, dict):
            choices = body.get('choices')
            if isinstance(choices, list) and choices:
                message = choices[0].get('message', {})
                content = message.get('content', '')
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text_parts.append(str(item.get('text', '')))
                    return ''.join(text_parts).strip()

            output = body.get('output')
            if isinstance(output, dict):
                text = output.get('text')
                if isinstance(text, str):
                    return text.strip()

        return ''

    def evaluate_signal(self, market_data, prompt_mode='aggressive'):
        if not self.api_key:
            logger.error('未配置 AI_API_KEY，跳过 AI 评估。')
            return None

        symbol = market_data.get('symbol', 'UNKNOWN')
        logger.info(f'🧠 [{symbol}] 启动 AI 独立盲测评估... (模式: {prompt_mode.upper()})')

        klines_1h_str = self._format_klines(market_data.get('raw_klines_1h', []))
        klines_4h_str = self._format_klines(market_data.get('raw_klines_4h', []))
        market_sentiment_summary = market_data.get('market_sentiment_summary', '\u3010\u5e02\u573a\u60c5\u7eea\u6458\u8981 - \u8d44\u91d1\u8d39\u7387\u3011\n- \u8d44\u91d1\u8d39\u7387\u6570\u636e\u6682\u4e0d\u53ef\u7528\n\n\u8bf7\u7ed3\u5408\u4ee5\u4e0a\u60c5\u7eea\u80cc\u666f\u5ba1\u6838\u5f53\u524d\u4ea4\u6613\u4fe1\u53f7\uff1a\n- \u60c5\u7eea\u6570\u636e\u4ec5\u4f5c\u4e3a\u8f85\u52a9\u53c2\u8003\uff0c\u4e0d\u662f\u786c\u6027\u5426\u51b3\u6761\u4ef6\uff0c\u6700\u7ec8\u4ecd\u4ee5\u6280\u672f\u9762\u7ed3\u6784\u4e3a\u6838\u5fc3\u4f9d\u636e\u3002')

        def _risk_status_lines():
            cci_1h = market_data.get('cci_1h')
            rsi_1h = market_data.get('rsi_1h')
            price = market_data.get('price')
            ema20_1h = market_data.get('ema20_1h')
            atr_1h = market_data.get('atr')
            boll_high_1h = market_data.get('boll_high_1h')
            boll_low_1h = market_data.get('boll_low_1h')
            # Prefer the canonical AI RR field and fall back to legacy aliases for compatibility.
            rr_value = market_data.get('realistic_rr', market_data.get('rr', market_data.get('current_rr')))
            pullback_flag = market_data.get('pullback_ok')

            try:
                cci_value = float(cci_1h)
                if cci_value > 180:
                    cci_status = '极端超买，追多风险高'
                elif cci_value < -180:
                    cci_status = '极端超卖，追空风险高'
                else:
                    cci_status = '正常'
                cci_text = f"{cci_value:.2f}"
            except (TypeError, ValueError):
                cci_status = '数据暂不可用'
                cci_text = '数据暂不可用'

            try:
                rsi_value = float(rsi_1h)
                if rsi_value > 75:
                    rsi_status = '超买区，谨慎追多'
                elif rsi_value < 25:
                    rsi_status = '超卖区，谨慎追空'
                else:
                    rsi_status = '正常'
                rsi_text = f"{rsi_value:.2f}"
            except (TypeError, ValueError):
                rsi_status = '数据暂不可用'
                rsi_text = '数据暂不可用'

            deviation_text = '数据暂不可用'
            deviation_status = '数据暂不可用'
            try:
                price_value = float(price)
                ema20_value = float(ema20_1h)
                atr_value = float(atr_1h)
                deviation_pct = ((price_value - ema20_value) / ema20_value) * 100 if ema20_value else 0.0
                atr_ratio = abs(price_value - ema20_value) / atr_value if atr_value else 0.0
                deviation_text = f"{deviation_pct:+.2f}%"
                deviation_status = '乖离过大，回归风险高' if atr_ratio > 1.5 else '正常'
            except (TypeError, ValueError, ZeroDivisionError):
                pass

            boll_position = '数据暂不可用'
            try:
                price_value = float(price)
                boll_high_value = float(boll_high_1h)
                boll_low_value = float(boll_low_1h)
                if price_value >= boll_high_value:
                    boll_position = '突破上轨，处于极端位置'
                elif price_value <= boll_low_value:
                    boll_position = '跌破下轨，处于极端位置'
                else:
                    boll_position = '位于布林带区间内'
            except (TypeError, ValueError):
                pass

            try:
                rr_float = float(rr_value)
                rr_text = f"{rr_float:.2f}"
                if rr_float < 1.0:
                    rr_status = '空间不足'
                elif rr_float <= 1.5:
                    rr_status = '空间偏窄'
                else:
                    rr_status = '空间充足'
            except (TypeError, ValueError):
                rr_text = '数据暂不可用'
                rr_status = '数据暂不可用'

            if pullback_flag is True:
                pullback_text = 'OK（价格贴近均线）'
            elif pullback_flag is False:
                pullback_text = 'NO（价格已拉开）'
            else:
                pullback_text = '数据暂不可用'

            return {
                'cci_text': cci_text,
                'cci_status': cci_status,
                'rsi_text': rsi_text,
                'rsi_status': rsi_status,
                'deviation_text': deviation_text,
                'deviation_status': deviation_status,
                'boll_position': boll_position,
                'rr_text': rr_text,
                'rr_status': rr_status,
                'pullback_text': pullback_text,
            }

        risk_summary = _risk_status_lines()

        if prompt_mode == 'aggressive':
            system_prompt = """你是一名偏激进但严格把关“入场位置”的合约交易风控审核员。

你的核心职责：
量化系统已经完成了方向判断和初步位置过滤。你的任务不是重新判断市场应该做多还是做空，而是审查“当前这个位置值不值得参与，以及更适合用什么强度参与”。

你需要像一个专业交易员一样问自己：
- 现在入场是否在追涨杀跌（追在趋势的最末端）？
- 当前价格是否已经过度偏离均线，存在强烈的回归风险？
- 盈亏比空间是否因为前方障碍位过近而严重不足？
- 市场情绪是否已经进入极端高潮（贪婪/恐慌），随时可能反转？
- 如果不是完美位置，是否仍值得轻仓/试仓参与，而不是直接放弃？

审核原则：
1. 你默认尊重量化系统给出的方向，除非当前位置风险过高。
2. 重点关注以下位置风险信号，出现任意一条即应警惕：
   - 1H CCI 处于极端超买区（>180）时做多，或极端超卖区（<-180）时做空。
   - 1H RSI 超过 75 仍做多，或低于 25 仍做空。
   - 价格距离 1H EMA20 过远（以 ATR 衡量为乖离过大）。
   - 价格已触及或突破布林带外轨，仍顺势追单。
   - 盈亏比（RR）低于 1.0，盈利空间不足以覆盖风险。
3. 如果位置不是最佳，但结构未坏、方向未错、RR 尚可，不要轻易 PASS。应通过降低 confidence 来表达“可参与但应缩仓/试仓”。
4. 只有当位置风险极高、RR 明显太差、或追涨杀跌特征非常集中时，才输出 PASS。
5. 不要输出“市场结构偏多/偏空”这类重复量化工作的描述，而应输出“当前位置适合重仓/轻仓/试仓/不适合入场”的判断。

confidence 评分标准（结合位置风险）：
- 0-49：位置风险极高，必须 PASS
- 50-57：位置较差，但若方向仍成立，可理解为仅适合极轻仓试单
- 58-64：位置一般，可理解为边缘参与，适合小仓试错
- 65-71：位置尚可，适合理性缩仓参与
- 72-85：位置舒适，风险可控，顺势通过
- 86-100：完美位置（回踩均线+动能同步+盈亏比优秀），仅在证据非常充分时使用

输出要求：
1. 只返回合法 JSON。
2. 不要输出 markdown、解释文字、代码块或 JSON 之外的任何内容。
3. action 只能是 LONG、SHORT、PASS。
4. confidence 必须为 0-100。
5. reason 必须简洁说明，重点描述位置风险评估，不超过 100 字。
6. leverage_factor 只能是 0.50、0.70、0.85、1.00 之一；边缘试仓用 0.50，中等机会用 0.70，较好机会用 0.85，高质量机会用 1.00。

你必须只返回如下格式：
{
  "action": "LONG" | "SHORT" | "PASS",
  "confidence": 0-100,
  "reason": "简洁说明位置风险，不超过100字",
  "leverage_factor": 0.50 | 0.70 | 0.85 | 1.00
}"""
        else:
            system_prompt = """你是一名偏保守的合约交易风控审核员。
你的核心任务是过滤掉低质量机会，只在方向明确、结构清晰、盈亏空间合理时才允许交易。

规则：
1. 如果趋势不清晰、信号冲突、支撑阻力过近、量价不匹配，优先 PASS。
2. 只有在多项证据一致时，才考虑 LONG 或 SHORT。
3. 结合 1H/4H 摘要、量能、MACD、CCI、EMA、支撑阻力综合判断。
4. confidence 标准：0-49 必须 PASS；50-59 仍建议 PASS；60-69 可谨慎通过；70-80 方向较明确；81-100 高质量确认机会。
5. 无法确认时必须 PASS。
6. reason 必须引用至少两个客观依据。

你必须只返回合法 JSON：
{
  "action": "LONG" | "SHORT" | "PASS",
  "confidence": 0-100,
  "reason": "简洁说明，不超过100字"
}"""

        direction = str(market_data.get('direction', '') or '').upper()
        direction_task_note = ''
        if prompt_mode == 'aggressive' and direction == 'SHORT':
            direction_task_note = (
                '【本次审核任务】\n你正在审核一个 SHORT（做空）候选信号。\n量化系统已经确认方向，你的任务不是重新判断空头是否成立，而是审查：当前这个做空位置是否值得立即执行。\n如果方向虽对，但位置已经过度延伸、追空风险高、或盈亏比过差，应降低 confidence，必要时直接 PASS。'
            )
        elif prompt_mode == 'aggressive' and direction == 'LONG':
            direction_task_note = (
                '【本次审核任务】\n你正在审核一个 LONG（做多）候选信号。\n量化系统已经确认方向，你的任务不是重新判断多头是否成立，而是审查：当前这个做多位置是否值得立即执行。\n如果方向虽对，但位置已经过度延伸、追多风险高、或盈亏比过差，应降低 confidence，必要时直接 PASS。'
            )
        elif direction == 'SHORT':
            direction_task_note = (
                '【本次审核任务】\n你正在审核一个 **SHORT（做空）** 候选信号。\n量化系统已经给出了明确的空头评分，你的任务是判断是否应该**执行这个做空交易**。\n除非出现致命硬伤（如极端反转形态、盈亏比严重不足、或信号方向与大周期完全相反且无任何反转支撑），否则应倾向于同意该方向。'
            )
        elif direction == 'LONG':
            direction_task_note = (
                '【本次审核任务】\n你正在审核一个 **LONG（做多）** 候选信号。\n量化系统已经给出了明确的多头评分，你的任务是判断是否应该**执行这个做多交易**。\n除非出现致命硬伤（如极端反转形态、盈亏比严重不足、或信号方向与大周期完全相反且无任何反转支撑），否则应倾向于同意该方向。'
            )

        funding_principle = '\n- 资金费率情绪标签为“多头极度拥挤”时，对做多信号应适当谨慎；为“空头极度拥挤”时，对做空信号应适当谨慎。情绪极端时不意味着必须反向，但应提高对技术面确认强度的要求。'
        system_prompt += funding_principle

        user_prompt = f"""
请审核以下市场数据，只返回 JSON。

要求：
- 不适合交易时直接返回 PASS。
- 若返回 LONG 或 SHORT，confidence 必须与证据强度匹配。
- reason 用一句话说明核心依据。

交易对: {symbol}
当前价格: {market_data.get('price', 'N/A')}
1H摘要: {klines_1h_str}
4H摘要: {klines_4h_str}
趋势状态: {market_data.get('trend_state', 'N/A')}
ATR: {market_data.get('atr', 'N/A')}
裸K形态: {market_data.get('naked_k', 'N/A')}
1H量能比: {market_data.get('vol_ratio_1h', 'N/A')}
EMA趋势: {market_data.get('ema_trend', 'N/A')}
MACD: {market_data.get('macd_info', 'N/A')}
CCI: {market_data.get('cci_info', 'N/A')}
1H支撑/阻力: {market_data.get('support_1h', 'N/A')} / {market_data.get('resistance_1h', 'N/A')}
4H支撑/阻力: {market_data.get('support_4h', 'N/A')} / {market_data.get('resistance_4h', 'N/A')}
资金流: {market_data.get('money_flow', '未知')}

"""
        if direction_task_note:
            user_prompt = f"{direction_task_note}\n\n{user_prompt.lstrip()}"
        if prompt_mode == 'aggressive':
            user_prompt += f"""
【位置风险摘要 - 请重点审查】
- 1H CCI：{risk_summary['cci_text']}（{risk_summary['cci_status']}）
- 1H RSI：{risk_summary['rsi_text']}（{risk_summary['rsi_status']}）
- 价格偏离 1H EMA20：{risk_summary['deviation_text']}（{risk_summary['deviation_status']}）
- 布林带位置：{risk_summary['boll_position']}
- 当前盈亏比（RR）：{risk_summary['rr_text']}（{risk_summary['rr_status']}）
- 15M 回踩状态：{risk_summary['pullback_text']}

请基于以上位置风险指标，判断当前是否适合立即入场。如果位置风险较高，请给出降低 confidence 或 PASS 的建议。

"""
        user_prompt += f"\n{market_sentiment_summary}\n"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            'temperature': 0.2,
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    proxies=self.proxies,
                    timeout=self.timeout,
                )
                if response.status_code >= 500:
                    response.raise_for_status()
                if response.status_code >= 400:
                    logger.error(f'AI API 客户端错误 {response.status_code}: {response.text[:500]}')
                    return None

                raw_text = response.text.strip()
                if not raw_text:
                    raise ValueError('AI API 返回空响应，请检查 AI_API_URL 是否正确，或接口是否需要不同协议。')

                try:
                    body = _repair_json_strings(self._decode_response_json(response))
                except json.JSONDecodeError:
                    content_type = response.headers.get('Content-Type', 'unknown')
                    preview = raw_text[:300].replace('\n', ' ')
                    raise ValueError(f'AI API 返回的不是 JSON，Content-Type={content_type}，内容预览: {preview}')

                reply_text = _repair_mojibake(self._extract_reply_text(body))
                if not reply_text:
                    preview = json.dumps(body, ensure_ascii=False)[:400]
                    raise ValueError(f'AI API JSON 结构中未找到有效回复内容，响应预览: {preview}')

                start_idx = reply_text.find('{')
                end_idx = reply_text.rfind('}')
                if start_idx == -1 or end_idx == -1:
                    raise ValueError(f'AI 回复中未找到 JSON 结果，内容预览: {reply_text[:300]}')

                ai_result = _repair_json_strings(json.loads(reply_text[start_idx:end_idx + 1]))
                if str(prompt_mode).lower() == 'aggressive':
                    ai_result = self._soften_aggressive_pass(market_data, ai_result)
                action = str(ai_result.get('action', 'PASS')).upper()
                if action not in ['LONG', 'SHORT', 'PASS']:
                    raise ValueError(f'非法 action: {action}')

                confidence = ai_result.get('confidence', 0)
                try:
                    ai_result['confidence'] = float(confidence)
                except (TypeError, ValueError):
                    ai_result['confidence'] = 0.0
                ai_result['action'] = action
                logger.info(f'🧠 [{symbol}] AI 评估成功 | 动作: {action} | 置信度: {ai_result["confidence"]:.1f}')
                return ai_result
            except Exception as e:
                logger.warning(f'⚠️ AI 评估异常 (第 {attempt + 1}/{self.max_retries} 次): {e}')
                time.sleep(2 ** attempt)

        logger.error(f'🛑 [{symbol}] AI 评估彻底失败，当前轮已降级为纯算法处理，不中断主循环。')
        return None

    def evaluate_position(self, position_data, prompt_mode='aggressive'):
        if not self.api_key:
            logger.error('未配置 AI_API_KEY，跳过持仓后 AI 评估。')
            return None

        symbol = position_data.get('symbol', 'UNKNOWN')
        logger.info(f'🧠 [{symbol}] 启动持仓后 AI 评估... (模式: {str(prompt_mode).upper()})')

        system_prompt = """你是一名合约交易持仓风控助手。
你的任务不是判断是否新开仓，而是判断当前这笔已持仓位应该如何管理。

你只能输出以下四种动作之一：
- 继续持有
- 减仓防守
- 推保本线
- 全部退出

判断重点：
1. 当前持仓方向是否仍然成立。
2. 1H/4H 结构是否破坏。
3. 浮盈是否明显回吐，或浮亏是否正在扩大。
4. 当前价格是否接近止损、阻力、支撑、EMA20 或 VWAP。
5. 是否应该保护已有利润。
6. 不要因为短线噪音频繁建议退出。

输出要求：
1. 只返回合法 JSON。
2. 不要输出 markdown、解释文字、代码块或 JSON 之外的任何内容。
3. action 只能是：继续持有、减仓防守、推保本线、全部退出。
4. confidence 必须为 0-100。
5. reason 必须一句话说明原因，不超过 100 字。
6. risk_level 只能是 low、normal、high。

你必须只返回如下格式：
{
  "action": "继续持有" | "减仓防守" | "推保本线" | "全部退出",
  "confidence": 0-100,
  "reason": "一句话说明原因，不超过100字",
  "risk_level": "low" | "normal" | "high"
}"""

        user_prompt = f"""
请评估当前持仓是否需要调整，只返回 JSON。

账户: {position_data.get('account_name', 'N/A')}
交易对: {symbol}
订单ID: {position_data.get('trade_id', 'N/A')}
持仓方向: {position_data.get('side', 'N/A')}
开仓价: {position_data.get('entry_price', 'N/A')}
当前价: {position_data.get('current_price', 'N/A')}
浮动盈亏: {position_data.get('unrealized_pnl', 'N/A')} U
浮动盈亏比例: {position_data.get('unrealized_pnl_pct', 'N/A')}%
持仓时间: {position_data.get('holding_minutes', 'N/A')} 分钟
当前止损价: {position_data.get('stop_loss', 'N/A')}
当前止盈价: {position_data.get('take_profit', 'N/A')}
止盈阶段: {position_data.get('tp_phase', 'N/A')}

1H趋势: {position_data.get('trend_state', 'N/A')}
EMA趋势: {position_data.get('ema_trend', 'N/A')}
MACD: {position_data.get('macd_info', 'N/A')}
CCI: {position_data.get('cci_info', 'N/A')}
1H支撑/阻力: {position_data.get('support_1h', 'N/A')} / {position_data.get('resistance_1h', 'N/A')}
4H支撑/阻力: {position_data.get('support_4h', 'N/A')} / {position_data.get('resistance_4h', 'N/A')}
15M价格: {position_data.get('price_15m', 'N/A')}
15M EMA20: {position_data.get('ema20_15m', 'N/A')}
15M VWAP: {position_data.get('vwap_15m', 'N/A')}
15M CCI: {position_data.get('cci_15m', 'N/A')}

请只判断这笔已持仓位的处理建议，不要输出新开仓方向。
"""

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            'temperature': 0.2,
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    proxies=self.proxies,
                    timeout=self.timeout,
                )
                if response.status_code >= 500:
                    response.raise_for_status()
                if response.status_code >= 400:
                    logger.error(f'持仓后 AI API 客户端错误 {response.status_code}: {response.text[:500]}')
                    return None

                raw_text = response.text.strip()
                if not raw_text:
                    raise ValueError('持仓后 AI API 返回空响应。')

                try:
                    body = _repair_json_strings(self._decode_response_json(response))
                except json.JSONDecodeError:
                    content_type = response.headers.get('Content-Type', 'unknown')
                    preview = raw_text[:300].replace('\n', ' ')
                    raise ValueError(f'持仓后 AI API 返回的不是 JSON，Content-Type={content_type}，内容预览: {preview}')

                reply_text = _repair_mojibake(self._extract_reply_text(body))
                if not reply_text:
                    preview = json.dumps(body, ensure_ascii=False)[:400]
                    raise ValueError(f'持仓后 AI JSON 结构中未找到有效回复内容，响应预览: {preview}')

                start_idx = reply_text.find('{')
                end_idx = reply_text.rfind('}')
                if start_idx == -1 or end_idx == -1:
                    raise ValueError(f'持仓后 AI 回复中未找到 JSON 结果，内容预览: {reply_text[:300]}')

                result = _repair_json_strings(json.loads(reply_text[start_idx:end_idx + 1]))
                action = str(result.get('action', '')).strip()
                if action not in POSITION_AI_ACTIONS:
                    raise ValueError(f'非法持仓后 action: {action}')

                try:
                    result['confidence'] = float(result.get('confidence', 0) or 0)
                except (TypeError, ValueError):
                    result['confidence'] = 0.0
                result['action'] = action
                risk_level = str(result.get('risk_level', 'normal') or 'normal').strip().lower()
                result['risk_level'] = risk_level if risk_level in POSITION_AI_RISK_LEVELS else 'normal'
                result['reason'] = str(result.get('reason', '') or '')[:100]
                logger.info(f'🧠 [{symbol}] 持仓后 AI 评估成功 | 动作: {action} | 置信度: {result["confidence"]:.1f} | 风险: {result["risk_level"]}')
                return result
            except Exception as e:
                logger.warning(f'⚠️ 持仓后 AI 评估异常 (第 {attempt + 1}/{self.max_retries} 次): {e}')
                time.sleep(2 ** attempt)

        logger.error(f'🛑 [{symbol}] 持仓后 AI 评估彻底失败，本轮跳过。')
        return None
