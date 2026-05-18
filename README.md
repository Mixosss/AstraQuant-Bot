# AstraQuant

> 基于 OKX 的 AI 辅助量化交易框架。
> AI-assisted quantitative trading framework for OKX crypto futures.

AstraQuant 是一个面向 OKX 合约交易的自动化量化框架，核心目标不是让 AI 独立自由开单，而是通过“量化信号先筛选、AI 再审核、风控最终放行”的方式，尽量把交易决策拆成可解释、可复盘、可控制的多个环节。

欢迎大家下载、研究和交流这个项目。如果你在使用过程中遇到问题，或有改进建议，可以通过邮件联系我：wadusec@outlook.com。

---

## 目录

- [项目定位](#项目定位)
- [核心特性](#核心特性)
- [策略框架](#策略框架)
- [风控体系](#风控体系)
- [AI 在系统中的作用](#ai-在系统中的作用)
- [Web 面板与通知](#web-面板与通知)
- [与其他量化框架的对比](#与其他量化框架的对比)
- [项目结构](#项目结构)
- [安装与配置](#安装与配置)
- [启动方式](#启动方式)
- [快速上手建议](#快速上手建议)
- [常用配置](#常用配置)
- [常见不开单原因](#常见不开单原因)
- [回测](#回测)
- [安全说明](#安全说明)
- [English Summary](#english-summary)
- [许可证](#许可证)
- [免责声明](#免责声明)

---

## 项目定位

AstraQuant 是一个偏实盘监控和执行的量化交易机器人框架，主要用于：

1. 多币种行情扫描；
2. 多周期技术面分析；
3. 量化信号打分；
4. AI 辅助开仓审核；
5. 动态仓位和 RR 过滤；
6. 持仓后的风险复评；
7. Web 面板监控和消息通知；
8. 历史数据回测与策略复盘。

它的设计重点是“多层过滤”和“风险优先”，而不是追求高频开单。系统会在开仓前经过量化分数、AI 置信度、RR、仓位、最小名义金额、趋势环境等多道检查，只有同时满足条件时才会进入执行环节。

---

## 核心特性

- OKX V5 行情与交易执行；
- 支持 `SIMULATE`、`TESTNET`、`REAL` 三种模式；
- 支持 BTC、ETH、SOL、XRP 等主流币种；
- 多周期行情分析，包括 15m、1h、4h、1d；
- 多空双向量化打分；
- AI 辅助开仓审核；
- AI 持仓后风险复评；
- BTC 专用分层 RR 过滤；
- ETH / SOL / XRP 的小账户最小开仓保护逻辑；
- 动态仓位系数；
- 1R / 2R 分阶段减仓与止损推进；
- 钉钉、飞书 Webhook 通知；
- Web Dashboard 监控面板；
- 回测工具。

---

## 策略框架

### 1. 多周期行情分析

系统不会只依赖单个指标或单个周期，而是结合多个周期判断行情状态：

- 15m：用于短线入场质量确认；
- 1h：用于当前交易节奏和短周期趋势判断；
- 4h：用于中周期趋势和关键支撑阻力判断；
- 1d：用于更高周期方向过滤。

常用分析信息包括：

- EMA 趋势结构；
- MACD、CCI、RSI 等技术指标；
- ATR 波动率；
- 成交量变化；
- 支撑位与阻力位；
- 裸 K 和反转形态；
- 近期高低点结构；
- 趋势延续或反转环境。

### 2. 量化信号打分

系统会分别计算多头分数和空头分数：

- `long_score`：多头方向质量；
- `short_score`：空头方向质量；
- `score_threshold`：当前环境下的动态开仓阈值。

只有当某一方向的分数达到阈值，并且策略文本能够形成明确方向时，系统才会进入后续开仓评估。分数未达到阈值时，通常只会作为观察信号或 AI 预检查信号，不会直接开仓。

### 3. AI 开仓审核

量化策略给出候选方向后，AI 会对信号进行二次审核。AI 的输出通常包括：

- 建议方向：`LONG`、`SHORT` 或 `PASS`；
- 置信度；
- 支撑位和阻力位；
- 交易理由；
- 是否偏谨慎或仅适合试仓。

AI 顺势审核大致分为：

| AI 置信度 | 系统含义 | 典型处理 |
| --- | --- | --- |
| ≥ 72 且理由不谨慎 | 强通过 | 可使用较完整仓位系数 |
| ≥ 65 | 谨慎通过 | 降低仓位后放行 |
| ≥ 60 | 边缘试仓 | 只允许小仓试探 |
| < 60 | 置信度不足 | 不开仓 |

AI 不是唯一决策者。即使 AI 看多或看空，如果量化分数、RR、趋势环境、仓位计算或最小名义金额不通过，系统仍然不会开单。

### 4. 低分 AI 覆盖逻辑

当量化分数略低于阈值时，系统可以对部分非 BTC 币种进行 AI 覆盖检查，但条件较严格：

- AI 必须明确给出 `LONG` 或 `SHORT`；
- AI 置信度需要达到较高水平；
- 需要出现强反转形态；
- BTC 不使用该低分覆盖逻辑。

这样设计是为了避免 AI 在低质量信号上过度放行，同时保留少量极端反转机会。

### 5. BTC 专用 RR 过滤

BTC 的波动结构和其他主流币不同，系统对 BTC 使用更严格的分层 RR 过滤：

- 近端 RR 太低时直接拦截；
- 近端 RR 中等时，需要检查扩展空间；
- 近端 RR 达标时，仍需要 AI 置信度配合；
- 如果 AI 置信度不足，即使 RR 接近或大于 1，也可能不开仓。

这个设计的目的不是让 BTC 永远少开单，而是避免在追涨追空、近端空间不足、回撤风险较高的位置盲目进场。

### 6. ETH / SOL / XRP 小账户开仓保护

对于 ETH、SOL、XRP 等非 BTC 主流币，系统在部分高质量条件下允许最小名义金额保护，避免小账户经过多重仓位压缩后，实际下单金额过小，导致交易结果没有意义。

但该逻辑仍然需要满足质量约束，例如：

- 净 RR 达到要求；
- 策略分类合适；
- 15m 入场确认通过，或属于强信号场景；
- AI 与策略方向一致；
- 高周期趋势不明显反向压制。

---

## 风控体系

### 1. 开仓前风控

每次开仓前，系统会依次检查：

- 当前交易模式；
- API 配置是否有效；
- 最大持仓数量；
- 币种是否允许交易；
- 量化分数是否达到阈值；
- AI 是否同向且置信度足够；
- RR 和扣除成本后的净 RR 是否达标；
- 预估下单金额是否满足最小名义金额；
- 杠杆和保证金是否在可接受范围内；
- 当前是否存在同向或反向持仓冲突。

### 2. 仓位控制

系统不会简单固定仓位，而是综合多个因素计算最终仓位：

- 币种基础配置；
- 账户余额；
- 保证金比例；
- 策略信号强度；
- AI 置信度；
- RR 质量；
- 波动率状态；
- 最小开仓金额；
- 当前持仓风险敞口。

因此，同样是一个 `LONG` 或 `SHORT` 信号，最终可能出现完整仓位、降低仓位、小仓试探或完全拒绝开仓四种结果。

### 3. 1R / 2R 动态管理

开仓后，系统可以根据盈亏进展进行分阶段风控：

| 阶段 | 条件 | 风控动作 |
| --- | --- | --- |
| 1R | 浮盈达到约 1 倍初始风险 | 部分减仓，并将止损推向保本区 |
| 2R | 浮盈达到约 2 倍初始风险 | 再次部分减仓，并移动止损锁定利润 |

这种方式的目标是在行情顺利时保留利润空间，同时在已经盈利后降低回撤风险。

### 4. 持仓后 AI 复评

Position AI 用于持仓后的风险复查。它可以结合当前价格、浮盈浮亏、止损距离、趋势状态、技术指标和市场结构，给出是否继续持有、减仓、警惕或退出的建议。

需要注意的是，Position AI 应被理解为风控辅助模块，而不是保证盈利的自动决策器。是否允许它执行真实减仓或平仓，需要通过配置明确控制，并且实盘前必须充分测试。

---

## AI 在系统中的作用

AstraQuant 中的 AI 主要承担三个角色：

1. **开仓审核**：对量化候选信号进行二次确认；
2. **方向反转判断**：在少数高置信度场景下识别可能的反向机会；
3. **持仓后复评**：对已有仓位进行风险检查和管理建议。

AI 不负责绕过全部风控，也不应该被理解为“AI 说开就开”。系统设计上保留了量化阈值、RR、趋势、仓位、最小金额和交易模式等硬性约束。

---

## Web 面板与通知

### Web Dashboard

Web 面板用于查看系统运行状态，包括：

- 当前交易模式；
- 当前持仓；
- 开仓审核记录；
- AI 决策记录；
- 持仓后 AI 复评；
- 最近 RR 记录；
- 拒绝开仓原因；
- 风控状态。

启动方式：

```bash
python dashboard_web.py
```

### 消息通知

系统支持通过 Webhook 推送通知：

- 钉钉；
- 飞书。

可用于推送开仓、平仓、风控、AI 复评、异常状态等信息。

---

## 与其他量化框架的对比

| 对比维度 | AstraQuant | 简单指标机器人 | 纯 AI 交易机器人 | 通用回测框架 |
| --- | --- | --- | --- | --- |
| 核心逻辑 | 量化打分 + AI 审核 + 风控放行 | 单指标或少量规则 | 主要依赖模型输出 | 用户自行编写策略 |
| 实盘执行 | 内置 OKX 执行层 | 视实现而定 | 视实现而定 | 通常不内置 |
| 风控强度 | 多层风控 | 通常较弱 | 取决于提示词和模型 | 需要自行实现 |
| 可解释性 | 较强，可查看分数、RR、AI 理由 | 中等 | 容易黑箱化 | 取决于策略代码 |
| 使用复杂度 | 中等偏高 | 低 | 中等 | 中等偏高 |
| 适合场景 | 小账户实盘验证、策略复盘、半自动交易 | 简单规则自动化 | AI 交易实验 | 策略研究和历史测试 |

### 优点

- 不是纯 AI 决策，降低模型幻觉直接开单的风险；
- 开仓前有多层过滤，便于复盘为什么开或不开；
- 同时包含策略、执行、风控、通知和面板；
- 支持模拟盘到实盘的逐步验证；
- 对小账户仓位过小的问题有针对性处理；
- 更适合做可控的策略迭代，而不是一次性脚本。

### 局限

- 系统参数较多，需要理解后再实盘；
- AI 结果依赖模型质量、接口稳定性和提示词质量；
- 风控过严时会错过一些事后看起来盈利的机会；
- 风控过松时会增加交易频率和回撤；
- 不是开箱即盈利系统；
- 仍需要人工监控、复盘和参数调整。

---

## 项目结构

```text
main.py              主程序循环和策略调度
config.py            环境变量配置
okx_engine.py        OKX V5 行情和交易执行层
market_analyzer.py   行情、指标和多周期分析
ai_advisor.py        AI 开仓审核接入
risk_controller.py   动态风控和持仓管理
tool.py              通知、持久化和辅助工具
dashboard_web.py     Web 面板服务
static/              Web 面板静态资源
templates/           Web 面板模板
backtest_tools/      回测工具
```

---

## 安装与配置

### 1. 安装依赖

建议使用 Python 3.10+。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 创建配置文件

复制示例配置：

```bash
copy .env.example .env
```

macOS / Linux：

```bash
cp .env.example .env
```

### 3. OKX 配置

你需要先在 OKX 创建 API Key：

1. 登录 OKX 账户，进入 **个人中心 / API** 页面；
2. 创建新的 API Key，并设置：
   - `API Key`；
   - `Secret Key`；
   - `Passphrase`；
3. 如果只做行情、回测或本地模拟，可以先不开启交易权限；
4. 如果要连接 OKX 模拟盘或实盘交易，需要按 OKX 页面要求开启对应权限；
5. 建议先绑定可信 IP，并优先使用 `TESTNET` 或 OKX 模拟盘验证；
6. 不要把真实 API Key、Secret、Passphrase 提交到 GitHub，也不要发给任何人。

然后在本地 `.env` 文件中填写：

```env
OKX_API_KEY_1=your_okx_api_key_here
OKX_API_SECRET_1=your_okx_api_secret_here
OKX_API_PASSPHRASE_1=your_okx_api_passphrase_here
```

### 4. 交易模式

```env
TRADING_MODE=SIMULATE
```

可选模式：

| 模式 | 含义 |
| --- | --- |
| `SIMULATE` | 本地模拟，不真实连接交易所下单 |
| `TESTNET` | OKX 模拟盘 |
| `REAL` | OKX 实盘 |

建议顺序：

1. 先使用 `SIMULATE`；
2. 再使用 `TESTNET`；
3. 完整验证后再考虑 `REAL`。

### 5. AI 配置

```env
AI_API_KEY=your_ai_api_key_here
AI_API_URL=https://your-openai-compatible-endpoint/v1/chat/completions
AI_MODEL=your_model_name
API_TIMEOUT=60
```

### 6. 通知配置

```env
DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=your_token_here
FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/your_token_here
```

---

## 启动方式

### 启动交易机器人

```bash
python main.py
```

如果 `ENABLE_WEB_DASHBOARD=True`，启动 [main.py](main.py) 后会自动启动 Web 面板，默认监听：

```text
http://127.0.0.1:5000
```

默认端口可在 `.env` 中调整：

```env
ENABLE_WEB_DASHBOARD=True
WEB_DASHBOARD_HOST=127.0.0.1
WEB_DASHBOARD_PORT=5000
```

### 单独启动 Web 面板

如果你只想单独启动 Web 面板，也可以运行：

```bash
python dashboard_web.py
```

启动后，打开终端输出的本地访问地址。

---

## 快速上手建议

建议新用户按下面顺序使用：

1. 先设置 `TRADING_MODE=SIMULATE`，确认程序、Web 面板和通知能正常运行；
2. 配置 AI 后，先观察开仓审核结果，不要急着上实盘；
3. 接入 OKX 模拟盘并运行一段时间，重点查看不开单原因、RR 记录和持仓管理记录；
4. 如果要使用实盘，建议先小资金运行，并持续观察 Web 面板和通知；
5. 持仓风控 AI 默认更适合作为观察和提醒，确认理解风险后再开启自动执行。

AstraQuant 的开仓不是由 AI 单独决定。系统通常会按下面顺序放行：

```text
行情分析 → 量化分数 → AI 开仓审核 → RR / 仓位 / 风控检查 → OKX 执行
```

平仓和持仓管理主要包括止损、1R / 2R 分阶段减仓、止损推进，以及可选的持仓风控 AI 复评。持仓风控 AI 会额外消耗 token，默认不建议新用户一开始就让它自动平仓或减仓。

---

## 常用配置

| 配置项 | 作用 | 新手建议 |
| --- | --- | --- |
| `TRADING_MODE` | 交易模式：`SIMULATE` / `TESTNET` / `REAL` | 先用 `SIMULATE` |
| `ENABLE_WEB_DASHBOARD` | 是否随主程序启动 Web 面板 | 建议开启 |
| `WEB_DASHBOARD_PORT` | Web 面板端口 | 默认 `5000` |
| `AI_API_KEY` | AI 开仓审核所需 Key | 可先配置后观察 |
| `MIN_OPEN_SCORE` | 开仓量化分数门槛 | 不理解前不要随意降低 |
| `MIN_OPEN_FLOOR_NET_RR` | 净 RR 基础门槛 | 不理解前不要随意降低 |
| `MAX_OPEN_POSITIONS` | 最大同时持仓数量 | 小账户建议保守 |
| `ENABLE_POSITION_AI_EXECUTION` | 是否允许持仓风控 AI 自动减仓 / 平仓 | 新手建议 `False` |
| `DINGTALK_WEBHOOK` / `FEISHU_WEBHOOK` | 通知推送 | 可选 |

如果只想降低 AI token 消耗，可以不开启持仓风控 AI 自动执行，或适当延长持仓 AI 的评估间隔。

---

## 常见不开单原因

如果 AI 给出 `LONG` 或 `SHORT` 但系统没有开单，通常不是异常，而是某个风控条件没有通过。常见原因包括：

- 量化分数没有达到开仓门槛；
- AI 方向和策略方向不一致，或 AI 置信度不足；
- 净 RR 不够，扣除手续费、滑点或资金费率后空间不足；
- 15m 入场确认不足，追涨追空风险较高；
- 当前已有同方向或反方向持仓冲突；
- 下单金额低于交易所最小名义金额；
- 单币种、同方向或组合风险敞口过高。

Web 面板中的开仓记录、AI 决策记录、RR 记录和拒单原因可以帮助你判断具体是哪一道检查拦截了交易。

---

## 回测

回测工具位于：

```text
backtest_tools/
```

示例：

```bash
python backtest_tools/backtestNew.py
```

回测输出属于本地运行产物，不建议提交到公开仓库。

---

## 安全说明

下载或 Fork 本项目后，请在你自己的本地环境中创建 `.env` 文件，并填入你自己的 API 和通知配置。

本公开仓库不包含真实 API Key、Webhook Token、交易日志、账户快照、持仓状态、行情缓存、个人笔记或本地测试脚本。使用者在二次开发或部署时，也应避免把这些敏感文件提交到公开仓库。

请不要提交以下内容：

- `.env`；
- 真实 OKX API Key、Secret、Passphrase；
- AI API Key；
- 钉钉或飞书真实 Webhook；
- 交易日志；
- 账户快照；
- 持仓状态；
- 行情缓存；
- 本地测试脚本；
- 个人笔记或提示词草稿。

如果真实密钥曾经暴露到公开仓库，请立即去对应平台重置。

---

## English Summary

AstraQuant is an AI-assisted quantitative trading framework for OKX crypto futures. It combines multi-timeframe market analysis, rule-based signal scoring, AI-assisted entry review, dynamic risk control, dashboard monitoring, webhook notifications, and backtesting utilities.

If you have questions, feedback, or suggestions, feel free to contact the maintainer by email: wadusec@outlook.com.

The public repository does not include real API keys, webhook tokens, trading logs, account snapshots, active-position state, cached market data, personal notes, or local test scripts.

The framework is not designed as a pure AI auto-trader. Its workflow is:

1. The quantitative strategy analyzes the market and creates a candidate direction.
2. The scoring system checks whether the signal reaches the required threshold.
3. AI reviews the candidate signal and provides action, confidence, and reasoning.
4. Risk-control logic checks RR, position size, leverage, minimum notional, and exposure.
5. Orders are sent through OKX only when all required gates pass.

Main components:

- `main.py`: main strategy loop and orchestration;
- `config.py`: environment-based configuration;
- `okx_engine.py`: OKX V5 execution layer;
- `market_analyzer.py`: market data and indicator analysis;
- `ai_advisor.py`: AI review integration;
- `risk_controller.py`: dynamic risk management;
- `dashboard_web.py`: web dashboard;
- `backtest_tools/`: backtesting utilities.

Supported modes:

- `SIMULATE`: local simulation;
- `TESTNET`: OKX demo trading;
- `REAL`: live trading.

Always validate with `SIMULATE` and `TESTNET` before using `REAL` mode. This software does not guarantee profit and should be used at your own risk.

---

## 许可证

本项目基于 MIT License 开源。你可以自由使用、修改和分发本项目代码，但需要保留原始版权声明和许可证文本。详情请查看 [LICENSE](LICENSE)。

This project is open-sourced under the MIT License. You may use, modify, and distribute the code, provided that the original copyright notice and license text are retained. See [LICENSE](LICENSE) for details.

---

## 免责声明

AstraQuant 仅供个人学习、技术研究、策略回测和自动化交易实验使用，不构成任何投资建议、交易建议、理财建议、资产管理服务或收益承诺。

加密货币及合约交易具有极高风险，可能因市场波动、杠杆、流动性、交易所规则、网络延迟、API 配置错误、程序缺陷、第三方 AI 输出或使用者操作不当而造成部分或全部本金损失，甚至发生强制平仓。

任何下载、Fork、修改、部署或运行本项目的使用者，都应在充分理解代码逻辑和交易风险后，自行决定是否使用，并自行承担由此产生的全部风险和后果。项目作者和维护者不对任何直接或间接损失、交易亏损、爆仓、数据错误、服务中断或其他后果承担责任。

本项目不保证盈利，不保证策略有效性，也不保证在任何市场环境下稳定运行。实盘前请务必先在 `SIMULATE` 或 `TESTNET` 环境中充分测试。

By downloading, forking, modifying, deploying, or running this project, you acknowledge that AstraQuant is provided for research and educational purposes only. It is not financial advice, investment advice, trading advice, asset management, or a profit guarantee. You are solely responsible for your own configuration, API permissions, leverage, position size, trading decisions, and any resulting gains or losses. The author and maintainers shall not be liable for any direct or indirect loss arising from the use of this software.
