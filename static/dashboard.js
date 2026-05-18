const state = {
  symbols: [],
  sortKey: 'symbol',
  sortDir: 'asc',
};

async function fetchJson(url) {
  const resp = await fetch(url, { cache: 'no-store' });
  if (!resp.ok) {
    throw new Error(`${url} -> ${resp.status}`);
  }
  return resp.json();
}

function formatValue(value) {
  return value ?? '--';
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || value === '') {
    return '--';
  }
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : String(value);
}

function formatSigned(value, digits = 2) {
  if (value === null || value === undefined || value === '') {
    return '--';
  }
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return String(value);
  }
  return `${num > 0 ? '+' : ''}${num.toFixed(digits)}`;
}

function sideClass(side) {
  const upper = String(side || '').toUpperCase();
  if (upper === 'LONG') return 'long';
  if (upper === 'SHORT') return 'short';
  if (upper === 'PASS') return 'pass';
  return 'none';
}

function fundingClass(text) {
  const content = String(text || '');
  if (content.includes('多头') || content.includes('空头')) {
    return content.includes('极度拥挤') || content.includes('偏热') ? 'hot' : 'cold';
  }
  return 'neutral';
}

function renderOverview(data) {
  const root = document.getElementById('overview-cards');
  const entries = [
    ['模式', data.mode],
    ['AI 开仓', data.ai_enabled ? '开启' : '关闭'],
    ['AI 平仓', data.ai_close_enabled ? '开启' : '关闭'],
    ['持仓AI执行', data.position_ai_execution_enabled ? '开启' : '关闭'],
    ['监控币种', data.monitored_symbol_count],
    ['账户数', data.account_count],
    ['总余额', formatNumber(data.total_balance)],
    ['机器人24H', formatSigned(data.trade_24h_pnl)],
    ['机器人总盈利', formatSigned(data.total_trade_pnl)],
    ['持仓数', data.open_position_count],
  ];
  root.innerHTML = entries.map(([label, value]) => `
    <div class="stat-card">
      <div class="stat-label">${label}</div>
      <div class="stat-value">${formatValue(value)}</div>
    </div>
  `).join('');
  document.getElementById('last-refresh').textContent = `最近刷新: ${data.last_refresh_time || '--'}`;
}

function renderBucketSummary(rows) {
  const root = document.getElementById('bucket-summary-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无策略桶统计</div>';
    return;
  }
  root.innerHTML = rows.map(row => `
    <div class="summary-card bucket-card">
      <div class="card-title">
        <strong>${formatValue(row.strategy_bucket)}</strong>
        <span class="metric-tag">信号 ${formatValue(row.signal_count)}</span>
      </div>
      <div class="data-row text-muted">
        <span>开仓 ${formatValue(row.trade_count)}</span>
        <span>胜率 ${formatValue(formatNumber(row.win_rate, 1))}%</span>
      </div>
      <div class="data-row text-muted">
        <span>均值净RR ${formatValue(formatNumber(row.avg_net_rr, 2))}</span>
      </div>
      <div class="bucket-pnl ${Number(row.realized_pnl || 0) >= 0 ? 'text-green' : 'text-red'}">
        已实现PnL ${formatSigned(row.realized_pnl, 2)}
      </div>
    </div>
  `).join('');
}

function renderRegimeSummary(rows) {
  const root = document.getElementById('regime-summary-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无市场状态统计</div>';
    return;
  }
  root.innerHTML = rows.map(row => `
    <div class="summary-card bucket-card">
      <div class="card-title">
        <strong>${formatValue(row.market_regime)}</strong>
        <span class="metric-tag">信号 ${formatValue(row.signal_count)}</span>
      </div>
      <div class="data-row text-muted">
        <span>开仓 ${formatValue(row.trade_count)}</span>
        <span>胜率 ${formatValue(formatNumber(row.win_rate, 1))}%</span>
      </div>
      <div class="bucket-pnl ${Number(row.realized_pnl || 0) >= 0 ? 'text-green' : 'text-red'}">
        已实现PnL ${formatSigned(row.realized_pnl, 2)}
      </div>
    </div>
  `).join('');
}

function renderRegimeBucketMatrix(rows) {
  const root = document.getElementById('regime-bucket-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无组合统计</div>';
    return;
  }
  root.innerHTML = rows.map(row => `
    <div class="event-card compact bucket-card">
      <div class="card-title">
        <strong>${formatValue(row.market_regime)}</strong>
        <span class="metric-tag">${formatValue(row.strategy_bucket)}</span>
      </div>
      <div class="data-row text-muted">
        <span>信号 ${formatValue(row.signal_count)}</span>
        <span>开仓 ${formatValue(row.trade_count)}</span>
      </div>
      <div class="data-row text-muted">
        <span>胜率 ${formatValue(formatNumber(row.win_rate, 1))}%</span>
        <span>净RR ${formatValue(formatNumber(row.avg_net_rr, 2))}</span>
      </div>
      <div class="bucket-pnl ${Number(row.realized_pnl || 0) >= 0 ? 'text-green' : 'text-red'}">
        已实现PnL ${formatSigned(row.realized_pnl, 2)}
      </div>
    </div>
  `).join('');
}

function renderOpenRejectionSummary(rows) {
  const root = document.getElementById('open-rejection-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无未开仓原因统计</div>';
    return;
  }
  root.innerHTML = rows.map(row => `
    <div class="summary-card rejection-card">
      <div class="card-title">
        <strong>${formatValue(row.reason)}</strong>
        <span class="metric-tag">${formatValue(row.count)} 次</span>
      </div>
    </div>
  `).join('');
}

function renderOpenRejectionStageSummary(rows) {
  const root = document.getElementById('open-rejection-stage-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无未开仓阶段统计</div>';
    return;
  }
  root.innerHTML = rows.map(row => `
    <div class="summary-card rejection-card">
      <div class="card-title">
        <strong>${formatValue(row.stage)}</strong>
        <span class="metric-tag">${formatValue(row.count)} 次</span>
      </div>
    </div>
  `).join('');
}

function renderBacktestParameterRecommendations(rows) {
  const root = document.getElementById('backtest-parameter-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无回测/复盘参数建议</div>';
    return;
  }
  root.innerHTML = rows.map(row => `
    <div class="summary-card recommendation-card ${String(row.severity || '').toLowerCase()}">
      <div class="card-title">
        <strong>${formatValue(row.severity)}</strong>
        <span class="metric-tag">${formatValue(row.source || 'review')}</span>
      </div>
      <div class="recommendation-text">${formatValue(row.recommendation)}</div>
    </div>
  `).join('');
}

function renderAiShadowQualitySummary(rows) {
  const root = document.getElementById('ai-shadow-quality-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无 AI质量×Shadow 统计</div>';
    return;
  }
  root.innerHTML = rows.map(row => `
    <div class="summary-card bucket-card">
      <div class="card-title">
        <strong>${formatValue(row.confidence_band)}</strong>
        <span class="metric-tag">样本 ${formatValue(row.sample_count)}</span>
      </div>
      <div class="data-row text-muted">
        <span>放行 ${formatValue(row.approved_count)}</span>
        <span>PASS ${formatValue(row.pass_count)}</span>
      </div>
      <div class="data-row text-muted">
        <span>放行胜率 ${formatValue(formatNumber(row.approved_win_rate, 1))}%</span>
        <span>Shadow错过 ${formatValue(formatNumber(row.shadow_missed_rate, 1))}%</span>
      </div>
      <div class="bucket-pnl ${Number(row.avg_shadow_return_pct || 0) >= 0 ? 'text-green' : 'text-red'}">
        平均机会 ${formatSigned(row.avg_shadow_return_pct, 2)}%
      </div>
    </div>
  `).join('');
}

function renderAiShadowRecommendations(rows) {
  const root = document.getElementById('ai-shadow-recommendation-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无 AI质量×Shadow 建议</div>';
    return;
  }
  root.innerHTML = rows.map((item, index) => `
    <div class="summary-card recommendation-card medium">
      <div class="card-title">
        <strong>建议 ${index + 1}</strong>
        <span class="metric-tag">review</span>
      </div>
      <div class="recommendation-text">${formatValue(item)}</div>
    </div>
  `).join('');
}

function renderCalibrationRecommendations(rows) {
  const root = document.getElementById('calibration-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无参数校准建议</div>';
    return;
  }
  root.innerHTML = rows.map((item, index) => `
    <div class="summary-card recommendation-card high">
      <div class="card-title">
        <strong>校准 ${index + 1}</strong>
        <span class="metric-tag">review</span>
      </div>
      <div class="recommendation-text">${formatValue(item)}</div>
    </div>
  `).join('');
}

function renderExecutionQualitySummary(rows) {
  const root = document.getElementById('execution-quality-summary-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无执行质量统计</div>';
    return;
  }
  root.innerHTML = rows.map(row => `
    <div class="summary-card bucket-card">
      <div class="card-title">
        <strong>${formatValue(row.scope)}</strong>
        <span class="metric-tag">${formatValue(row.trade_count)} 笔</span>
      </div>
      <div class="data-row text-muted">
        <span>预估 ${formatValue(formatNumber(row.avg_estimated_cost_u, 4))}U</span>
        <span>实际 ${formatValue(formatNumber(row.avg_actual_cost_u, 4))}U</span>
      </div>
      <div class="data-row text-muted">
        <span>偏差 ${formatValue(formatNumber(row.avg_cost_delta_u, 4))}U</span>
        <span>开平偏差 ${formatValue(formatNumber(row.avg_open_close_cost_delta_u, 4))}U</span>
      </div>
    </div>
  `).join('');
}

function renderExecutionQualityBucketSummary(rows) {
  const root = document.getElementById('execution-quality-bucket-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无策略桶执行质量统计</div>';
    return;
  }
  root.innerHTML = rows.map(row => `
    <div class="summary-card bucket-card">
      <div class="card-title">
        <strong>${formatValue(row.strategy_bucket)}</strong>
        <span class="metric-tag">${formatValue(row.trade_count)} 笔</span>
      </div>
      <div class="data-row text-muted">
        <span>预估 ${formatValue(formatNumber(row.avg_estimated_cost_u, 4))}U</span>
        <span>实际 ${formatValue(formatNumber(row.avg_actual_cost_u, 4))}U</span>
      </div>
      <div class="bucket-pnl ${Number(row.avg_cost_delta_u || 0) <= 0 ? 'text-green' : 'text-red'}">
        平均偏差 ${formatSigned(row.avg_cost_delta_u, 4)}U
      </div>
    </div>
  `).join('');
}

function renderExecutionQualityRegimeSummary(rows) {
  const root = document.getElementById('execution-quality-regime-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无市场状态执行质量统计</div>';
    return;
  }
  root.innerHTML = rows.map(row => `
    <div class="summary-card bucket-card">
      <div class="card-title">
        <strong>${formatValue(row.market_regime)}</strong>
        <span class="metric-tag">${formatValue(row.trade_count)} 笔</span>
      </div>
      <div class="data-row text-muted">
        <span>预估 ${formatValue(formatNumber(row.avg_estimated_cost_u, 4))}U</span>
        <span>实际 ${formatValue(formatNumber(row.avg_actual_cost_u, 4))}U</span>
      </div>
      <div class="bucket-pnl ${Number(row.avg_cost_delta_u || 0) <= 0 ? 'text-green' : 'text-red'}">
        平均偏差 ${formatSigned(row.avg_cost_delta_u, 4)}U
      </div>
    </div>
  `).join('');
}

function renderShadowOpportunitySummary(rows) {
  const root = document.getElementById('shadow-opportunity-summary-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无 Shadow 机会统计</div>';
    return;
  }
  root.innerHTML = rows.map(row => `
    <div class="summary-card bucket-card">
      <div class="card-title">
        <strong>${formatValue(row.scope)}</strong>
        <span class="metric-tag">样本 ${formatValue(row.sample_count)}</span>
      </div>
      <div class="data-row text-muted">
        <span>错过 ${formatValue(row.missed_count)}</span>
        <span>避免亏损 ${formatValue(row.avoided_count)}</span>
      </div>
      <div class="data-row text-muted">
        <span>错过率 ${formatValue(formatNumber(row.missed_rate, 1))}%</span>
        <span>平均机会 ${formatSigned(row.avg_opportunity_return_pct, 2)}%</span>
      </div>
    </div>
  `).join('');
}

function renderShadowOpportunityBucketSummary(rows) {
  const root = document.getElementById('shadow-opportunity-bucket-panel');
  if (!root) {
    return;
  }
  if (!rows || !rows.length) {
    root.innerHTML = '<div class="summary-card text-muted">暂无策略桶 Shadow 统计</div>';
    return;
  }
  root.innerHTML = rows.map(row => `
    <div class="summary-card bucket-card">
      <div class="card-title">
        <strong>${formatValue(row.strategy_bucket)}</strong>
        <span class="metric-tag">样本 ${formatValue(row.sample_count)}</span>
      </div>
      <div class="data-row text-muted">
        <span>错过 ${formatValue(row.missed_count)}</span>
        <span>避免亏损 ${formatValue(row.avoided_count)}</span>
      </div>
      <div class="bucket-pnl ${Number(row.avg_opportunity_return_pct || 0) <= 0 ? 'text-green' : 'text-red'}">
        平均机会 ${formatSigned(row.avg_opportunity_return_pct, 2)}% · 错过率 ${formatNumber(row.missed_rate, 1)}%
      </div>
    </div>
  `).join('');
}

function renderBtcRr(payload) {
  const panel = document.getElementById('btc-rr-panel');
  const tbody = document.querySelector('#btc-rr-table tbody');
  const latest = payload?.latest;
  const modes = payload?.mode_distribution || [];
  const recent = payload?.recent || [];

  if (!latest) {
    if (panel) panel.innerHTML = '<div class="summary-card text-muted">暂无 BTC RR 数据</div>';
    if (tbody) tbody.innerHTML = '<tr><td colspan="8" class="text-muted">暂无 BTC RR 记录</td></tr>';
    return;
  }

  if (panel) {
    panel.innerHTML = `
      <div class="summary-card bucket-card">
        <div class="stat-label">最近近端净RR</div>
        <div class="stat-value">${formatNumber(latest.btc_near_net_rr, 2)}</div>
        <div class="text-muted">目标 ${formatNumber(latest.btc_near_obstacle, 4)}</div>
      </div>
      <div class="summary-card bucket-card">
        <div class="stat-label">最近扩展净RR</div>
        <div class="stat-value">${formatNumber(latest.btc_extended_net_rr, 2)}</div>
        <div class="text-muted">目标 ${formatNumber(latest.btc_extended_obstacle, 4)}</div>
      </div>
      <div class="summary-card bucket-card">
        <div class="stat-label">BTC RR模式</div>
        <div class="stat-value">${formatValue(latest.btc_rr_mode)}</div>
        <div class="text-muted">仓位上限 ${formatNumber(latest.btc_rr_cap, 2)}x</div>
      </div>
      <div class="summary-card bucket-card">
        <div class="stat-label">模式分布</div>
        <div class="event-reason">${modes.map(row => `${formatValue(row.mode)}: ${formatValue(row.count)}`).join('<br>') || '暂无分布'}</div>
      </div>
    `;
  }

  if (tbody) {
    tbody.innerHTML = recent.map(row => `
      <tr>
        <td>${formatValue(row.time)}</td>
        <td><span class="event-badge ${sideClass(row.ai_action)}">${formatValue(row.ai_action)}</span></td>
        <td>${formatNumber(row.ai_confidence, 0)}</td>
        <td>${formatNumber(row.net_rr, 2)}</td>
        <td>${formatNumber(row.btc_near_net_rr, 2)}</td>
        <td>${formatNumber(row.btc_extended_net_rr, 2)}</td>
        <td>${formatValue(row.btc_rr_mode)}</td>
        <td>${formatNumber(row.btc_rr_cap, 2)}</td>
      </tr>
    `).join('') || '<tr><td colspan="8" class="text-muted">暂无 BTC RR 记录</td></tr>';
  }
}

function renderPositionAi(payload) {
  const panel = document.getElementById('position-ai-summary-panel');
  const tbody = document.querySelector('#position-ai-table tbody');
  const summary = payload?.summary || {};
  const rows = payload?.latest_by_trade || [];

  if (panel) {
    panel.innerHTML = `
      <div class="summary-card bucket-card">
        <div class="stat-label">AI监控持仓</div>
        <div class="stat-value">${formatValue(summary.monitored_positions)}</div>
        <div class="text-muted">平均置信度 ${formatNumber(summary.avg_confidence, 1)}</div>
      </div>
      <div class="summary-card bucket-card">
        <div class="stat-label">正常 / 观察</div>
        <div class="stat-value">${formatValue(summary.normal_holding)} / ${formatValue(summary.risk_watching)}</div>
        <div class="text-muted">高风险 ${formatValue(summary.high_risk)}</div>
      </div>
      <div class="summary-card rejection-card">
        <div class="stat-label">确认风险</div>
        <div class="stat-value">${formatValue(summary.confirm_reduce + summary.confirm_breakeven + summary.confirm_exit)}</div>
        <div class="text-muted">减仓 ${formatValue(summary.confirm_reduce)} · 推保本 ${formatValue(summary.confirm_breakeven)} · 退出 ${formatValue(summary.confirm_exit)}</div>
      </div>
    `;
  }

  if (tbody) {
    tbody.innerHTML = rows.map(row => `
      <tr>
        <td>${formatValue(row.time)}</td>
        <td>${formatValue(row.account_name)}</td>
        <td>${formatValue(row.symbol)}</td>
        <td><span class="side-badge ${sideClass(row.side)}">${formatValue(row.side)}</span></td>
        <td class="${Number(row.unrealized_pnl_pct || 0) >= 0 ? 'text-green' : 'text-red'}">${formatSigned(row.unrealized_pnl_pct, 2)}%</td>
        <td>${formatNumber(row.holding_minutes, 0)}</td>
        <td>${formatValue(row.ai_action)}</td>
        <td>${formatNumber(row.ai_confidence, 0)}</td>
        <td>${formatValue(row.risk_level)}</td>
        <td>${formatValue(row.confirmed_action)}</td>
        <td>${formatValue(row.execution_state)}</td>
      </tr>
    `).join('') || '<tr><td colspan="11" class="text-muted">暂无持仓后 AI 记录</td></tr>';
  }
}

function renderSymbols(rows) {
  const tbody = document.querySelector('#symbol-table tbody');
  const query = document.getElementById('symbol-search').value.trim().toUpperCase();
  const positionsOnly = document.getElementById('positions-only').checked;
  let filtered = rows.filter(row => String(row.symbol || '').includes(query));
  if (positionsOnly) {
    filtered = filtered.filter(row => row.has_position);
  }

  filtered.sort((a, b) => {
    const av = a[state.sortKey] ?? '';
    const bv = b[state.sortKey] ?? '';
    const dir = state.sortDir === 'asc' ? 1 : -1;
    if (typeof av === 'number' && typeof bv === 'number') {
      return (av - bv) * dir;
    }
    return String(av).localeCompare(String(bv)) * dir;
  });

  tbody.innerHTML = filtered.map(row => `
    <tr>
      <td>
        <div class="card-title">
          <strong>${formatValue(row.symbol)}</strong>
          <span class="metric-tag">AI ${formatValue(row.latest_ai_action)}</span>
        </div>
      </td>
      <td>${formatNumber(row.latest_price, 4)}</td>
      <td>${formatValue(row.trend_state)}</td>
      <td>${formatNumber(row.long_score, 2)}</td>
      <td>${formatNumber(row.short_score, 2)}</td>
      <td>${formatNumber(row.realistic_rr, 2)}</td>
      <td><span class="funding-badge ${fundingClass(row.funding_sentiment)}">${formatValue(row.funding_sentiment)}</span></td>
      <td><span class="side-badge ${sideClass(row.position_side)}">${row.has_position ? (row.position_side || '持仓中') : '无'}</span></td>
      <td>${formatValue(row.last_update_time)}</td>
    </tr>
  `).join('');
}

function renderPositions(rows) {
  const root = document.getElementById('positions-panel');
  root.innerHTML = rows.map(row => {
    const pnlClass = Number(row.unrealized_pnl || 0) >= 0 ? 'text-green' : 'text-red';
    return `
      <div class="position-card">
        <div class="card-title">
          <strong>${formatValue(row.account_name)} · ${formatValue(row.symbol)}</strong>
          <span class="side-badge ${sideClass(row.side)}">${formatValue(row.side)}</span>
        </div>
        <div class="data-row text-muted"><span>Entry ${formatNumber(row.entry_price, 4)}</span><span>Mark ${formatNumber(row.current_price, 4)}</span></div>
        <div class="data-row text-muted"><span>Qty ${formatNumber(row.quantity, 4)}</span><span>Notional ${formatNumber(row.notional, 2)}</span></div>
        <div class="data-row text-muted"><span>SL ${formatValue(row.stop_loss)}</span><span>TP ${formatValue(row.take_profit)}</span><span>Phase ${formatValue(row.tp_phase)}</span></div>
        <div class="${pnlClass}">PnL ${formatSigned(row.unrealized_pnl)} (${formatSigned(row.unrealized_pnl_percent)}%)</div>
      </div>
    `;
  }).join('') || '<div class="text-muted">暂无持仓</div>';
}

function renderAccountStats(rows) {
  const root = document.getElementById('account-stats-panel');
  root.innerHTML = rows.map(row => `
    <div class="account-card">
      <div class="card-title">
        <strong>${formatValue(row.account_name)}</strong>
        <span class="metric-tag">${row.cooldown_state ? '冷却中' : '活跃'}</span>
      </div>
      <div class="data-row"><span class="text-muted">余额</span><span>${formatNumber(row.current_balance)}</span></div>
      <div class="data-row"><span class="text-muted">机器人24H</span><span>${formatSigned(row.trade_24h_pnl)}</span></div>
      <div class="data-row"><span class="text-muted">机器人总盈利</span><span>${formatSigned(row.total_trade_pnl)}</span></div>
      <div class="data-row"><span class="text-muted">持仓数</span><span>${formatValue(row.open_position_count)}</span></div>
      <div class="data-row"><span class="text-muted">暴露比</span><span>${formatNumber(row.exposure_ratio, 4)}</span></div>
      <div class="data-row"><span class="text-muted">连续亏损</span><span>${formatValue(row.loss_streak_context)}</span></div>
    </div>
  `).join('');
}

function renderLatestAiSummary(rows) {
  const root = document.getElementById('latest-ai-summary');
  root.innerHTML = rows.map(row => `
    <div class="summary-card">
      <div class="card-title">
        <strong>${formatValue(row.symbol)}</strong>
        <span class="event-badge ${sideClass(row.ai_action)}">${formatValue(row.ai_action)}</span>
      </div>
      <div class="data-row text-muted">
        <span>${formatValue(row.time)}</span>
        <span>${formatValue(row.direction)}</span>
      </div>
      <div class="data-row">
        <span class="text-muted">置信度</span>
        <span>${formatValue(row.ai_confidence === null ? '--' : formatNumber(row.ai_confidence, 0))}</span>
      </div>
      <div class="data-row text-muted">
        <span>净RR ${formatValue(row.net_rr === null ? '--' : formatNumber(row.net_rr, 2))}</span>
        <span>成本 ${formatValue(row.estimated_cost_u === null ? '--' : formatNumber(row.estimated_cost_u, 4))}U</span>
      </div>
      <div class="data-row text-muted">
        <span>${formatValue(row.ai_entry_label || '--')}</span>
        <span>杠杆系数 ${formatValue(row.ai_leverage_factor === null ? '--' : formatNumber(row.ai_leverage_factor, 2))}</span>
      </div>
      <div class="data-row text-muted">
        <span>状态 ${formatValue(row.market_regime || '--')}</span>
        <span>策略 ${formatValue(row.strategy_bucket || '--')}</span>
      </div>
      <div class="event-reason">${formatValue(row.reason_summary)}</div>
    </div>
  `).join('');
}

function renderEvents(payload) {
  const summaryRoot = document.getElementById('latest-ai-summary');
  const root = document.getElementById('events-feed');
  const latestRows = payload?.latest_by_symbol || [];
  const groupedEvents = payload?.grouped_events || {};

  renderLatestAiSummary(latestRows);

  const symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT'];
  root.innerHTML = symbols.map(symbol => {
    const rows = groupedEvents[symbol] || [];
    const items = rows.map(row => `
      <div class="event-card compact">
        <div class="event-top">
          <strong>${formatValue(row.time)}</strong>
          <span class="event-badge ${sideClass(row.ai_action)}">${formatValue(row.ai_action)}</span>
        </div>
        <div class="data-row text-muted">
          <span>${formatValue(row.direction)}</span>
          <span>置信度 ${formatValue(row.ai_confidence === null ? '--' : formatNumber(row.ai_confidence, 0))}</span>
        </div>
        <div class="data-row text-muted">
          <span>净RR ${formatValue(row.net_rr === null ? '--' : formatNumber(row.net_rr, 2))}</span>
          <span>成本 ${formatValue(row.estimated_cost_u === null ? '--' : formatNumber(row.estimated_cost_u, 4))}U</span>
        </div>
        <div class="data-row text-muted">
          <span>${formatValue(row.ai_entry_label || '--')}</span>
          <span>杠杆系数 ${formatValue(row.ai_leverage_factor === null ? '--' : formatNumber(row.ai_leverage_factor, 2))}</span>
        </div>
        <div class="data-row text-muted">
          <span>状态 ${formatValue(row.market_regime || '--')}</span>
          <span>策略 ${formatValue(row.strategy_bucket || '--')}</span>
        </div>
        <div class="event-reason">${formatValue(row.reason_summary)}</div>
      </div>
    `).join('') || '<div class="text-muted">暂无事件</div>';

    return `
      <section class="event-group-card">
        <div class="panel-header compact event-group-header">
          <div>
            <p class="section-kicker">${symbol.replace('USDT', '')}</p>
            <h3>${symbol}</h3>
          </div>
          <div class="meta-pill">最近 5 条</div>
        </div>
        <div class="events-feed inner">${items}</div>
      </section>
    `;
  }).join('');
}

async function refreshDashboard() {
  const [overview, symbols, positions, events, accountStats, btcRr, positionAi] = await Promise.all([
    fetchJson('/api/overview'),
    fetchJson('/api/symbols'),
    fetchJson('/api/positions'),
    fetchJson('/api/events'),
    fetchJson('/api/account-stats'),
    fetchJson('/api/btc-rr'),
    fetchJson('/api/position-ai'),
  ]);
  state.symbols = symbols;
  renderOverview(overview);
  renderBucketSummary(overview?.bucket_summary || []);
  renderRegimeSummary(overview?.regime_summary || []);
  renderRegimeBucketMatrix(overview?.regime_bucket_matrix || []);
  renderOpenRejectionSummary(overview?.open_rejection_summary || []);
  renderOpenRejectionStageSummary(overview?.open_rejection_stage_summary || []);
  renderExecutionQualitySummary(overview?.execution_quality_summary || []);
  renderExecutionQualityBucketSummary(overview?.execution_quality_bucket_summary || []);
  renderExecutionQualityRegimeSummary(overview?.execution_quality_regime_summary || []);
  renderBacktestParameterRecommendations(overview?.backtest_parameter_recommendations || []);
  renderAiShadowQualitySummary(overview?.ai_shadow_quality_summary || []);
  renderAiShadowRecommendations(overview?.ai_shadow_quality_recommendations || []);
  renderCalibrationRecommendations(overview?.calibration_recommendations || []);
  renderShadowOpportunitySummary(overview?.shadow_opportunity_summary || []);
  renderShadowOpportunityBucketSummary(overview?.shadow_opportunity_bucket_summary || []);
  renderSymbols(symbols);
  renderPositions(positions);
  renderEvents(events);
  renderBtcRr(btcRr);
  renderPositionAi(positionAi);
  renderAccountStats(accountStats);
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('#symbol-table th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.sort;
      state.sortDir = state.sortKey === key && state.sortDir === 'asc' ? 'desc' : 'asc';
      state.sortKey = key;
      renderSymbols(state.symbols);
    });
  });
  document.getElementById('symbol-search').addEventListener('input', () => renderSymbols(state.symbols));
  document.getElementById('positions-only').addEventListener('change', () => renderSymbols(state.symbols));
  refreshDashboard().catch(err => {
    document.getElementById('events-feed').innerHTML = `<div class="event-card text-red">${err.message}</div>`;
  });
  window.setInterval(() => {
    refreshDashboard().catch(err => {
      document.getElementById('events-feed').innerHTML = `<div class="event-card text-red">${err.message}</div>`;
    });
  }, 3000);
});
