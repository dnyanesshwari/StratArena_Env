'use strict';

const SCENARIO_UI = {
  auction: {
    subtitle: 'Generalized multi-agent RL for auction, opponent modeling, and theory-of-mind training.',
    participantsHeading: 'Auction Participants',
    roundInfoHeading: 'Auction Round Info',
    winnerLabel: 'This Round Winner',
    tomHeading: 'Theory-of-Mind: What Smart Agent Believes',
    tomNote: 'Smart agent infers opponent budgets and strategy from observing bids in a competitive auction.',
    signalsHeading: 'Auction Decision Signals',
    exploitLabel: 'Exploit Signal',
    uncertaintyLabel: 'Market Uncertainty',
    trainingHeading: 'Auction RL Training Signals',
    rewardHeading: 'Reward Components (Last Round)',
    metricsHeading: 'Auction Episode Metrics',
    adaptationHeading: 'Auction Strategy Adaptations',
    rewardChartHeading: 'Cumulative Reward During Auction Play',
    bidChartHeading: 'Bid Comparison Over Time',
  },
  negotiation: {
    subtitle: 'Generalized multi-agent RL for negotiation dynamics, leverage tracking, and theory-of-mind training.',
    participantsHeading: 'Negotiation Agents',
    roundInfoHeading: 'Negotiation Round Info',
    winnerLabel: 'This Round Advantage',
    tomHeading: 'Theory-of-Mind: What Smart Agent Infers',
    tomNote: 'Smart agent infers counterpart leverage, confidence, and likely stance from observed behavior.',
    signalsHeading: 'Negotiation Decision Signals',
    exploitLabel: 'Leverage Signal',
    uncertaintyLabel: 'Stance Uncertainty',
    trainingHeading: 'Negotiation RL Training Signals',
    rewardHeading: 'Negotiation Reward Components',
    metricsHeading: 'Negotiation Episode Metrics',
    adaptationHeading: 'Negotiation Strategy Adaptations',
    rewardChartHeading: 'Cumulative Reward During Negotiation',
    bidChartHeading: 'Offer Pressure Over Time',
  },
  resource_allocation: {
    subtitle: 'Generalized multi-agent RL for resource allocation under scarcity, contention, and theory-of-mind training.',
    participantsHeading: 'Resource Allocation Agents',
    roundInfoHeading: 'Allocation Round Info',
    winnerLabel: 'This Round Allocation Winner',
    tomHeading: 'Theory-of-Mind: What Smart Agent Estimates',
    tomNote: 'Smart agent tracks competing agents, scarcity pressure, and hidden budget state while allocating resources.',
    signalsHeading: 'Allocation Decision Signals',
    exploitLabel: 'Allocation Opportunity',
    uncertaintyLabel: 'Scarcity Uncertainty',
    trainingHeading: 'Allocation RL Training Signals',
    rewardHeading: 'Allocation Reward Components',
    metricsHeading: 'Allocation Episode Metrics',
    adaptationHeading: 'Allocation Strategy Adaptations',
    rewardChartHeading: 'Cumulative Reward During Allocation',
    bidChartHeading: 'Allocation Intensity Over Time',
  },
};

let currentTask = 'medium';
let currentScenario = 'auction';
let episodeId = null;
let history = [];
let rewardChart = null;
let bidChart = null;
let isPlaying = false;
let playbackTimer = null;

const PLAYBACK_DELAY_MS = 450;

window.addEventListener('DOMContentLoaded', () => {
  console.log('[Dashboard] Initialized');
  document.getElementById('status').textContent = 'Ready';
  syncScenarioUI();
});

async function startEpisode() {
  stopPlayback();
  const rounds = getConfiguredRounds();

  const res = await fetch('/api/episode/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      task: currentTask,
      scenario: currentScenario,
      rounds,
      seed: null,
    }),
  });
  const data = await res.json();

  if (!data.success) {
    alert('Error: ' + data.error);
    return;
  }

  episodeId = data.episode_id;
  history = [];
  isPlaying = true;
  document.getElementById('stepBtn').disabled = false;
  document.getElementById('stepBtn').textContent = 'Pause';
  document.getElementById('status').textContent = `${formatScenario(data.scenario)} | ${data.task.toUpperCase()} | ${data.max_steps} rounds`;
  document.getElementById('round-num').textContent = `0 / ${data.max_steps}`;
  document.getElementById('transitions-list').innerHTML = '';
  updateUI({ step: 0, max_steps: data.max_steps });
  initCharts();
  queueNextStep();
}

async function stepOne() {
  if (!episodeId) return;

  const res = await fetch(`/api/episode/${episodeId}/step`, { method: 'POST' });
  const data = await res.json();

  if (!data.success) {
    stopPlayback();
    isPlaying = false;
    document.getElementById('stepBtn').textContent = 'Resume';
    alert('Error: ' + data.error);
    return;
  }

  const round = data.round;
  history.push(round);
  updateUI(round);
  updateCharts();

  if (data.done) {
    stopPlayback();
    isPlaying = false;
    document.getElementById('stepBtn').disabled = true;
    document.getElementById('stepBtn').textContent = 'Done';
    const summary = await fetch(`/api/episode/${episodeId}/summary`).then((r) => r.json());
    document.getElementById('status').textContent = `${formatScenario(summary.scenario)} score ${summary.score.toFixed(4)}`;
    alert(`Episode finished!\nScenario: ${formatScenario(summary.scenario)}\nScore: ${summary.score.toFixed(4)}\nSteps: ${summary.total_steps}`);
    return;
  }

  if (isPlaying) {
    queueNextStep();
  }
}

function updateUI(round) {
  const safeRound = {
    step: 0,
    max_steps: getConfiguredRounds(),
    winner: 'none',
    phase: 'early',
    agg_bid: 0,
    con_bid: 0,
    sm_bid: 0,
    agg_budget_remaining: 500,
    con_budget_remaining: 500,
    sm_budget_remaining: 500,
    agg_wins: 0,
    con_wins: 0,
    sm_wins: 0,
    reward: 0,
    reward_breakdown: {},
    metrics: {},
    ...round,
  };

  updateAgent('agg', safeRound.agg_bid, safeRound.agg_budget_remaining, safeRound.agg_wins);
  updateAgent('con', safeRound.con_bid, safeRound.con_budget_remaining, safeRound.con_wins);
  updateAgent('sm', safeRound.sm_bid, safeRound.sm_budget_remaining, safeRound.sm_wins);

  const trophy = { aggressive: '🔴', conservative: '🔵', me: '⭐', none: '—' }[safeRound.winner] || '—';
  document.getElementById('trophy').textContent = trophy;
  const winnerNames = { aggressive: 'Aggressive Agent', conservative: 'Conservative Agent', me: 'Smart Agent', none: 'None' };
  document.getElementById('winner-name').textContent = winnerNames[safeRound.winner] || 'None';
  if (safeRound.winner !== 'none' && safeRound.resource_value) {
    document.getElementById('winner-prize').textContent = `Value won: ${safeRound.resource_value.toFixed(1)}`;
  } else {
    document.getElementById('winner-prize').textContent = '—';
  }

  document.getElementById('resource-val').textContent = safeRound.resource_value?.toFixed(1) || '—';
  document.getElementById('scarcity').textContent = safeRound.scarcity?.toFixed(2) || '—';
  document.getElementById('market-press').textContent = safeRound.market_pressure?.toFixed(2) || '—';
  document.getElementById('phase').textContent = safeRound.phase || 'early';
  document.getElementById('round-num').textContent = `${safeRound.step} / ${safeRound.max_steps}`;

  const tom = safeRound.tom_beliefs || {};
  if (tom.aggressive) {
    document.getElementById('tom-agg-budget').textContent = tom.aggressive.budget_belief?.toFixed(3) || '—';
    document.getElementById('tom-agg-aggr').textContent = tom.aggressive.aggression_belief?.toFixed(3) || '—';
    document.getElementById('tom-agg-conf').textContent = tom.aggressive.confidence?.toFixed(3) || '—';
    document.getElementById('tom-agg-style').textContent = tom.aggressive.inferred_style || '—';
  }
  if (tom.conservative) {
    document.getElementById('tom-con-budget').textContent = tom.conservative.budget_belief?.toFixed(3) || '—';
    document.getElementById('tom-con-aggr').textContent = tom.conservative.aggression_belief?.toFixed(3) || '—';
    document.getElementById('tom-con-conf').textContent = tom.conservative.confidence?.toFixed(3) || '—';
    document.getElementById('tom-con-style').textContent = tom.conservative.inferred_style || '—';
  }

  const exploitSig = safeRound.exploit_signal || 0;
  const uncertaintySig = safeRound.uncertainty_signal || 0;
  document.getElementById('exploit-sig').textContent = exploitSig.toFixed(3);
  document.getElementById('uncert-sig').textContent = uncertaintySig.toFixed(3);
  document.getElementById('exploit-sig-bar').style.setProperty('--signal-width', (exploitSig * 100) + '%');
  document.getElementById('uncert-sig-bar').style.setProperty('--signal-width', (uncertaintySig * 100) + '%');

  const rb = safeRound.reward_breakdown || {};
  updateRewardBar('rew-value', rb.value || 0);
  updateRewardBar('rew-eff', rb.efficiency || 0);
  updateRewardBar('rew-strat', rb.strategy || 0);
  updateRewardBar('rew-penal', rb.penalty || 0);
  document.getElementById('rew-val-num').textContent = (rb.value || 0).toFixed(2);
  document.getElementById('rew-eff-num').textContent = (rb.efficiency || 0).toFixed(2);
  document.getElementById('rew-strat-num').textContent = (rb.strategy || 0).toFixed(2);
  document.getElementById('rew-penal-num').textContent = (rb.penalty || 0).toFixed(2);
  document.getElementById('total-reward').textContent = (safeRound.reward || 0).toFixed(2);

  const metrics = safeRound.metrics || {};
  document.getElementById('total-val').textContent = (metrics.total_value_won || 0).toFixed(1);
  document.getElementById('eff-ratio').textContent = (metrics.efficiency_ratio || 0).toFixed(2);
  document.getElementById('exploit-rate').textContent = (metrics.exploit_success_rate || 0).toFixed(2);
  document.getElementById('pass-rate').textContent = (metrics.smart_pass_rate || 0).toFixed(2);
  document.getElementById('belief-align').textContent = (metrics.belief_alignment || 0).toFixed(2);

  document.getElementById('strategy').textContent = safeRound.strategy || 'PROBE';

  if (safeRound.strategy_transitions?.length > 0) {
    const trans = safeRound.strategy_transitions[safeRound.strategy_transitions.length - 1];
    const list = document.getElementById('transitions-list');
    const item = document.createElement('div');
    item.className = 'transition-item';
    item.textContent = `Step ${trans.step}: ${trans.from} → ${trans.to} (${trans.trigger})`;
    list.insertBefore(item, list.firstChild);
    if (list.children.length > 8) list.removeChild(list.lastChild);
  }
}

function updateRewardBar(id, value) {
  const bar = document.getElementById(id + '-bar');
  if (bar) {
    const normalized = Math.max(0, Math.min(1, (value + 5) / 10));
    bar.style.width = (normalized * 100) + '%';
  }
}

function updateAgent(id, bid, budget, wins) {
  const spent = 500 - budget;
  const ratio = spent / 500;
  document.getElementById(`bid-${id}`).textContent = bid?.toFixed(2) || '—';
  document.getElementById(`fill-${id}`).style.width = (ratio * 100) + '%';
  document.getElementById(`budget-label-${id}`).textContent = `Budget: ${(ratio * 100).toFixed(0)}%`;
  document.getElementById(`budget-${id}`).textContent = budget?.toFixed(0) || '0';
  document.getElementById(`wins-${id}`).textContent = wins || 0;
}

function onTaskChange() {
  currentTask = document.getElementById('taskSel').value;
}

function onScenarioChange() {
  currentScenario = document.getElementById('scenarioSel').value;
  syncScenarioUI();
}

function getConfiguredRounds() {
  const input = document.getElementById('roundsInput');
  const parsed = Number.parseInt(input.value, 10);
  const rounds = Number.isFinite(parsed) ? Math.max(15, parsed) : 15;
  input.value = String(rounds);
  return rounds;
}

function formatScenario(value) {
  if (value === 'resource_allocation') return 'RESOURCE ALLOCATION';
  return String(value || 'auction').replace('_', ' ').toUpperCase();
}

function syncScenarioUI() {
  const ui = SCENARIO_UI[currentScenario] || SCENARIO_UI.auction;
  document.getElementById('scenarioSubtitle').textContent = ui.subtitle;
  document.getElementById('participantsHeading').textContent = ui.participantsHeading;
  document.getElementById('roundInfoHeading').textContent = ui.roundInfoHeading;
  document.getElementById('winnerLabel').textContent = ui.winnerLabel;
  document.getElementById('tomHeading').textContent = ui.tomHeading;
  document.getElementById('tomNote').textContent = ui.tomNote;
  document.getElementById('signalsHeading').textContent = ui.signalsHeading;
  document.getElementById('exploitLabel').textContent = ui.exploitLabel;
  document.getElementById('uncertaintyLabel').textContent = ui.uncertaintyLabel;
  document.getElementById('trainingHeading').textContent = ui.trainingHeading;
  document.getElementById('rewardHeading').textContent = ui.rewardHeading;
  document.getElementById('metricsHeading').textContent = ui.metricsHeading;
  document.getElementById('adaptationHeading').textContent = ui.adaptationHeading;
  document.getElementById('rewardChartHeading').textContent = ui.rewardChartHeading;
  document.getElementById('bidChartHeading').textContent = ui.bidChartHeading;
}

function queueNextStep() {
  stopPlayback();
  playbackTimer = window.setTimeout(() => {
    if (isPlaying) {
      stepOne();
    }
  }, PLAYBACK_DELAY_MS);
}

function stopPlayback() {
  if (playbackTimer !== null) {
    window.clearTimeout(playbackTimer);
    playbackTimer = null;
  }
}

function togglePlayback() {
  if (!episodeId) return;
  isPlaying = !isPlaying;
  document.getElementById('stepBtn').textContent = isPlaying ? 'Pause' : 'Resume';
  document.getElementById('status').textContent = isPlaying ? 'Autoplay running' : 'Autoplay paused';
  if (isPlaying) {
    queueNextStep();
  } else {
    stopPlayback();
  }
}

function resetUI() {
  stopPlayback();
  if (episodeId) {
    fetch(`/api/episode/${episodeId}/reset`, { method: 'POST' });
  }
  episodeId = null;
  history = [];
  isPlaying = false;
  document.getElementById('stepBtn').disabled = true;
  document.getElementById('stepBtn').textContent = 'Pause';
  document.getElementById('status').textContent = 'Reset';
  document.getElementById('transitions-list').innerHTML = '';
  document.getElementById('round-num').textContent = `0 / ${getConfiguredRounds()}`;
}

function initCharts() {
  const rcCtx = document.getElementById('rewardChart');
  if (rewardChart) rewardChart.destroy();
  rewardChart = new Chart(rcCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Total Reward',
          data: [],
          borderColor: '#4ade80',
          backgroundColor: 'rgba(74, 222, 128, 0.1)',
          borderWidth: 2,
          tension: 0.3,
          fill: true,
          pointRadius: 3,
          pointBackgroundColor: '#4ade80',
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        filler: { propagate: true }
      },
      scales: {
        y: { beginAtZero: true, max: 10 },
        x: { display: true }
      }
    }
  });
  
  const bcCtx = document.getElementById('bidChart');
  if (bidChart) bidChart.destroy();
  bidChart = new Chart(bcCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Aggressive', data: [], borderColor: '#ef4444', borderWidth: 2, tension: 0.3 },
        { label: 'Conservative', data: [], borderColor: '#3b82f6', borderWidth: 2, tension: 0.3 },
        { label: 'Smart', data: [], borderColor: '#fbbf24', borderWidth: 3, tension: 0.3 },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: 'top' }
      },
      scales: { y: { beginAtZero: true }, x: { display: true } }
    }
  });
}

function updateCharts() {
  if (rewardChart) {
    rewardChart.data.labels = history.map((_, i) => i + 1);
    rewardChart.data.datasets[0].data = history.map((h) => h.reward || 0);
    rewardChart.update('none');
  }
  
  if (bidChart) {
    bidChart.data.labels = history.map((_, i) => i + 1);
    bidChart.data.datasets[0].data = history.map((h) => h.agg_bid || 0);
    bidChart.data.datasets[1].data = history.map((h) => h.con_bid || 0);
    bidChart.data.datasets[2].data = history.map((h) => h.sm_bid || 0);
    bidChart.update('none');
  }
}
