'use strict';

let currentTask = 'medium';
let episodeId = null;
let history = [];
let rewardChart = null;
let bidChart = null;
let isPlaying = false;

// Initialize on load
window.addEventListener('DOMContentLoaded', () => {
  console.log('[Dashboard] Initialized');
  document.getElementById('status').textContent = 'Ready';
});

async function startEpisode() {
  const res = await fetch('/api/episode/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task: currentTask, seed: null }),
  });
  const data = await res.json();
  
  if (data.success) {
    episodeId = data.episode_id;
    history = [];
    isPlaying = true;
    document.getElementById('stepBtn').disabled = false;
    document.getElementById('status').textContent = `Ep: ${data.task.toUpperCase()}`;
    document.getElementById('round-num').textContent = '0 / ' + data.max_steps;
    updateUI({ step: 0 });
    initCharts();
    console.log('[Episode] Started:', episodeId);
  } else {
    alert('Error: ' + data.error);
  }
}

async function stepOne() {
  if (!episodeId) return;
  
  const res = await fetch(`/api/episode/${episodeId}/step`, { method: 'POST' });
  const data = await res.json();
  
  if (data.success) {
    const round = data.round;
    history.push(round);
    updateUI(round);
    updateCharts();
    
    if (data.done) {
      document.getElementById('stepBtn').disabled = true;
      isPlaying = false;
      const summary = await fetch(`/api/episode/${episodeId}/summary`).then(r => r.json());
      document.getElementById('status').textContent = `Final Score: ${summary.score.toFixed(4)}`;
      alert(`Episode finished!\nScore: ${summary.score.toFixed(4)}\nSteps: ${summary.total_steps}`);
    }
  } else {
    alert('Error: ' + data.error);
  }
}

function updateUI(round) {
  // Update agents
  updateAgent('agg', round.agg_bid, round.agg_budget_remaining, round.agg_wins);
  updateAgent('con', round.con_bid, round.con_budget_remaining, round.con_wins);
  updateAgent('sm', round.sm_bid, round.sm_budget_remaining, round.sm_wins);
  
  // Winner
  const trophy = { 'aggressive': '🔴', 'conservative': '🔵', 'me': '⭐', 'none': '—' }[round.winner] || '—';
  document.getElementById('trophy').textContent = trophy;
  const winnerNames = { 'aggressive': 'Aggressive Agent', 'conservative': 'Conservative Agent', 'me': 'Smart Agent', 'none': 'None' };
  document.getElementById('winner-name').textContent = winnerNames[round.winner] || 'None';
  if (round.winner !== 'none' && round.resource_value) {
    document.getElementById('winner-prize').textContent = `Value won: ${round.resource_value.toFixed(1)}`;
  }
  
  // Resource info
  document.getElementById('resource-val').textContent = round.resource_value?.toFixed(1) || '—';
  document.getElementById('scarcity').textContent = round.scarcity?.toFixed(2) || '—';
  document.getElementById('market-press').textContent = round.market_pressure?.toFixed(2) || '—';
  document.getElementById('phase').textContent = round.phase || 'early';
  document.getElementById('round-num').textContent = round.step + ' / ' + round.max_steps;
  
  // ToM beliefs (TRUTHFUL - directly from environment)
  const tom = round.tom_beliefs || {};
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
  
  // Signals (what drives agent decisions)
  const exploitSig = round.exploit_signal || 0;
  const uncertaintySig = round.uncertainty_signal || 0;
  document.getElementById('exploit-sig').textContent = exploitSig.toFixed(3);
  document.getElementById('uncert-sig').textContent = uncertaintySig.toFixed(3);
  document.getElementById('exploit-sig-bar').style.width = (exploitSig * 100) + '%';
  document.getElementById('uncert-sig-bar').style.width = (uncertaintySig * 100) + '%';
  
  // Reward breakdown (TRUTHFUL - training signal)
  const rb = round.reward_breakdown || {};
  updateRewardBar('rew-value', rb.value || 0);
  updateRewardBar('rew-eff', rb.efficiency || 0);
  updateRewardBar('rew-strat', rb.strategy || 0);
  updateRewardBar('rew-penal', rb.penalty || 0);
  document.getElementById('rew-val-num').textContent = (rb.value || 0).toFixed(2);
  document.getElementById('rew-eff-num').textContent = (rb.efficiency || 0).toFixed(2);
  document.getElementById('rew-strat-num').textContent = (rb.strategy || 0).toFixed(2);
  document.getElementById('rew-penal-num').textContent = (rb.penalty || 0).toFixed(2);
  document.getElementById('total-reward').textContent = (round.reward || 0).toFixed(2);
  
  // Metrics (what gets graded)
  const m = round.metrics || {};
  document.getElementById('total-val').textContent = (m.total_value_won || 0).toFixed(1);
  document.getElementById('eff-ratio').textContent = (m.efficiency_ratio || 0).toFixed(2);
  document.getElementById('exploit-rate').textContent = (m.exploit_success_rate || 0).toFixed(2);
  document.getElementById('pass-rate').textContent = (m.smart_pass_rate || 0).toFixed(2);
  document.getElementById('belief-align').textContent = (m.belief_alignment || 0).toFixed(2);
  
  // Strategy (core RL mechanism)
  document.getElementById('strategy').textContent = round.strategy || 'PROBE';
  
  // Strategy transitions (bandit learning)
  if (round.strategy_transitions?.length > 0) {
    const trans = round.strategy_transitions[round.strategy_transitions.length - 1];
    const list = document.getElementById('transitions-list');
    const item = document.createElement('div');
    item.className = 'transition-item';
    item.textContent = `Step ${trans.step}: ${trans.from} → ${trans.to} (${trans.trigger})`;
    list.insertBefore(item, list.firstChild);
    if (list.children.length > 8) list.removeChild(list.lastChild);
  }
  
  // Bandit Q-values visualization
  if (round.bandit_q) {
    console.log('[Bandit] Q-values:', round.bandit_q);
  }
}

function updateRewardBar(id, value) {
  const bar = document.getElementById(id + '-bar');
  if (bar) {
    const normalized = Math.max(0, Math.min(1, (value + 5) / 10)); // Normalize to 0-1
    bar.style.width = (normalized * 100) + '%';
  }
}

function updateAgent(id, bid, budget, wins) {
  const spent = 500 - budget;
  const ratio = spent / 500;
  document.getElementById(`bid-${id}`).textContent = bid?.toFixed(2) || '—';
  document.getElementById(`fill-${id}`).style.width = (ratio * 100) + '%';
  document.getElementById(`budget-label-${id}`).textContent = `Budget: ${(ratio*100).toFixed(0)}%`;
  document.getElementById(`budget-${id}`).textContent = budget?.toFixed(0) || '0';
  document.getElementById(`wins-${id}`).textContent = wins || 0;
}

function onTaskChange() {
  currentTask = document.getElementById('taskSel').value;
}

function resetUI() {
  if (episodeId) {
    fetch(`/api/episode/${episodeId}/reset`, { method: 'POST' });
  }
  episodeId = null;
  history = [];
  isPlaying = false;
  document.getElementById('stepBtn').disabled = true;
  document.getElementById('status').textContent = 'Reset';
  document.getElementById('transitions-list').innerHTML = '';
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
    rewardChart.data.datasets[0].data = history.map(h => h.reward || 0);
    rewardChart.update('none');
  }
  
  if (bidChart) {
    bidChart.data.labels = history.map((_, i) => i + 1);
    bidChart.data.datasets[0].data = history.map(h => h.agg_bid || 0);
    bidChart.data.datasets[1].data = history.map(h => h.con_bid || 0);
    bidChart.data.datasets[2].data = history.map(h => h.sm_bid || 0);
    bidChart.update('none');
  }
}
