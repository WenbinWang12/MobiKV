/* Interactive views of the final evaluation aggregates. No external services. */
(() => {
  'use strict';
  const escape = (value) => String(value).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const state = {metric: 'ttft', percentile: 'p90'};
  const chart = document.getElementById('replay-bars');
  const controls = document.querySelectorAll('#replay-explorer button');
  const metricName = () => state.metric === 'ttft' ? 'time to first token' : 'time per output token';
  const unit = () => state.metric === 'ttft' ? 's' : 'ms';
  const format = (value) => value.toFixed(state.metric === 'ttft' ? 2 : 1);

  function render(data) {
    const percentile = state.percentile.toUpperCase();
    const visible = data.replay.methods.filter((method) => !['follow_me_cloud','wang_ton19_mdp'].includes(method.id));
    const maximum = Math.max(...visible.map((method) => method[state.metric][state.percentile].ci[1])) * 1.05;
    const description = [];
    chart.innerHTML = visible.map((method) => {
      const point = method[state.metric][state.percentile];
      const isMobiKV = method.id === 'mobikv';
      description.push(`${method.label}: ${format(point.value)} ${unit()}, 95% interval ${format(point.ci[0])} to ${format(point.ci[1])}`);
      return `<div class="replay-row${isMobiKV ? ' is-mobikv' : ''}" aria-hidden="true"><span class="replay-label">${escape(method.label)}</span><div class="replay-track"><span class="replay-fill" style="width:${point.value / maximum * 100}%"></span><span class="replay-interval" style="left:${point.ci[0] / maximum * 100}%;width:${(point.ci[1] - point.ci[0]) / maximum * 100}%"></span></div><span class="replay-value">${format(point.value)}</span></div>`;
    }).join('');
    chart.setAttribute('aria-label', `${percentile} ${metricName()}. ${description.join('. ')}.`);
    document.getElementById('replay-chart-title').textContent = `${percentile} ${metricName()}`;
    document.getElementById('replay-chart-unit').textContent = `${state.metric === 'ttft' ? 'Seconds' : 'Milliseconds'} · lower is better`;
    document.getElementById('replay-table-caption').textContent = `${percentile} ${state.metric.toUpperCase()} in ${state.metric === 'ttft' ? 'seconds' : 'milliseconds'}; estimates with 95% bootstrap intervals.`;
    document.querySelector('#replay-table tbody').innerHTML = data.replay.methods.map((method) => {
      const point = method[state.metric][state.percentile];
      return `<tr${method.id === 'mobikv' ? ' class="is-mobikv"' : ''}><th scope="row">${escape(method.label)}</th><td>${state.metric === 'ttft' ? method.n : method.nTpot}</td><td>${format(point.value)}</td><td>[${format(point.ci[0])}, ${format(point.ci[1])}]</td></tr>`;
    }).join('');
    const readings = {
      ttft: {
        p50: 'MobiKV has the lowest median TTFT. Its 95% ratio interval is below 1 against every baseline.',
        p90: 'MobiKV reduces P90 TTFT by 54% relative to Mooncake, the best baseline. The 95% ratio interval is [0.30, 0.78].',
        p99: 'MobiKV has the lowest P99 TTFT point estimate. Its advantage over the four load-aware baselines is not statistically significant.'
      },
      tpot: {
        p50: 'Median TPOT is 33.9 ms for MobiKV and 33.6 ms for Mooncake. The 95% ratio interval includes 1.',
        p90: 'MobiKV has the lowest P90 TPOT point estimate. The intervals do not establish an advantage over Mooncake or ESP.',
        p99: 'Tail TPOT has wide intervals. MobiKV’s P99 advantage over the four load-aware baselines is not statistically significant.'
      }
    };
    document.getElementById('chart-reading').textContent = readings[state.metric][state.percentile];
    controls.forEach((button) => {
      const selected = button.dataset.metric ? button.dataset.metric === state.metric : button.dataset.percentile.toLowerCase() === state.percentile;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
  }

  fetch('./assets/results-final.json').then((response) => {
    if (!response.ok) throw new Error('Could not load results');
    return response.json();
  }).then((data) => {
    document.getElementById('model-bars').innerHTML = data.fullSpeedP90.models.map((model) => {
      const label = model.model.replace('-v0.3','').replace('-it','');
      const reduction = model.reductionPercent.value;
      return `<div class="model-row" aria-label="${escape(model.model)}: ${reduction.toFixed(1)} percent lower P90 TTFT than ${escape(model.baselineLabel)}"><span>${escape(label)}</span><span class="model-track" aria-hidden="true"><span class="model-fill" style="width:${reduction / 50 * 100}%"></span></span><strong>${reduction.toFixed(1)}%</strong></div>`;
    }).join('');
    render(data);
    controls.forEach((button) => button.addEventListener('click', () => {
      if (button.dataset.metric) state.metric = button.dataset.metric;
      if (button.dataset.percentile) state.percentile = button.dataset.percentile.toLowerCase();
      render(data);
    }));
  }).catch(() => {
    controls.forEach((button) => { button.disabled = true; });
    document.getElementById('chart-reading').textContent = 'The interactive chart could not load. The full measurements are available through the CSV link below.';
    document.querySelector('#replay-table tbody').innerHTML = '<tr><td colspan="4">Use the CSV download below to view the complete measurements.</td></tr>';
  });

  const copyButton = document.getElementById('copy-citation');
  copyButton.addEventListener('click', async () => {
    const code = document.getElementById('bibtex');
    const status = document.getElementById('copy-status');
    try {
      await navigator.clipboard.writeText(code.textContent);
      status.textContent = 'BibTeX copied to clipboard.';
      copyButton.textContent = 'Copied';
      setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 2500);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = 'Citation selected. Press Ctrl+C (or ⌘C) to copy.';
    }
  });
})();
