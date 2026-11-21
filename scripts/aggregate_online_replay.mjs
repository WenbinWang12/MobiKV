#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

const methods = {
  replay: 'MobiKV (RPP)',
  direct: 'MobiKV (direct control)',
  esp: 'ESP',
  lyap: 'Lyapunov',
};

function arg(name, fallback) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

const input = path.resolve(arg('--input', 'online_results'));
const output = path.resolve(arg('--output', 'docs/data/aggregate/online_replay_qwen3.csv'));

function quantile(values, probability) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function sampleStd(values) {
  if (values.length < 2) return null;
  const average = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1));
}

function metrics(rows, noRpp = false) {
  const ttft = rows.map((row) => Number(row.ttft_meas)).filter(Number.isFinite);
  const tpot = rows.map((row) => Number(row.tpot_meas) * 1000).filter(Number.isFinite);
  const rpp = rows.map((row) => Number(row.rpp)).filter(Number.isFinite);
  const boundary = rows.map((row) => Number(row.boundary)).filter(Number.isFinite);
  return {
    n_requests: rows.length,
    n_ttft: ttft.length,
    n_tpot: tpot.length,
    ttft_mean_s: mean(ttft),
    ttft_std_s: sampleStd(ttft),
    ttft_p50_s: quantile(ttft, 0.50),
    ttft_p90_s: quantile(ttft, 0.90),
    ttft_p99_s: quantile(ttft, 0.99),
    tpot_mean_ms: mean(tpot),
    tpot_std_ms: sampleStd(tpot),
    tpot_p50_ms: quantile(tpot, 0.50),
    tpot_p90_ms: quantile(tpot, 0.90),
    tpot_p99_ms: quantile(tpot, 0.99),
    rpp_fraction: noRpp ? 0 : mean(rpp),
    boundary_fraction: mean(boundary),
  };
}

const records = [];
for (const [directory, method] of Object.entries(methods)) {
  const directoryPath = path.join(input, directory);
  const files = fs.readdirSync(directoryPath)
    .filter((file) => /^replay_me\d+\.json$/.test(file))
    .sort();
  const pooled = [];
  for (const file of files) {
    const source = JSON.parse(fs.readFileSync(path.join(directoryPath, file), 'utf8'));
    const rows = (Array.isArray(source.reqs) ? source.reqs : []).filter((row) => Number(row.eval));
    pooled.push(...rows);
    records.push({
      model: 'Qwen3-8B',
      method,
      aggregation_unit: 'edge_shard',
      edge: source.edge,
      edge_shard: source.me,
      gpu: source.gpu,
      service_rate_profile_pct: Number(source.share),
      no_rpp: Number(source.no_rpp),
      send_err: Number(source.send_err) || 0,
      wait_timeouts: Number(source.wait_timeouts) || 0,
      ...metrics(rows, directory === 'direct'),
    });
  }
  records.push({
    model: 'Qwen3-8B',
    method,
    aggregation_unit: 'pooled',
    edge: 'all',
    edge_shard: 'all',
    gpu: 'NVIDIA A800 80GB PCIe',
    service_rate_profile_pct: 25,
    no_rpp: method === 'MobiKV (direct control)' ? 1 : 0,
    send_err: 0,
    wait_timeouts: 0,
    ...metrics(pooled, directory === 'direct'),
  });
}

const fields = [
  'model', 'method', 'aggregation_unit', 'edge', 'edge_shard', 'gpu',
  'service_rate_profile_pct', 'no_rpp', 'n_requests', 'n_ttft', 'n_tpot', 'send_err', 'wait_timeouts',
  'ttft_mean_s', 'ttft_std_s', 'ttft_p50_s', 'ttft_p90_s', 'ttft_p99_s',
  'tpot_mean_ms', 'tpot_std_ms', 'tpot_p50_ms', 'tpot_p90_ms', 'tpot_p99_ms',
  'rpp_fraction', 'boundary_fraction',
];
const csvCell = (value) => value == null ? '' : String(value).replaceAll('"', '""');
const csv = [
  fields.join(','),
  ...records.map((row) => fields.map((field) => {
    const value = csvCell(row[field]);
    return /[,"\n]/.test(value) ? `"${value}"` : value;
  }).join(',')),
].join('\n') + '\n';
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, csv);
console.log(`wrote ${records.length} rows to ${output}`);
