#!/usr/bin/env python3
"""Evolution dashboard — real-time monitoring (stdlib only, no Flask).

Reads:
  - /tmp/multi_evo.log (live progress)
  - /root/ai-trader-evolution/training/results/cycle_N_*.json
  - /root/ai-trader-evolution/training/results/profitable/*.json

Serves on port 8080.
"""
import os
import json
import re
import glob
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

LOG_PATH = "/tmp/multi_evo.log"
RESULTS_DIR = "/root/ai-trader-evolution/training/results"
PROFITABLE_DIR = os.path.join(RESULTS_DIR, "profitable")


def parse_log_progress():
    try:
        with open(LOG_PATH) as f:
            lines = f.readlines()
    except Exception:
        return {"error": "log not found", "lines": []}

    launch_params = {}
    for line in lines[:20]:
        m = re.search(r"Total budget:\s*([\d.]+)h", line)
        if m: launch_params["hours"] = float(m.group(1))
        m = re.search(r"Cycles:\s*(\d+)\s*\(([^)]+)\)", line)
        if m:
            launch_params["n_cycles"] = int(m.group(1))
            launch_params["structures"] = m.group(2)
        m = re.search(r"Per cycle:\s*([\d.]+)h,\s*(\d+) models × (\d+) gens", line)
        if m:
            launch_params["hours_per_cycle"] = float(m.group(1))
            launch_params["models"] = int(m.group(2))
            launch_params["generations"] = int(m.group(3))
        m = re.search(r"Data:\s*(\d+) days MOEX", line)
        if m: launch_params["data_days"] = int(m.group(1))

    current_cycle = None
    current_gen = None
    cycles_status = {}

    for line in lines:
        m = re.search(r"CYCLE (\d+):\s*(\w+)", line)
        if m:
            cn = int(m.group(1))
            cycles_status[cn] = {
                "structure": m.group(2), "total_profitable": 0,
                "best_val_pnl": None, "status": "running", "generations_done": 0,
            }
            current_cycle = cn

        m = re.search(r"\[cycle (\d+) gen (\d+)/(\d+)\]\s*best_fitness=([\d.-]+)\s*best_val_pnl=([+-]?[\d.]+)\s*trades=(\d+)\s*profitable=(\d+)\s*\(([\d.]+)s\)", line)
        if m:
            cn, gen = int(m.group(1)), int(m.group(2))
            if cn in cycles_status:
                cycles_status[cn].update({
                    "generations_done": gen,
                    "last_gen_fitness": float(m.group(4)),
                    "last_gen_val_pnl": float(m.group(5)),
                    "last_gen_trades": int(m.group(6)),
                    "profitable_in_gen": int(m.group(7)),
                    "gen_time_sec": float(m.group(8)),
                })
            current_cycle = cn
            current_gen = gen

        m = re.search(r"\[cycle (\d+)\] total profitable:\s*(\d+)", line)
        if m:
            cn = int(m.group(1))
            if cn in cycles_status:
                cycles_status[cn]["total_profitable"] = int(m.group(2))
                cycles_status[cn]["status"] = "completed"

        m = re.search(r"\[cycle (\d+)\] BEST:\s*val_pnl=([+-]?[\d.]+)\s*trades=(\d+)\s*fitness=([\d.-]+)", line)
        if m:
            cn = int(m.group(1))
            if cn in cycles_status:
                cycles_status[cn]["best_val_pnl"] = float(m.group(2))
                cycles_status[cn]["best_trades"] = int(m.group(3))
                cycles_status[cn]["best_fitness"] = float(m.group(4))

    # Convert keys to string for JSON
    cycles_status_str = {str(k): v for k, v in cycles_status.items()}

    return {
        "launch_params": launch_params,
        "current_cycle": current_cycle,
        "current_gen": current_gen,
        "cycles_status": cycles_status_str,
        "recent_lines": [l.rstrip() for l in lines[-30:] if l.strip()],
        "log_size": os.path.getsize(LOG_PATH) if os.path.exists(LOG_PATH) else 0,
        "log_mtime": datetime.fromtimestamp(os.path.getmtime(LOG_PATH)).isoformat() if os.path.exists(LOG_PATH) else None,
    }


def load_cycle_details():
    cycles = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "cycle_*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
            cycles.append({
                "file": os.path.basename(path),
                "cycle": data.get("cycle"),
                "structure": data.get("structure"),
                "description": data.get("description"),
                "models": data.get("models"),
                "generations": data.get("generations"),
                "total_profitable": data.get("total_profitable", 0),
                "total_time_sec": data.get("total_time_sec"),
                "generations_log": data.get("generations_log", []),
                "best_model": data.get("profitable_models", [None])[0] if data.get("profitable_models") else None,
            })
        except Exception as e:
            cycles.append({"file": os.path.basename(path), "error": str(e)})
    return cycles


def load_top_profitable(limit=20):
    models = []
    for path in sorted(glob.glob(os.path.join(PROFITABLE_DIR, "*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
            for m in data:
                m["source_file"] = os.path.basename(path)
                models.append(m)
        except Exception:
            pass
    models.sort(key=lambda x: -x.get("val_pnl", 0))
    return models[:limit]


HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evolution Dashboard — AI Trader</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: #0a0a0a; color: #e5e5e5; padding: 20px; }
  .header { background: #FFDD2D; color: #0a0a0a; padding: 16px 20px; border-radius: 12px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }
  .header h1 { font-size: 22px; font-weight: 900; }
  .header .status { font-size: 13px; font-weight: 600; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  .card { background: #1a1a1a; border: 1px solid #333; border-radius: 10px; padding: 16px; }
  .card h2 { font-size: 14px; text-transform: uppercase; color: #999; margin-bottom: 12px; letter-spacing: 0.5px; }
  .param { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #222; font-size: 13px; }
  .param:last-child { border-bottom: none; }
  .param .key { color: #999; }
  .param .val { font-weight: 600; font-family: monospace; }
  .cycle { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 12px; margin-bottom: 8px; }
  .cycle.active { border-color: #FFDD2D; box-shadow: 0 0 0 1px #FFDD2D; }
  .cycle.completed { border-color: #0DBC4C; }
  .cycle-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .cycle-name { font-weight: 700; font-size: 14px; }
  .cycle-status { font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
  .status-running { background: #FFDD2D; color: #0a0a0a; }
  .status-completed { background: #0DBC4C; color: #0a0a0a; }
  .status-pending { background: #333; color: #999; }
  .cycle-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; font-size: 12px; }
  .stat .label { color: #666; font-size: 10px; text-transform: uppercase; }
  .stat .value { font-family: monospace; font-weight: 600; }
  .positive { color: #0DBC4C; }
  .negative { color: #E53935; }
  .progress-bar { background: #222; height: 6px; border-radius: 3px; margin-top: 8px; overflow: hidden; }
  .progress-fill { background: #FFDD2D; height: 100%; transition: width 0.5s; }
  .log { background: #0a0a0a; border: 1px solid #222; border-radius: 8px; padding: 12px; font-family: 'Courier New', monospace; font-size: 11px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; }
  .log-line { padding: 2px 0; }
  .log-line.gen { color: #FFDD2D; }
  .log-line.profit { color: #0DBC4C; }
  .log-line.error { color: #E53935; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; padding: 8px; border-bottom: 2px solid #333; color: #999; font-size: 11px; text-transform: uppercase; }
  td { padding: 8px; border-bottom: 1px solid #222; font-family: monospace; }
  .refresh-indicator { display: inline-block; width: 8px; height: 8px; background: #0DBC4C; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  .chart { height: 120px; display: flex; align-items: flex-end; gap: 2px; padding: 8px 0; }
  .bar { flex: 1; background: #FFDD2D; min-height: 2px; border-radius: 2px 2px 0 0; position: relative; }
  .bar:hover { background: #fff; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>🧬 Evolution Dashboard</h1>
    <div class="status"><span class="refresh-indicator"></span><span id="last-update">loading...</span></div>
  </div>
  <div style="text-align: right; font-size: 12px;">
    <div><strong>Server:</strong> 2.26.123.205</div>
    <div><strong>Log:</strong> /tmp/multi_evo.log</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Launch Parameters (твои настройки)</h2>
    <div id="params">loading...</div>
  </div>
  <div class="card">
    <h2>Current Status</h2>
    <div id="status">loading...</div>
  </div>
</div>

<div class="card" style="margin-bottom: 20px;">
  <h2>Cycles Progress</h2>
  <div id="cycles">loading...</div>
</div>

<div class="grid">
  <div class="card">
    <h2>Fitness History (current cycle)</h2>
    <div id="chart">loading...</div>
  </div>
  <div class="card">
    <h2>Top Profitable Models</h2>
    <div id="top-models" style="max-height: 250px; overflow-y: auto;">loading...</div>
  </div>
</div>

<div class="card">
  <h2>Live Log (last 30 lines)</h2>
  <div class="log" id="log">loading...</div>
</div>

<script>
async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('last-update').textContent = 
      'Updated: ' + new Date().toLocaleTimeString() + ' | log: ' + (d.log_size || 0) + ' bytes';
    const p = d.launch_params || {};
    document.getElementById('params').innerHTML = `
      <div class="param"><span class="key">Total budget</span><span class="val">${p.hours || '?'}h</span></div>
      <div class="param"><span class="key">Cycles</span><span class="val">${p.n_cycles || '?'}</span></div>
      <div class="param"><span class="key">Structures</span><span class="val" style="font-size: 11px">${p.structures || '?'}</span></div>
      <div class="param"><span class="key">Models/cycle</span><span class="val">${p.models || '?'}</span></div>
      <div class="param"><span class="key">Generations/cycle</span><span class="val">${p.generations || '?'}</span></div>
      <div class="param"><span class="key">Data days</span><span class="val">${p.data_days || '?'} days MOEX</span></div>
      <div class="param"><span class="key">Hours/cycle</span><span class="val">${p.hours_per_cycle || '?'}h</span></div>
    `;
    const cs = d.cycles_status || {};
    const totalProfitable = Object.values(cs).reduce((s, c) => s + (c.total_profitable || 0), 0);
    const completedCycles = Object.values(cs).filter(c => c.status === 'completed').length;
    document.getElementById('status').innerHTML = `
      <div class="param"><span class="key">Current cycle</span><span class="val">${d.current_cycle || '-'}/${p.n_cycles || '?'}</span></div>
      <div class="param"><span class="key">Current generation</span><span class="val">${d.current_gen || '-'}/${p.generations || '?'}</span></div>
      <div class="param"><span class="key">Cycles completed</span><span class="val">${completedCycles}/${p.n_cycles || '?'}</span></div>
      <div class="param"><span class="key">Total profitable</span><span class="val positive">${totalProfitable}</span></div>
    `;
    const cyclesHtml = Object.entries(cs).map(([num, c]) => {
      const isCurrent = parseInt(num) === d.current_cycle;
      const progress = c.generations_done / (p.generations || 15) * 100;
      const pnlClass = c.best_val_pnl >= 0 ? 'positive' : 'negative';
      const pnlStr = c.best_val_pnl !== null ? (c.best_val_pnl >= 0 ? '+' : '') + c.best_val_pnl.toFixed(0) + ' RUB' : '-';
      return `
        <div class="cycle ${isCurrent ? 'active' : ''} ${c.status === 'completed' ? 'completed' : ''}">
          <div class="cycle-header">
            <span class="cycle-name">Cycle ${num}: ${c.structure}</span>
            <span class="cycle-status status-${c.status}">${c.status}${isCurrent ? ' - gen ' + (c.generations_done || 0) : ''}</span>
          </div>
          <div class="cycle-stats">
            <div class="stat"><div class="label">Gen done</div><div class="value">${c.generations_done || 0}/${p.generations || '?'}</div></div>
            <div class="stat"><div class="label">Profitable</div><div class="value positive">${c.total_profitable || 0}</div></div>
            <div class="stat"><div class="label">Best val P&L</div><div class="value ${pnlClass}">${pnlStr}</div></div>
            <div class="stat"><div class="label">Last gen P&L</div><div class="value ${c.last_gen_val_pnl >= 0 ? 'positive' : 'negative'}">${c.last_gen_val_pnl !== undefined ? (c.last_gen_val_pnl >= 0 ? '+' : '') + c.last_gen_val_pnl.toFixed(0) : '-'}</div></div>
          </div>
          <div class="progress-bar"><div class="progress-fill" style="width: ${progress}%"></div></div>
        </div>
      `;
    }).join('');
    document.getElementById('cycles').innerHTML = cyclesHtml || '<div style="color:#666">no cycles yet</div>';
    const logLines = (d.recent_lines || []).map(l => {
      let cls = 'log-line';
      if (l.includes('gen')) cls += ' gen';
      if (l.includes('profitable') || l.includes('BEST')) cls += ' profit';
      if (l.includes('error') || l.includes('FAIL')) cls += ' error';
      return '<div class="' + cls + '">' + escapeHtml(l) + '</div>';
    }).join('');
    document.getElementById('log').innerHTML = logLines || '<div style="color:#666">log empty</div>';
    document.getElementById('log').scrollTop = document.getElementById('log').scrollHeight;
  } catch (e) {
    document.getElementById('last-update').textContent = 'ERROR: ' + e.message;
  }
}

async function fetchTopModels() {
  try {
    const r = await fetch('/api/top-models');
    const models = await r.json();
    if (!models || models.length === 0) {
      document.getElementById('top-models').innerHTML = '<div style="color:#666">no profitable models yet</div>';
      return;
    }
    let html = '<table><thead><tr><th>#</th><th>Structure</th><th>Val P&L</th><th>Trades</th><th>Sortino</th><th>Gen</th></tr></thead><tbody>';
    models.forEach((m, i) => {
      const pnlClass = m.val_pnl >= 0 ? 'positive' : 'negative';
      html += `<tr><td>${i+1}</td><td>${m.structure || '-'}</td><td class="${pnlClass}">${m.val_pnl >= 0 ? '+' : ''}${m.val_pnl.toFixed(0)} RUB</td><td>${m.val_trades}</td><td>${m.val_sortino.toFixed(2)}</td><td>${m.generation}</td></tr>`;
    });
    html += '</tbody></table>';
    document.getElementById('top-models').innerHTML = html;
  } catch (e) {
    document.getElementById('top-models').innerHTML = '<div style="color:#E53935">ERROR: ' + e.message + '</div>';
  }
}

async function fetchChart() {
  try {
    const r = await fetch('/api/cycle-details');
    const cycles = await r.json();
    const activeCycle = cycles.filter(c => c.generations_log && c.generations_log.length > 0).pop();
    if (!activeCycle) {
      document.getElementById('chart').innerHTML = '<div style="color:#666">no generation data yet</div>';
      return;
    }
    const gens = activeCycle.generations_log;
    const maxFitness = Math.max(...gens.map(g => g.best_fitness || 0), 0.1);
    let html = `<div style="font-size: 12px; margin-bottom: 8px; color: #999;">Cycle ${activeCycle.cycle}: ${activeCycle.structure} - fitness by generation</div>`;
    html += '<div class="chart">';
    gens.forEach(g => {
      const h = (g.best_fitness / maxFitness) * 100;
      html += `<div class="bar" style="height: ${h}%" title="gen ${g.generation}: fitness=${g.best_fitness.toFixed(2)}, pnl=${g.best_val_pnl.toFixed(0)}, profitable=${g.profitable_count}"></div>`;
    });
    html += '</div>';
    html += `<div style="font-size: 11px; color: #666; margin-top: 20px;">${gens.length} generations - best fitness: ${gens[gens.length-1].best_fitness.toFixed(2)} - last gen P&L: ${gens[gens.length-1].best_val_pnl.toFixed(0)} RUB</div>`;
    document.getElementById('chart').innerHTML = html;
  } catch (e) {
    document.getElementById('chart').innerHTML = '<div style="color:#E53935">ERROR: ' + e.message + '</div>';
  }
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

fetchStatus();
fetchTopModels();
fetchChart();
setInterval(fetchStatus, 10000);
setInterval(fetchTopModels, 30000);
setInterval(fetchChart, 30000);
</script>

</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
        elif path == "/api/status":
            self._json(parse_log_progress())
        elif path == "/api/cycle-details":
            self._json(load_cycle_details())
        elif path == "/api/top-models":
            self._json(load_top_profitable(20))
        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, data):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # silence default logging


if __name__ == "__main__":
    print("Evolution dashboard: http://2.26.123.205:8080")
    server = HTTPServer(("0.0.0.0", 8080), Handler)
    server.serve_forever()
