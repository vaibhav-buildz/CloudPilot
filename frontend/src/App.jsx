import React, { useState, useEffect, useCallback, useMemo } from 'react';
import './App.css';

const API_BASE = (import.meta.env.VITE_API_URL !== undefined
  ? import.meta.env.VITE_API_URL
  : 'http://127.0.0.1:8000').replace(/\/$/, '');

const POLICIES = [
  { key: 'predictive_ar', label: 'Predictive (LinearAR ML)', short: 'LinearAR', color: '#38bdf8' },
  { key: 'predictive_ewma', label: 'Predictive (EWMA)', short: 'EWMA', color: '#818cf8' },
  { key: 'ucb_bandit', label: 'UCB-Bandit (Contextual RL)', short: 'UCB-RL', color: '#c084fc' },
  { key: 'first_fit', label: 'FirstFit', short: 'FirstFit', color: '#64748b' },
  { key: 'best_fit', label: 'BestFit', short: 'BestFit', color: '#94a3b8' },
  { key: 'worst_fit', label: 'WorstFit', short: 'WorstFit', color: '#cbd5e1' },
  { key: 'round_robin', label: 'RoundRobin', short: 'RoundRobin', color: '#475569' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('cluster');
  const [backendStatus, setBackendStatus] = useState('checking'); // 'healthy' | 'waking' | 'down'
  const [healthData, setHealthData] = useState(null);

  // Cluster View State
  const [clusterData, setClusterData] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [clusterLoading, setClusterLoading] = useState(false);

  // Simulation Runner State
  const [simPolicy, setSimPolicy] = useState('predictive_ar');
  const [simWorkloads, setSimWorkloads] = useState(200);
  const [simSeed, setSimSeed] = useState(42);
  const [simLoading, setSimLoading] = useState(false);
  const [simResult, setSimResult] = useState(null);
  const [simError, setSimError] = useState(null);

  // Compare View State
  const [compareWorkloads, setCompareWorkloads] = useState(200);
  const [compareSeed, setCompareSeed] = useState(42);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareProgress, setCompareProgress] = useState('');
  const [compareResults, setCompareResults] = useState([]);
  const [chartMetric, setChartMetric] = useState('placed_count');

  // History State
  const [historyRuns, setHistoryRuns] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Admin State (In-memory API key only)
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [adminPolicy, setAdminPolicy] = useState('predictive_ar');
  const [allocId, setAllocId] = useState('');
  const [allocCpu, setAllocCpu] = useState(2.0);
  const [allocRam, setAllocRam] = useState(4.0);
  const [releaseId, setReleaseId] = useState('');
  const [adminFeedback, setAdminFeedback] = useState(null);

  // Fetch Health
  const checkHealth = useCallback(async () => {
    let timer;
    try {
      timer = setTimeout(() => {
        setBackendStatus('waking');
      }, 2500);
      const res = await fetch(`${API_BASE}/api/v1/health`);
      clearTimeout(timer);
      if (res.ok) {
        const data = await res.json();
        setHealthData(data);
        setBackendStatus('healthy');
      } else {
        setBackendStatus('down');
      }
    } catch {
      clearTimeout(timer);
      setBackendStatus('waking');
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  // Fetch Cluster Telemetry
  const fetchCluster = useCallback(async () => {
    try {
      setClusterLoading(true);
      const res = await fetch(`${API_BASE}/api/v1/cluster`);
      if (res.ok) {
        const data = await res.json();
        setClusterData(data);
        setBackendStatus('healthy');
      }
    } catch {
      // Handled by health status
    } finally {
      setClusterLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCluster();
  }, [fetchCluster]);

  useEffect(() => {
    if (!autoRefresh) return;
    const timer = setInterval(() => {
      if (activeTab === 'cluster') {
        fetchCluster();
      }
    }, 3000);
    return () => clearInterval(timer);
  }, [autoRefresh, activeTab, fetchCluster]);

  // Run Single Simulation
  const handleRunSimulation = async (e) => {
    if (e) e.preventDefault();
    setSimLoading(true);
    setSimError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          policy: simPolicy,
          num_workloads: parseInt(simWorkloads, 10),
          seed: parseInt(simSeed, 10),
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      setSimResult(data);
    } catch (err) {
      setSimError(err.message);
    } finally {
      setSimLoading(false);
    }
  };

  // Run Multi-Policy Comparison Benchmark
  const handleRunComparison = async () => {
    setCompareLoading(true);
    setCompareResults([]);
    const results = [];
    try {
      for (let i = 0; i < POLICIES.length; i++) {
        const pol = POLICIES[i];
        setCompareProgress(`Evaluating ${pol.label} (${i + 1}/${POLICIES.length})...`);
        const res = await fetch(`${API_BASE}/api/v1/simulate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            policy: pol.key,
            num_workloads: parseInt(compareWorkloads, 10),
            seed: parseInt(compareSeed, 10),
          }),
        });
        if (res.ok) {
          const run = await res.json();
          results.push(run);
          setCompareResults([...results]);
        }
      }
    } catch {
      // Handled by finally
    } finally {
      setCompareLoading(false);
      setCompareProgress('');
    }
  };

  // Fetch History from Supabase
  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/history`);
      if (res.ok) {
        const data = await res.json();
        setHistoryRuns(Array.isArray(data) ? data : []);
      }
    } catch {
      setHistoryRuns([]);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeTab === 'history') {
      fetchHistory();
    }
  }, [activeTab, fetchHistory]);

  // Admin Actions
  const handleAdminPolicySwitch = async () => {
    setAdminFeedback(null);
    try {
      const headers = { 'Content-Type': 'application/json' };
      if (apiKey) headers['X-API-Key'] = apiKey;
      const res = await fetch(`${API_BASE}/api/v1/policy`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ policy: adminPolicy }),
      });
      const data = await res.json();
      if (res.ok) {
        setAdminFeedback({ type: 'success', msg: `Policy successfully set to ${data.active_policy}` });
        fetchCluster();
      } else {
        setAdminFeedback({ type: 'error', msg: `[${res.status}] ${data.error || 'Failed to switch policy'}` });
      }
    } catch (err) {
      setAdminFeedback({ type: 'error', msg: err.message });
    }
  };

  const handleAdminAllocate = async (e) => {
    if (e) e.preventDefault();
    setAdminFeedback(null);
    try {
      const headers = { 'Content-Type': 'application/json' };
      if (apiKey) headers['X-API-Key'] = apiKey;
      const res = await fetch(`${API_BASE}/api/v1/allocate`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          workload_id: allocId.trim(),
          cpu: parseFloat(allocCpu),
          ram: parseFloat(allocRam),
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setAdminFeedback({
          type: 'success',
          msg: `Allocated ${data.workload_id} -> ${data.server_id} (Policy: ${data.active_policy})`,
        });
        setAllocId('');
        fetchCluster();
      } else {
        setAdminFeedback({ type: 'error', msg: `[${res.status}] ${data.error || 'Allocation rejected'}` });
      }
    } catch (err) {
      setAdminFeedback({ type: 'error', msg: err.message });
    }
  };

  const handleAdminRelease = async (e) => {
    if (e) e.preventDefault();
    setAdminFeedback(null);
    try {
      const headers = { 'Content-Type': 'application/json' };
      if (apiKey) headers['X-API-Key'] = apiKey;
      const res = await fetch(`${API_BASE}/api/v1/release`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ workload_id: releaseId.trim() }),
      });
      const data = await res.json();
      if (res.ok) {
        setAdminFeedback({
          type: 'success',
          msg: `Released workload ${data.workload_id} from ${data.server_id}`,
        });
        setReleaseId('');
        fetchCluster();
      } else {
        setAdminFeedback({ type: 'error', msg: `[${res.status}] ${data.error || 'Release failed'}` });
      }
    } catch (err) {
      setAdminFeedback({ type: 'error', msg: err.message });
    }
  };

  const handleAdminReset = async () => {
    if (!window.confirm('Are you sure you want to reset the cluster? All allocations will be cleared.')) return;
    setAdminFeedback(null);
    try {
      const headers = { 'Content-Type': 'application/json' };
      if (apiKey) headers['X-API-Key'] = apiKey;
      const res = await fetch(`${API_BASE}/api/v1/reset`, { method: 'POST', headers });
      const data = await res.json();
      if (res.ok) {
        setAdminFeedback({ type: 'success', msg: 'Cluster has been reset to empty initial state.' });
        fetchCluster();
      } else {
        setAdminFeedback({ type: 'error', msg: `[${res.status}] ${data.error || 'Reset failed'}` });
      }
    } catch (err) {
      setAdminFeedback({ type: 'error', msg: err.message });
    }
  };

  // SVG Chart Calculations
  const chartData = useMemo(() => {
    if (!compareResults.length) return null;
    const values = compareResults.map((r) => r[chartMetric] || 0);
    const maxVal = Math.max(...values, 1);
    return {
      items: compareResults.map((r, i) => {
        const pol = POLICIES.find((p) => p.key === r.policy_key) || { short: r.algorithm, color: '#38bdf8' };
        const val = r[chartMetric] || 0;
        return {
          key: r.policy_key || `item-${i}`,
          label: pol.short,
          val: Number(val.toFixed(2)),
          color: pol.color,
          heightPct: (val / maxVal) * 100,
        };
      }),
      maxVal,
    };
  }, [compareResults, chartMetric]);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-icon">CP</div>
          <div>
            <div className="brand-title">CloudPilot Control Plane</div>
            <div className="brand-subtitle">ML-Guided Cloud Resource Allocation & Simulation Engine</div>
          </div>
        </div>
        <div className="status-section">
          {backendStatus === 'healthy' && (
            <span className="backend-badge">
              <span className="status-dot"></span> Backend Online ({healthData?.active_policy || 'ready'})
            </span>
          )}
          {backendStatus === 'waking' && (
            <span className="backend-badge waking">
              <span className="status-dot"></span> Free-tier Spun Down (Waking up...)
            </span>
          )}
          {backendStatus === 'down' && (
            <span className="backend-badge down">
              <span className="status-dot"></span> Backend Unreachable
            </span>
          )}
        </div>
      </header>

      {/* Free-tier wake-up notification */}
      {backendStatus === 'waking' && (
        <div className="wakeup-alert">
          <span>⚡</span>
          <span>
            <strong>Waking up backend service on free tier</strong> (e.g. Render spin-up may take up to 60 seconds). Requests will automatically resolve once active.
          </span>
        </div>
      )}

      {/* Navigation Tabs */}
      <nav className="nav-tabs">
        <button className={`tab-btn ${activeTab === 'cluster' ? 'active' : ''}`} onClick={() => setActiveTab('cluster')}>
          📊 Live Cluster
        </button>
        <button className={`tab-btn ${activeTab === 'simulation' ? 'active' : ''}`} onClick={() => setActiveTab('simulation')}>
          ⚡ Simulation Runner
        </button>
        <button className={`tab-btn ${activeTab === 'compare' ? 'active' : ''}`} onClick={() => setActiveTab('compare')}>
          📈 Multi-Policy Compare
        </button>
        <button className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>
          📜 Run History
        </button>
        <button className={`tab-btn ${activeTab === 'admin' ? 'active' : ''}`} onClick={() => setActiveTab('admin')}>
          ⚙️ Admin Control
        </button>
      </nav>

      {/* TAB 1: LIVE CLUSTER */}
      {activeTab === 'cluster' && (
        <div>
          <div className="panel-header">
            <h2 className="panel-title">Cluster State & Real-Time Telemetry</h2>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <label style={{ fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                />
                Auto-refresh (3s)
              </label>
              <button className="secondary" onClick={fetchCluster} disabled={clusterLoading}>
                {clusterLoading ? 'Refreshing...' : '🔄 Refresh'}
              </button>
            </div>
          </div>

          {clusterData && (
            <>
              <div className="metrics-grid">
                <div className="metric-card">
                  <span className="metric-label">Active Policy</span>
                  <span className="metric-val" style={{ color: '#38bdf8', fontSize: '20px' }}>
                    {clusterData.active_policy}
                  </span>
                  <span className="metric-sub">Dynamic Allocation Strategy</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">CPU Utilization</span>
                  <span className="metric-val">{clusterData.cpu_utilization_pct}%</span>
                  <span className="metric-sub">{clusterData.allocated_cpu_cores} / {clusterData.total_cpu_cores} Cores</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">RAM Utilization</span>
                  <span className="metric-val">{clusterData.ram_utilization_pct}%</span>
                  <span className="metric-sub">{clusterData.allocated_ram_gb} / {clusterData.total_ram_gb} GB</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Active Nodes</span>
                  <span className="metric-val">{clusterData.servers_in_use} / {clusterData.total_servers}</span>
                  <span className="metric-sub">Nodes with Allocated Workloads</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Active Jobs</span>
                  <span className="metric-val">{clusterData.active_workloads_count}</span>
                  <span className="metric-sub">Currently Resident Containers</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Load Imbalance</span>
                  <span className="metric-val">{clusterData.load_imbalance}</span>
                  <span className="metric-sub">Utilization Std Dev</span>
                </div>
              </div>

              <div className="panel">
                <h3 className="panel-title" style={{ marginBottom: '14px' }}>Server Node Topology</h3>
                <div className="servers-grid">
                  {clusterData.servers.map((srv) => (
                    <div className="server-card" key={srv.server_id}>
                      <div className="server-header">
                        <span className="server-id">{srv.server_id}</span>
                        <span className={`server-badge ${srv.active_workloads > 0 ? 'in-use' : ''}`}>
                          {srv.active_workloads > 0 ? `${srv.active_workloads} jobs` : 'idle'}
                        </span>
                      </div>
                      <div className="bar-wrapper">
                        <div className="bar-label">
                          <span>CPU: {srv.cpu_used} / {srv.cpu_capacity} Cores</span>
                          <span>{srv.cpu_utilization_pct}%</span>
                        </div>
                        <div className="progress-track">
                          <div className="progress-fill cpu" style={{ width: `${srv.cpu_utilization_pct}%` }}></div>
                        </div>
                      </div>
                      <div className="bar-wrapper">
                        <div className="bar-label">
                          <span>RAM: {srv.ram_used} / {srv.ram_capacity} GB</span>
                          <span>{srv.ram_utilization_pct}%</span>
                        </div>
                        <div className="progress-track">
                          <div className="progress-fill ram" style={{ width: `${srv.ram_utilization_pct}%` }}></div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* TAB 2: SIMULATION RUNNER */}
      {activeTab === 'simulation' && (
        <div className="panel">
          <h2 className="panel-title">On-Demand Discrete-Event Trace Simulator</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '13px', margin: '6px 0 16px 0' }}>
            Simulate realistic heterogeneous workload traces against candidate scheduling policies using heap-driven event progression.
          </p>

          <form onSubmit={handleRunSimulation} className="controls-row" style={{ marginBottom: '20px' }}>
            <div className="control-group">
              <label>Allocation Policy</label>
              <select value={simPolicy} onChange={(e) => setSimPolicy(e.target.value)}>
                {POLICIES.map((p) => (
                  <option key={p.key} value={p.key}>
                    {p.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="control-group">
              <label>Workloads Count</label>
              <input
                type="number"
                min="10"
                max="2000"
                value={simWorkloads}
                onChange={(e) => setSimWorkloads(e.target.value)}
              />
            </div>
            <div className="control-group">
              <label>Random Seed</label>
              <input
                type="number"
                value={simSeed}
                onChange={(e) => setSimSeed(e.target.value)}
              />
            </div>
            <button type="submit" disabled={simLoading}>
              {simLoading ? 'Simulating Trace...' : '▶ Run Simulation'}
            </button>
          </form>

          {simError && <div className="feedback-banner error">{simError}</div>}

          {simResult && (
            <div>
              <div className="metrics-grid">
                <div className="metric-card">
                  <span className="metric-label">Placed Workloads</span>
                  <span className="metric-val" style={{ color: '#10b981' }}>
                    {simResult.placed_count} / {simResult.total_workloads}
                  </span>
                  <span className="metric-sub">Rejection Rate: {simResult.rejection_rate_pct}%</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">SLA Efficiency</span>
                  <span className="metric-val" style={{ color: '#38bdf8' }}>
                    {simResult.sla_efficiency_score}
                  </span>
                  <span className="metric-sub">Comprehensive Performance Metric</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Avg CPU Utilization</span>
                  <span className="metric-val">{simResult.avg_cpu_utilization_pct}%</span>
                  <span className="metric-sub">Peak: {simResult.peak_cpu_utilization_pct}%</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Avg RAM Utilization</span>
                  <span className="metric-val">{simResult.avg_ram_utilization_pct}%</span>
                  <span className="metric-sub">Peak: {simResult.peak_ram_utilization_pct}%</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Peak Active Nodes</span>
                  <span className="metric-val">{simResult.peak_active_servers}</span>
                  <span className="metric-sub">Imbalance: {simResult.avg_load_imbalance}</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Allocated Core-Hours</span>
                  <span className="metric-val">{simResult.allocated_cpu_hours}</span>
                  <span className="metric-sub">Duration: {simResult.simulation_duration_sec}s</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: COMPARE VIEW */}
      {activeTab === 'compare' && (
        <div className="panel">
          <h2 className="panel-title">Multi-Policy Benchmarking Suite</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '13px', margin: '6px 0 16px 0' }}>
            Benchmark all 7 heuristic and ML-guided allocation algorithms on identical traces and visualize placement efficiency with pure SVG graphics.
          </p>

          <div className="controls-row" style={{ marginBottom: '20px' }}>
            <div className="control-group">
              <label>Workloads Count</label>
              <input
                type="number"
                min="20"
                max="1000"
                value={compareWorkloads}
                onChange={(e) => setCompareWorkloads(e.target.value)}
              />
            </div>
            <div className="control-group">
              <label>Random Seed</label>
              <input
                type="number"
                value={compareSeed}
                onChange={(e) => setCompareSeed(e.target.value)}
              />
            </div>
            <button onClick={handleRunComparison} disabled={compareLoading}>
              {compareLoading ? compareProgress || 'Benchmarking...' : '📊 Run Comparative Benchmark'}
            </button>
          </div>

          {compareResults.length > 0 && (
            <div>
              <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)', alignSelf: 'center', marginRight: '6px' }}>
                  Chart Metric:
                </span>
                {[
                  { id: 'placed_count', label: 'Placed Workloads' },
                  { id: 'sla_efficiency_score', label: 'SLA Score' },
                  { id: 'avg_cpu_utilization_pct', label: 'Avg CPU %' },
                  { id: 'avg_ram_utilization_pct', label: 'Avg RAM %' },
                  { id: 'peak_cpu_utilization_pct', label: 'Peak CPU %' },
                ].map((m) => (
                  <button
                    key={m.id}
                    className={chartMetric === m.id ? '' : 'secondary'}
                    style={{ fontSize: '12px', padding: '4px 10px' }}
                    onClick={() => setChartMetric(m.id)}
                  >
                    {m.label}
                  </button>
                ))}
              </div>

              {/* Pure SVG Bar Chart (zero external libraries) */}
              {chartData && (
                <div className="chart-container">
                  <svg className="chart-svg" viewBox="0 0 740 260" width="100%" height="240">
                    {/* Background grid lines */}
                    {[0, 25, 50, 75, 100].map((pct) => {
                      const y = 200 - (pct / 100) * 160;
                      return (
                        <g key={pct}>
                          <line x1="40" y1={y} x2="720" y2={y} stroke="#1e293b" strokeDasharray="3,3" />
                          <text x="32" y={y + 4} fill="#64748b" fontSize="10" textAnchor="end">
                            {((pct / 100) * chartData.maxVal).toFixed(0)}
                          </text>
                        </g>
                      );
                    })}

                    {/* Bars */}
                    {chartData.items.map((item, idx) => {
                      const barWidth = 62;
                      const x = 55 + idx * 95;
                      const barH = Math.max(4, (item.val / chartData.maxVal) * 160);
                      const y = 200 - barH;
                      return (
                        <g key={item.key}>
                          <rect
                            x={x}
                            y={y}
                            width={barWidth}
                            height={barH}
                            fill={item.color}
                            rx="4"
                            opacity="0.9"
                          />
                          <text
                            x={x + barWidth / 2}
                            y={y - 6}
                            fill="#f8fafc"
                            fontSize="11"
                            fontWeight="700"
                            textAnchor="middle"
                          >
                            {item.val}
                          </text>
                          <text
                            x={x + barWidth / 2}
                            y="222"
                            fill="#94a3b8"
                            fontSize="11"
                            fontWeight="600"
                            textAnchor="middle"
                          >
                            {item.label}
                          </text>
                        </g>
                      );
                    })}
                    <line x1="40" y1="200" x2="720" y2="200" stroke="#334155" strokeWidth="1" />
                  </svg>
                </div>
              )}

              {/* Comparison Table */}
              <div className="table-container" style={{ marginTop: '20px' }}>
                <table>
                  <thead>
                    <tr>
                      <th>Algorithm</th>
                      <th>Placed / Total</th>
                      <th>Rejection Rate</th>
                      <th>SLA Score</th>
                      <th>Avg CPU %</th>
                      <th>Avg RAM %</th>
                      <th>Peak CPU %</th>
                      <th>Peak RAM %</th>
                      <th>Imbalance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {compareResults.map((r) => (
                      <tr key={r.policy_key}>
                        <td style={{ fontWeight: 600, color: '#38bdf8' }}>{r.algorithm}</td>
                        <td>{r.placed_count} / {r.total_workloads}</td>
                        <td style={{ color: r.rejection_rate_pct === 0 ? '#10b981' : '#f59e0b' }}>
                          {r.rejection_rate_pct}%
                        </td>
                        <td style={{ fontWeight: 700 }}>{r.sla_efficiency_score}</td>
                        <td>{r.avg_cpu_utilization_pct}%</td>
                        <td>{r.avg_ram_utilization_pct}%</td>
                        <td>{r.peak_cpu_utilization_pct}%</td>
                        <td>{r.peak_ram_utilization_pct}%</td>
                        <td>{r.avg_load_imbalance}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 4: RUN HISTORY */}
      {activeTab === 'history' && (
        <div className="panel">
          <div className="panel-header">
            <div>
              <h2 className="panel-title">Simulation Run History</h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '13px', marginTop: '4px' }}>
                Persistent historical execution records retrieved from Supabase backend storage.
              </p>
            </div>
            <button className="secondary" onClick={fetchHistory} disabled={historyLoading}>
              {historyLoading ? 'Loading...' : '🔄 Refresh History'}
            </button>
          </div>

          {historyRuns.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-muted)' }}>
              <p style={{ fontSize: '15px', marginBottom: '8px' }}>No simulation runs recorded yet.</p>
              <p style={{ fontSize: '12px', maxWidth: '500px', margin: '0 auto' }}>
                Runs will appear here automatically when executed on the backend if <code>SUPABASE_URL</code> and <code>SUPABASE_SERVICE_KEY</code> are configured.
              </p>
            </div>
          ) : (
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Algorithm</th>
                    <th>Workloads</th>
                    <th>Placed</th>
                    <th>Rejection %</th>
                    <th>SLA Score</th>
                    <th>Avg CPU %</th>
                    <th>Avg RAM %</th>
                  </tr>
                </thead>
                <tbody>
                  {historyRuns.map((run, i) => (
                    <tr key={run.id || run.created_at || i}>
                      <td style={{ fontFamily: 'monospace', fontSize: '11px' }}>
                        {run.created_at ? new Date(run.created_at).toLocaleString() : 'N/A'}
                      </td>
                      <td style={{ fontWeight: 600, color: '#38bdf8' }}>{run.algorithm || run.policy}</td>
                      <td>{run.total_workloads}</td>
                      <td>{run.placed_count}</td>
                      <td>{run.rejection_rate_pct}%</td>
                      <td style={{ fontWeight: 700 }}>{run.sla_efficiency_score}</td>
                      <td>{run.avg_cpu_utilization_pct}%</td>
                      <td>{run.avg_ram_utilization_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* TAB 5: ADMIN PANEL */}
      {activeTab === 'admin' && (
        <div className="panel">
          <h2 className="panel-title">Authenticated Control-Plane Actions</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '13px', margin: '6px 0 16px 0' }}>
            Execute protected state-mutating cluster operations. API key is stored strictly in component memory and forwarded as <code>X-API-Key</code>.
          </p>

          <div className="control-group" style={{ maxWidth: '420px', marginBottom: '24px' }}>
            <label>API Key (Stored in React state only)</label>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                type={showKey ? 'text' : 'password'}
                placeholder="Enter CLOUDPILOT_API_KEY if configured..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                style={{ flex: 1 }}
              />
              <button type="button" className="secondary" onClick={() => setShowKey(!showKey)}>
                {showKey ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>

          {adminFeedback && (
            <div className={`feedback-banner ${adminFeedback.type}`}>
              <span>{adminFeedback.msg}</span>
              <button
                type="button"
                className="secondary"
                style={{ padding: '2px 8px', fontSize: '11px' }}
                onClick={() => setAdminFeedback(null)}
              >
                ✕
              </button>
            </div>
          )}

          <div className="form-grid">
            {/* Policy Switch */}
            <div className="form-box">
              <h4>Switch Active Policy</h4>
              <div className="control-group">
                <label>Target Policy</label>
                <select value={adminPolicy} onChange={(e) => setAdminPolicy(e.target.value)}>
                  {POLICIES.map((p) => (
                    <option key={p.key} value={p.key}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </div>
              <button onClick={handleAdminPolicySwitch}>Apply Policy</button>
            </div>

            {/* Allocate Workload */}
            <div className="form-box">
              <h4>Direct Workload Allocation</h4>
              <form onSubmit={handleAdminAllocate} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div className="control-group">
                  <label>Workload ID</label>
                  <input
                    type="text"
                    placeholder="e.g. job-web-01"
                    required
                    value={allocId}
                    onChange={(e) => setAllocId(e.target.value)}
                  />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <div className="control-group">
                    <label>CPU Cores</label>
                    <input
                      type="number"
                      step="0.5"
                      min="0.5"
                      value={allocCpu}
                      onChange={(e) => setAllocCpu(e.target.value)}
                    />
                  </div>
                  <div className="control-group">
                    <label>RAM (GB)</label>
                    <input
                      type="number"
                      step="0.5"
                      min="0.5"
                      value={allocRam}
                      onChange={(e) => setAllocRam(e.target.value)}
                    />
                  </div>
                </div>
                <button type="submit">Allocate Workload</button>
              </form>
            </div>

            {/* Release Workload */}
            <div className="form-box">
              <h4>Direct Workload Release</h4>
              <form onSubmit={handleAdminRelease} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div className="control-group">
                  <label>Workload ID to Evict</label>
                  <input
                    type="text"
                    placeholder="e.g. job-web-01"
                    required
                    value={releaseId}
                    onChange={(e) => setReleaseId(e.target.value)}
                  />
                </div>
                <button type="submit" className="secondary">Release Workload</button>
              </form>
            </div>

            {/* Cluster Reset */}
            <div className="form-box">
              <h4 style={{ color: 'var(--accent-rose)' }}>Cluster Eviction & Reset</h4>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Evict all resident workloads and reset server utilization counters to zero.
              </p>
              <button className="danger" onClick={handleAdminReset} style={{ marginTop: 'auto' }}>
                ⚠️ Reset Entire Cluster
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
