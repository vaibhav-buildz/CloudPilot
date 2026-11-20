import React, { useState, useEffect, useCallback, useMemo } from 'react';
import './App.css';

const API_BASE = (import.meta.env.VITE_API_URL !== undefined && import.meta.env.VITE_API_URL !== ''
  ? import.meta.env.VITE_API_URL
  : (typeof window !== 'undefined' && window.location.origin ? '/api/backend' : 'http://127.0.0.1:8000')
).replace(/\/$/, '');

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
  const [wakeAttempts, setWakeAttempts] = useState(0);

  // Cluster View State
  const [clusterData, setClusterData] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [clusterLoading, setClusterLoading] = useState(false);
  const [clusterError, setClusterError] = useState(null);

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
  const [compareError, setCompareError] = useState(null);
  const [chartMetric, setChartMetric] = useState('placed_count');

  // History State
  const [historyRuns, setHistoryRuns] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState(null);

  // Admin State (Serverless proxy attaches authentication automatically)
  const [adminPolicy, setAdminPolicy] = useState('predictive_ar');
  const [adminFeedback, setAdminFeedback] = useState(null);
  const [adminLoading, setAdminLoading] = useState(false);

  // Fetch Health (cold-start resilience)
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
        setWakeAttempts(0);
      } else {
        setBackendStatus('down');
        setWakeAttempts((prev) => prev + 1);
      }
    } catch {
      clearTimeout(timer);
      setBackendStatus('waking');
      setWakeAttempts((prev) => prev + 1);
    }
  }, []);

  // Poll /health aggressively (every 3s) during wake-up, or every 30s when healthy
  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  useEffect(() => {
    const delay = backendStatus === 'healthy' ? 30000 : 3000;
    const interval = setInterval(() => {
      checkHealth();
    }, delay);
    return () => clearInterval(interval);
  }, [backendStatus, checkHealth]);

  // Fetch Cluster Telemetry
  const fetchCluster = useCallback(async () => {
    setClusterLoading(true);
    setClusterError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/cluster`);
      if (res.ok) {
        const data = await res.json();
        setClusterData(data);
        setBackendStatus('healthy');
      } else {
        const errData = await res.json().catch(() => ({}));
        setClusterError(errData.error || `HTTP ${res.status}: Failed to load cluster telemetry`);
      }
    } catch (err) {
      setClusterError(err.message || 'Network error fetching cluster telemetry');
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
      const headers = { 'Content-Type': 'application/json' };
      if (apiKey) headers['X-API-Key'] = apiKey;
      const count = Math.min(Math.max(parseInt(simWorkloads, 10) || 1, 1), 1000);
      const res = await fetch(`${API_BASE}/api/v1/simulate`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          policy: simPolicy,
          num_workloads: count,
          seed: parseInt(simSeed, 10) || 42,
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
    setCompareError(null);
    setCompareResults([]);
    const results = [];
    try {
      const count = Math.min(Math.max(parseInt(compareWorkloads, 10) || 1, 1), 1000);
      for (let i = 0; i < POLICIES.length; i++) {
        const pol = POLICIES[i];
        setCompareProgress(`Evaluating ${pol.label} (${i + 1}/${POLICIES.length})...`);
        const headers = { 'Content-Type': 'application/json' };
        if (apiKey) headers['X-API-Key'] = apiKey;
        const res = await fetch(`${API_BASE}/api/v1/simulate`, {
          method: 'POST',
          headers,
          body: JSON.stringify({
            policy: pol.key,
            num_workloads: count,
            seed: parseInt(compareSeed, 10) || 42,
          }),
        });
        if (res.ok) {
          const run = await res.json();
          results.push(run);
          setCompareResults([...results]);
        } else {
          const errData = await res.json().catch(() => ({}));
          throw new Error(`Failed on ${pol.label}: ${errData.error || res.statusText}`);
        }
      }
    } catch (err) {
      setCompareError(err.message);
    } finally {
      setCompareLoading(false);
      setCompareProgress('');
    }
  };

  // Fetch History from Supabase
  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/history`);
      if (res.ok) {
        const data = await res.json();
        setHistoryRuns(Array.isArray(data) ? data : []);
      } else {
        const errData = await res.json().catch(() => ({}));
        setHistoryError(errData.error || `HTTP ${res.status}: Failed to retrieve history`);
      }
    } catch (err) {
      setHistoryError(err.message || 'Network error fetching run history');
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
    setAdminLoading(true);
    try {
      const headers = { 'Content-Type': 'application/json' };
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
    } finally {
      setAdminLoading(false);
    }
  };

  const handleAdminReset = async () => {
    if (!window.confirm('Are you sure you want to reset the cluster? All allocations will be cleared.')) return;
    setAdminFeedback(null);
    setAdminLoading(true);
    try {
      const headers = { 'Content-Type': 'application/json' };
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
    } finally {
      setAdminLoading(false);
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
          <div style={{ flex: 1 }}>
            <strong>Backend is waking up (Render free tier cold start may take 30–60s)</strong>
            <div style={{ fontSize: '12px', marginTop: '2px', opacity: 0.9 }}>
              Retrying /health (Attempt #{wakeAttempts})... Requests will resume automatically once online.
            </div>
          </div>
          <button className="retry-btn" onClick={() => checkHealth(true)}>
            Retry Now
          </button>
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

          {clusterLoading && !clusterData && (
            <div className="loading-box">
              <div className="spinner"></div>
              <span>Connecting to cluster telemetry...</span>
            </div>
          )}

          {clusterError && !clusterData && (
            <div className="feedback-banner error">
              <span>{clusterError}</span>
              <button className="retry-btn" onClick={fetchCluster}>Retry</button>
            </div>
          )}

          {!clusterData && !clusterLoading && !clusterError && (
            <div className="empty-state">
              <div className="empty-state-icon">📡</div>
              <div className="empty-state-title">No Cluster Telemetry Available</div>
              <div className="empty-state-desc">The cluster state could not be loaded. Please ensure the backend is running and reachable.</div>
              <button onClick={fetchCluster}>Connect to Cluster</button>
            </div>
          )}

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
              <label>Workloads Count (Max 1000)</label>
              <input
                type="number"
                min="1"
                max="1000"
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

          {simLoading && (
            <div className="loading-box">
              <div className="spinner"></div>
              <span>Executing trace simulation against policy {simPolicy}...</span>
            </div>
          )}

          {!simResult && !simLoading && !simError && (
            <div className="empty-state">
              <div className="empty-state-icon">⚡</div>
              <div className="empty-state-title">No Simulation Executed Yet</div>
              <div className="empty-state-desc">
                Select an allocation algorithm, configure workload size (up to 1,000 workloads to protect free-tier instances), and click <strong>Run Simulation</strong> to compute placement efficiency.
              </div>
            </div>
          )}

          {simResult && (
            <div>
              {(simResult.store_status === 'not saved' || simResult.persisted === false) && (
                <div
                  className="feedback-banner warning"
                  style={{
                    background: 'rgba(245, 158, 11, 0.15)',
                    border: '1px solid rgba(245, 158, 11, 0.4)',
                    color: '#fef08a',
                    marginBottom: '16px',
                  }}
                >
                  <span>
                    ⚠️ Simulation computed successfully, but results were <strong>not saved</strong> to persistent database (storage offline or unreachable).
                  </span>
                </div>
              )}
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
              <label>Workloads Count (Max 1000)</label>
              <input
                type="number"
                min="1"
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

          {compareError && <div className="feedback-banner error">{compareError}</div>}

          {compareLoading && (
            <div className="loading-box">
              <div className="spinner"></div>
              <span>{compareProgress || 'Evaluating benchmark suites across all 7 policies...'}</span>
            </div>
          )}

          {compareResults.length === 0 && !compareLoading && !compareError && (
            <div className="empty-state">
              <div className="empty-state-icon">📈</div>
              <div className="empty-state-title">No Benchmark Evaluated Yet</div>
              <div className="empty-state-desc">
                Click <strong>Run Comparative Benchmark</strong> to simulate all 7 heuristic and ML policies side-by-side on identical workloads and view visual comparisons.
              </div>
            </div>
          )}

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

          {historyError && (
            <div className="feedback-banner error">
              <span>{historyError}</span>
              <button className="retry-btn" onClick={fetchHistory}>Retry</button>
            </div>
          )}

          {historyLoading && historyRuns.length === 0 && (
            <div className="loading-box">
              <div className="spinner"></div>
              <span>Fetching simulation runs from Supabase...</span>
            </div>
          )}

          {historyRuns.length === 0 && !historyLoading && !historyError && (
            <div className="empty-state">
              <div className="empty-state-icon">📜</div>
              <div className="empty-state-title">No Simulation Runs Recorded Yet</div>
              <div className="empty-state-desc">
                Runs will appear here automatically when executed on the backend if <code>SUPABASE_URL</code> and <code>SUPABASE_SERVICE_KEY</code> are configured.
              </div>
            </div>
          )}

          {historyRuns.length > 0 && (
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
                  {historyRuns.map((run, i) => {
                    const res = run.result || {};
                    const pol = run.policy || run.algorithm || res.policy_key || res.algorithm || 'unknown';
                    const totalW = run.num_workloads ?? run.total_workloads ?? res.total_workloads ?? '-';
                    const placed = run.placed_count ?? res.placed_count ?? '-';
                    const rejection = run.rejection_rate_pct ?? res.rejection_rate_pct ?? 0;
                    const sla = run.sla_efficiency_score ?? res.sla_efficiency_score ?? '-';
                    const avgCpu = run.avg_cpu_utilization_pct ?? res.avg_cpu_utilization_pct ?? '-';
                    const avgRam = run.avg_ram_utilization_pct ?? res.avg_ram_utilization_pct ?? '-';
                    return (
                      <tr key={run.id || run.created_at || i}>
                        <td style={{ fontFamily: 'monospace', fontSize: '11px' }}>
                          {run.created_at ? new Date(run.created_at).toLocaleString() : 'N/A'}
                        </td>
                        <td style={{ fontWeight: 600, color: '#38bdf8' }}>{pol}</td>
                        <td>{totalW}</td>
                        <td>{placed}</td>
                        <td>{rejection}%</td>
                        <td style={{ fontWeight: 700 }}>{sla}</td>
                        <td>{avgCpu}%</td>
                        <td>{avgRam}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* TAB 5: ADMIN PANEL */}
      {activeTab === 'admin' && (
        <div className="panel">
          <h2 className="panel-title">Cluster Control Actions</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '13px', margin: '6px 0 16px 0' }}>
            Operational cluster management. Authenticated securely via Vercel serverless proxy with server-side API key.
          </p>

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
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                Change the active allocation strategy evaluated across the live cluster.
              </p>
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
              <button onClick={handleAdminPolicySwitch} disabled={adminLoading}>
                {adminLoading ? 'Applying...' : 'Apply Policy'}
              </button>
            </div>

            {/* Cluster Reset */}
            <div className="form-box">
              <h4 style={{ color: 'var(--accent-rose)' }}>Cluster Eviction & Reset</h4>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                Evict all resident workloads and reset server utilization counters back to zero.
              </p>
              <button className="danger" onClick={handleAdminReset} disabled={adminLoading} style={{ marginTop: 'auto' }}>
                {adminLoading ? 'Resetting...' : '⚠️ Reset Entire Cluster'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
