import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, act } from '@testing-library/react';
import App from './App';

describe('App Cold-Start Resilience and Store Status Warning', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    global.fetch = vi.fn();
  });

  it('renders cold-start wake-up notification with retry counter when backend is waking up', async () => {
    // Simulate delayed / hung response triggering waking state
    global.fetch.mockImplementation(() => new Promise(() => {}));

    render(<App />);

    // Fast-forward timeout past 2500ms to trigger 'waking' status
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 2600));
    });

    const wakeupBanner = screen.getByText(/Backend is waking up \(Render free tier cold start may take 30–60s\)/i);
    expect(wakeupBanner).toBeTruthy();

    const retryBtn = screen.getByText(/Retry Now/i);
    expect(retryBtn).toBeTruthy();
  });

  it('displays a visible warning banner when simulation returns store_status not saved', async () => {
    // Mock health check and cluster
    global.fetch.mockImplementation((url) => {
      if (url.includes('/health')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ status: 'ok', active_policy: 'predictive_ar' }),
        });
      }
      if (url.includes('/cluster')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            active_policy: 'predictive_ar',
            cpu_utilization_pct: 10,
            ram_utilization_pct: 20,
            allocated_cpu_cores: 2,
            total_cpu_cores: 16,
            allocated_ram_gb: 4,
            total_ram_gb: 32,
            servers_in_use: 1,
            total_servers: 2,
            active_workloads_count: 1,
            load_imbalance: 0.1,
            servers: [],
          }),
        });
      }
      if (url.includes('/simulate')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            placed_count: 50,
            total_workloads: 50,
            rejection_rate_pct: 0,
            sla_efficiency_score: 95.5,
            avg_cpu_utilization_pct: 45.2,
            avg_ram_utilization_pct: 38.1,
            peak_cpu_utilization_pct: 60,
            peak_ram_utilization_pct: 50,
            peak_active_servers: 2,
            avg_load_imbalance: 0.05,
            allocated_cpu_hours: 12.5,
            simulation_duration_sec: 1800,
            store_status: 'not saved',
            persisted: false,
          }),
        });
      }
      return Promise.reject(new Error('Unknown URL'));
    });

    render(<App />);

    // Switch to simulation tab
    const simTab = screen.getByText(/⚡ Simulation Runner/i);
    await act(async () => {
      simTab.click();
    });

    // Run simulation
    const runBtn = screen.getByRole('button', { name: /Run Simulation/i });
    await act(async () => {
      runBtn.click();
    });

    // Verify "not saved" visible warning is displayed
    const warning = screen.getByText(/results were not saved to persistent database/i);
    expect(warning).toBeTruthy();
  });
});
