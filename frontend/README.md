# CloudPilot Frontend (Vite + React)

A zero-UI-dependency, high-performance dark-themed control-plane dashboard and benchmark visualization interface for CloudPilot.

## Features

1. **Live Cluster View**: Auto-refreshing (3s) telemetry showing cluster CPU/RAM utilization, load imbalance, active policy, and per-server utilization bars.
2. **Simulation Runner**: On-demand trace discrete-event simulation across 7 allocation policies with deep SLA efficiency and utilization metrics.
3. **Compare View**: Multi-policy benchmarking on identical traces with pure SVG interactive bar charts and comparative metrics.
4. **History Table**: Historical simulation run records fetched from the backend Supabase persistence store.
5. **Admin Panel**: In-memory authenticated control plane for dynamic policy switching, manual workload allocation/release, and cluster reset.
6. **Free-Tier Wakeup Guard**: Non-blocking detection and notification during backend cold-starts (e.g. Render free tier).

## Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `VITE_API_URL` | Base URL of the CloudPilot backend REST API (e.g., `https://cloudpilot-backend.onrender.com`) | `http://127.0.0.1:8000` in dev |

## Local Development

```bash
# Install dependencies
npm install

# Run Vite development server
npm run dev
```

## Production Build

```bash
# Build optimized static assets into dist/
npm run build

# Preview production build locally
npm run preview
```

## Deployment on Vercel

1. Connect repository on [Vercel](https://vercel.com).
2. Set **Root Directory** to `frontend`.
3. Add environment variable `VITE_API_URL` pointing to your deployed backend (e.g. `https://cloudpilot-api.onrender.com`).
4. Deploy! `vercel.json` provides automatic SPA client routing.
