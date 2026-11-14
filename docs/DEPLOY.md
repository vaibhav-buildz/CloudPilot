# CloudPilot Production Deployment Guide

CloudPilot is architected as a high-performance, decoupled cloud-native system:
- **Backend**: Zero-dependency Python standard library REST API microservice (Render / Hugging Face Spaces / Google Cloud Run).
- **Frontend**: Lightweight Vite + React single-page dashboard with zero external chart/UI dependencies (Vercel).
- **Persistence**: Managed Supabase PostgreSQL database store via lightweight standard library REST calls.

---

## Architecture Overview

```
+------------------------------------+
|  Vercel Frontend (Vite + React)    |
|  - Live Cluster Telemetry          |
|  - Pure SVG Multi-Policy Compare   |
|  - Trace Simulation Runner         |
+-----------------+------------------+
                  | HTTPS (CORS)
                  v
+------------------------------------+
|  Render / Cloud Run (Python API)   |
|  - Discrete-Event Simulator        |
|  - ML / UCB Allocation Policies    |
|  - Rate-Limited & Token-Guarded    |
+-----------------+------------------+
                  | REST (Urllib)
                  v
+------------------------------------+
|  Supabase Database Store           |
|  - simulation_runs                 |
|  - allocation_events               |
+------------------------------------+
```

---

## 1. Supabase Persistence Setup (Optional)

If persistent history is desired across restarts, create a free project on [Supabase](https://supabase.com).

### Database Tables:
Run this SQL script in the Supabase SQL Editor:

```sql
create table if not exists public.simulation_runs (
    id bigint generated always as identity primary key,
    created_at timestamp with time zone default now(),
    algorithm text,
    policy_key text,
    total_workloads int,
    placed_count int,
    rejected_count int,
    rejection_rate_pct double precision,
    avg_cpu_utilization_pct double precision,
    avg_ram_utilization_pct double precision,
    peak_cpu_utilization_pct double precision,
    peak_ram_utilization_pct double precision,
    avg_load_imbalance double precision,
    peak_active_servers int,
    allocated_cpu_hours double precision,
    allocated_ram_gb_hours double precision,
    simulation_duration_sec double precision,
    sla_efficiency_score double precision
);

create table if not exists public.allocation_events (
    id bigint generated always as identity primary key,
    created_at timestamp with time zone default now(),
    workload_id text,
    cpu double precision,
    ram double precision,
    placed boolean,
    server_id text,
    policy text
);
```

### Credentials:
- Copy the **Project URL** (`https://<project-ref>.supabase.co`).
- Copy the **Service Role Secret Key** (`SUPABASE_SERVICE_KEY`).

---

## 2. Backend Deployment on Render

Render provides free containerized web service hosting using `render.yaml`.

### Step-by-Step Instructions:
1. Push your repository to GitHub.
2. Sign in to [Render](https://render.com) and navigate to **Blueprints**.
3. Click **New Blueprint Instance** and select your repository.
4. Render automatically configures the web service from `render.yaml`:
   - **Environment Variables**:
     - `PORT`: `8000`
     - `ALLOWED_ORIGIN`: `*` (or your Vercel frontend URL, e.g. `https://cloudpilot.vercel.app`)
     - `CLOUDPILOT_API_KEY`: Secret token for state-mutating endpoints (`/allocate`, `/release`, `/reset`, `/policy`).
     - `SUPABASE_URL`: (Optional) `https://<ref>.supabase.co`
     - `SUPABASE_SERVICE_KEY`: (Optional) Supabase service key.
5. Click **Apply**.
6. Note the backend URL (e.g. `https://cloudpilot-backend.onrender.com`).

---

## 3. Frontend Deployment on Vercel

The frontend is a Vite + React application in `/frontend`.

### Step-by-Step Instructions:
1. Sign in to [Vercel](https://vercel.com) and click **Add New Project**.
2. Import your GitHub repository.
3. Configure the project settings:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
4. In **Environment Variables**, add:
   - `VITE_API_URL`: Your deployed Render backend URL (e.g. `https://cloudpilot-backend.onrender.com`).
5. Click **Deploy**.
6. Vercel automatically deploys the static single-page app and configures SPA routing via `vercel.json`.

---

## 4. Alternative Backend: Google Cloud Run

Deploy backend directly from source to Cloud Run:

```bash
gcloud run deploy cloudpilot \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8000 \
  --set-env-vars ALLOWED_ORIGIN="https://cloudpilot.vercel.app"
```

---

## 5. Environment Variables Reference

| Variable | Target | Scope | Description |
| :--- | :--- | :--- | :--- |
| `VITE_API_URL` | Frontend | Build / Runtime | Backend REST URL (e.g. `https://api.cloudpilot.io`). |
| `ALLOWED_ORIGIN` | Backend | Runtime | Allowed CORS origin (default: `*`). |
| `CLOUDPILOT_API_KEY` | Backend | Runtime | Secret token guarding `POST /allocate`, `/release`, `/reset`, `/policy`. |
| `SUPABASE_URL` | Backend | Runtime | Supabase project REST URL. |
| `SUPABASE_SERVICE_KEY` | Backend | Runtime | Supabase service role key (supports JWT and `sb_secret_` formats). |
| `PORT` | Backend | Runtime | Port to bind HTTP server (default: `8000`). |
