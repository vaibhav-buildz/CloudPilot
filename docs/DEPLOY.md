# CloudPilot Deployment Guide

CloudPilot is packaged as a zero-dependency containerized microservice that can be deployed to any container hosting platform or serverless container platform.

---

## 1. Render Deployment (Free Tier)

Render provides free containerized web service hosting using the blueprint specification `render.yaml`.

### Step-by-Step Instructions:
1. Push your repository to GitHub.
2. Sign in to [Render](https://render.com) and navigate to **Blueprints**.
3. Click **New Blueprint Instance** and connect your `CloudPilot` repository.
4. Render automatically parses `render.yaml`:
   - **Service Type**: Web Service
   - **Environment**: Docker
   - **Healthcheck Path**: `/api/v1/health`
   - **Environment Variable**: `CLOUDPILOT_API_KEY` (automatically generated secret).
5. Click **Apply**.
6. Once deployed, open your `https://<service-name>.onrender.com` dashboard in a browser.

---

## 2. Hugging Face Spaces Deployment (Free Tier)

Hugging Face Spaces supports Docker containers natively via repository front-matter.

### Step-by-Step Instructions:
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose **Docker** as the Space SDK and set license / visibility.
3. In your Space repository, ensure `README.md` contains the metadata header:
   ```yaml
   ---
   title: CloudPilot Control Plane
   emoji: 🚀
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 8000
   ---
   ```
4. Push your codebase to the Hugging Face Space repository:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push space main
   ```
5. Hugging Face automatically builds `Dockerfile` and binds port 8000 to the interactive web iframe.

---

## 3. Google Cloud Run Deployment

Google Cloud Run offers fully managed serverless container execution with auto-scaling down to zero instances.

### Prerequisites:
- `gcloud` CLI installed and authenticated (`gcloud auth login`).
- Active Google Cloud project with Cloud Run and Cloud Build APIs enabled.

### Deployment Command:
From the root of the `CloudPilot` repository, deploy directly from source:

```bash
# Set project and region
gcloud config set project YOUR_PROJECT_ID
export REGION="us-central1"

# Deploy to Cloud Run
gcloud run deploy cloudpilot \
  --source . \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --port 8000 \
  --set-env-vars PORT=8000
```

### Health & Telemetry Verification:
```bash
# Retrieve service URL
SERVICE_URL=$(gcloud run services describe cloudpilot --region $REGION --format 'value(status.url)')

# Run smoke test against deployed Cloud Run service
python -m backend.smoke_test --url $SERVICE_URL
```

