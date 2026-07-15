# GoldenSense Frontend Deployment

This Vite app is the public retail-facing UI only. Deploy it to Vercel as a static frontend and point it at the Python Agent Gateway running on Docker or a container platform.

## Vercel Settings

- Root directory: `modern_showcase_site`
- Build command: `npm run build`
- Output directory: `dist`
- Node.js: `20`

Do not deploy from the repository root. The repository also contains Python services, model checkpoints, and research data that are not part of the public static frontend.

## Required Environment Variables

```bash
VITE_AGENT_API_URL=https://your-backend.example.com/api/v1/agent/analyze
VITE_AGENT_DASHBOARD_URL=https://your-backend.example.com/api/v1/agent/dashboard/current
VITE_AGENT_FEEDBACK_URL=https://your-backend.example.com/api/v1/agent/feedback
VITE_AGENT_API_KEY=your-public-api-key
```

The backend must include the Vercel domain in `AGENT_ALLOW_ORIGINS`.

Current public frontend:

```text
https://goldensense-public-site.vercel.app
```

Current Railway demo backend:

```text
https://agent-gateway-production-fa79.up.railway.app
```

## Backend Readiness

Before routing user traffic to the frontend, verify the backend:

```bash
curl https://agent-gateway-production-fa79.up.railway.app/health/live
curl https://agent-gateway-production-fa79.up.railway.app/health/ready
```

If `/health/ready` fails, keep the frontend deployed but do not treat the system as production-ready.
