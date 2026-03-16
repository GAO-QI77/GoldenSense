# GoldenSense Live Platform

This is the high-performance, public-facing real-time dashboard for GoldenSense.

## Features
- **FastAPI Backend**: High concurrency support with async I/O.
- **SSE (Server-Sent Events)**: Real-time price updates pushed to the client.
- **Modern UI**: HTML5, Tailwind CSS, and Chart.js.
- **No Auth**: Open access for public demonstration.

## How to Run Locally

1. Ensure you are in the project root.
2. Run the server:
   ```bash
   python3 -m uvicorn live_platform.server:app --host 0.0.0.0 --port 8000 --reload
   ```
3. Open http://localhost:8000 in your browser.

## Deployment (Docker)

1. Build the image from project root:
   ```bash
   docker build -t goldensense-live -f live_platform/Dockerfile .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 goldensense-live
   ```
