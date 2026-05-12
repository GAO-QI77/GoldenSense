# Security Policy

GoldenSense is an educational market research assistant. It does not execute
trades, store brokerage credentials, or provide personalized investment advice.

## Supported Versions

Security fixes are handled on the `main` branch. Historical branches are kept
for traceability and are not supported.

## Reporting a Vulnerability

Please report security issues privately to the repository owner rather than
opening a public issue with exploit details. Include:

- A concise description of the issue
- Steps to reproduce it
- The affected endpoint, service, or configuration
- Any relevant logs or screenshots with secrets removed

Do not include real API keys, access tokens, private data, or production
credentials in reports.

## Production Hardening

Before exposing a deployment publicly:

- Set `APP_ENV=production`
- Replace all development API keys
- Disable synthetic/sample fallbacks unless explicitly intended
- Restrict trace and trigger endpoints to trusted operators
- Use TLS and an external gateway or reverse proxy for rate limiting
- Keep market/news providers and OpenAI credentials outside the repository
