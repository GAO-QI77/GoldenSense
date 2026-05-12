# Contributing to GoldenSense

Thanks for helping keep GoldenSense clean, auditable, and useful.

## Development Flow

1. Create a feature branch from `main`.
2. Keep changes focused and avoid unrelated rewrites.
3. Update tests and docs when behavior changes.
4. Run the relevant checks before opening a pull request.
5. Use clear commit messages such as `feat:`, `fix:`, `docs:`, or `chore:`.

## Local Checks

Run the Python test suite from the repository root:

```bash
python3 -m pytest
```

Run the consumer web build and e2e tests from `modern_showcase_site`:

```bash
npm run build
npm run test:e2e
```

## Data and Model Assets

The tracked checkpoints and sample data are lightweight demo/research assets
used by the local development path. Large model versions, private datasets, and
production data should be published as release artifacts, stored in object
storage, or managed with Git LFS rather than committed directly.

## Security

Never commit real API keys, access tokens, `.env` files, production credentials,
or private user data. See `SECURITY.md` for vulnerability reporting guidance.

## License

By contributing, you agree that your contributions are licensed under the MIT
License included in this repository.
