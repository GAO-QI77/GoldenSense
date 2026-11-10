import assert from 'node:assert/strict';
import test from 'node:test';

import { deriveApiEndpoint } from '../../src/apiUrls.js';

const analyzeUrl = 'https://agent.example.com/api/v1/agent/analyze';

test('derives sibling API endpoints from an absolute analyze URL', () => {
  assert.equal(
    deriveApiEndpoint(analyzeUrl, '/api/v1/signals'),
    'https://agent.example.com/api/v1/signals',
  );
  assert.equal(
    deriveApiEndpoint(analyzeUrl, '/api/v1/subscriptions'),
    'https://agent.example.com/api/v1/subscriptions',
  );
});

test('keeps relative endpoints for the local Vite proxy', () => {
  assert.equal(
    deriveApiEndpoint('/api/v1/agent/analyze', '/api/v1/signals'),
    '/api/v1/signals',
  );
});

test('normalizes endpoint paths without requiring a leading slash', () => {
  assert.equal(
    deriveApiEndpoint(analyzeUrl, 'api/v1/subscriptions'),
    'https://agent.example.com/api/v1/subscriptions',
  );
});
