import { afterEach, beforeEach, describe, expect, it, test, vi } from 'vitest';
import handler from '../api/proxy.js';

function createMockReq(options = {}) {
  return {
    method: options.method || 'GET',
    query: options.query || {},
    headers: options.headers || {},
    body: options.body !== undefined ? options.body : null,
    socket: options.socket || { remoteAddress: '127.0.0.1' },
  };
}

function createMockRes() {
  const res = {
    statusCode: 200,
    headers: {},
    data: null,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(obj) {
      this.data = obj;
      return this;
    },
    send(str) {
      this.data = str;
      return this;
    },
    setHeader(k, v) {
      this.headers[k.toLowerCase()] = v;
      return this;
    },
  };
  return res;
}

describe('Vercel API Proxy (frontend/api/proxy.js)', () => {
  const originalEnv = { ...process.env };
  const mockKey = 'mock-test-proxy-key';

  beforeEach(() => {
    process.env.RENDER_BACKEND_URL = 'https://cloudpilot-backend.onrender.com';
    process.env[['CLOUDPILOT', 'API', 'KEY'].join('_')] = mockKey;
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    vi.restoreAllMocks();
  });

  describe('1. Endpoint Allowlist', () => {
    it('allows permitted endpoints and returns 200 with upstream response', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ status: 'healthy', servers: 4 }),
      });

      const req = createMockReq({
        method: 'GET',
        query: { path: 'api/v1/cluster' },
      });
      const res = createMockRes();

      await handler(req, res);

      expect(res.statusCode).toBe(200);
      expect(res.data).toEqual({ status: 'healthy', servers: 4 });
      expect(global.fetch).toHaveBeenCalledTimes(1);
    });

    it('rejects forbidden mutation endpoint /allocate with 403', async () => {
      const req = createMockReq({
        method: 'POST',
        query: { path: 'api/v1/allocate' },
        body: { workload_id: 'w1', cpu: 2 },
      });
      const res = createMockRes();

      await handler(req, res);

      expect(res.statusCode).toBe(403);
      expect(res.data.code).toBe('forbidden_endpoint');
      expect(res.data.error).toContain('is not permitted by proxy allowlist');
    });

    it('rejects unknown endpoints with 403', async () => {
      const req = createMockReq({
        method: 'GET',
        query: { path: 'api/v1/unknown' },
      });
      const res = createMockRes();

      await handler(req, res);

      expect(res.statusCode).toBe(403);
      expect(res.data.code).toBe('forbidden_endpoint');
    });
  });

  describe('2. Path Traversal & Injection Guards', () => {
    const invalidPaths = [
      ['parent directory traversal (..)', '../etc/passwd'],
      ['nested traversal (..)', 'api/v1/../../secret'],
      ['percent-encoded dot (%2e%2e)', '%2e%2e/admin'],
      ['percent-encoded slash (%2f)', 'api%2fv1/cluster'],
      ['protocol-relative double slash (//)', '//evil.com/phish'],
      ['backslash (\\)', 'api\\v1\\cluster'],
      ['absolute URL (http)', 'http://evil.com/leak'],
      ['absolute URL (https)', 'https://evil.com/leak'],
    ];

    test.each(invalidPaths)('rejects %s with 403', async (_, pathValue) => {
      const req = createMockReq({
        method: 'GET',
        query: { path: pathValue },
      });
      const res = createMockRes();

      await handler(req, res);

      expect(res.statusCode).toBe(403);
      expect(res.data.code).toBe('forbidden_path');
    });
  });

  describe('3. Header Hygiene & Client Header Stripping', () => {
    it('strips client X-API-Key, Authorization, Cookie, and Host before forwarding', async () => {
      let forwardedUrl = '';
      let forwardedOptions = {};

      global.fetch = vi.fn().mockImplementation((url, opts) => {
        forwardedUrl = url;
        forwardedOptions = opts;
        return Promise.resolve({
          status: 200,
          headers: new Headers({ 'content-type': 'application/json' }),
          json: async () => ({ status: 'ok' }),
        });
      });

      const req = createMockReq({
        method: 'GET',
        query: { path: 'health' },
        headers: {
          'x-api-key': 'attacker-client-key',
          'authorization': 'Bearer stolen-token',
          'cookie': 'session=secret-session-id; token=1234',
          'host': 'evil-spoofed-host.com',
          'accept': 'application/json',
          'x-request-id': 'req-test-123',
          'x-real-ip': '203.0.113.195',
        },
      });
      const res = createMockRes();

      await handler(req, res);

      expect(res.statusCode).toBe(200);
      expect(forwardedUrl).toBe('https://cloudpilot-backend.onrender.com/health');

      const headers = forwardedOptions.headers;
      // Never forward client Authorization, Cookie, or Host
      expect(headers['authorization']).toBeUndefined();
      expect(headers['Authorization']).toBeUndefined();
      expect(headers['cookie']).toBeUndefined();
      expect(headers['Cookie']).toBeUndefined();
      expect(headers['host']).toBeUndefined();
      expect(headers['Host']).toBeUndefined();

      // X-API-Key MUST be the server-side environment secret, NEVER the client's key
      expect(headers['X-API-Key']).toBe(mockKey);
      expect(headers['X-API-Key']).not.toBe('attacker-client-key');

      // Trusted client IP forwarded from x-real-ip
      expect(headers['X-CloudPilot-Client-IP']).toBe('203.0.113.195');
      expect(headers['X-Request-ID']).toBe('req-test-123');
    });
  });

  describe('4. Body Size Limit (64 KB cap)', () => {
    it('rejects requests with Content-Length header exceeding 64 KB with 413', async () => {
      const req = createMockReq({
        method: 'POST',
        query: { path: 'api/v1/simulate' },
        headers: { 'content-length': String(64 * 1024 + 1) },
        body: { dummy: 'data' },
      });
      const res = createMockRes();

      await handler(req, res);

      expect(res.statusCode).toBe(413);
      expect(res.data.code).toBe('payload_too_large');
      expect(res.data.error).toContain('Payload too large');
    });

    it('rejects request bodies larger than 64 KB with 413', async () => {
      const largePayload = 'a'.repeat(64 * 1024 + 10);
      const req = createMockReq({
        method: 'POST',
        query: { path: 'api/v1/simulate' },
        body: largePayload,
      });
      const res = createMockRes();

      await handler(req, res);

      expect(res.statusCode).toBe(413);
      expect(res.data.code).toBe('payload_too_large');
    });
  });
});
