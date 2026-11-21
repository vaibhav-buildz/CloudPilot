// Vercel Serverless API Proxy for CloudPilot
// Securely proxies API requests from the Vercel frontend to the Render backend,
// injecting the server-side CLOUDPILOT_API_KEY without exposing it to the browser bundle.

const ALLOWED_ENDPOINTS = {
  GET: new Set([
    '/health',
    '/ready',
    '/api/v1/health',
    '/api/v1/ready',
    '/api/v1/cluster',
    '/api/v1/history',
  ]),
  POST: new Set([
    '/api/v1/simulate',
    '/api/v1/policy',
    '/api/v1/reset',
  ]),
};

const MAX_BODY_BYTES = 64 * 1024; // 64 KB
const UPSTREAM_TIMEOUT_MS = 15000; // 15 seconds

export default async function handler(req, res) {
  // 1. Resolve Backend Base URL strictly from RENDER_BACKEND_URL
  const backendBase = process.env.RENDER_BACKEND_URL
    ? process.env.RENDER_BACKEND_URL.replace(/\/$/, '')
    : '';

  if (!backendBase) {
    return res.status(502).json({
      error: 'Upstream gateway configuration missing: RENDER_BACKEND_URL is not set.',
      code: 'proxy_misconfigured',
    });
  }

  // 2. Validate Path Parameter & Guard against Path Traversal / Injection
  const rawPath = req.query.path || '';
  const targetPathStr = Array.isArray(rawPath) ? rawPath.join('/') : String(rawPath);

  // Check for raw backslashes or null bytes
  if (targetPathStr.includes('\0') || targetPathStr.includes('\\')) {
    return res.status(403).json({
      error: 'Forbidden: Invalid path characters.',
      code: 'forbidden_path',
    });
  }

  // Reject double encoded % (e.g. %252e) or encoded slashes/dots before decoding
  const lowerRaw = targetPathStr.toLowerCase();
  if (
    lowerRaw.includes('%2e') ||
    lowerRaw.includes('%2f') ||
    lowerRaw.includes('%5c') ||
    lowerRaw.includes('%00') ||
    lowerRaw.includes('%25')
  ) {
    return res.status(403).json({
      error: 'Forbidden: Encoded dots, slashes, or null bytes are rejected.',
      code: 'forbidden_path',
    });
  }

  let decodedPath;
  try {
    decodedPath = decodeURIComponent(targetPathStr);
  } catch {
    return res.status(403).json({
      error: 'Forbidden: Malformed path encoding.',
      code: 'forbidden_path',
    });
  }

  // Check for traversal, backslashes, absolute schemes, protocol-relative paths
  if (
    decodedPath.includes('\0') ||
    decodedPath.includes('\\') ||
    decodedPath.includes('..') ||
    decodedPath.includes('://') ||
    decodedPath.startsWith('//')
  ) {
    return res.status(403).json({
      error: 'Forbidden: Path traversal or absolute URL schemes are rejected.',
      code: 'forbidden_path',
    });
  }

  const cleanPath = decodedPath.startsWith('/') ? decodedPath : `/${decodedPath}`;

  // Validate path against strict regex ^/[a-zA-Z0-9/_-]*$
  if (!/^\/[a-zA-Z0-9/_-]*$/.test(cleanPath)) {
    return res.status(403).json({
      error: 'Forbidden: Path contains unapproved characters.',
      code: 'forbidden_path',
    });
  }

  // 3. Enforce Strict Endpoint Allowlist
  const allowedSet = ALLOWED_ENDPOINTS[req.method];
  if (!allowedSet || !allowedSet.has(cleanPath)) {
    return res.status(403).json({
      error: `Forbidden: Endpoint '${req.method} ${cleanPath}' is not permitted by proxy allowlist.`,
      code: 'forbidden_endpoint',
    });
  }

  // 4. Client Body Size Cap (64 KB)
  const incomingContentLength = parseInt(req.headers['content-length'] || '0', 10);
  if (incomingContentLength > MAX_BODY_BYTES) {
    return res.status(413).json({
      error: `Payload too large. Maximum allowed size is ${MAX_BODY_BYTES} bytes.`,
      code: 'payload_too_large',
    });
  }

  let bodyStr = null;
  if (req.method !== 'GET' && req.method !== 'HEAD' && req.body) {
    bodyStr = typeof req.body === 'string' ? req.body : JSON.stringify(req.body);
    if (Buffer.byteLength(bodyStr, 'utf8') > MAX_BODY_BYTES) {
      return res.status(413).json({
        error: `Payload too large. Maximum allowed size is ${MAX_BODY_BYTES} bytes.`,
        code: 'payload_too_large',
      });
    }
  }

  // 5. Header Hygiene: Strip sensitive client headers & construct minimal forward set
  // Client X-API-Key, Authorization, Cookie, and Host are completely stripped.
  const forwardHeaders = {
    'Content-Type': req.headers['content-type'] || 'application/json',
    'Accept': req.headers['accept'] || 'application/json',
  };

  // Attach server-side CLOUDPILOT_API_KEY only (never client-supplied)
  const serverApiKey = process.env.CLOUDPILOT_API_KEY || '';
  if (serverApiKey) {
    forwardHeaders['X-API-Key'] = serverApiKey;
  }

  if (req.headers['x-request-id']) {
    forwardHeaders['X-Request-ID'] = String(req.headers['x-request-id']).slice(0, 64);
  }

  // 6. Trusted Client IP Forwarding
  // On Vercel, x-real-ip is asserted directly by Vercel edge infrastructure from the client socket.
  // Any client-supplied x-real-ip is overwritten by the edge proxy, making it spoof-proof.
  const trustedClientIp = (
    req.headers['x-real-ip'] ||
    (req.headers['x-vercel-forwarded-for'] ? req.headers['x-vercel-forwarded-for'].split(',')[0].trim() : '') ||
    req.socket?.remoteAddress ||
    '127.0.0.1'
  );
  forwardHeaders['X-CloudPilot-Client-IP'] = trustedClientIp;

  // 7. Construct Upstream Target URL
  const targetUrl = new URL(`${backendBase}${cleanPath}`);
  for (const [k, v] of Object.entries(req.query || {})) {
    if (k !== 'path') {
      targetUrl.searchParams.set(k, Array.isArray(v) ? v[0] : v);
    }
  }

  // 8. Forward Request with 15s Timeout AbortController
  const controller = new AbortController();
  const timeoutTimer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  try {
    const fetchOptions = {
      method: req.method,
      headers: forwardHeaders,
      signal: controller.signal,
    };
    if (bodyStr !== null) {
      fetchOptions.body = bodyStr;
    }

    const backendRes = await fetch(targetUrl.toString(), fetchOptions);
    clearTimeout(timeoutTimer);

    const contentType = backendRes.headers.get('content-type') || '';
    res.status(backendRes.status);

    if (contentType.includes('application/json')) {
      const data = await backendRes.json();
      return res.json(data);
    } else {
      const text = await backendRes.text();
      return res.send(text);
    }
  } catch {
    clearTimeout(timeoutTimer);
    return res.status(502).json({
      error: 'Bad Gateway: Upstream Render backend service failure or timeout.',
      code: 'upstream_unavailable',
    });
  }
}
