// Vercel Serverless API Proxy for CloudPilot
// Securely proxies API requests from the Vercel frontend to the Render backend,
// injecting the server-side CLOUDPILOT_API_KEY without exposing it to the browser bundle.

export default async function handler(req, res) {
  const backendBase = (
    process.env.RENDER_BACKEND_URL ||
    process.env.BACKEND_URL ||
    'https://cloudpilot-2whk.onrender.com'
  ).replace(/\/$/, '');

  const apiKey = process.env.CLOUDPILOT_API_KEY || process.env.BACKEND_API_KEY || '';

  // Extract path from query (e.g. path=api/v1/cluster) or req.url
  const rawPath = req.query.path || '';
  const targetPath = Array.isArray(rawPath) ? rawPath.join('/') : rawPath;
  const cleanPath = targetPath.startsWith('/') ? targetPath : `/${targetPath}`;

  const targetUrl = new URL(`${backendBase}${cleanPath}`);
  for (const [k, v] of Object.entries(req.query || {})) {
    if (k !== 'path') {
      targetUrl.searchParams.set(k, Array.isArray(v) ? v[0] : v);
    }
  }

  const forwardHeaders = {
    'Content-Type': req.headers['content-type'] || 'application/json',
    'Accept': req.headers['accept'] || 'application/json',
  };

  // Attach secret API key from server-side Vercel environment variable
  if (apiKey) {
    forwardHeaders['X-API-Key'] = apiKey;
  }
  // Allow manual client override if supplied in headers
  if (req.headers['x-api-key']) {
    forwardHeaders['X-API-Key'] = req.headers['x-api-key'];
  }
  if (req.headers['x-request-id']) {
    forwardHeaders['X-Request-ID'] = req.headers['x-request-id'];
  }

  try {
    const fetchOptions = {
      method: req.method,
      headers: forwardHeaders,
    };
    if (req.method !== 'GET' && req.method !== 'HEAD' && req.body) {
      fetchOptions.body = typeof req.body === 'string' ? req.body : JSON.stringify(req.body);
    }

    const backendRes = await fetch(targetUrl.toString(), fetchOptions);
    const contentType = backendRes.headers.get('content-type') || '';

    res.status(backendRes.status);
    if (contentType.includes('application/json')) {
      const data = await backendRes.json();
      return res.json(data);
    } else {
      const text = await backendRes.text();
      return res.send(text);
    }
  } catch (err) {
    return res.status(502).json({
      error: `Proxy error forwarding to backend: ${err.message}`,
      status: 502,
      code: 'bad_gateway',
    });
  }
}
