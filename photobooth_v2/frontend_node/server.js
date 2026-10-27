const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');

const app = express();
const port = parseInt(process.env.PORT || '3000', 10);
const remoteGpuUrl = process.env.REMOTE_GPU_URL || 'http://127.0.0.1:8000';

app.use(cors());

// Ensure .mjs files are served with the proper JavaScript MIME type
express.static.mime.define({ 'application/javascript': ['mjs'] });

// Serve static assets from public/
app.use(express.static(path.join(__dirname, 'public')));

// Configure Multer for pure in-memory buffer storage (zero disk writes)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 } // 25 MB ceiling
});

// Proxy GET /api/health
app.get('/api/health', async (req, res) => {
  try {
    const healthUrl = `${remoteGpuUrl.replace(/\/+$/, '')}/health`;
    const resp = await fetch(healthUrl, { signal: AbortSignal.timeout(5000) });
    if (!resp.ok) {
      return res.status(resp.status).json({
        status: 'degraded',
        node_proxy: 'ok',
        gpu_backend_status: resp.statusText
      });
    }
    const data = await resp.json();
    return res.json({
      status: 'ready',
      node_proxy: 'ok',
      gpu_backend: data
    });
  } catch (err) {
    return res.status(503).json({
      status: 'unavailable',
      node_proxy: 'ok',
      gpu_backend_error: err.message
    });
  }
});

// Proxy POST /api/zombify (zero disk writes)
app.post('/api/zombify', upload.single('file'), async (req, res) => {
  if (!req.file || !req.file.buffer) {
    return res.status(400).json({ error: 'No image file uploaded under field "file"' });
  }

  const startTime = Date.now();
  try {
    const targetUrl = `${remoteGpuUrl.replace(/\/+$/, '')}/zombify`;
    const formData = new FormData();
    const blob = new Blob([req.file.buffer], { type: req.file.mimetype || 'image/jpeg' });
    formData.append('file', blob, req.file.originalname || 'capture.jpg');

    const remoteResp = await fetch(targetUrl, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(60000)
    });

    if (!remoteResp.ok) {
      const errText = await remoteResp.text();
      console.error(`Remote GPU error (${remoteResp.status}):`, errText);
      return res.status(remoteResp.status).send(errText);
    }

    const genTime = remoteResp.headers.get('x-gen-time') || `${Date.now() - startTime}ms`;
    res.setHeader('X-Gen-Time', genTime);
    res.setHeader('Content-Type', 'image/jpeg');

    const arrayBuffer = await remoteResp.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    return res.send(buffer);
  } catch (err) {
    console.error('Error forwarding zombify request to GPU backend:', err);
    return res.status(502).json({
      error: 'Failed to communicate with GPU backend',
      details: err.message
    });
  }
});

const host = process.env.HOST || '127.0.0.1';

const server = app.listen(port, host, () => {
  console.log(`[Photobooth Node.js Proxy] Listening on http://${host}:${port}`);
  console.log(`[Photobooth Node.js Proxy] Upstream GPU URL: ${remoteGpuUrl}`);
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`Port ${port} in use, retrying in 1.5s...`);
    setTimeout(() => {
      server.close();
      server.listen(port, host);
    }, 1500);
  } else {
    console.error('Server error:', err);
  }
});

