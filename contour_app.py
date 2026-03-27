#!/usr/bin/env python3
"""
Contour Explorer Web App
Upload any image in the browser → Python runs the pipeline → interactive explorer appears.

Usage: python3 contour_app.py
Then open http://localhost:8765
"""
import http.server
import json
import os
import tempfile
import urllib.parse
from generate_explorer import run_pipeline, generate_html

PORT = 8765

UPLOAD_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Contour Explorer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0d1117; color: #e6edf3;
         display: flex; flex-direction: column; align-items: center; justify-content: center;
         min-height: 100vh; padding: 20px; }
  h1 { font-size: 1.5em; margin-bottom: 8px; }
  .sub { color: #8b949e; font-size: 0.9em; margin-bottom: 24px; text-align: center; line-height: 1.5; }
  .sub code { color: #58a6ff; }
  .drop {
    width: 500px; max-width: 90vw; padding: 60px 40px; text-align: center;
    border: 2px dashed #30363d; border-radius: 16px; background: #161b22;
    cursor: pointer; transition: border-color 0.2s, background 0.2s;
  }
  .drop:hover, .drop.over { border-color: #58a6ff; background: #1c2333; }
  .drop p { color: #8b949e; font-size: 1em; margin-bottom: 8px; }
  .drop .hint { font-size: 0.8em; color: #484f58; }
  .drop input { display: none; }
  #status { margin-top: 16px; font-size: 0.9em; color: #f0883e; min-height: 20px; }
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #f0883e;
             border-top-color: transparent; border-radius: 50%; animation: spin 0.8s linear infinite;
             vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<h1>Contour Explorer</h1>
<p class="sub">
  Pipeline: <code>Image</code> → <code>Grayscale</code> → <code>Blur</code> →
  <code>Threshold (BINARY)</code> → <code>findContours</code> → <code>Solid Fill Explorer</code><br>
  Deterministic. Same image = same results. Every time.
</p>

<div class="drop" id="drop">
  <p>Drop an image here or click to upload</p>
  <p class="hint">PNG, JPG, any size</p>
  <input type="file" id="file" accept="image/*">
</div>
<div id="status"></div>

<script>
const drop = document.getElementById('drop');
const file = document.getElementById('file');
const status = document.getElementById('status');

drop.addEventListener('click', () => file.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault();
  drop.classList.remove('over');
  if (e.dataTransfer.files.length > 0) upload(e.dataTransfer.files[0]);
});
file.addEventListener('change', e => {
  if (e.target.files.length > 0) upload(e.target.files[0]);
});

function upload(f) {
  status.innerHTML = '<span class="spinner"></span> Processing ' + f.name + '...';
  const form = new FormData();
  form.append('image', f);
  fetch('/process', { method: 'POST', body: form })
    .then(r => {
      if (!r.ok) throw new Error('Server error');
      return r.text();
    })
    .then(html => {
      document.open();
      document.write(html);
      document.close();
    })
    .catch(err => {
      status.textContent = 'Error: ' + err.message;
    });
}
</script>
</body>
</html>'''


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(UPLOAD_PAGE.encode())

    def do_POST(self):
        if self.path != '/process':
            self.send_error(404)
            return

        # Parse multipart form data
        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self.send_error(400, 'Expected multipart/form-data')
            return

        # Read the body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        # Extract boundary
        boundary = content_type.split('boundary=')[1].encode()

        # Find the file data between boundaries
        parts = body.split(b'--' + boundary)
        file_data = None
        file_name = 'image'
        for part in parts:
            if b'filename=' in part:
                # Extract filename
                header_end = part.find(b'\r\n\r\n')
                if header_end == -1:
                    continue
                header = part[:header_end].decode('utf-8', errors='replace')
                for line in header.split('\r\n'):
                    if 'filename=' in line:
                        fn = line.split('filename=')[1].strip('"').strip("'")
                        if fn:
                            file_name = fn
                file_data = part[header_end + 4:]
                # Remove trailing \r\n
                if file_data.endswith(b'\r\n'):
                    file_data = file_data[:-2]
                break

        if file_data is None:
            self.send_error(400, 'No image found in upload')
            return

        # Save to temp file and process
        suffix = os.path.splitext(file_name)[1] or '.png'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        try:
            data = run_pipeline(tmp_path)
            html = generate_html(data, file_name)

            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
        finally:
            os.unlink(tmp_path)

    def log_message(self, format, *args):
        print(f"  {args[0]}")


if __name__ == '__main__':
    print(f"Contour Explorer running at http://localhost:{PORT}")
    print("Upload any image to explore its contours.")
    print("Press Ctrl+C to stop.\n")
    server = http.server.HTTPServer(('', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
