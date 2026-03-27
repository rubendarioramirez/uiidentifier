#!/usr/bin/env python3
"""
Contour Explorer Web App
Upload any image in the browser → Python runs the pipeline → interactive explorer appears.

Usage: python3 contour_app.py
Then open http://localhost:8765
"""
import base64
import http.server
import json
import os
import tempfile
import time
import urllib.parse
import urllib.request
from generate_explorer import run_pipeline, generate_html

# Pre-load rembg session once at startup (u2netp = fast lightweight model)
print("Loading BG removal model (u2netp)...", flush=True)
try:
    from rembg import new_session, remove as rembg_remove
    _REMBG_SESSION = new_session("u2netp")
    print("BG removal model ready.", flush=True)
except Exception as e:
    _REMBG_SESSION = None
    print(f"BG removal unavailable: {e}", flush=True)

PORT = 8765

# The ComfyUI workflow. The LoadImage entry node is detected automatically —
# its image input is replaced at runtime with the uploaded filename(s).
BASE_WORKFLOW = {
    "1766": {"inputs": {"image": ""}, "class_type": "LoadImage"},
    "1758": {"inputs": {"red": 0, "green": 0, "blue": 0, "threshold": 10, "image": ["1766", 0]}, "class_type": "MaskFromColor+"},
    "1762": {"inputs": {"amount": 4, "device": "auto", "mask": ["1758", 0]}, "class_type": "MaskBlur+"},
    "1764": {"inputs": {"upscale_method": "bicubic", "width": 512, "height": 512, "crop": "disabled", "image": ["1766", 0]}, "class_type": "ImageScale"},
    "1765": {"inputs": {"height": 512, "width": 512, "interpolation_mode": "bicubic", "mask": ["1762", 0]}, "class_type": "JWMaskResize"},
    "1740": {"inputs": {"num_columns": 4, "match_image_size": False, "max_resolution": 2048, "images": ["1764", 0]}, "class_type": "ImageConcatFromBatch"},
    "1741": {"inputs": {"columns": 4, "rows": 1, "image": ["1732:8", 0]}, "class_type": "ImageGridtoBatch"},
    "1742": {"inputs": {"image": ["1741", 0], "alpha": ["1765", 0]}, "class_type": "JoinImageWithAlpha"},
    "1747": {"inputs": {"factor": 1, "method": "luminance (Rec.709)", "image": ["1740", 0]}, "class_type": "ImageDesaturate+"},
    "1767": {"inputs": {"filename_prefix": "ComfyUI", "images": ["1742", 0]}, "class_type": "SaveImage"},
    "1732:75":  {"inputs": {"strength": 1, "model": ["1732:66", 0]}, "class_type": "CFGNorm"},
    "1732:39":  {"inputs": {"vae_name": "qwen_image_vae.safetensors"}, "class_type": "VAELoader"},
    "1732:38":  {"inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}, "class_type": "CLIPLoader"},
    "1732:37":  {"inputs": {"unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "1732:66":  {"inputs": {"shift": 3, "model": ["1732:89", 0]}, "class_type": "ModelSamplingAuraFlow"},
    "1732:8":   {"inputs": {"samples": ["1732:3", 0], "vae": ["1732:39", 0]}, "class_type": "VAEDecode"},
    "1732:89":  {"inputs": {"lora_name": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1, "model": ["1732:37", 0]}, "class_type": "LoraLoaderModelOnly"},
    "1732:110": {"inputs": {"prompt": "", "clip": ["1732:38", 0], "vae": ["1732:39", 0], "image1": ["1747", 0]}, "class_type": "TextEncodeQwenImageEditPlus"},
    "1732:111": {"inputs": {"prompt": "Imagine these are all icons for a mobile videogame. Colour them in a cohesive, consistent, modern palette. Cyberpunk style. Black background", "clip": ["1732:38", 0], "vae": ["1732:39", 0], "image1": ["1747", 0]}, "class_type": "TextEncodeQwenImageEditPlus"},
    "1732:1491": {"inputs": {"conditioning": ["1732:111", 0], "latent": ["1732:1729", 0]}, "class_type": "ReferenceLatent"},
    "1732:1493": {"inputs": {"image": ["1747", 0]}, "class_type": "GetImageSize"},
    "1732:1490": {"inputs": {"conditioning": ["1732:110", 0], "latent": ["1732:1729", 0]}, "class_type": "ReferenceLatent"},
    "1732:88":   {"inputs": {"pixels": ["1747", 0], "vae": ["1732:39", 0]}, "class_type": "VAEEncode"},
    "1732:1729": {"inputs": {"noise_seed": 195622442258469, "noise_strength": 1, "latent": ["1732:88", 0]}, "class_type": "InjectLatentNoise+"},
    "1732:1492": {"inputs": {"width": ["1732:1493", 0], "height": ["1732:1493", 1], "batch_size": 1}, "class_type": "EmptySD3LatentImage"},
    "1732:3":    {"inputs": {"seed": 615229316583434, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1, "model": ["1732:75", 0], "positive": ["1732:1491", 0], "negative": ["1732:1490", 0], "latent_image": ["1732:1730", 0]}, "class_type": "KSampler"},
    "1732:1731": {"inputs": {"confidence_threshold": 0.2, "text_prompt": "", "max_detections": -1, "offload_model": False}, "class_type": "SAM3Grounding"},
    "1732:1730": {"inputs": {"noise_seed": 981899384112205, "noise_strength": 1, "latent": ["1732:1492", 0]}, "class_type": "InjectLatentNoise+"},
}

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
        if self.path == '/process':
            self._handle_process()
        elif self.path == '/comfyui_run':
            self._handle_comfyui_run()
        elif self.path == '/comfyui_batch':
            self._handle_comfyui_batch()
        elif self.path == '/remove_bg':
            self._handle_remove_bg()
        else:
            self.send_error(404)

    def _handle_process(self):
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

    def _handle_comfyui_run(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_length))

        b64_data = body.get('base64', '')
        comfyui_url = body.get('comfyui_url', 'http://localhost:8188').rstrip('/')
        api_key = body.get('api_key', '')
        custom_workflow = body.get('workflow')
        print(f"  url={comfyui_url} key={'SET('+str(len(api_key))+'chars)' if api_key else 'EMPTY'} workflow={'custom' if custom_workflow else 'default'}", flush=True)

        # Strip data URL prefix (data:image/png;base64,...)
        if ',' in b64_data:
            b64_data = b64_data.split(',', 1)[1]

        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['X-API-Key'] = api_key
        auth_headers = {k: v for k, v in headers.items() if k != 'Content-Type'}

        try:
            # Upload image to ComfyUI, get back the filename
            img_bytes = base64.b64decode(b64_data)
            filename = self._upload_image(comfyui_url, auth_headers, img_bytes)
            print(f"  Uploaded as: {filename}", flush=True)

            # Build workflow — use custom if provided, else BASE_WORKFLOW
            import copy
            workflow = copy.deepcopy(custom_workflow if custom_workflow else BASE_WORKFLOW)
            entry_id = self._find_entry_node(workflow)
            if entry_id:
                workflow[entry_id]["inputs"]["image"] = filename

            prompt_payload = json.dumps({"prompt": workflow}).encode()
            req = urllib.request.Request(
                comfyui_url + '/api/prompt',
                data=prompt_payload,
                headers=headers,
                method='POST'
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    result = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8', errors='replace')
                raise ValueError(f'HTTP {e.code} from /api/prompt: {err_body[:300]}')

            prompt_id = result.get('prompt_id')
            if not prompt_id:
                raise ValueError('No prompt_id in response: ' + str(result))

            print(f"  ComfyUI prompt_id: {prompt_id}")

            images = self._poll_and_fetch_images(comfyui_url, prompt_id, headers)
            response = {'prompt_id': prompt_id, 'images': images, 'image_b64': images[0] if images else None}

        except Exception as e:
            print(f"  ComfyUI error: {e}")
            response = {'error': str(e)}

        resp_bytes = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(resp_bytes)

    def _upload_image(self, comfyui_url, auth_headers, img_bytes, filename='crop.png'):
        boundary = b'----CropBoundary'
        body_parts = (
            b'--' + boundary + b'\r\n'
            b'Content-Disposition: form-data; name="image"; filename="' + filename.encode() + b'"\r\n'
            b'Content-Type: image/png\r\n\r\n' +
            img_bytes + b'\r\n' +
            b'--' + boundary + b'--\r\n'
        )
        req = urllib.request.Request(
            comfyui_url + '/api/upload/image',
            data=body_parts,
            headers={**auth_headers, 'Content-Type': f'multipart/form-data; boundary={boundary.decode()}'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='replace')
            raise ValueError(f'Upload failed HTTP {e.code}: {err_body[:300]}')
        return result.get('name', result.get('filename', filename))

    def _find_entry_node(self, workflow):
        """Find the LoadImage node with a string filename (the entry point)."""
        for node_id, node in workflow.items():
            if node.get('class_type') == 'LoadImage':
                image_input = node.get('inputs', {}).get('image', '')
                if isinstance(image_input, str):
                    return node_id
        return None

    def _build_batch_workflow(self, filenames, custom_workflow=None):
        """Take BASE_WORKFLOW (or custom), find entry LoadImage node, replace with
        a LoadImage+ImageBatch chain, patch all references to the entry node."""
        import copy
        workflow = copy.deepcopy(custom_workflow if custom_workflow else BASE_WORKFLOW)
        entry_id = self._find_entry_node(workflow)
        if not entry_id:
            raise ValueError('No LoadImage entry node found in workflow')
        del workflow[entry_id]

        # LoadImage nodes: IDs 10, 11, 12, ...
        for i, fname in enumerate(filenames):
            workflow[str(10 + i)] = {"inputs": {"image": fname}, "class_type": "LoadImage"}

        if len(filenames) == 1:
            img_out = ["10", 0]
        else:
            # IMAGE batch chain: nodes 20, 21, ...
            workflow["20"] = {"inputs": {"image1": ["10", 0], "image2": ["11", 0]}, "class_type": "ImageBatch"}
            for i in range(2, len(filenames)):
                workflow[str(20 + i - 1)] = {
                    "inputs": {"image1": [str(20 + i - 2), 0], "image2": [str(10 + i), 0]},
                    "class_type": "ImageBatch"
                }
            img_out = [str(20 + len(filenames) - 2), 0]

        # Patch all references to [entry_id, slot] with img_out
        for node in workflow.values():
            for key, val in node.get("inputs", {}).items():
                if isinstance(val, list) and len(val) == 2 and str(val[0]) == entry_id:
                    node["inputs"][key] = list(img_out)

        return workflow

    def _handle_comfyui_batch(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_length))

        images_b64 = body.get('images', [])
        comfyui_url = body.get('comfyui_url', 'http://localhost:8188').rstrip('/')
        api_key = body.get('api_key', '')
        custom_workflow = body.get('workflow')
        print(f"  Batch: {len(images_b64)} images, key={'SET' if api_key else 'EMPTY'} workflow={'custom' if custom_workflow else 'default'}", flush=True)

        auth_headers = {}
        if api_key:
            auth_headers['X-API-Key'] = api_key
        headers = {**auth_headers, 'Content-Type': 'application/json'}

        try:
            # Upload all images
            filenames = []
            for i, b64_data in enumerate(images_b64):
                if ',' in b64_data:
                    b64_data = b64_data.split(',', 1)[1]
                img_bytes = base64.b64decode(b64_data)
                fname = self._upload_image(comfyui_url, auth_headers, img_bytes, f'crop_{i}.png')
                filenames.append(fname)
                print(f"  Uploaded [{i+1}/{len(images_b64)}]: {fname}", flush=True)

            workflow = self._build_batch_workflow(filenames, custom_workflow)

            # Submit
            prompt_payload = json.dumps({"prompt": workflow}).encode()
            req = urllib.request.Request(
                comfyui_url + '/api/prompt',
                data=prompt_payload,
                headers=headers,
                method='POST'
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    result = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8', errors='replace')
                raise ValueError(f'HTTP {e.code} from /api/prompt: {err_body[:300]}')

            prompt_id = result.get('prompt_id')
            if not prompt_id:
                raise ValueError('No prompt_id: ' + str(result))
            print(f"  Batch prompt_id: {prompt_id}", flush=True)

            images = self._poll_and_fetch_images(comfyui_url, prompt_id, headers, timeout=90)
            response = {'prompt_id': prompt_id, 'images': images}

        except Exception as e:
            print(f"  Batch error: {e}", flush=True)
            response = {'error': str(e)}

        resp_bytes = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(resp_bytes)

    def _handle_remove_bg(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_length))
        b64_data = body.get('base64', '')
        if ',' in b64_data:
            b64_data = b64_data.split(',', 1)[1]
        try:
            if _REMBG_SESSION is None:
                raise RuntimeError("BG removal model failed to load at startup")
            img_bytes = base64.b64decode(b64_data)
            t0 = time.time()
            result_bytes = rembg_remove(img_bytes, session=_REMBG_SESSION)
            print(f"  BG removed in {time.time()-t0:.1f}s: {len(img_bytes)//1024}KB → {len(result_bytes)//1024}KB", flush=True)
            response = {'image_b64': base64.b64encode(result_bytes).decode()}
        except Exception as e:
            print(f"  BG removal error: {e}", flush=True)
            response = {'error': str(e)}
        resp_bytes = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(resp_bytes)

    def _poll_and_fetch_images(self, comfyui_url, prompt_id, headers, timeout=60):
        """Poll history until complete, fetch and return all result images as base64 list."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.5)
            try:
                hist_req = urllib.request.Request(
                    comfyui_url + '/api/history/' + prompt_id, headers=headers)
                with urllib.request.urlopen(hist_req, timeout=5) as resp:
                    hist = json.loads(resp.read())
            except Exception:
                continue
            if prompt_id not in hist:
                continue
            entry = hist[prompt_id]
            if not entry.get('status', {}).get('completed'):
                continue
            images = []
            for node_out in entry.get('outputs', {}).values():
                for img_info in node_out.get('images', []):
                    if img_info.get('type') != 'output':
                        continue
                    img_url = (comfyui_url + '/api/view?filename=' +
                               urllib.parse.quote(img_info['filename']) +
                               '&type=' + img_info.get('type', 'output'))
                    if img_info.get('subfolder'):
                        img_url += '&subfolder=' + urllib.parse.quote(img_info['subfolder'])
                    img_req = urllib.request.Request(img_url, headers=headers)
                    with urllib.request.urlopen(img_req, timeout=5) as img_resp:
                        images.append(base64.b64encode(img_resp.read()).decode())
            print(f"  Fetched {len(images)} result images", flush=True)
            return images
        return []

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
