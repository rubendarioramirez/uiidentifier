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
import math
import os
import tempfile
import time
import uuid
import urllib.parse
import urllib.request
import websocket
import cv2
import numpy as np
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
    # ── Icon pipeline ──────────────────────────────────────────────────────────
    "1766": {"inputs": {"image": ""}, "class_type": "LoadImage", "_meta": {"title": "Load Icons"}},
    "1758": {"inputs": {"red": 0, "green": 0, "blue": 0, "threshold": 10, "image": ["1766", 0]}, "class_type": "MaskFromColor+"},
    "1762": {"inputs": {"amount": 4, "device": "auto", "mask": ["1758", 0]}, "class_type": "MaskBlur+"},
    "1764": {"inputs": {"upscale_method": "bicubic", "width": 512, "height": 512, "crop": "disabled", "image": ["1766", 0]}, "class_type": "ImageScale"},
    "1765": {"inputs": {"height": 512, "width": 512, "interpolation_mode": "bicubic", "mask": ["1762", 0]}, "class_type": "JWMaskResize"},
    "1740": {"inputs": {"num_columns": 4, "match_image_size": False, "max_resolution": 2048, "images": ["1764", 0]}, "class_type": "ImageConcatFromBatch"},
    "1741": {"inputs": {"columns": 4, "rows": 1, "image": ["1732:8", 0]}, "class_type": "ImageGridtoBatch"},
    "1742": {"inputs": {"image": ["1741", 0], "alpha": ["1765", 0]}, "class_type": "JoinImageWithAlpha"},
    "1747": {"inputs": {"factor": 1, "method": "luminance (Rec.709)", "image": ["1740", 0]}, "class_type": "ImageDesaturate+"},
    "1769": {"inputs": {"image": ["1766", 0]}, "class_type": "GetImageSize"},
    "1770": {"inputs": {"width": ["1769", 0], "height": ["1769", 1], "interpolation": "lanczos", "method": "stretch", "condition": "always", "multiple_of": 0, "image": ["1742", 0]}, "class_type": "ImageResize+"},
    "1767": {"inputs": {"filename_prefix": "ComfyUI", "images": ["1770", 0]}, "class_type": "SaveImage", "_meta": {"title": "Save Image - Icons"}},
    # ── Background inpaint (Flux Fill) ─────────────────────────────────────────
    "1771": {"inputs": {"image": "34970ec1f3093a5c8828a41c994e0a8f83c6c9ef596056a2d9816f58e051d7b1.png"}, "class_type": "LoadImage", "_meta": {"title": "Load Background"}},
    "1793": {"inputs": {"image": "14924d635a44e056c4681ce7ef530cc87f007b349490710d0872265974511552.png"}, "class_type": "LoadImage", "_meta": {"title": "Load Image"}},
    "1794": {"inputs": {"channel": "red", "image": ["1793", 0]}, "class_type": "ImageToMask"},
    "1796": {"inputs": {"expand": -5, "incremental_expandrate": 0, "tapered_corners": True, "flip_input": False, "blur_radius": 4, "lerp_alpha": 1, "decay_factor": 1, "fill_holes": False, "mask": ["1794", 0]}, "class_type": "GrowMaskWithBlur"},
    "1801:34": {"inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors", "type": "flux", "device": "default"}, "class_type": "DualCLIPLoader"},
    "1801:23": {"inputs": {"text": "\n", "clip": ["1801:34", 0]}, "class_type": "CLIPTextEncode"},
    "1801:26": {"inputs": {"guidance": 30, "conditioning": ["1801:23", 0]}, "class_type": "FluxGuidance"},
    "1801:46": {"inputs": {"conditioning": ["1801:23", 0]}, "class_type": "ConditioningZeroOut"},
    "1801:32": {"inputs": {"vae_name": "ae.safetensors"}, "class_type": "VAELoader"},
    "1801:31": {"inputs": {"unet_name": "flux1-fill-dev.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "1801:1797": {"inputs": {"strength": 1, "model": ["1801:31", 0]}, "class_type": "DifferentialDiffusion"},
    "1801:1799": {"inputs": {"noise_mask": True, "positive": ["1801:26", 0], "negative": ["1801:46", 0], "vae": ["1801:32", 0], "pixels": ["1771", 0], "mask": ["1796", 1]}, "class_type": "InpaintModelConditioning"},
    "1801:1800": {"inputs": {"seed": 890913210584896, "steps": 20, "cfg": 1, "sampler_name": "euler", "scheduler": "normal", "denoise": 1, "model": ["1801:1797", 0], "positive": ["1801:1799", 0], "negative": ["1801:1799", 1], "latent_image": ["1801:1799", 2]}, "class_type": "KSampler"},
    "1801:1798": {"inputs": {"samples": ["1801:1800", 0], "vae": ["1801:32", 0]}, "class_type": "VAEDecode"},
    "1806": {"inputs": {"filename_prefix": "ComfyUI", "images": ["1801:1798", 0]}, "class_type": "SaveImage", "_meta": {"title": "Save Image"}},
    # ── Environment restyle (Qwen) ─────────────────────────────────────────────
    "1822:1814": {"inputs": {"unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "1822:1821": {"inputs": {"lora_name": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1, "model": ["1822:1814", 0]}, "class_type": "LoraLoaderModelOnly"},
    "1822:1816": {"inputs": {"shift": 3, "model": ["1822:1821", 0]}, "class_type": "ModelSamplingAuraFlow"},
    "1822:1811": {"inputs": {"strength": 1, "model": ["1822:1816", 0]}, "class_type": "CFGNorm"},
    "1822:1812": {"inputs": {"vae_name": "qwen_image_vae.safetensors"}, "class_type": "VAELoader"},
    "1822:1813": {"inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}, "class_type": "CLIPLoader"},
    "1822:1815": {"inputs": {"prompt": "", "clip": ["1822:1813", 0], "vae": ["1822:1812", 0], "image1": ["1801:1798", 0], "image2": ["1801:1798", 0], "image3": ["1801:1798", 0]}, "class_type": "TextEncodeQwenImageEditPlus"},
    "1822:1817": {"inputs": {"prompt": "Keep the proportions of every element in this mobile video game screenshot, but convert this environment to feature a colour palette and theme akin to that of a medieval fantasty one.", "clip": ["1822:1813", 0], "vae": ["1822:1812", 0], "image1": ["1801:1798", 0], "image2": ["1801:1798", 0], "image3": ["1801:1798", 0]}, "class_type": "TextEncodeQwenImageEditPlus"},
    "1822:112":  {"inputs": {"width": 2560, "height": 1440, "batch_size": 1}, "class_type": "EmptySD3LatentImage"},
    "1822:1818": {"inputs": {"pixels": ["1801:1798", 0], "vae": ["1822:1812", 0]}, "class_type": "VAEEncode"},
    "1822:1819": {"inputs": {"seed": 1118877715456453, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 0.5, "model": ["1822:1811", 0], "positive": ["1822:1817", 0], "negative": ["1822:1815", 0], "latent_image": ["1822:1818", 0]}, "class_type": "KSampler"},
    "1822:1820": {"inputs": {"samples": ["1822:1819", 0], "vae": ["1822:1812", 0]}, "class_type": "VAEDecode"},
    "1823": {"inputs": {"filename_prefix": "ComfyUI", "images": ["1822:1820", 0]}, "class_type": "SaveImage", "_meta": {"title": "Save Image - Environment"}},
    # ── Shared Qwen icon pipeline ──────────────────────────────────────────────
    "1732:75":  {"inputs": {"strength": 1, "model": ["1732:66", 0]}, "class_type": "CFGNorm"},
    "1732:39":  {"inputs": {"vae_name": "qwen_image_vae.safetensors"}, "class_type": "VAELoader"},
    "1732:38":  {"inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}, "class_type": "CLIPLoader"},
    "1732:37":  {"inputs": {"unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "1732:66":  {"inputs": {"shift": 3, "model": ["1732:89", 0]}, "class_type": "ModelSamplingAuraFlow"},
    "1732:8":   {"inputs": {"samples": ["1732:3", 0], "vae": ["1732:39", 0]}, "class_type": "VAEDecode"},
    "1732:89":  {"inputs": {"lora_name": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1, "model": ["1732:37", 0]}, "class_type": "LoraLoaderModelOnly"},
    "1732:110": {"inputs": {"prompt": "", "clip": ["1732:38", 0], "vae": ["1732:39", 0], "image1": ["1747", 0]}, "class_type": "TextEncodeQwenImageEditPlus"},
    "1732:111": {"inputs": {"prompt": "Imagine these are all icons for a mobile videogame. Colour them in a cohesive, consistent, modern palette. Nature materiality. Mossy green and oak brown palette. Black background", "clip": ["1732:38", 0], "vae": ["1732:39", 0], "image1": ["1747", 0]}, "class_type": "TextEncodeQwenImageEditPlus"},
    "1732:1491": {"inputs": {"conditioning": ["1732:111", 0], "latent": ["1732:1729", 0]}, "class_type": "ReferenceLatent"},
    "1732:1493": {"inputs": {"image": ["1747", 0]}, "class_type": "GetImageSize"},
    "1732:1490": {"inputs": {"conditioning": ["1732:110", 0], "latent": ["1732:1729", 0]}, "class_type": "ReferenceLatent"},
    "1732:88":   {"inputs": {"pixels": ["1747", 0], "vae": ["1732:39", 0]}, "class_type": "VAEEncode"},
    "1732:1729": {"inputs": {"noise_seed": 830488168659880, "noise_strength": 1, "latent": ["1732:88", 0]}, "class_type": "InjectLatentNoise+"},
    "1732:1492": {"inputs": {"width": ["1732:1493", 0], "height": ["1732:1493", 1], "batch_size": 1}, "class_type": "EmptySD3LatentImage"},
    "1732:3":    {"inputs": {"seed": 600727535083022, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1, "model": ["1732:75", 0], "positive": ["1732:1491", 0], "negative": ["1732:1490", 0], "latent_image": ["1732:1730", 0]}, "class_type": "KSampler"},
    "1732:1731": {"inputs": {"confidence_threshold": 0.2, "text_prompt": "", "max_detections": -1, "offload_model": False}, "class_type": "SAM3Grounding"},
    "1732:1730": {"inputs": {"noise_seed": 20878200432539, "noise_strength": 1, "latent": ["1732:1492", 0]}, "class_type": "InjectLatentNoise+"},
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
        elif self.path == '/proxy_image':
            self._handle_proxy_image()
        elif self.path == '/upload_original':
            self._handle_upload_original()
        elif self.path == '/detect_ui':
            self._handle_detect_ui()
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
            headers['Authorization'] = f'Bearer {api_key}'
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

            images, node_ids = self._wait_and_fetch_images(comfyui_url, prompt_id, headers)
            response = {'prompt_id': prompt_id, 'images': images, 'node_ids': node_ids, 'image_b64': images[0] if images else None}

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

    def _build_batch_workflow(self, filenames, custom_workflow=None, original_filename=None, mask_filename=None):
        """Take BASE_WORKFLOW (or custom), find entry LoadImage node, replace with
        a LoadImage+ImageBatch chain, patch all references to the entry node.
        If original_filename is given, inject it into the background LoadImage node.
        If mask_filename is given, inject it into the mask LoadImage node."""
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

        # Inject original screenshot into the background LoadImage node
        if original_filename:
            for node_id, node in workflow.items():
                if node.get('class_type') == 'LoadImage':
                    title = (node.get('_meta') or {}).get('title', '').lower()
                    if 'background' in title:
                        node['inputs']['image'] = original_filename
                        print(f"  Injected original into background node {node_id}", flush=True)
                        break

        # Inject mask into the mask LoadImage node (node 1793 "Load Image Mask")
        if mask_filename:
            for node_id, node in workflow.items():
                if node.get('class_type') == 'LoadImage':
                    title = (node.get('_meta') or {}).get('title', '').lower()
                    if 'mask' in title:
                        node['inputs']['image'] = mask_filename
                        print(f"  Injected mask into mask node {node_id}", flush=True)
                        break

        return workflow

    def _pack_icon_grid(self, images_b64_list, cols=4):
        """Pack individual crop images into a cols-column grid PNG.
        Returns (grid_bytes, mask_bytes, count) where mask_bytes is a solid-white PNG
        of the same dimensions (node 1936 — individual masks, no special treatment)."""
        imgs = []
        for b64 in images_b64_list:
            if ',' in b64:
                b64 = b64.split(',', 1)[1]
            arr = np.frombuffer(base64.b64decode(b64), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                imgs.append(img)

        if not imgs:
            raise ValueError('No valid images to pack into grid')

        count  = len(imgs)
        cell_h = max(img.shape[0] for img in imgs)
        cell_w = max(img.shape[1] for img in imgs)
        rows   = math.ceil(count / cols)

        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for i, img in enumerate(imgs):
            r, c = divmod(i, cols)
            h, w = img.shape[:2]
            grid[r * cell_h: r * cell_h + h, c * cell_w: c * cell_w + w] = img

        _, grid_enc = cv2.imencode('.png', grid)
        mask = np.full_like(grid, 255)
        _, mask_enc = cv2.imencode('.png', mask)
        return grid_enc.tobytes(), mask_enc.tobytes(), count

    def _inject_workflow_inputs(self, base_wf, bg_filename, bg_mask_filename,
                                style_string, icon_count, custom_workflow=None):
        """Inject fixed inputs by node ID. Icons/masks (1766, 1936) are handled
        separately by _build_image_batch_chain."""
        import copy
        wf = copy.deepcopy(custom_workflow if custom_workflow else base_wf)

        def set_image(node_id, filename):
            if not filename:
                return
            if node_id in wf:
                wf[node_id]['inputs']['image'] = filename
                print(f"  Injected image into node {node_id}: {filename}", flush=True)
            else:
                print(f"  Warning: node {node_id} not found in workflow", flush=True)

        def set_primitive(node_id, value):
            if node_id not in wf:
                print(f"  Warning: node {node_id} not found in workflow", flush=True)
                return
            inputs = wf[node_id]['inputs']
            hints = ['text', 'string', 'replacement', 'value'] if isinstance(value, str) else ['int', 'value', 'number']
            for key in hints:
                if key in inputs and not isinstance(inputs[key], list):
                    inputs[key] = value
                    print(f"  Injected {type(value).__name__} into node {node_id}[{key}]: {value}", flush=True)
                    return
            for key, val in inputs.items():
                if not isinstance(val, list) and isinstance(val, type(value)):
                    inputs[key] = value
                    print(f"  Injected {type(value).__name__} into node {node_id}[{key}] (fallback): {value}", flush=True)
                    return
            print(f"  Warning: no suitable field for {type(value).__name__} in node {node_id}", flush=True)

        set_image('1771', bg_filename)
        set_image('1793', bg_mask_filename)
        set_primitive('1945', style_string)
        set_primitive('1947', style_string)
        set_primitive('1934', icon_count)
        return wf

    def _build_image_batch_chain(self, workflow, filenames, replace_node_id, start_id):
        """Replace a single LoadImage node (replace_node_id) with N individual LoadImage
        nodes chained together via ImageBatch, wiring the final output into all downstream
        references that previously pointed at replace_node_id.
        LoadImage nodes get IDs start_id, start_id+1, ...
        ImageBatch nodes get IDs start_id+100, start_id+101, ..."""
        import copy
        wf = copy.deepcopy(workflow)
        n = len(filenames)
        batch_start = start_id + 1000  # far enough to never collide with other chains

        # Create one LoadImage per file
        for i, fname in enumerate(filenames):
            wf[str(start_id + i)] = {'inputs': {'image': fname}, 'class_type': 'LoadImage'}

        # Chain them with ImageBatch
        if n == 1:
            final_out = [str(start_id), 0]
        else:
            wf[str(batch_start)] = {
                'inputs': {'image1': [str(start_id), 0], 'image2': [str(start_id + 1), 0]},
                'class_type': 'ImageBatch'
            }
            for i in range(2, n):
                wf[str(batch_start + i - 1)] = {
                    'inputs': {
                        'image1': [str(batch_start + i - 2), 0],
                        'image2': [str(start_id + i), 0]
                    },
                    'class_type': 'ImageBatch'
                }
            final_out = [str(batch_start + n - 2), 0]

        # Rewire all downstream references from replace_node_id to the chain output
        for node in wf.values():
            for key, val in node.get('inputs', {}).items():
                if isinstance(val, list) and len(val) == 2 and str(val[0]) == str(replace_node_id):
                    node['inputs'][key] = list(final_out)

        # Remove the original placeholder node
        wf.pop(str(replace_node_id), None)
        print(f"  Built ImageBatch chain: {n} files → node {replace_node_id} replaced (LoadImage IDs {start_id}–{start_id+n-1})", flush=True)
        return wf

    def _handle_comfyui_batch(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_length))

        images_b64       = body.get('images', [])
        icon_masks_b64   = body.get('icon_masks', [])      # per-icon mask crops from browser
        bg_filename      = body.get('original_filename')   # BG image, pre-uploaded → node 1771
        bg_mask_filename = body.get('mask_filename')       # BG mask, pre-uploaded → node 1793
        style_string     = body.get('style_string', '')    # style for both pipelines
        comfyui_url      = body.get('comfyui_url', 'http://localhost:8188').rstrip('/')
        api_key          = body.get('api_key', '')
        custom_workflow  = body.get('workflow')
        print(f"  Batch: {len(images_b64)} icons | bg={'yes' if bg_filename else 'no'} | "
              f"bg_mask={'yes' if bg_mask_filename else 'no'} | style='{style_string}'", flush=True)

        auth_headers = {}
        if api_key:
            auth_headers['X-API-Key'] = api_key
            auth_headers['Authorization'] = f'Bearer {api_key}'
        headers = {**auth_headers, 'Content-Type': 'application/json'}

        try:
            icon_filenames = []
            mask_filenames = []

            # Upload each icon + a matching white mask individually
            for i, b64 in enumerate(images_b64):
                if ',' in b64:
                    b64 = b64.split(',', 1)[1]
                img_bytes = base64.b64decode(b64)

                arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"  Skipping icon {i+1}: could not decode", flush=True)
                    continue
                h, w = img.shape[:2]

                # Use browser-provided mask crop if available, else fall back to solid white
                if i < len(icon_masks_b64) and icon_masks_b64[i]:
                    m64 = icon_masks_b64[i]
                    if ',' in m64:
                        m64 = m64.split(',', 1)[1]
                    mask_bytes = base64.b64decode(m64)
                else:
                    _, mask_enc = cv2.imencode('.png', np.full((h, w, 3), 255, dtype=np.uint8))
                    mask_bytes = mask_enc.tobytes()

                icon_fname = self._upload_image(comfyui_url, auth_headers, img_bytes,  f'icon_{i}.png')
                mask_fname = self._upload_image(comfyui_url, auth_headers, mask_bytes, f'icon_mask_{i}.png')
                icon_filenames.append(icon_fname)
                mask_filenames.append(mask_fname)
                print(f"  [{i+1}/{len(images_b64)}] icon={icon_fname}  mask={mask_fname}", flush=True)

            # Build workflow: inject fixed inputs (BG, style, count) then wire icon/mask chains
            workflow = self._inject_workflow_inputs(
                BASE_WORKFLOW,
                bg_filename=bg_filename,
                bg_mask_filename=bg_mask_filename,
                style_string=style_string,
                icon_count=len(icon_filenames),
                custom_workflow=custom_workflow,
            )

            # Replace nodes 1766 and 1936 with ImageBatch chains (one LoadImage per file)
            workflow = self._build_image_batch_chain(workflow, icon_filenames, '1766', start_id=10)
            workflow = self._build_image_batch_chain(workflow, mask_filenames, '1936', start_id=110)

            prompt_payload = json.dumps({'prompt': workflow}).encode()
            req = urllib.request.Request(
                comfyui_url + '/api/prompt',
                data=prompt_payload, headers=headers, method='POST'
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

            images, node_ids = self._wait_and_fetch_images(comfyui_url, prompt_id, headers)
            response = {'prompt_id': prompt_id, 'images': images, 'node_ids': node_ids}

        except Exception as e:
            print(f"  Batch error: {e}", flush=True)
            response = {'error': str(e)}

        resp_bytes = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(resp_bytes)

    def _handle_upload_original(self):
        """Upload the original screenshot to ComfyUI and return the stored filename."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_length))
        comfyui_url = body.get('comfyui_url', 'http://localhost:8188').rstrip('/')
        api_key = body.get('api_key', '')
        b64_data = body.get('base64', '')
        filename_hint = body.get('filename_hint', 'original.png')
        if ',' in b64_data:
            b64_data = b64_data.split(',', 1)[1]
        auth_headers = {}
        if api_key:
            auth_headers['X-API-Key'] = api_key
            auth_headers['Authorization'] = f'Bearer {api_key}'
        try:
            img_bytes = base64.b64decode(b64_data)
            filename = self._upload_image(comfyui_url, auth_headers, img_bytes, filename_hint)
            print(f"  Uploaded {filename_hint}: {filename}", flush=True)
            response = {'filename': filename}
        except Exception as e:
            print(f"  Upload original error: {e}", flush=True)
            response = {'error': str(e)}
        resp_bytes = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(resp_bytes)

    def _handle_proxy_image(self):
        """Fetch an image from a remote URL (e.g. ComfyUI cloud) and return it as base64.
        Used to work around CORS restrictions when the browser can't fetch directly."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_length))
        url = body.get('url', '')
        api_key = body.get('api_key', '')
        req_headers = {}
        if api_key:
            req_headers['X-API-Key'] = api_key
            req_headers['Authorization'] = f'Bearer {api_key}'
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                image_b64 = base64.b64encode(resp.read()).decode()
            response = {'image_b64': image_b64}
            print(f"  Proxied image: {url.split('?')[0]}", flush=True)
        except Exception as e:
            print(f"  Proxy error: {e}", flush=True)
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

    def _render_boxes_on_image(self, img_bytes, boxes, img_w, img_h, max_side=768):
        """Draw labeled boxes onto image bytes, return resized PNG b64 for correction round."""
        import numpy as np
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # Scale boxes to match resized image
        scale = min(max_side / img_w, max_side / img_h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(img_w * scale), int(img_h * scale)))
        for b in boxes:
            x1 = max(0, int(b['x'] * scale))
            y1 = max(0, int(b['y'] * scale))
            x2 = min(img.shape[1], int((b['x'] + b['w']) * scale))
            y2 = min(img.shape[0], int((b['y'] + b['h']) * scale))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(img, b['label'], (x1 + 2, max(y1 + 14, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        _, out = cv2.imencode('.png', img)
        return base64.b64encode(out).decode('utf-8')

    def _call_gemini(self, gemini_key, model, contents, tools, tool_config):
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}'
        payload = json.dumps({'contents': contents, 'tools': tools, 'tool_config': tool_config,
                              'generationConfig': {'temperature': 0}}).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

    def _handle_detect_ui(self):
        """Detect UI elements via Gemini function calling with a self-correction loop."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_length))
        gemini_key = body.get('gemini_key', '')
        b64_data = body.get('image_b64', '')
        img_w = body.get('width', 1)
        img_h = body.get('height', 1)
        excl = body.get('excluded_zone')  # {x, y, w, h} in original image pixels
        if ',' in b64_data:
            b64_data = b64_data.split(',', 1)[1]

        model = body.get('model', 'gemini-3.1-pro-preview')
        max_rounds = 3

        tools = [{'function_declarations': [
            {
                'name': 'add_ui_element',
                'description': 'Mark a UI element bounding box in the screenshot',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'label': {'type': 'string', 'description': 'Short name for this UI element'},
                        'x_min': {'type': 'integer', 'description': f'Left edge px (0–{img_w})'},
                        'y_min': {'type': 'integer', 'description': f'Top edge px (0–{img_h})'},
                        'x_max': {'type': 'integer', 'description': f'Right edge px (0–{img_w})'},
                        'y_max': {'type': 'integer', 'description': f'Bottom edge px (0–{img_h})'},
                    },
                    'required': ['label', 'x_min', 'y_min', 'x_max', 'y_max']
                }
            },
            {
                'name': 'confirm_done',
                'description': 'Call this when all boxes are correct and no more changes needed',
                'parameters': {'type': 'object', 'properties': {}, 'required': []}
            }
        ]}]

        try:
            boxes = []
            img_bytes = base64.b64decode(b64_data)

            for round_num in range(max_rounds):
                excl_note = ''
                if excl:
                    ex, ey, ew, eh = excl['x'], excl['y'], excl['w'], excl['h']
                    excl_note = (
                        f' IMPORTANT: ignore everything inside the excluded rectangle '
                        f'x={ex} y={ey} w={ew} h={eh} (x2={ex+ew}, y2={ey+eh}). '
                        'Do NOT call add_ui_element() for anything whose center falls inside that area.'
                    )
                if round_num == 0:
                    # First round: detect from contour image
                    prompt = (
                        f'This image shows colored contour regions detected in a mobile game UI ({img_w}x{img_h} px). '
                        'Each colored shape represents a detected UI zone. '
                        'Call add_ui_element() for each distinct colored region that looks like a UI element '
                        '— buttons, icons, HUD bars, timers, score cards, badges. '
                        'Ignore faint/thin lines (those are game world noise, not UI). '
                        'x=0 is LEFT, y=0 is TOP. Coordinates must be in original image pixels.' +
                        excl_note
                    )
                    contents = [{'parts': [
                        {'text': prompt},
                        {'inline_data': {'mime_type': 'image/png', 'data': b64_data}}
                    ]}]
                    tool_config = {'function_calling_config': {'mode': 'ANY', 'allowed_function_names': ['add_ui_element']}}
                else:
                    # Correction round: send annotated image, ask to fix
                    annotated_b64 = self._render_boxes_on_image(img_bytes, boxes, img_w, img_h)
                    prompt = (
                        f'Round {round_num+1}: Scaled preview with your detected boxes in blue. '
                        f'Original image is {img_w}x{img_h} px — give all coordinates in original pixel space. '
                        'Are all UI elements correctly boxed? '
                        'If yes, call confirm_done(). '
                        'If any boxes are wrong/missing/misaligned, call add_ui_element() for the FULL corrected list.' +
                        excl_note
                    )
                    contents = [{'parts': [
                        {'text': prompt},
                        {'inline_data': {'mime_type': 'image/png', 'data': annotated_b64}}
                    ]}]
                    tool_config = {'function_calling_config': {'mode': 'ANY'}}

                print(f"  Gemini round {round_num+1}/{max_rounds}...", flush=True)
                result = self._call_gemini(gemini_key, model, contents, tools, tool_config)

                new_boxes = []
                confirmed = False
                for part in result['candidates'][0]['content']['parts']:
                    fc = part.get('functionCall')
                    if not fc:
                        continue
                    if fc['name'] == 'confirm_done':
                        confirmed = True
                    elif fc['name'] == 'add_ui_element':
                        a = fc['args']
                        x1, y1 = int(a['x_min']), int(a['y_min'])
                        x2, y2 = int(a['x_max']), int(a['y_max'])
                        new_boxes.append({'label': a.get('label', ''), 'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1})

                if new_boxes:
                    boxes = new_boxes
                    print(f"    → {len(boxes)} elements", flush=True)
                    for b in boxes:
                        print(f"      {b['label']}: ({b['x']},{b['y']}) {b['w']}x{b['h']}", flush=True)

                if confirmed:
                    print(f"  Gemini confirmed done at round {round_num+1}", flush=True)
                    break

                # Stop after round 1 unless verify mode is enabled
                if round_num == 0 and boxes and not body.get('verify', False):
                    print(f"  Round 1 done ({len(boxes)} elements) — verify mode off, stopping", flush=True)
                    break

            response = {'boxes': boxes}
        except Exception as e:
            print(f"  Gemini detect error: {e}", flush=True)
            response = {'error': str(e)}

        resp_bytes = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(resp_bytes)

    def _wait_and_fetch_images(self, comfyui_url, prompt_id, headers, timeout=600):
        """Listen on ComfyUI WebSocket for execution_success, then fetch output images."""
        api_key = headers.get('X-API-Key', '')

        # Build WebSocket URL
        ws_url = comfyui_url.replace('https://', 'wss://').replace('http://', 'ws://')
        client_id = str(uuid.uuid4())
        ws_url = ws_url.rstrip('/') + '/ws?clientId=' + client_id
        if api_key:
            ws_url += '&api_key=' + urllib.parse.quote(api_key)

        print(f"  WS connecting: {ws_url[:60]}...", flush=True)

        completed = False
        output_images = []  # collected from 'executed' WS messages
        last_activity = time.time()
        IDLE_TIMEOUT = 30  # return what we have after 30s of silence
        try:
            ws_headers = {}
            if api_key:
                ws_headers['X-API-Key'] = api_key
                ws_headers['Authorization'] = f'Bearer {api_key}'

            ws = websocket.create_connection(
                ws_url, timeout=timeout,
                header=[f'{k}: {v}' for k, v in ws_headers.items()]
            )
            deadline = time.time() + timeout
            while time.time() < deadline:
                # If we have results and nothing came in for IDLE_TIMEOUT seconds, give up waiting
                if output_images and (time.time() - last_activity) > IDLE_TIMEOUT:
                    print(f"  WS idle {IDLE_TIMEOUT}s with {len(output_images)} result(s) — returning early", flush=True)
                    completed = True
                    break
                ws.settimeout(min(5, deadline - time.time()))
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                last_activity = time.time()
                # Binary frames (preview images) — skip
                if isinstance(raw, bytes):
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                mtype = data.get('type', '')
                d = data.get('data') or {}
                # Log all messages for our prompt for visibility
                if d.get('prompt_id') == prompt_id or mtype in ('status',):
                    print(f"  WS [{mtype}] node={d.get('node')} err={d.get('exception_message','')[:80] if mtype=='execution_error' else ''}", flush=True)
                if mtype == 'executed' and d.get('prompt_id') == prompt_id:
                    output = d.get('output') or {}
                    node_id = d.get('node')
                    for img_info in output.get('images', []):
                        if img_info.get('type') != 'temp':
                            output_images.append({**img_info, 'node_id': node_id})
                            print(f"  WS executed node {node_id}: {img_info.get('filename')}", flush=True)
                elif mtype == 'execution_success' and d.get('prompt_id') == prompt_id:
                    print(f"  WS: execution_success — draining 5s for any remaining outputs", flush=True)
                    completed = True
                    drain_deadline = time.time() + 5
                    ws.settimeout(1)
                    while time.time() < drain_deadline:
                        try:
                            drain_raw = ws.recv()
                            if isinstance(drain_raw, bytes):
                                continue
                            drain_data = json.loads(drain_raw)
                            if drain_data.get('type') == 'executed':
                                dd = drain_data.get('data') or {}
                                if dd.get('prompt_id') == prompt_id:
                                    out = dd.get('output') or {}
                                    dn = dd.get('node')
                                    for img_info in out.get('images', []):
                                        if img_info.get('type') != 'temp':
                                            output_images.append({**img_info, 'node_id': dn})
                                            print(f"  WS drain node {dn}: {img_info.get('filename')}", flush=True)
                        except Exception:
                            break
                    break
                elif mtype == 'execution_error' and d.get('prompt_id') == prompt_id:
                    node_id = d.get('node', '?')
                    msg = d.get('exception_message', 'unknown')
                    print(f"  WS execution_error node {node_id}: {msg}", flush=True)
                    if not output_images:
                        raise ValueError(f"ComfyUI node {node_id} error: {msg}")
                    # Partial results — return what we have
                    completed = True
                    break
            ws.close()
        except Exception as e:
            print(f"  WS error: {e}", flush=True)
            if not completed and not output_images:
                return [], []

        if not output_images:
            print(f"  WS finished with no results", flush=True)
            return [], []

        # Fetch each output image via /api/view
        images = []
        node_ids = []
        base = comfyui_url.rstrip('/')
        for img_info in output_images:
            img_url = (base + '/api/view?filename=' +
                       urllib.parse.quote(img_info['filename']) +
                       '&type=' + img_info.get('type', 'output'))
            if img_info.get('subfolder'):
                img_url += '&subfolder=' + urllib.parse.quote(img_info['subfolder'])
            if api_key:
                img_url += '&api_key=' + urllib.parse.quote(api_key)
            try:
                img_req = urllib.request.Request(img_url, headers=headers)
                with urllib.request.urlopen(img_req, timeout=30) as img_resp:
                    images.append(base64.b64encode(img_resp.read()).decode())
                    node_ids.append(img_info.get('node_id', ''))
                    print(f"  Fetched node {img_info.get('node_id')}: {img_info['filename']}", flush=True)
            except Exception as e:
                print(f"  Failed to fetch {img_info['filename']}: {e}", flush=True)

        print(f"  Fetched {len(images)} result images total", flush=True)
        return images, node_ids

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
