#!/usr/bin/env python3
"""
Contour Explorer Generator
Pipeline: Image → Grayscale → Blur → Adaptive Threshold (BINARY) → findContours → JSON → HTML

Usage: python3 generate_explorer.py <image_path> [output.html]
"""
import cv2
import numpy as np
import json
import sys
import os
import base64

def run_pipeline(image_path):
    """Run the deterministic contour detection pipeline."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")

    # Step 1: Grayscale + Blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Step 2: Adaptive Threshold (BINARY)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    # Step 3: findContours (RETR_TREE)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contours: {len(contours)}")

    # Extract contour data
    contour_data = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        x, y, cw, ch = cv2.boundingRect(cnt)
        circularity = (4 * np.pi * area / (peri * peri)) if peri > 0 else 0
        aspect = float(cw) / ch if ch > 0 else 0
        extent = area / (cw * ch) if cw * ch > 0 else 0
        parent = int(hierarchy[0][i][3])

        # Count children
        num_children = 0
        child = int(hierarchy[0][i][2])
        while child != -1:
            num_children += 1
            child = int(hierarchy[0][child][0])

        # Depth
        depth = 0
        p = parent
        while p != -1:
            depth += 1
            p = int(hierarchy[0][p][3])

        # Simplify points
        approx = cv2.approxPolyDP(cnt, 1.5, True)
        points = approx.reshape(-1, 2).tolist()

        contour_data.append({
            'id': i, 'area': round(area), 'perimeter': round(peri, 1),
            'x': x, 'y': y, 'w': cw, 'h': ch,
            'circularity': round(circularity, 3),
            'aspect': round(aspect, 2),
            'extent': round(extent, 3),
            'parent': parent, 'children': num_children, 'depth': depth,
            'vertices': len(approx), 'points': points
        })

    # Encode original image as base64 for inline HTML
    _, img_encoded = cv2.imencode('.png', img)
    img_b64 = base64.b64encode(img_encoded).decode('utf-8')

    return {'width': w, 'height': h, 'contours': contour_data, 'image_b64': img_b64}


def generate_html(data, image_name="uploaded image"):
    """Generate the explorer HTML with contours, original image, and crop grid."""
    # Separate b64 image from contour JSON (keep JSON smaller)
    img_b64 = data.pop('image_b64', '')
    data_json = json.dumps(data)

    # Canvas size: scale to fit side by side
    scale = min(450 / data['width'], 650 / data['height'])
    canvas_w = int(data['width'] * scale)
    canvas_h = int(data['height'] * scale)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Contour Explorer — {image_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #0d1117; color: #e6edf3; padding: 16px; }}
  h1 {{ text-align: center; font-size: 1.2em; margin-bottom: 4px; }}
  .sub {{ text-align: center; color: #8b949e; font-size: 0.85em; margin-bottom: 12px; }}
  .controls {{
    max-width: 1200px; margin: 0 auto 12px;
    background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px;
  }}
  .slider-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
  .slider-row label {{ min-width: 110px; font-size: 0.85em; color: #8b949e; }}
  .slider-row input[type=range] {{ flex: 1; }}
  .slider-row .val {{ min-width: 80px; font-size: 0.9em; font-weight: bold; text-align: right; }}
  .stats {{ font-size: 0.85em; color: #58a6ff; text-align: center; margin-top: 8px; }}

  .side-by-side {{
    max-width: 1600px; margin: 0 auto 16px; display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;
  }}
  .panel {{ text-align: center; }}
  .panel h2 {{ font-size: 0.85em; color: #8b949e; margin-bottom: 6px; }}
  canvas {{ border: 1px solid #30363d; border-radius: 4px; cursor: crosshair; }}

  .crops-section {{
    max-width: 1200px; margin: 0 auto;
    background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px;
  }}
  .crops-section h2 {{ font-size: 0.95em; color: #58a6ff; margin-bottom: 12px; text-align: center; }}
  .crops-grid {{
    display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;
  }}
  .crop-item {{
    background: #0d1117; border: 2px solid #30363d; border-radius: 6px;
    padding: 4px; text-align: center; cursor: pointer; transition: border-color 0.2s;
  }}
  .crop-item:hover {{ border-color: #58a6ff; }}
  .crop-item canvas {{ border-radius: 4px; display: block; }}
  .crop-item .crop-label {{
    font-size: 0.7em; color: #8b949e; margin-top: 4px; white-space: nowrap;
  }}
  .comfy-config {{ display: flex; align-items: center; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }}
  .comfy-config label {{ font-size: 0.85em; color: #8b949e; }}
  .comfy-config input {{ background: #0d1117; border: 1px solid #30363d; border-radius: 4px; color: #e6edf3; padding: 3px 8px; font-size: 0.85em; width: 200px; }}
  .comfy-config button {{ padding: 4px 12px; background: #21262d; color: #58a6ff; border: 1px solid #30363d; border-radius: 4px; cursor: pointer; font-size: 0.85em; }}
  .comfy-config button:hover {{ background: #30363d; }}
  .comfy-btn {{ margin-top: 4px; font-size: 0.7em; padding: 2px 6px; background: #21262d; color: #58a6ff; border: 1px solid #30363d; border-radius: 4px; cursor: pointer; width: 100%; }}
  .comfy-btn:hover {{ background: #30363d; }}
  .bg-btn {{ margin-top: 4px; font-size: 0.7em; padding: 2px 6px; background: #21262d; color: #a5d6a7; border: 1px solid #30363d; border-radius: 4px; cursor: pointer; width: 100%; }}
  .bg-btn:hover {{ background: #30363d; }}
  .comfy-status {{ font-size: 0.65em; color: #8b949e; min-height: 14px; margin-top: 2px; text-align: center; }}
  .comfy-result img {{ max-width: 120px; max-height: 120px; border-radius: 4px; margin-top: 4px; display: block; }}
  .crop-checkbox {{ position: absolute; top: 4px; left: 4px; width: 16px; height: 16px; cursor: pointer; accent-color: #ffa657; }}
  .crop-item {{ position: relative; }}
  .workflow-panel {{ max-width: 1200px; margin: 0 auto 12px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
  .workflow-toggle {{ width: 100%; background: none; border: none; color: #8b949e; font-size: 0.85em; padding: 8px 16px; text-align: left; cursor: pointer; display: flex; align-items: center; gap: 6px; }}
  .workflow-toggle:hover {{ color: #e6edf3; background: #1c2333; }}
  .workflow-body {{ display: none; padding: 12px 16px; border-top: 1px solid #30363d; }}
  .workflow-body.open {{ display: block; }}
  .workflow-body textarea {{ width: 100%; height: 180px; background: #0d1117; border: 1px solid #30363d; border-radius: 4px; color: #e6edf3; font-family: monospace; font-size: 0.75em; padding: 8px; resize: vertical; }}
  .workflow-body .wf-actions {{ display: flex; gap: 8px; margin-top: 8px; align-items: center; }}
  #resultsSection {{ max-width: 1600px; margin: 24px auto; padding: 0 16px; display: none; }}
  #resultsSection h2 {{ font-size: 1em; color: #8b949e; margin-bottom: 12px; }}
  .results-grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .result-card {{
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    overflow: hidden; display: flex; flex-direction: column;
  }}
  .result-card img {{ display: block; max-width: 100%; max-height: 480px; object-fit: contain; background: #0d1117; }}
  .result-card .rc-footer {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 12px; font-size: 0.8em; color: #8b949e;
  }}
  .result-card .rc-footer a {{
    color: #58a6ff; text-decoration: none; font-size: 0.85em;
  }}
  .result-card .rc-footer a:hover {{ text-decoration: underline; }}
  .workflow-body .wf-status {{ font-size: 0.8em; color: #8b949e; }}

  #info {{
    position: fixed; bottom: 16px; right: 16px; background: #161b22ee; border: 1px solid #30363d;
    border-radius: 8px; padding: 12px; font-size: 0.8em; min-width: 220px; display: none; z-index: 10;
  }}
  #info div {{ margin-bottom: 2px; }}
</style>
</head>
<body>
<h1>Contour Explorer — {image_name}</h1>
<p class="sub">{data['width']}x{data['height']} — {len(data['contours'])} contours — Pipeline: Grayscale → Blur → Threshold (BINARY) → findContours</p>

<div class="controls">
  <div class="slider-row">
    <label>Min area:</label>
    <input type="range" id="minArea" min="0" max="10000" value="2000" step="50">
    <div class="val" id="minAreaVal">2,000</div>
  </div>
  <div class="slider-row">
    <label>Max area:</label>
    <input type="range" id="maxArea" min="1000" max="200000" value="50000" step="500">
    <div class="val" id="maxAreaVal">50,000</div>
  </div>
  <div class="slider-row">
    <label>Only parents:</label>
    <label><input type="checkbox" id="onlyParents" checked> Skip contours whose parent is also visible</label>
  </div>
  <div class="slider-row">
    <label>Merge nearby:</label>
    <label><input type="checkbox" id="mergeNearby"> Gap ≤</label>
    <input type="range" id="proximity" min="1" max="20" value="5" step="1">
    <div class="val" id="proxVal">5px</div>
  </div>
  <div class="slider-row">
    <label>Max size ratio:</label>
    <input type="range" id="sizeRatio" min="1" max="20" value="5" step="1">
    <div class="val" id="sizeRatioVal">5×</div>
  </div>
  <div class="stats" id="stats"></div>
</div>

<div class="side-by-side">
  <div class="panel">
    <h2>Contours (solid fill)</h2>
    <canvas id="canvas" width="{canvas_w}" height="{canvas_h}"></canvas>
  </div>
  <div class="panel">
    <h2>Mask preview</h2>
    <canvas id="maskCanvas" width="{canvas_w}" height="{canvas_h}"></canvas>
  </div>
  <div class="panel">
    <h2>Original image</h2>
    <canvas id="origCanvas" width="{canvas_w}" height="{canvas_h}"></canvas>
  </div>
  <div class="panel">
    <h2>Modified <span id="modCount" style="color:#ffa657;font-size:0.8em;"></span></h2>
    <canvas id="modCanvas" width="{canvas_w}" height="{canvas_h}"></canvas>
  </div>
</div>

<div class="workflow-panel">
  <button class="workflow-toggle" onclick="toggleWorkflow()">
    <span id="wfArrow">&rsaquo;</span> ComfyUI Workflow JSON
    <span id="wfNodeHint" style="margin-left:auto;color:#58a6ff;font-size:0.8em;"></span>
  </button>
  <div class="workflow-body" id="workflowBody">
    <textarea id="workflowJson" placeholder="Paste your ComfyUI workflow JSON here. Node &quot;1733&quot; (LoadImage) will receive the uploaded image."></textarea>
    <div class="wf-actions">
      <button onclick="loadWorkflowFromFile()" style="color:#58a6ff;">Load JSON file</button>
      <button onclick="applyWorkflow()" style="color:#3fb950;">Apply</button>
      <button onclick="clearWorkflow()" style="color:#f85149;">Clear</button>
      <input type="file" id="wfFileInput" accept=".json" style="display:none" onchange="readWorkflowFile(event)">
      <span class="wf-status" id="wfStatus"></span>
    </div>
  </div>
</div>

<div class="crops-section">
  <h2>Cropped Icons from Original (<span id="cropCount">0</span> cuts)</h2>
  <div class="comfy-config">
    <label>Gemini Key:</label>
    <input type="password" id="geminiKey" placeholder="AIza..." style="width:160px;">
    <button onclick="detectUIWithAI()" style="color:#c9d1d9;background:#388bfd22;border-color:#388bfd;">&#x1F916; Detect UI</button>
    <button onclick="downloadViewDescriptionJSON()" style="color:#a5d6a7;background:#3fb95022;border-color:#3fb950;">&#x2B07; Download JSON</button>
    <span id="aiStatus" style="font-size:0.8em;color:#8b949e;margin-right:16px;"></span>
    <label>ComfyUI URL:</label>
    <input type="text" id="comfyUrl" value="https://cloud.comfy.org">
    <label>API Key:</label>
    <input type="password" id="comfyApiKey" placeholder="comfyui-..." style="width:180px;">
    <button onclick="removeAllBg()" style="color:#a5d6a7;">Remove All BG</button>
    <button onclick="sendAllToComfyUI()">Send All &rarr; ComfyUI</button>
    <button onclick="toggleSelectAll()" id="selectAllBtn" style="color:#ffa657;">Select All</button>
    <button onclick="sendBatchToComfyUI()" style="color:#ffa657;">Send Batch &rarr; ComfyUI</button>
    <span id="batchStatus" style="font-size:0.8em;color:#8b949e;"></span>
  </div>
  <div class="crops-grid" id="cropsGrid"></div>
</div>

<div id="resultsSection">
  <h2>ComfyUI Results <span id="resultsCount" style="color:#ffa657;"></span></h2>
  <div class="results-grid" id="resultsGrid"></div>
</div>

<div id="info">
  <div id="iText"></div>
</div>

<script>
const DATA = {data_json};
const IMG_B64 = "data:image/png;base64,{img_b64}";

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const origCanvas = document.getElementById('origCanvas');
const origCtx = origCanvas.getContext('2d');
const modCanvas = document.getElementById('modCanvas');
const modCtx = modCanvas.getContext('2d');
const maskPreviewCanvas = document.getElementById('maskCanvas');
const maskPreviewCtx = maskPreviewCanvas.getContext('2d');
let _modCount = 0;
let _currentGroups = [];
let _aiBoxes = [];
const scaleX = {canvas_w} / DATA.width;
const scaleY = {canvas_h} / DATA.height;

// ── ROI (crop-exclusion zone) ─────────────────────────────────────────────────
// Default: full width, center 60% of height
let roi = {{ x: 0, y: Math.round({canvas_h} * 0.2), w: {canvas_w}, h: Math.round({canvas_h} * 0.6) }};
let roiDrag = null;
const HANDLE_R = 7;

function clampRoi() {{
  roi.w = Math.max(20, roi.w);
  roi.h = Math.max(20, roi.h);
  roi.x = Math.max(0, Math.min(roi.x, canvas.width - roi.w));
  roi.y = Math.max(0, Math.min(roi.y, canvas.height - roi.h));
  roi.w = Math.min(roi.w, canvas.width - roi.x);
  roi.h = Math.min(roi.h, canvas.height - roi.y);
}}

function roiHandles() {{
  const r = roi;
  return [
    {{ id:'nw', x:r.x,           y:r.y         }},
    {{ id:'n',  x:r.x + r.w/2,   y:r.y         }},
    {{ id:'ne', x:r.x + r.w,     y:r.y         }},
    {{ id:'e',  x:r.x + r.w,     y:r.y + r.h/2 }},
    {{ id:'se', x:r.x + r.w,     y:r.y + r.h   }},
    {{ id:'s',  x:r.x + r.w/2,   y:r.y + r.h   }},
    {{ id:'sw', x:r.x,           y:r.y + r.h   }},
    {{ id:'w',  x:r.x,           y:r.y + r.h/2 }},
  ];
}}

const HANDLE_CURSORS = {{ nw:'nw-resize', n:'n-resize', ne:'ne-resize', e:'e-resize',
                          se:'se-resize', s:'s-resize', sw:'sw-resize', w:'w-resize' }};

function hitROI(cx, cy) {{
  for (const h of roiHandles()) {{
    if (Math.abs(cx - h.x) <= HANDLE_R + 3 && Math.abs(cy - h.y) <= HANDLE_R + 3) return h.id;
  }}
  if (cx > roi.x && cx < roi.x + roi.w && cy > roi.y && cy < roi.y + roi.h) return 'move';
  return null;
}}

function drawROI() {{
  const r = roi;
  ctx.save();
  // Dim excluded zones
  ctx.fillStyle = 'rgba(0,0,0,0.58)';
  ctx.fillRect(0, 0, canvas.width, r.y);
  ctx.fillRect(0, r.y + r.h, canvas.width, canvas.height - r.y - r.h);
  ctx.fillRect(0, r.y, r.x, r.h);
  ctx.fillRect(r.x + r.w, r.y, canvas.width - r.x - r.w, r.h);
  // Dashed border
  ctx.strokeStyle = '#58a6ff';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 3]);
  ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
  ctx.setLineDash([]);
  // Corner + edge handles
  ctx.fillStyle = '#58a6ff';
  for (const h of roiHandles()) {{
    ctx.beginPath();
    ctx.arc(h.x, h.y, HANDLE_R, 0, Math.PI * 2);
    ctx.fill();
  }}
  // Label
  ctx.fillStyle = 'rgba(88,166,255,0.85)';
  ctx.font = '11px system-ui';
  ctx.fillText('excluded zone — drag to adjust', r.x + 8, r.y + 16);
  ctx.restore();
}}
// ─────────────────────────────────────────────────────────────────────────────

// Load original image
const origImg = new Image();
let origLoaded = false;

// Hidden full-size canvas for cropping at original resolution
const fullCanvas = document.createElement('canvas');
fullCanvas.width = DATA.width;
fullCanvas.height = DATA.height;
const fullCtx = fullCanvas.getContext('2d');

origImg.onload = function() {{
  origLoaded = true;
  // Draw scaled version
  origCtx.drawImage(origImg, 0, 0, {canvas_w}, {canvas_h});
  // Initialize modified canvas with original image
  modCtx.drawImage(origImg, 0, 0, {canvas_w}, {canvas_h});
  // Draw full-size for cropping
  fullCtx.drawImage(origImg, 0, 0, DATA.width, DATA.height);
  draw();
}};
origImg.src = IMG_B64;

function hslStr(id) {{
  const hue = (id * 137.508) % 360;
  return {{ h: hue, css: 'hsl(' + hue + ', 75%, 55%)' }};
}}

function getFiltered() {{
  const minA = +document.getElementById('minArea').value;
  const maxA = +document.getElementById('maxArea').value;
  const onlyParents = document.getElementById('onlyParents').checked;

  const visibleSet = new Set();
  for (const c of DATA.contours) {{
    if (c.area >= minA && c.area <= maxA) visibleSet.add(c.id);
  }}

  let filtered;
  if (onlyParents) {{
    filtered = DATA.contours.filter(c => {{
      if (!visibleSet.has(c.id)) return false;
      let p = c.parent;
      while (p !== -1) {{
        if (visibleSet.has(p)) return false;
        p = DATA.contours[p].parent;
      }}
      return true;
    }});
  }} else {{
    filtered = DATA.contours.filter(c => visibleSet.has(c.id));
  }}

  filtered.sort((a, b) => b.area - a.area);
  return filtered;
}}

function draw() {{
  const minA = +document.getElementById('minArea').value;
  const maxA = +document.getElementById('maxArea').value;
  const onlyParents = document.getElementById('onlyParents').checked;

  document.getElementById('minAreaVal').textContent = minA.toLocaleString();
  document.getElementById('maxAreaVal').textContent = maxA.toLocaleString();

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const filtered = getFiltered();
  const groups   = computeGroups(filtered);
  const doMerge  = document.getElementById('mergeNearby').checked;

  let colorIdx = 0;
  for (const g of groups) {{
    const color = hslStr(colorIdx++);
    g._color = color.css;
    if (doMerge) {{
      // Draw merged bounding box as solid rectangle
      ctx.fillStyle = color.css;
      ctx.fillRect(g.x * scaleX, g.y * scaleY, g.w * scaleX, g.h * scaleY);
    }} else {{
      // Draw each contour shape individually
      for (const c of g.members) {{
        if (c.points.length < 3) continue;
        c._color = color.css;
        ctx.beginPath();
        ctx.moveTo(c.points[0][0] * scaleX, c.points[0][1] * scaleY);
        for (let i = 1; i < c.points.length; i++) {{
          ctx.lineTo(c.points[i][0] * scaleX, c.points[i][1] * scaleY);
        }}
        ctx.closePath();
        ctx.fillStyle = color.css;
        ctx.fill();
      }}
    }}
  }}

  window._filtered = filtered;
  window._groups   = groups;
  document.getElementById('stats').textContent =
    'Showing ' + (doMerge ? groups.length + ' groups' : filtered.length + ' contours') +
    ' / ' + DATA.contours.length + ' total (area ' +
    minA.toLocaleString() + ' – ' + maxA.toLocaleString() + ')' +
    (onlyParents ? ' — outermost only' : '');

  drawROI();
  // Draw AI-detected boxes as blue outlines (skip excluded zone)
  if (_aiBoxes.length > 0) {{
    const roiXai = roi.x / scaleX, roiYai = roi.y / scaleY;
    const roiWai = roi.w / scaleX, roiHai = roi.h / scaleY;
    ctx.save();
    ctx.lineWidth = 2;
    ctx.font = 'bold 10px sans-serif';
    for (const b of _aiBoxes) {{
      const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
      const excluded = cx >= roiXai && cx <= roiXai + roiWai && cy >= roiYai && cy <= roiYai + roiHai;
      ctx.strokeStyle = excluded ? '#666' : '#388bfd';
      ctx.fillStyle   = excluded ? '#666' : '#388bfd';
      ctx.strokeRect(b.x * scaleX, b.y * scaleY, b.w * scaleX, b.h * scaleY);
      ctx.fillText(b.label || '', b.x * scaleX + 2, b.y * scaleY + 11);
    }}
    ctx.restore();
  }}
  _currentGroups = groups;
  drawMaskPreview(groups);
  generateCrops(groups);
}}

function computeGroups(filtered) {{
  const thresh = +document.getElementById('proximity').value;
  const items  = filtered.filter(c => c.points.length >= 3);
  const doMerge = document.getElementById('mergeNearby').checked;

  if (!doMerge) {{
    return items.map(c => ({{ x:c.x, y:c.y, w:c.w, h:c.h, id:c.id, _color:c._color, members:[c], count:1 }}));
  }}

  const par = items.map((_, i) => i);
  function find(i) {{ return par[i] === i ? i : (par[i] = find(par[i])); }}
  function unite(i, j) {{ par[find(i)] = find(j); }}

  const maxRatio = +document.getElementById('sizeRatio').value;

  for (let i = 0; i < items.length; i++) {{
    for (let j = i + 1; j < items.length; j++) {{
      const a = items[i], b = items[j];
      // Skip if size ratio is too large (large bg element absorbing small icons)
      const areaA = a.area || 1, areaB = b.area || 1;
      if (Math.max(areaA, areaB) / Math.min(areaA, areaB) > maxRatio) continue;
      const hgap = Math.max(0, Math.max(a.x, b.x) - Math.min(a.x + a.w, b.x + b.w));
      const vgap = Math.max(0, Math.max(a.y, b.y) - Math.min(a.y + a.h, b.y + b.h));
      if (hgap <= thresh && vgap <= thresh) unite(i, j);
    }}
  }}

  const map = new Map();
  for (let i = 0; i < items.length; i++) {{
    const root = find(i);
    if (!map.has(root)) map.set(root, []);
    map.get(root).push(items[i]);
  }}

  return [...map.values()].map(members => {{
    const x  = Math.min(...members.map(c => c.x));
    const y  = Math.min(...members.map(c => c.y));
    const x2 = Math.max(...members.map(c => c.x + c.w));
    const y2 = Math.max(...members.map(c => c.y + c.h));
    const rep = members.reduce((a, b) => a.area > b.area ? a : b);
    return {{ x, y, w: x2 - x, h: y2 - y, id: rep.id, _color: rep._color, members, count: members.length }};
  }});
}}

function generateCrops(groups) {{
  const grid = document.getElementById('cropsGrid');
  grid.innerHTML = '';
  if (!origLoaded) return;

  const roiX = roi.x / scaleX, roiY = roi.y / scaleY;
  const roiW = roi.w / scaleX, roiH = roi.h / scaleY;

  let count = 0;

  for (const g of groups) {{
    // Skip groups whose center falls INSIDE the exclusion rectangle
    const ccx = g.x + g.w / 2, ccy = g.y + g.h / 2;
    if (ccx >= roiX && ccx <= roiX + roiW && ccy >= roiY && ccy <= roiY + roiH) continue;

    const color = g._color || '#58a6ff';

    // Crop with 10% padding
    const padX = Math.round(g.w * 0.10), padY = Math.round(g.h * 0.10);
    const cropX = Math.max(0, g.x - padX);
    const cropY = Math.max(0, g.y - padY);
    const cropW = Math.min(DATA.width  - cropX, g.w + padX * 2);
    const cropH = Math.min(DATA.height - cropY, g.h + padY * 2);
    const cropData = fullCtx.getImageData(cropX, cropY, cropW, cropH);

    // Display size: max 120px
    const maxSize = 120;
    const displayScale = Math.min(maxSize / cropW, maxSize / cropH, 1);
    const dw = Math.round(cropW * displayScale);
    const dh = Math.round(cropH * displayScale);

    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = dw; cropCanvas.height = dh;
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = cropW; tmpCanvas.height = cropH;
    tmpCanvas.getContext('2d').putImageData(cropData, 0, 0);
    cropCanvas.getContext('2d').drawImage(tmpCanvas, 0, 0, dw, dh);

    const item = document.createElement('div');
    item.className = 'crop-item';
    item.style.borderColor = color;
    item.appendChild(cropCanvas);

    const lbl = document.createElement('div');
    lbl.className = 'crop-label';
    lbl.textContent = '#' + g.id + (g.count > 1 ? ' +' + (g.count - 1) : '') + ' (' + g.w + '×' + g.h + ')';
    item.appendChild(lbl);

    item.title = 'Click canvas to download' + (g.count > 1 ? ' (' + g.count + ' merged)' : '');
    cropCanvas.addEventListener('click', (function(cd, cw, ch, gid) {{
      return function() {{
        const dl = document.createElement('canvas');
        dl.width = cw; dl.height = ch;
        dl.getContext('2d').putImageData(cd, 0, 0);
        const a = document.createElement('a');
        a.download = 'crop_' + gid + '_' + cw + 'x' + ch + '.png';
        a.href = dl.toDataURL('image/png');
        a.click();
      }};
    }})(cropData, cropW, cropH, g.id));

    item._group = g;

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.className = 'crop-checkbox';
    checkbox.addEventListener('change', () => {{
      item.style.outline = checkbox.checked ? '2px solid #ffa657' : 'none';
    }});
    item.appendChild(checkbox);

    const bgBtn = document.createElement('button');
    bgBtn.className = 'bg-btn';
    bgBtn.textContent = 'Remove BG';
    const comfyBtn = document.createElement('button');
    comfyBtn.className = 'comfy-btn';
    comfyBtn.textContent = '\u2192 ComfyUI';
    const comfyStatus = document.createElement('div');
    comfyStatus.className = 'comfy-status';
    const comfyResult = document.createElement('div');
    comfyResult.className = 'comfy-result';
    bgBtn.addEventListener('click', (function(it, c, s) {{
      return function() {{ removeCropBg(it, c, s); }};
    }})(item, cropCanvas, comfyStatus));
    comfyBtn.addEventListener('click', (function(it, c, s, r) {{
      return function() {{
        const b64 = it._rgbaB64 || c.toDataURL('image/png');
        sendCropToComfyUI(b64, s, r);
      }};
    }})(item, cropCanvas, comfyStatus, comfyResult));
    item.appendChild(bgBtn);
    item.appendChild(comfyBtn);
    item.appendChild(comfyStatus);
    item.appendChild(comfyResult);

    grid.appendChild(item);
    count++;
  }}

  document.getElementById('cropCount').textContent = count;
}}

function drawCheckerboard(ctx, w, h) {{
  const size = 8;
  for (let y = 0; y < h; y += size) {{
    for (let x = 0; x < w; x += size) {{
      ctx.fillStyle = ((Math.floor(x/size) + Math.floor(y/size)) % 2 === 0) ? '#cccccc' : '#ffffff';
      ctx.fillRect(x, y, size, size);
    }}
  }}
}}

async function removeCropBg(item, cropCanvas, statusEl) {{
  statusEl.textContent = 'Removing BG\u2026';
  statusEl.style.color = '#f0883e';
  try {{
    const resp = await fetch('/remove_bg', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{base64: cropCanvas.toDataURL('image/png')}})
    }});
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    item._rgbaB64 = 'data:image/png;base64,' + data.image_b64;
    const img = new Image();
    img.onload = function() {{
      const ctx = cropCanvas.getContext('2d');
      ctx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
      drawCheckerboard(ctx, cropCanvas.width, cropCanvas.height);
      ctx.drawImage(img, 0, 0, cropCanvas.width, cropCanvas.height);
      statusEl.textContent = 'BG removed \u2713';
      statusEl.style.color = '#3fb950';
    }};
    img.src = item._rgbaB64;
  }} catch(e) {{
    statusEl.textContent = e.message;
    statusEl.style.color = '#f85149';
  }}
}}

// ── Workflow editor ──────────────────────────────────────────────────────────
let _customWorkflow = null;

function toggleWorkflow() {{
  const body = document.getElementById('workflowBody');
  const arrow = document.getElementById('wfArrow');
  const open = body.classList.toggle('open');
  arrow.textContent = open ? '\u2039' : '\u203a';
}}

function applyWorkflow() {{
  const raw = document.getElementById('workflowJson').value.trim();
  const statusEl = document.getElementById('wfStatus');
  if (!raw) {{ _customWorkflow = null; statusEl.textContent = 'Cleared.'; document.getElementById('wfNodeHint').textContent = ''; return; }}
  try {{
    _customWorkflow = JSON.parse(raw);
    const nodeCount = Object.keys(_customWorkflow).length;
    statusEl.textContent = '\u2713 Loaded (' + nodeCount + ' nodes)';
    statusEl.style.color = '#3fb950';
    document.getElementById('wfNodeHint').textContent = nodeCount + ' nodes';
  }} catch(e) {{
    statusEl.textContent = 'Invalid JSON: ' + e.message;
    statusEl.style.color = '#f85149';
    _customWorkflow = null;
  }}
}}

function clearWorkflow() {{
  _customWorkflow = null;
  document.getElementById('workflowJson').value = '';
  document.getElementById('wfStatus').textContent = 'Using server default.';
  document.getElementById('wfStatus').style.color = '#8b949e';
  document.getElementById('wfNodeHint').textContent = '';
}}

function loadWorkflowFromFile() {{ document.getElementById('wfFileInput').click(); }}

function readWorkflowFile(e) {{
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(ev) {{
    document.getElementById('workflowJson').value = ev.target.result;
    applyWorkflow();
  }};
  reader.readAsText(file);
  e.target.value = '';
}}
// ─────────────────────────────────────────────────────────────────────────────

function toggleSelectAll() {{
  const checkboxes = document.querySelectorAll('.crop-checkbox');
  const anyUnchecked = [...checkboxes].some(c => !c.checked);
  checkboxes.forEach(c => {{
    c.checked = anyUnchecked;
    c.closest('.crop-item').style.outline = anyUnchecked ? '2px solid #ffa657' : 'none';
  }});
  document.getElementById('selectAllBtn').textContent = anyUnchecked ? 'Deselect All' : 'Select All';
}}

function drawMaskPreview(groups) {{
  maskPreviewCtx.fillStyle = '#ffffff';
  maskPreviewCtx.fillRect(0, 0, maskPreviewCanvas.width, maskPreviewCanvas.height);
  maskPreviewCtx.fillStyle = '#000000';
  for (const g of groups) {{
    for (const c of g.members) {{
      if (c.points.length < 3) continue;
      maskPreviewCtx.beginPath();
      maskPreviewCtx.moveTo(c.points[0][0] * scaleX, c.points[0][1] * scaleY);
      for (let i = 1; i < c.points.length; i++) {{
        maskPreviewCtx.lineTo(c.points[i][0] * scaleX, c.points[i][1] * scaleY);
      }}
      maskPreviewCtx.closePath();
      maskPreviewCtx.fill();
    }}
  }}
}}

function generateMaskCanvas() {{
  // Creates a B&W mask: white background, black contour shapes at original image resolution.
  const mc = document.createElement('canvas');
  mc.width = DATA.width;
  mc.height = DATA.height;
  const mctx = mc.getContext('2d');
  mctx.fillStyle = '#ffffff';
  mctx.fillRect(0, 0, mc.width, mc.height);
  mctx.fillStyle = '#000000';
  for (const g of _currentGroups) {{
    for (const c of g.members) {{
      if (c.points.length < 3) continue;
      mctx.beginPath();
      mctx.moveTo(c.points[0][0], c.points[0][1]);
      for (let i = 1; i < c.points.length; i++) {{
        mctx.lineTo(c.points[i][0], c.points[i][1]);
      }}
      mctx.closePath();
      mctx.fill();
    }}
  }}
  return mc.toDataURL('image/png');
}}

async function detectUIWithAI() {{
  const geminiKey = document.getElementById('geminiKey').value.trim();
  const statusEl = document.getElementById('aiStatus');
  if (!geminiKey) {{ statusEl.textContent = 'Enter Gemini key first'; statusEl.style.color = '#f85149'; return; }}
  statusEl.textContent = 'Asking Gemini\u2026'; statusEl.style.color = '#f0883e';

  try {{
    // Send the contour canvas (shows detected zones) instead of raw original
    const contourB64 = canvas.toDataURL('image/png');
    // Pass excluded zone in image coordinates
    const exZone = {{
      x: Math.round(roi.x / scaleX), y: Math.round(roi.y / scaleY),
      w: Math.round(roi.w / scaleX), h: Math.round(roi.h / scaleY)
    }};
    const resp = await fetch('/detect_ui', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        gemini_key: geminiKey,
        image_b64: contourB64,
        width: DATA.width,
        height: DATA.height,
        excluded_zone: exZone
      }})
    }});
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    const boxes = data.boxes || [];
    statusEl.textContent = `Gemini found ${{boxes.length}} UI elements`;
    statusEl.style.color = '#3fb950';

    // Draw boxes on original image canvas for accurate visual verification
    origCtx.drawImage(origImg, 0, 0, origCanvas.width, origCanvas.height);
    const roiXv = roi.x / scaleX, roiYv = roi.y / scaleY;
    const roiWv = roi.w / scaleX, roiHv = roi.h / scaleY;
    origCtx.save();
    origCtx.lineWidth = 2;
    origCtx.font = 'bold 10px sans-serif';
    for (const b of boxes) {{
      const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
      const excluded = cx >= roiXv && cx <= roiXv + roiWv && cy >= roiYv && cy <= roiYv + roiHv;
      origCtx.strokeStyle = excluded ? 'rgba(150,150,150,0.6)' : '#00e5ff';
      origCtx.fillStyle   = excluded ? 'rgba(150,150,150,0.6)' : '#00e5ff';
      origCtx.strokeRect(b.x * scaleX, b.y * scaleY, b.w * scaleX, b.h * scaleY);
      origCtx.fillText(b.label || '', b.x * scaleX + 2, b.y * scaleY + 11);
    }}
    origCtx.restore();

    // Also store for contour canvas overlay
    _aiBoxes = boxes;
    draw();

    // Generate crops directly from Gemini boxes (bypass contour system)
    const roiX = roi.x / scaleX, roiY = roi.y / scaleY;
    const roiW = roi.w / scaleX, roiH = roi.h / scaleY;
    const grid = document.getElementById('cropsGrid');
    grid.innerHTML = '';
    let count = 0;
    boxes.forEach((b, i) => {{
      // Skip boxes whose center falls inside the excluded zone
      const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
      if (cx >= roiX && cx <= roiX + roiW && cy >= roiY && cy <= roiY + roiH) return;
      const cropX = Math.max(0, b.x);
      const cropY = Math.max(0, b.y);
      const cropW = Math.min(DATA.width - cropX, b.w);
      const cropH = Math.min(DATA.height - cropY, b.h);
      if (cropW < 5 || cropH < 5) return;

      const cropData = fullCtx.getImageData(cropX, cropY, cropW, cropH);
      const maxSize = 120;
      const displayScale = Math.min(maxSize / cropW, maxSize / cropH, 1);
      const dw = Math.round(cropW * displayScale), dh = Math.round(cropH * displayScale);

      const cropCanvas = document.createElement('canvas');
      cropCanvas.width = dw; cropCanvas.height = dh;
      const tmp = document.createElement('canvas');
      tmp.width = cropW; tmp.height = cropH;
      tmp.getContext('2d').putImageData(cropData, 0, 0);
      cropCanvas.getContext('2d').drawImage(tmp, 0, 0, dw, dh);

      const item = document.createElement('div');
      item.className = 'crop-item';
      item.style.borderColor = '#388bfd';
      item.appendChild(cropCanvas);

      const lbl = document.createElement('div');
      lbl.className = 'crop-label';
      lbl.textContent = (b.label || 'ui') + ' (' + cropW + '×' + cropH + ')';
      item.appendChild(lbl);

      item._group = {{ x: cropX, y: cropY, w: cropW, h: cropH, id: 'ai_' + i, members: [], count: 1 }};

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.className = 'crop-checkbox';
      checkbox.checked = true;
      checkbox.addEventListener('change', () => {{
        item.style.outline = checkbox.checked ? '2px solid #388bfd' : 'none';
      }});
      item.style.outline = '2px solid #388bfd';
      item.appendChild(checkbox);

      const bgBtn = document.createElement('button');
      bgBtn.className = 'bg-btn';
      bgBtn.textContent = 'Remove BG';
      const comfyStatus = document.createElement('div');
      comfyStatus.className = 'comfy-status';
      const comfyResult = document.createElement('div');
      comfyResult.className = 'comfy-result';
      bgBtn.addEventListener('click', (function(it, c, s) {{
        return function() {{ removeCropBg(it, c, s); }};
      }})(item, cropCanvas, comfyStatus));
      item.appendChild(bgBtn);
      item.appendChild(comfyStatus);
      item.appendChild(comfyResult);
      grid.appendChild(item);
      count++;
    }});

    document.getElementById('cropCount').textContent = count;
    statusEl.textContent = `Gemini found ${{boxes.length}} UI elements — ${{count}} crops`;

  }} catch(e) {{
    statusEl.textContent = 'Error: ' + e.message;
    statusEl.style.color = '#f85149';
  }}
}}

function downloadViewDescriptionJSON() {{
  // Collect checked crop items from the grid
  const checkedItems = [...document.querySelectorAll('.crop-item')].filter(item => {{
    const cb = item.querySelector('.crop-checkbox');
    return cb && cb.checked;
  }});

  if (checkedItems.length === 0) {{
    alert('No crops selected. Check at least one crop first.');
    return;
  }}

  const imgW = DATA.width, imgH = DATA.height;

  function deriveAnchor(cx, cy) {{
    // cx/cy are percentage coords with bottom-left origin (y=0 bottom, y=100 top)
    const vert  = cy > 66 ? 'top'    : cy < 33 ? 'bottom' : 'middle';
    const horiz = cx < 33 ? 'left'   : cx > 66 ? 'right'  : 'center';
    if (vert === 'middle' && horiz === 'center') return 'center';
    if (vert === 'middle') return 'middle-' + horiz;
    return vert + '-' + horiz;
  }}

  function deriveType(label) {{
    const l = (label || '').toLowerCase();
    if (l.includes('button') || l.includes('btn') || l.includes('cta')) return 'button';
    if (l.includes('text') || l.includes('label') || l.includes('title') || l.includes('score')) return 'text';
    if (l.includes('panel') || l.includes('card') || l.includes('container') || l.includes('modal')) return 'container';
    return 'image';
  }}

  const elements = [];
  let sortOrder = 1;
  const usedIds = {{}};

  checkedItems.forEach(item => {{
    const g = item._group;
    if (!g) return;

    // Get label from the crop-label element
    const lblEl = item.querySelector('.crop-label');
    const rawLabel = lblEl ? lblEl.textContent.replace(/\s*\(.*\)$/, '').trim() : '';

    const cx = g.x + g.w / 2, cy = g.y + g.h / 2;

    // Percentage coordinates — bottom-left origin
    const xPct = Math.round(cx / imgW * 10000) / 100;
    const yPct = Math.round((1 - cy / imgH) * 10000) / 100;
    const wPct = Math.round(g.w / imgW * 10000) / 100;
    const hPct = Math.round(g.h / imgH * 10000) / 100;

    const anchor = deriveAnchor(xPct, yPct);
    const type   = deriveType(rawLabel);

    // Sanitise label into a valid id; deduplicate
    let baseId = rawLabel.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || 'element';
    usedIds[baseId] = (usedIds[baseId] || 0) + 1;
    const id = usedIds[baseId] > 1 ? baseId + '_' + usedIds[baseId] : baseId;

    const el = {{ id, type, anchor, x: xPct, y: yPct, width: wPct, height: hPct, color: '#FFFFFFFF', sortOrder: sortOrder++ }};
    if (type === 'button' || type === 'text') el.text = rawLabel;
    elements.push(el);
  }});

  const viewDesc = {{
    orientation: imgH > imgW ? 'portrait' : 'landscape',
    referenceResolution: {{ width: imgW, height: imgH }},
    elements
  }};

  const json = JSON.stringify(viewDesc, null, 2);
  const blob = new Blob([json], {{ type: 'application/json' }});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = 'view_description.json';
  a.click();
  URL.revokeObjectURL(url);
}}

async function sendBatchToComfyUI() {{
  const comfyUrl = (document.getElementById('comfyUrl').value || 'http://localhost:8188').replace(/[/]+$/, '');
  const apiKey = document.getElementById('comfyApiKey').value;
  const statusEl = document.getElementById('batchStatus');

  const items = [...document.querySelectorAll('.crop-item')].filter(item => {{
    const cb = item.querySelector('.crop-checkbox');
    return cb && cb.checked;
  }});

  if (items.length === 0) {{
    statusEl.textContent = 'No crops selected.';
    statusEl.style.color = '#f85149';
    return;
  }}

  statusEl.textContent = 'Uploading original\u2026';
  statusEl.style.color = '#f0883e';

  const images = items.map(item => {{
    const c = item.querySelector('canvas');
    return item._rgbaB64 || c.toDataURL('image/png');
  }});

  try {{
    // Step 1: upload original screenshot separately (avoids huge JSON payload)
    const origResp = await fetch('/upload_original', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{base64: IMG_B64, comfyui_url: comfyUrl, api_key: apiKey}})
    }});
    const origData = await origResp.json();
    if (origData.error) throw new Error('Original upload: ' + origData.error);
    const originalFilename = origData.filename;

    // Step 2: upload B&W mask from contour solid fill
    statusEl.textContent = 'Uploading mask\u2026';
    const maskB64 = generateMaskCanvas();
    const maskResp = await fetch('/upload_original', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{base64: maskB64, comfyui_url: comfyUrl, api_key: apiKey, filename_hint: 'mask.png'}})
    }});
    const maskData = await maskResp.json();
    if (maskData.error) throw new Error('Mask upload: ' + maskData.error);
    const maskFilename = maskData.filename;

    statusEl.textContent = 'Uploading ' + items.length + ' crops\u2026';

    // Step 3: send batch with original_filename and mask_filename references
    const resp = await fetch('/comfyui_batch', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{images, original_filename: originalFilename, mask_filename: maskFilename, comfyui_url: comfyUrl, api_key: apiKey, workflow: _customWorkflow}})
    }});
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    const groups = items.map(it => it._group).filter(Boolean);
    const batchImgs = data.images || [];
    const nodeIds = data.node_ids || [];
    applyComfyResultsToCanvas(batchImgs, nodeIds, groups);
    showComfyResults(batchImgs, nodeIds);
    statusEl.textContent = 'Batch done \u2713 (' + batchImgs.length + ' results)';
    statusEl.style.color = '#3fb950';
  }} catch(e) {{
    statusEl.textContent = 'Batch error: ' + e.message;
    statusEl.style.color = '#f85149';
  }}
}}

async function removeAllBg() {{
  const bgBtns = document.querySelectorAll('.bg-btn');
  for (const btn of bgBtns) {{
    btn.click();
    await new Promise(r => setTimeout(r, 800));
  }}
}}

function compositeResultsOnCanvas(images, groups) {{
  if (!images || images.length === 0) return;
  images.forEach(function(b64, i) {{
    if (!groups[i]) return;
    const g = groups[i];
    const img = new Image();
    img.onload = function() {{
      modCtx.drawImage(img,
        Math.round(g.x * scaleX),
        Math.round(g.y * scaleY),
        Math.round(g.w * scaleX),
        Math.round(g.h * scaleY)
      );
      _modCount += 1;
      document.getElementById('modCount').textContent = '(' + _modCount + ' results)';
    }};
    img.src = 'data:image/png;base64,' + b64;
  }});
}}

function applyComfyResultsToCanvas(images, nodeIds, groups) {{
  // Separate background from icon images by node ID
  const bgB64 = images[nodeIds.indexOf('1806')] || null;
  const iconImgs = images.filter((_, i) => nodeIds[i] === '1767');
  // Icon crops align to the groups in order
  const iconGroups = groups.slice(0, iconImgs.length);

  if (bgB64) {{
    const bgImg = new Image();
    bgImg.onload = function() {{
      // Fill modCanvas with clean background at original ratio
      modCtx.drawImage(bgImg, 0, 0, modCanvas.width, modCanvas.height);
      compositeResultsOnCanvas(iconImgs, iconGroups);
    }};
    bgImg.src = 'data:image/png;base64,' + bgB64;
  }} else {{
    compositeResultsOnCanvas(iconImgs.length ? iconImgs : images, iconImgs.length ? iconGroups : groups);
  }}
}}

async function sendCropToComfyUI(b64DataUrl, statusEl, resultEl) {{
  const comfyUrl = (document.getElementById('comfyUrl').value || 'http://localhost:8188').replace(/[/]+$/, '');
  statusEl.textContent = 'Sending\u2026';
  statusEl.style.color = '#f0883e';
  resultEl.innerHTML = '';
  try {{
    const resp = await fetch('/comfyui_run', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        base64: b64DataUrl,
        comfyui_url: comfyUrl,
        api_key: document.getElementById('comfyApiKey').value,
        workflow: _customWorkflow
      }})
    }});
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    const imgs = data.images || (data.image_b64 ? [data.image_b64] : []);
    if (imgs.length > 0) {{
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + imgs[0];
      resultEl.appendChild(img);
      statusEl.textContent = 'Done \u2713';
      statusEl.style.color = '#3fb950';
      const item = statusEl.closest('.crop-item');
      if (item && item._group) {{
        compositeResultsOnCanvas(imgs, [item._group]);
      }}
      showComfyResults(imgs, data.node_ids || []);
    }} else {{
      statusEl.textContent = 'No output';
      statusEl.style.color = '#8b949e';
    }}
  }} catch(e) {{
    statusEl.textContent = e.message;
    statusEl.style.color = '#f85149';
  }}
}}

async function sendAllToComfyUI() {{
  const btns = document.querySelectorAll('.comfy-btn');
  for (const btn of btns) {{
    btn.click();
    await new Promise(r => setTimeout(r, 300));
  }}
}}

async function pollHistory(comfyuiUrl, promptId, statusEl) {{
  const apiKey = document.getElementById('comfyApiKey').value;
  const historyUrl = comfyuiUrl.replace(/\/+$/, '') + '/api/history/' + promptId;
  const deadline = Date.now() + 10 * 60 * 1000; // 10 min
  let poll = 0;
  while (Date.now() < deadline) {{
    await new Promise(r => setTimeout(r, 3000));
    poll++;
    try {{
      const resp = await fetch(historyUrl, {{
        headers: apiKey ? {{'X-API-Key': apiKey, 'Authorization': 'Bearer ' + apiKey}} : {{}}
      }});
      if (!resp.ok) {{
        if (statusEl && poll % 5 === 0) statusEl.textContent = 'Waiting\u2026 (poll #' + poll + ', ' + resp.status + ')';
        continue;
      }}
      const hist = await resp.json();
      if (!hist[promptId]) continue;
      const entry = hist[promptId];
      const status = entry.status || {{}};
      if (!status.completed && status.status_str !== 'success') continue;
      // Collect all output images
      const images = [];
      for (const nodeOut of Object.values(entry.outputs || {{}})) {{
        for (const img of (nodeOut.images || [])) {{
          if (img.type === 'temp') continue;
          let viewUrl = comfyuiUrl.replace(/\/+$/, '') + '/api/view?filename=' + encodeURIComponent(img.filename) + '&type=' + (img.type || 'output');
          if (img.subfolder) viewUrl += '&subfolder=' + encodeURIComponent(img.subfolder);
          // Fetch via our proxy to avoid CORS
          const proxyResp = await fetch('/proxy_image', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{url: viewUrl, api_key: apiKey}})
          }});
          if (!proxyResp.ok) continue;
          const pd = await proxyResp.json();
          if (pd.image_b64) images.push(pd.image_b64);
        }}
      }}
      return images;
    }} catch(e) {{
      if (statusEl && poll % 5 === 0) statusEl.textContent = 'Waiting\u2026 (poll #' + poll + ')';
    }}
  }}
  throw new Error('Timed out waiting for ComfyUI result');
}}

const NODE_LABELS = {{'1767': 'Icons', '1806': 'Background', '1823': 'Environment'}};

function showComfyResults(images, nodeIds) {{
  if (!images || images.length === 0) return;
  nodeIds = nodeIds || [];
  const section = document.getElementById('resultsSection');
  const grid = document.getElementById('resultsGrid');
  const countEl = document.getElementById('resultsCount');
  grid.innerHTML = '';
  const iconCount = {{}};
  images.forEach(function(b64, i) {{
    const nodeId = nodeIds[i] || '';
    let label = NODE_LABELS[nodeId];
    if (!label) label = 'Output ' + (i + 1);
    // Number duplicate labels (e.g. multiple icons)
    if (nodeId === '1767') {{
      iconCount[nodeId] = (iconCount[nodeId] || 0) + 1;
      label = 'Icon ' + iconCount[nodeId];
    }}

    const card = document.createElement('div');
    card.className = 'result-card';

    const img = document.createElement('img');
    img.src = 'data:image/png;base64,' + b64;

    const footer = document.createElement('div');
    footer.className = 'rc-footer';

    const labelEl = document.createElement('span');
    labelEl.textContent = label;

    const dl = document.createElement('a');
    dl.textContent = 'Download';
    dl.href = img.src;
    dl.download = label.replace(/\s+/g, '_').toLowerCase() + '.png';

    footer.appendChild(labelEl);
    footer.appendChild(dl);
    card.appendChild(img);
    card.appendChild(footer);
    grid.appendChild(card);
  }});
  countEl.textContent = '(' + images.length + ')';
  section.style.display = 'block';
  section.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
}}

['minArea', 'maxArea'].forEach(id => {{
  document.getElementById(id).addEventListener('input', draw);
}});
document.getElementById('onlyParents').addEventListener('change', draw);
document.getElementById('mergeNearby').addEventListener('change', draw);
document.getElementById('proximity').addEventListener('input', e => {{
  document.getElementById('proxVal').textContent = e.target.value + 'px';
  draw();
}});
document.getElementById('sizeRatio').addEventListener('input', e => {{
  document.getElementById('sizeRatioVal').textContent = e.target.value + '×';
  draw();
}});

canvas.addEventListener('mousedown', e => {{
  if (e.button !== 0) return;
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const hit = hitROI(cx, cy);
  if (hit) {{
    e.preventDefault();
    roiDrag = {{ mode: hit, startCX: cx, startCY: cy, startRoi: {{...roi}} }};
  }}
}});

window.addEventListener('mouseup', () => {{ roiDrag = null; }});

canvas.addEventListener('mousemove', e => {{
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;

  // ── ROI drag ──
  if (roiDrag) {{
    const dx = cx - roiDrag.startCX;
    const dy = cy - roiDrag.startCY;
    const sr = roiDrag.startRoi;
    const mode = roiDrag.mode;
    roi.x = sr.x; roi.y = sr.y; roi.w = sr.w; roi.h = sr.h;
    if (mode === 'move') {{
      roi.x = sr.x + dx; roi.y = sr.y + dy;
    }} else {{
      if (mode.includes('e')) {{ roi.w = sr.w + dx; }}
      if (mode.includes('w')) {{ roi.x = sr.x + dx; roi.w = sr.w - dx; }}
      if (mode.includes('s')) {{ roi.h = sr.h + dy; }}
      if (mode.includes('n')) {{ roi.y = sr.y + dy; roi.h = sr.h - dy; }}
    }}
    clampRoi();
    draw();
    return;
  }}

  // ── Cursor ──
  const hit = hitROI(cx, cy);
  canvas.style.cursor = hit ? (hit === 'move' ? 'move' : (HANDLE_CURSORS[hit] || 'crosshair')) : 'crosshair';

  // ── Tooltip (image coords) ──
  const mx = cx / scaleX;
  const my = cy / scaleY;
  const filtered = window._filtered || [];
  let best = null;
  for (const c of filtered) {{
    if (mx >= c.x && mx <= c.x + c.w && my >= c.y && my <= c.y + c.h) {{
      if (!best || c.area < best.area) best = c;
    }}
  }}
  const info = document.getElementById('info');
  if (best) {{
    info.style.display = 'block';
    info.style.borderColor = best._color || '#30363d';
    document.getElementById('iText').innerHTML =
      '<div style="width:100%;height:8px;background:' + best._color + ';border-radius:3px;margin-bottom:6px"></div>' +
      '<b>Contour #' + best.id + '</b><br>' +
      'Area: ' + best.area.toLocaleString() + '<br>' +
      'Size: ' + best.w + '×' + best.h + '<br>' +
      'Circularity: ' + best.circularity + '<br>' +
      'Aspect: ' + best.aspect + '<br>' +
      'Extent: ' + best.extent + '<br>' +
      'Vertices: ' + best.vertices + '<br>' +
      'Children: ' + best.children + '<br>' +
      'Depth: ' + best.depth + '<br>' +
      'Parent: ' + best.parent;
  }} else {{
    info.style.display = 'none';
  }}
}});

draw();
</script>
</body>
</html>'''


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 generate_explorer.py <image_path> [output.html]")
        sys.exit(1)

    image_path = sys.argv[1]
    image_name = os.path.basename(image_path)

    output_path = sys.argv[2] if len(sys.argv) > 2 else image_path.rsplit('.', 1)[0] + '_explorer.html'

    data = run_pipeline(image_path)
    html = generate_html(data, image_name)

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Saved: {output_path} ({len(html)//1024}KB)")
