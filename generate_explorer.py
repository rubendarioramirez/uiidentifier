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

    return {'width': w, 'height': h, 'contours': contour_data}


def generate_html(data, image_name="uploaded image"):
    """Generate the v2 explorer HTML with inline data."""
    data_json = json.dumps(data)

    # Canvas size: scale to max 500px wide
    scale = min(500 / data['width'], 700 / data['height'])
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
  .canvas-wrap {{ max-width: 1200px; margin: 0 auto; display: flex; gap: 12px; justify-content: center; }}
  .canvas-panel {{ text-align: center; }}
  .canvas-panel h2 {{ font-size: 0.85em; color: #8b949e; margin-bottom: 6px; }}
  canvas {{ border: 1px solid #30363d; border-radius: 4px; cursor: crosshair; }}
  #info {{
    position: fixed; bottom: 16px; right: 16px; background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 12px; font-size: 0.8em; min-width: 220px; display: none; z-index: 10;
  }}
  #info div {{ margin-bottom: 2px; }}
  .upload-btn {{
    display: block; margin: 12px auto 0; padding: 8px 20px; background: #238636; color: #fff;
    border: none; border-radius: 6px; font-size: 0.9em; cursor: pointer;
  }}
  .upload-btn:hover {{ background: #2ea043; }}
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
  <div class="stats" id="stats"></div>
</div>

<div class="canvas-wrap">
  <div class="canvas-panel">
    <h2>Solid-filled contours on black background</h2>
    <canvas id="canvas" width="{canvas_w}" height="{canvas_h}"></canvas>
  </div>
</div>

<div id="info">
  <div id="iText"></div>
</div>

<script>
const DATA = {data_json};

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const scaleX = {canvas_w} / DATA.width;
const scaleY = {canvas_h} / DATA.height;

function hslStr(id) {{
  const hue = (id * 137.508) % 360;
  return {{ h: hue, css: 'hsl(' + hue + ', 75%, 55%)' }};
}}

function draw() {{
  const minA = +document.getElementById('minArea').value;
  const maxA = +document.getElementById('maxArea').value;
  const onlyParents = document.getElementById('onlyParents').checked;

  document.getElementById('minAreaVal').textContent = minA.toLocaleString();
  document.getElementById('maxAreaVal').textContent = maxA.toLocaleString();

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

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

  let colorIdx = 0;
  for (const c of filtered) {{
    if (c.points.length < 3) continue;
    const color = hslStr(colorIdx);
    colorIdx++;
    ctx.beginPath();
    ctx.moveTo(c.points[0][0] * scaleX, c.points[0][1] * scaleY);
    for (let i = 1; i < c.points.length; i++) {{
      ctx.lineTo(c.points[i][0] * scaleX, c.points[i][1] * scaleY);
    }}
    ctx.closePath();
    ctx.fillStyle = color.css;
    ctx.fill();
    c._color = color.css;
  }}

  window._filtered = filtered;
  document.getElementById('stats').textContent =
    'Showing ' + filtered.length + ' / ' + DATA.contours.length + ' contours (area ' +
    minA.toLocaleString() + ' – ' + maxA.toLocaleString() + ')' +
    (onlyParents ? ' — outermost only' : '');
}}

['minArea', 'maxArea'].forEach(id => {{
  document.getElementById(id).addEventListener('input', draw);
}});
document.getElementById('onlyParents').addEventListener('change', draw);

canvas.addEventListener('mousemove', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) / scaleX;
  const my = (e.clientY - rect.top) / scaleY;
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
