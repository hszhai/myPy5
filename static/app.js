import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------------------------------------------------------------- state ----
let SCENE = null;
let CAMERA = null;
let SCHEMA = null;
let COMPOSITION = null;
const $ = (id) => document.getElementById(id);
const _MASK3D_KEYS = new Set([
  'mask3d_enabled', 'mask3d_x', 'mask3d_y', 'mask3d_z',
  'mask3d_r_in', 'mask3d_r_out', 'mask3d_invert',
]);

// -------------------------------------------------------------- api wrap ---
async function api(path, opts = {}) {
  const res = await fetch(path, {
    method: opts.method || 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = j.error || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

// -------------------------------------------------------------- bootstrap -
async function init() {
  const data = await api('/api/init');
  SCENE = data.scene;
  CAMERA = data.camera;
  SCHEMA = data.schema;
  COMPOSITION = data.composition;

  $('scene-name').textContent = SCENE.json_path;
  $('bbox-info').textContent = formatBbox(SCENE.bbox_min, SCENE.bbox_max);
  if (data.log && data.log.length) appendLog(data.log.join('\n'));

  setupCameraInputs();
  setupWireframe();
  renderLayers();
  bindActions();
}

function formatBbox(min, max) {
  const f = (v) => v.toFixed(2);
  return `(${min.map(f).join(',')}) → (${max.map(f).join(',')})`;
}

// -------------------------------------------------------------- camera ----
function setupCameraInputs() {
  syncInputsFromCamera();
  ['elev', 'azim', 'fov', 'dist', 'bias-x', 'bias-y'].forEach((k) => {
    $('cam-' + k).addEventListener('change', commitCameraFromInputs);
  });
}

function syncInputsFromCamera() {
  // Store full precision values in the input fields so they're preserved
  // when read back via commitCameraFromInputs. Display is formatted by CSS.
  $('cam-elev').value = CAMERA.elev_deg;
  $('cam-azim').value = CAMERA.azim_deg;
  $('cam-fov').value  = CAMERA.fov_deg;
  $('cam-dist').value = CAMERA.distance_k;
  $('cam-bias-x').value = CAMERA.head_bias_x ?? 0;
  $('cam-bias-y').value = CAMERA.head_bias_y ?? 0;
}

async function commitCameraFromInputs({ silent = false } = {}) {
  const localCamera = {
    elev_deg: +$('cam-elev').value,
    azim_deg: +$('cam-azim').value,
    fov_deg:  +$('cam-fov').value,
    distance_k: +$('cam-dist').value,
    head_bias_x: +$('cam-bias-x').value,
    head_bias_y: +$('cam-bias-y').value,
  };
  try {
    const res = await api('/api/camera', { method: 'POST', body: localCamera });
    // Use server's response to ensure state is synchronized
    CAMERA = res.camera;
    syncInputsFromCamera();
    positionThreeCameraFromState();
    if (!silent && res.changed) {
      appendLog(`camera changed → committed`);
    }
  } catch (err) {
    appendLog('camera commit failed: ' + err.message);
  }
}

// -------------------------------------------------------------- wireframe -
const three = {
  renderer: null, scene: null, camera: null, controls: null,
  bbCenter: null, bbDiag: 1, maskGroup: null,
  frameGroup: null, markerGroup: null, zlineGroup: null,
};

const MARKERS = [];

function setupWireframe() {
  const canvas = $('bbox-canvas');
  const wrap = $('bbox-wrap');
  const { clientWidth: w, clientHeight: h } = wrap;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(w, h, false);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x09090b, 1);

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(CAMERA.fov_deg, w / h, 0.001, 100);

  const bbMin = new THREE.Vector3(...SCENE.bbox_min);
  const bbMax = new THREE.Vector3(...SCENE.bbox_max);
  const bbCenter = bbMin.clone().add(bbMax).multiplyScalar(0.5);
  const bbDiag = bbMax.distanceTo(bbMin);

  // Full bbox (dim grey)
  addBoxWire(scene, bbMin, bbMax, 0x52525b);
  // Density bbox (warm amber)
  const dMin = new THREE.Vector3(...SCENE.density_min);
  const dMax = new THREE.Vector3(...SCENE.density_max);
  addBoxWire(scene, dMin, dMax, 0xd97706);

  // Axes triad at scene centre
  const axes = new THREE.AxesHelper(bbDiag * 0.15);
  axes.position.copy(bbCenter);
  scene.add(axes);

  // Mask spheres go into a Group so we can rebuild on composition change
  const maskGroup = new THREE.Group();
  scene.add(maskGroup);

  // Composition frame indicator
  const frameGroup = new THREE.Group();
  scene.add(frameGroup);

  // Custom markers
  const markerGroup = new THREE.Group();
  scene.add(markerGroup);

  // Z-line endpoint indicators
  const zlineGroup = new THREE.Group();
  scene.add(zlineGroup);

  // Orbit controls — target the bbox centre
  const controls = new OrbitControls(cam, canvas);
  controls.target.copy(bbCenter);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.addEventListener('end', () => {
    syncCameraFromOrbit();
    commitCameraFromInputs();
  });

  three.renderer = renderer;
  three.scene = scene;
  three.camera = cam;
  three.controls = controls;
  three.bbCenter = bbCenter;
  three.bbDiag = bbDiag;
  three.maskGroup = maskGroup;
  three.frameGroup = frameGroup;
  three.markerGroup = markerGroup;
  three.zlineGroup = zlineGroup;

  positionThreeCameraFromState();
  rebuildMaskSpheres();
  rebuildZlineEndpoints();
  rebuildFrame();
  setupMarkerUI();

  function loop() {
    controls.update();
    renderer.render(scene, cam);
    requestAnimationFrame(loop);
  }
  loop();

  const ro = new ResizeObserver(() => {
    const w2 = wrap.clientWidth, h2 = wrap.clientHeight;
    if (w2 === 0 || h2 === 0) return;
    renderer.setSize(w2, h2, false);
    cam.aspect = w2 / h2;
    cam.updateProjectionMatrix();
  });
  ro.observe(wrap);
}

function addBoxWire(scene, min, max, color) {
  const size = max.clone().sub(min);
  const center = min.clone().add(max).multiplyScalar(0.5);
  const geo = new THREE.BoxGeometry(size.x, size.y, size.z);
  const edges = new THREE.EdgesGeometry(geo);
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.75 });
  const lines = new THREE.LineSegments(edges, mat);
  lines.position.copy(center);
  scene.add(lines);
}

// CAMERA.azim_deg stores values in gsplat's convention (Ry rotates world
// CW around +Y for positive azim). Three.js's natural spherical math goes
// the opposite way, so we negate azim at the JS↔state boundary -- this is
// the ONLY place the convention difference lives.
function positionThreeCameraFromState() {
  if (!three.camera) return;
  const elev = CAMERA.elev_deg * Math.PI / 180;
  const azim = -CAMERA.azim_deg * Math.PI / 180;          // gsplat → three.js
  const r = three.bbDiag * CAMERA.distance_k;
  const target = three.bbCenter.clone();
  target.x += (CAMERA.head_bias_x || 0);
  target.y += (CAMERA.head_bias_y || 0);
  three.camera.position.set(
    target.x + r * Math.cos(elev) * Math.sin(azim),
    target.y + r * Math.sin(elev),
    target.z + r * Math.cos(elev) * Math.cos(azim),
  );
  three.camera.lookAt(target);
  three.controls.target.copy(target);
  three.camera.fov = CAMERA.fov_deg;
  three.camera.updateProjectionMatrix();
  three.controls.update();
  rebuildFrame();
}

function syncCameraFromOrbit() {
  const biasedCenter = three.bbCenter.clone();
  biasedCenter.x += (CAMERA.head_bias_x || 0);
  biasedCenter.y += (CAMERA.head_bias_y || 0);
  const dir = three.camera.position.clone().sub(biasedCenter);
  const r = dir.length();
  const elev = Math.atan2(dir.y, Math.sqrt(dir.x * dir.x + dir.z * dir.z));
  const azim = -Math.atan2(dir.x, dir.z);                 // three.js → gsplat
  CAMERA = {
    elev_deg: elev * 180 / Math.PI,
    azim_deg: azim * 180 / Math.PI,
    fov_deg: three.camera.fov,
    distance_k: r / three.bbDiag,
    head_bias_x: CAMERA.head_bias_x,
    head_bias_y: CAMERA.head_bias_y,
  };
  syncInputsFromCamera();
  rebuildFrame();
}

function rebuildMaskSpheres() {
  if (!three.maskGroup) return;
  three.maskGroup.clear();
  for (const layer of (COMPOSITION.layers || [])) {
    if (!layer.mask3d_enabled) continue;
    const c = new THREE.Vector3(layer.mask3d_x, layer.mask3d_y, layer.mask3d_z);
    const rIn = +layer.mask3d_r_in;
    const rOut = +layer.mask3d_r_out;
    const invert = !!layer.mask3d_invert;
    const colorIn  = invert ? 0xef4444 : 0x06b6d4;
    const colorOut = invert ? 0x7f1d1d : 0x155e75;

    for (const [r, color, opacity] of [
      [rIn,  colorIn,  0.85],
      [rOut, colorOut, 0.4 ],
    ]) {
      const geo = new THREE.SphereGeometry(r, 24, 16);
      const edges = new THREE.EdgesGeometry(geo);
      const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
      const lines = new THREE.LineSegments(edges, mat);
      lines.position.copy(c);
      three.maskGroup.add(lines);
    }

    // Centre marker
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(three.bbDiag * 0.008, 8, 6),
      new THREE.MeshBasicMaterial({ color: colorIn }),
    );
    dot.position.copy(c);
    three.maskGroup.add(dot);
  }
}

function rebuildZlineEndpoints() {
  if (!three.zlineGroup) return;
  three.zlineGroup.clear();
  for (const layer of (COMPOSITION.layers || [])) {
    if (layer.type !== "zline") continue;
    const params = layer.params || {};
    if (!params.show_endpoints) continue;
    const center = SCENE.center || [0, 0, 0];
    const p1 = new THREE.Vector3(
      center[0] + (params.p1_x ?? 0),
      center[1] + (params.p1_y ?? 0),
      center[2] + (params.p1_z ?? 0)
    );
    const p2 = new THREE.Vector3(
      center[0] + (params.p2_x ?? 0),
      center[1] + (params.p2_y ?? 0),
      center[2] + (params.p2_z ?? 0)
    );

    const r = three.bbDiag * 0.012;

    // P1 sphere (cyan)
    const s1 = new THREE.Mesh(
      new THREE.SphereGeometry(r, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0x06b6d4 }),
    );
    s1.position.copy(p1);
    three.zlineGroup.add(s1);

    // P2 sphere (orange)
    const s2 = new THREE.Mesh(
      new THREE.SphereGeometry(r, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0xf97316 }),
    );
    s2.position.copy(p2);
    three.zlineGroup.add(s2);

    // Connecting line
    const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
    const lineMat = new THREE.LineBasicMaterial({ color: 0xe4e4e7, transparent: true, opacity: 0.5 });
    three.zlineGroup.add(new THREE.Line(lineGeo, lineMat));

    // Drop lines for depth perception
    const groundY = three.bbCenter.y - three.bbDiag * 0.5;
    const dropMat = new THREE.LineBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.2 });
    three.zlineGroup.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([p1, new THREE.Vector3(p1.x, groundY, p1.z)]),
      dropMat,
    ));
    const dropMat2 = new THREE.LineBasicMaterial({ color: 0xf97316, transparent: true, opacity: 0.2 });
    three.zlineGroup.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([p2, new THREE.Vector3(p2.x, groundY, p2.z)]),
      dropMat2,
    ));
  }
}

// -------------------------------------------------------------- layers ---
function renderLayers() {
  const list = $('layers-list');
  list.innerHTML = '';

  // Background color card (always at the top, conceptually the canvas base)
  list.appendChild(buildBackgroundCard());

  (COMPOSITION.layers || []).forEach((layer, idx) => {
    list.appendChild(buildLayerCard(layer, idx));
  });
}

function buildBackgroundCard() {
  const card = document.createElement('div');
  card.className = 'bg-base border border-line rounded';

  const bg = COMPOSITION.background || [0.97, 0.96, 0.93];
  const r = (+bg[0]).toFixed(2);
  const g = (+bg[1]).toFixed(2);
  const b = (+bg[2]).toFixed(2);
  const cssColor = `rgb(${Math.round(bg[0]*255)}, ${Math.round(bg[1]*255)}, ${Math.round(bg[2]*255)})`;

  const header = document.createElement('div');
  header.className = 'px-3 py-2 flex items-center gap-2';
  header.innerHTML = `
    <span class="w-4 h-4 shrink-0 rounded-sm border border-line" style="background:${cssColor};"></span>
    <span class="text-[14px] text-ink flex-1 min-w-0">background</span>
    <span class="text-mute text-[12px] font-mono shrink-0">base</span>
  `;
  card.appendChild(header);

  const body = document.createElement('div');
  body.className = 'p-3 space-y-2';

  const row = document.createElement('div');
  row.className = 'flex items-center gap-1';
  const lab = document.createElement('label');
  lab.className = 'text-mute text-[12px] w-20 shrink-0';
  lab.textContent = 'color';
  row.appendChild(lab);

  ['r', 'g', 'b'].forEach((s, i) => {
    const inp = document.createElement('input');
    inp.type = 'number';
    inp.step = '0.01';
    inp.min = '0';
    inp.max = '1';
    inp.value = [r, g, b][i];
    inp.title = `background ${s}`;
    inp.className = 'flex-1 min-w-0 w-16 bg-base border border-line rounded px-2 py-1 text-right text-[13px] focus:outline-none focus:border-faint';
    inp.addEventListener('change', () => {
      const newBg = [
        +((COMPOSITION.background || [0.97, 0.96, 0.93])[0]),
        +((COMPOSITION.background || [0.97, 0.96, 0.93])[1]),
        +((COMPOSITION.background || [0.97, 0.96, 0.93])[2]),
      ];
      newBg[i] = +inp.value;
      COMPOSITION.background = newBg;
      commitComposition();
    });
    row.appendChild(inp);
  });

  body.appendChild(row);
  card.appendChild(body);
  return card;
}

function buildLayerCard(layer, idx) {
  const card = document.createElement('div');
  card.className = 'bg-base border border-line rounded';

  const collapsed = !!layer._ui_collapsed;
  const header = document.createElement('div');
  header.className = 'px-3 py-2 flex items-center gap-2' + (collapsed ? '' : ' border-b border-line');
  header.innerHTML = `
    <button type="button" data-role="collapse" title="${collapsed ? 'expand' : 'collapse'}" class="text-mute hover:text-ink leading-none w-4 shrink-0 text-[12px]">${collapsed ? '▸' : '▾'}</button>
    <input type="checkbox" data-role="enabled" ${layer.enabled !== false ? 'checked' : ''} title="enable layer" />
    <input type="text" data-role="name" value="${escapeAttr(layer.name || layer.type)}" class="bg-transparent text-[14px] text-ink focus:outline-none focus:border-faint border border-transparent rounded px-1 py-0.5 flex-1 min-w-0" />
    <span class="text-mute text-[12px] font-mono shrink-0">${layer.type}</span>
    <label class="text-mute text-[12px] shrink-0">α</label>
    <input type="number" data-role="alpha" step="0.05" min="0" max="1" value="${(+layer.alpha).toFixed(2)}" title="layer alpha" class="w-12 bg-base border border-line rounded px-1.5 py-0.5 text-right text-[13px] focus:outline-none focus:border-faint" />
    <button type="button" data-role="remove" title="remove layer" class="text-mute hover:text-ink text-base leading-none px-1">×</button>
  `;
  card.appendChild(header);

  header.querySelector('[data-role=collapse]').addEventListener('click', () => {
    layer._ui_collapsed = !layer._ui_collapsed;
    renderLayers();
  });
  header.querySelector('[data-role=enabled]').addEventListener('change', (e) => {
    layer.enabled = e.target.checked;
    commitComposition();
  });
  header.querySelector('[data-role=name]').addEventListener('change', (e) => {
    layer.name = e.target.value;
    commitComposition();
  });
  header.querySelector('[data-role=alpha]').addEventListener('change', (e) => {
    layer.alpha = +e.target.value;
    commitComposition();
  });
  header.querySelector('[data-role=remove]').addEventListener('click', () => removeLayer(idx));

  if (!collapsed) {
    const schema = SCHEMA[layer.type];
    if (schema && schema.fields) {
      const body = document.createElement('div');
      body.className = 'p-3 space-y-2';
      for (const [key, label, type, fmt, choices] of schema.fields) {
        const el = buildField(layer, schema.params_in, key, label, type, fmt, choices);
        if (el) body.appendChild(el);
      }
      card.appendChild(body);
    }
  }

  return card;
}

function readValue(layer, paramsIn, key) {
  if (_MASK3D_KEYS.has(key)) return layer[key];
  if (paramsIn && layer[paramsIn] && key in layer[paramsIn]) return layer[paramsIn][key];
  return layer[key];
}

function writeValue(layer, paramsIn, key, value) {
  if (paramsIn && !_MASK3D_KEYS.has(key)) {
    (layer[paramsIn] = layer[paramsIn] || {})[key] = value;
  } else {
    layer[key] = value;
  }
}

function buildField(layer, paramsIn, key, label, type, fmt, choices) {
  const row = document.createElement('div');

  if (type === 'rgb' || type === 'vec3') {
    const suffixes = type === 'rgb' ? ['r', 'g', 'b'] : ['x', 'y', 'z'];
    row.className = 'flex items-center gap-1';
    const lab = document.createElement('label');
    lab.className = 'text-mute text-[12px] w-20 shrink-0';
    lab.textContent = label;
    row.appendChild(lab);
    suffixes.forEach((s) => {
      const subKey = `${key}_${s}`;
      const v = readValue(layer, paramsIn, subKey) ?? 0;
      const inp = document.createElement('input');
      inp.type = 'number'; inp.step = '0.01'; inp.value = String(v);
      inp.title = `${label} ${s}`;
      inp.className = 'flex-1 min-w-0 w-16 bg-base border border-line rounded px-2 py-1 text-right text-[13px] focus:outline-none focus:border-faint';
      inp.addEventListener('change', () => {
        writeValue(layer, paramsIn, subKey, +inp.value);
        commitComposition();
      });
      row.appendChild(inp);
    });
    return row;
  }

  if (type === 'bool') {
    row.className = 'flex items-center justify-between gap-2';
    const id = `f-${Math.random().toString(36).slice(2, 8)}`;
    row.innerHTML = `<label for="${id}" class="text-mute text-[12px]">${label}</label>`;
    const inp = document.createElement('input');
    inp.type = 'checkbox'; inp.id = id;
    inp.checked = !!readValue(layer, paramsIn, key);
    inp.addEventListener('change', (e) => {
      writeValue(layer, paramsIn, key, e.target.checked);
      commitComposition();
    });
    row.appendChild(inp);
    return row;
  }

  if (type === 'str' && choices) {
    row.className = 'flex items-center justify-between gap-2';
    const id = `f-${Math.random().toString(36).slice(2, 8)}`;
    row.innerHTML = `<label for="${id}" class="text-mute text-[12px]">${label}</label>`;
    const sel = document.createElement('select');
    sel.id = id;
    sel.title = label;
    sel.className = 'bg-base border border-line rounded px-2 py-1 text-[13px] focus:outline-none focus:border-faint';
    const v = readValue(layer, paramsIn, key);
    for (const c of choices) {
      const opt = document.createElement('option');
      opt.value = c; opt.textContent = c;
      if (c === v) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.addEventListener('change', (e) => {
      writeValue(layer, paramsIn, key, e.target.value);
      commitComposition();
    });
    row.appendChild(sel);
    return row;
  }

  // float / int / str
  row.className = 'flex items-center justify-between gap-2';
  const id = `f-${Math.random().toString(36).slice(2, 8)}`;
  row.innerHTML = `<label for="${id}" class="text-mute text-[12px]">${label}</label>`;
  const inp = document.createElement('input');
  inp.id = id;
  inp.type = type === 'str' ? 'text' : 'number';
  inp.step = type === 'int' ? '1' : '0.01';
  inp.title = label;
  const v = readValue(layer, paramsIn, key);
  inp.value = v == null ? '' : String(v);
  inp.className = 'w-20 bg-base border border-line rounded px-2 py-1 text-right text-[13px] focus:outline-none focus:border-faint';
  inp.addEventListener('change', (e) => {
    let val = e.target.value;
    if (type === 'int') val = parseInt(val, 10);
    else if (type === 'float') val = parseFloat(val);
    writeValue(layer, paramsIn, key, val);
    commitComposition();
  });
  row.appendChild(inp);
  return row;
}

// -------------------------------------------------------------- actions --
let _commitTimer = null;
function commitComposition() {
  // debounce so typing doesn't fire a request per keystroke (we still
  // commit on 'change' but fast-typed batches collapse).
  clearTimeout(_commitTimer);
  _commitTimer = setTimeout(flushComposition, 80);
  // visual updates are immediate
  rebuildMaskSpheres();
  rebuildZlineEndpoints();
}

async function flushComposition() {
  clearTimeout(_commitTimer);
  _commitTimer = null;
  try { await api('/api/composition', { method: 'POST', body: COMPOSITION }); }
  catch (err) { appendLog('composition sync failed: ' + err.message); }
}

async function addLayer(type) {
  try {
    const data = await api('/api/add_layer', { method: 'POST', body: { type } });
    COMPOSITION.layers.push(data.layer);
    renderLayers();
    rebuildMaskSpheres();
    rebuildZlineEndpoints();
  } catch (err) { appendLog('add layer failed: ' + err.message); }
}

async function removeLayer(idx) {
  const layer = COMPOSITION.layers[idx];
  if (!layer) return;
  const label = layer.name || layer.type;
  if (!confirm(`Delete layer "${label}"?`)) return;
  try {
    await api('/api/remove_layer', { method: 'POST', body: { index: idx } });
    COMPOSITION.layers.splice(idx, 1);
    renderLayers();
    rebuildMaskSpheres();
    rebuildZlineEndpoints();
  } catch (err) { appendLog('remove layer failed: ' + err.message); }
}

// -------------------------------------------------------------- frame ----
function rebuildFrame() {
  if (!three.frameGroup) return;
  three.frameGroup.clear();
  if (!$('show-frame').checked) return;

  // Use the renderer's actual projection parameters when available.
  const center = new THREE.Vector3(...SCENE.center);
  const Rcam = SCENE.Rcam;           // 3x3 world->camera rotation (or null)
  const focal = SCENE.focal;         // focal length in pixels (or null)
  const distance = SCENE.distance;   // camera->center distance (or null)
  const ysign = SCENE.ysign ?? -1.0;
  const [W, H] = SCENE.render_size;

  let planeCenter, right, up;
  let hw, hh;

  if (Rcam && focal && distance) {
    // Accurate focal-plane geometry from gsplat projection.
    const r0 = new THREE.Vector3(Rcam[0][0], Rcam[0][1], Rcam[0][2]);
    const r1 = new THREE.Vector3(Rcam[1][0], Rcam[1][1], Rcam[1][2]);
    const r2 = new THREE.Vector3(Rcam[2][0], Rcam[2][1], Rcam[2][2]);
    // Camera looks along +r2; focal plane is at z = distance in camera space.
    planeCenter = new THREE.Vector3().copy(center).addScaledVector(r2, distance);
    hw = distance * (W / 2) / focal;
    hh = distance * (H / 2) / focal;
    right = r0;
    // ysign: image y = H/2 + ysign * focal * y_cam / z.
    // Top of image (y_pix=0) => y_cam = -(H/2)*z/(ysign*focal) = -hh/ysign.
    up = new THREE.Vector3().copy(r1).multiplyScalar(-1.0 / ysign);
  } else {
    // Fallback using camera angles (less accurate if biases exist).
    const d = SCENE.extent * CAMERA.distance_k;
    const aspect = W / H;
    const fovRad = (CAMERA.fov_deg * Math.PI) / 180;
    hh = d * Math.tan(fovRad / 2);
    hw = hh * aspect;
    const elev = CAMERA.elev_deg * Math.PI / 180;
    const azim = CAMERA.azim_deg * Math.PI / 180;
    const ce = Math.cos(elev), se = Math.sin(elev);
    const ca = Math.cos(azim), sa = Math.sin(azim);
    const forward = new THREE.Vector3(ce * sa, -se, -ce * ca).normalize();
    right = new THREE.Vector3(ca, 0, sa).normalize();
    up = new THREE.Vector3().crossVectors(right, forward).normalize();
    planeCenter = new THREE.Vector3().copy(center).addScaledVector(forward, -d);
  }

  const corners = [
    new THREE.Vector3().copy(planeCenter).addScaledVector(right,  hw).addScaledVector(up,  hh),
    new THREE.Vector3().copy(planeCenter).addScaledVector(right,  hw).addScaledVector(up, -hh),
    new THREE.Vector3().copy(planeCenter).addScaledVector(right, -hw).addScaledVector(up, -hh),
    new THREE.Vector3().copy(planeCenter).addScaledVector(right, -hw).addScaledVector(up,  hh),
  ];

  // Frame rectangle
  const geo = new THREE.BufferGeometry().setFromPoints([
    corners[0], corners[1],
    corners[1], corners[2],
    corners[2], corners[3],
    corners[3], corners[0],
  ]);
  const mat = new THREE.LineBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.6 });
  three.frameGroup.add(new THREE.LineSegments(geo, mat));

  // Diagonal crosshair
  const diagGeo = new THREE.BufferGeometry().setFromPoints([
    corners[0], corners[2],
    corners[1], corners[3],
  ]);
  const diagMat = new THREE.LineBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.2 });
  three.frameGroup.add(new THREE.LineSegments(diagGeo, diagMat));

  // Centre dot at the camera position (where the renderer camera sits)
  const dotGeo = new THREE.SphereGeometry(three.bbDiag * 0.006, 8, 6);
  const dotMat = new THREE.MeshBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.5 });
  const dot = new THREE.Mesh(dotGeo, dotMat);
  dot.position.copy(center);
  three.frameGroup.add(dot);

  // Line from camera centre to frame plane centre
  const lineGeo = new THREE.BufferGeometry().setFromPoints([center, planeCenter]);
  const lineMat = new THREE.LineBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.3 });
  three.frameGroup.add(new THREE.Line(lineGeo, lineMat));
}

// Toggle frame visibility
function setupFrameToggle() {
  $('show-frame').addEventListener('change', rebuildFrame);
}

// -------------------------------------------------------------- markers ----
function setupMarkerUI() {
  $('btn-add-marker').addEventListener('click', () => {
    const x = parseFloat($('marker-x').value);
    const y = parseFloat($('marker-y').value);
    const y2 = parseFloat($('marker-z').value);
    if (Number.isNaN(x) || Number.isNaN(y) || Number.isNaN(y2)) return;
    addMarker(x, y, y2);
    $('marker-x').value = '';
    $('marker-y').value = '';
    $('marker-z').value = '';
  });
  renderMarkerList();
}

function addMarker(x, y, z) {
  const id = Date.now() + Math.random();
  MARKERS.push({ id, x, y, z });
  rebuildMarkerSpheres();
  renderMarkerList();
}

function removeMarker(id) {
  const idx = MARKERS.findIndex((m) => m.id === id);
  if (idx >= 0) {
    MARKERS.splice(idx, 1);
    rebuildMarkerSpheres();
    renderMarkerList();
  }
}

function rebuildMarkerSpheres() {
  if (!three.markerGroup) return;
  three.markerGroup.clear();
  for (const m of MARKERS) {
    const geo = new THREE.SphereGeometry(three.bbDiag * 0.012, 12, 8);
    const mat = new THREE.MeshBasicMaterial({ color: 0xf43f5e });
    const sphere = new THREE.Mesh(geo, mat);
    sphere.position.set(m.x, m.y, m.z);
    three.markerGroup.add(sphere);

    // Label line to help with depth perception
    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(m.x, m.y, m.z),
      new THREE.Vector3(m.x, three.bbCenter.y - three.bbDiag * 0.5, m.z),
    ]);
    const lineMat = new THREE.LineBasicMaterial({ color: 0xf43f5e, transparent: true, opacity: 0.3 });
    three.markerGroup.add(new THREE.Line(lineGeo, lineMat));
  }
}

function renderMarkerList() {
  const list = $('markers-list');
  list.innerHTML = '';
  for (const m of MARKERS) {
    const row = document.createElement('div');
    row.className = 'flex items-center justify-between text-[12px] text-mute font-mono';
    row.innerHTML = `
      <span>${m.x.toFixed(2)}, ${m.y.toFixed(2)}, ${m.z.toFixed(2)}</span>
      <button type="button" data-mid="${m.id}" class="text-mute hover:text-ink px-1">×</button>
    `;
    row.querySelector('button').addEventListener('click', () => removeMarker(m.id));
    list.appendChild(row);
  }
}

function bindActions() {
  $('btn-render').addEventListener('click', doRender);
  $('btn-save').addEventListener('click', doSave);
  $('btn-save-img').addEventListener('click', doSaveImage);
  $('btn-reload').addEventListener('click', doReload);
  for (const btn of document.querySelectorAll('[data-add]')) {
    btn.addEventListener('click', () => addLayer(btn.dataset.add));
  }
  document.addEventListener('keydown', (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
    if (e.key === 'r') { e.preventDefault(); doRender(); }
    if (e.key === 's' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); doSave(); }
  });
  setupFrameToggle();
}

async function doRender() {
  const btn = $('btn-render');
  $('render-status').textContent = 'rendering…';
  btn.disabled = true;
  try {
    clearTimeout(_commitTimer);
    _commitTimer = null;
    // Read camera straight from the inputs
    const camera = {
      elev_deg: +$('cam-elev').value,
      azim_deg: +$('cam-azim').value,
      fov_deg:  +$('cam-fov').value,
      distance_k: +$('cam-dist').value,
      head_bias_x: +$('cam-bias-x').value,
      head_bias_y: +$('cam-bias-y').value,
    };
    console.log('[doRender] sending camera:', camera);
    CAMERA = camera;

    // Create async render job
    const createRes = await api('/api/render', {
      method: 'POST',
      body: { camera, composition: COMPOSITION },
    });
    const jobId = createRes.job_id;

    // Poll for completion
    const poll = async () => {
      const statusRes = await api(`/api/render/status/${jobId}`);
      const job = statusRes.job;

      if (job.status === 'done') {
        const img = $('preview-image');
        img.src = job.image;
        img.classList.remove('hidden');
        $('preview-placeholder').classList.add('hidden');
        $('render-status').textContent = `${(job.time || 0).toFixed(1)}s`;
        if (statusRes.log) appendLog(statusRes.log.join('\n'));
        btn.disabled = false;
      } else if (job.status === 'error') {
        $('render-status').textContent = 'failed';
        appendLog('render failed: ' + (job.error || 'unknown error'));
        btn.disabled = false;
      } else {
        // Still pending — poll again in 1.5s
        setTimeout(poll, 1500);
      }
    };

    setTimeout(poll, 500);
  } catch (err) {
    $('render-status').textContent = 'failed';
    appendLog('render failed: ' + err.message);
    btn.disabled = false;
  }
}

async function doSave() {
  try {
    // Same as doRender: make sure the latest edits + camera have reached
    // the backend before we ask it to serialise composition to disk.
    await commitCameraFromInputs({ silent: true });
    await flushComposition();
    const data = await api('/api/save_scene', { method: 'POST' });
    appendLog(`saved -> ${data.path}`);
  } catch (err) { appendLog('save failed: ' + err.message); }
}

async function doSaveImage() {
  try {
    const data = await api('/api/save_image', { method: 'POST' });
    appendLog(`saved image -> ${data.path}`);
  } catch (err) { appendLog('save image failed: ' + err.message); }
}

async function doReload() {
  if (!confirm('Reload scene from disk? Any unsaved UI changes will be lost.')) return;
  try {
    await api('/api/reload', { method: 'POST' });
    // Simplest correct re-init: just reload the page. init() then re-reads
    // /api/init and rebuilds the whole UI + Three.js scene from scratch.
    window.location.reload();
  } catch (err) {
    appendLog('reload failed: ' + err.message);
  }
}

function appendLog(text) {
  if (!text) return;
  const el = $('log');
  const merged = (el.textContent ? el.textContent + '\n' : '') + text;
  const trimmed = merged.split('\n').slice(-300).join('\n');
  el.textContent = trimmed;
  el.scrollTop = el.scrollHeight;
}

function escapeAttr(s) {
  return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
}

init().catch((err) => {
  console.error(err);
  document.body.innerHTML =
    `<div class="p-8 text-red-400 font-mono text-sm">init failed: ${err.message}</div>`;
});
