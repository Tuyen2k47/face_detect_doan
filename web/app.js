'use strict';

const $ = (id) => document.getElementById(id);
const UNKNOWN_COLOR = '#E2711D';
const KNOWN_COLOR = '#5CC8FF';

const state = {
  view: 'live',
  info: null,
  threshold: 0.55,
  paused: false,
  logOn: true,
  lastFaces: [],
  ws: null,
  busy: false,
  lastReply: 0,
  fps: 0,
  enroll: { step: 1, name: '', session: null, clips: [], summary: null },
  reportT: null,
};

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === 'style') Object.assign(node.style, value);
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) if (child) node.append(child);
  return node;
}

const pct = (x) => (x * 100).toFixed(1) + '%';

// ---- Điều hướng ----------------------------------------------------------------------------

function showView() {
  const view = (location.hash || '#live').slice(1);
  state.view = ['live', 'enroll', 'report'].includes(view) ? view : 'live';
  for (const name of ['live', 'enroll', 'report']) $('view-' + name).hidden = name !== state.view;
  document.querySelectorAll('.tabs a').forEach((a) => {
    if (a.dataset.view === state.view) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
  if (state.view === 'report') loadInfo();
}
window.addEventListener('hashchange', showView);

// ---- Thông tin chung -------------------------------------------------------------------------

async function loadInfo() {
  const response = await fetch('/api/info');
  state.info = await response.json();
  const info = state.info;
  $('chip-provider').textContent = (info.providers[0] || 'CPU').replace('ExecutionProvider', '') + ' · ONNX Runtime';
  $('chip-people').textContent = info.people.length + ' người đã đăng ký';
  if (state.reportT === null) {
    state.threshold = info.threshold;
    state.reportT = Math.round(info.threshold * 20) / 20;
    $('thr').value = info.threshold.toFixed(2);
  }
  $('people').replaceChildren(...info.people.map((p) => el('span', { text: p })));
  renderThreshold();
  renderLog(info.log);
  renderReport();
}

function calibrationAt(t) {
  const rows = (state.info && state.info.report.calibration) || [];
  let best = null;
  for (const row of rows) if (!best || Math.abs(row.t - t) < Math.abs(best.t - t)) best = row;
  return best;
}

// ---- 1. Nhận diện trực tiếp --------------------------------------------------------------

let stream = null;

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (error) {
    $('live-placeholder').textContent = 'Không mở được webcam: ' + error.message + '. Hãy cho phép quyền camera rồi tải lại trang.';
    return;
  }
  for (const id of ['live-video', 'enroll-video']) $(id).srcObject = stream;
  const video = $('live-video');
  video.addEventListener('loadedmetadata', () => {
    $('live-placeholder').hidden = true;
    $('live-cam').style.aspectRatio = video.videoWidth + ' / ' + video.videoHeight;
  });
}

function grabCanvas(video) {
  const canvas = $('grab');
  const width = Math.min(640, video.videoWidth);
  canvas.width = width;
  canvas.height = Math.round(width * video.videoHeight / video.videoWidth);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function connect() {
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    state.busy = false;
    sendSettings();
  };
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    state.busy = false;
    const now = performance.now();
    if (state.lastReply) state.fps = state.fps * 0.8 + (1000 / (now - state.lastReply)) * 0.2;
    state.lastReply = now;
    $('live-ms').textContent = data.ms + ' ms/frame';
    $('chip-fps').textContent = state.fps.toFixed(1) + ' FPS';
    renderFaces(data.faces);
    renderLog(data.log);
    pump();
  };
  ws.onclose = () => {
    state.busy = false;
    setTimeout(connect, 1000);
  };
  state.ws = ws;
}

function sendSettings() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ threshold: state.threshold, log: state.logOn }));
  }
}

function pump() {
  const video = $('live-video');
  const ws = state.ws;
  if (!ws || ws.readyState !== WebSocket.OPEN || state.busy || state.paused) return;
  if (state.view !== 'live' || !video.videoWidth) return;
  state.busy = true;
  grabCanvas(video).toBlob((blob) => {
    if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
    else state.busy = false;
  }, 'image/jpeg', 0.8);
}
setInterval(pump, 200);

function renderFaces(faces) {
  state.lastFaces = faces;
  const boxes = faces.map((f) => {
    const color = f.known ? KNOWN_COLOR : UNKNOWN_COLOR;
    return el('div', {
      class: 'box',
      style: {
        left: ((1 - f.x - f.w) * 100) + '%', top: (f.y * 100) + '%',
        width: (f.w * 100) + '%', height: (f.h * 100) + '%', borderColor: color,
      },
    }, el('div', { class: 'tag', text: f.name, style: { background: color } }));
  });
  $('live-boxes').replaceChildren(...boxes);

  const cards = faces.map((f) => {
    const color = f.known ? KNOWN_COLOR : UNKNOWN_COLOR;
    return el('div', { class: 'face-card' }, [
      el('div', { class: 'card-head' }, [
        el('span', { text: f.name, style: { fontWeight: 600, fontSize: '16px', color } }),
        el('span', { class: 'mono small', text: f.score.toFixed(2), style: { color: '#B4C1C9' } }),
      ]),
      el('div', { class: 'meter' }, [
        el('div', { class: 'fill', style: { width: Math.max(0, f.score) * 100 + '%', background: color } }),
        el('div', { class: 'mark', style: { left: state.threshold * 100 + '%' } }),
      ]),
      el('div', { class: 'small muted', text: 'Gần nhất: ' + f.best }),
    ]);
  });
  $('live-cards').replaceChildren(...cards);
}

function renderLog(rows) {
  if (!rows || !rows.length) return;
  $('log-list').replaceChildren(...rows.map((row) => el('div', { class: 'log-row' }, [
    el('span', { class: 'mono small muted', text: row.time }),
    el('span', { class: 'name', text: row.name, style: { color: row.name === 'Người lạ' ? UNKNOWN_COLOR : '' } }),
    el('span', { class: 'mono small', text: Number(row.score).toFixed(2), style: { color: '#B4C1C9' } }),
  ])));
}

function renderThreshold() {
  $('thr-value').textContent = state.threshold.toFixed(2);
  const row = calibrationAt(state.threshold);
  if (!row) return;
  $('thr-tar').textContent = pct(row.tar);
  $('thr-far').textContent = pct(row.far);
  $('thr-far').style.color = row.far > 0 ? UNKNOWN_COLOR : '';
  $('thr-note').textContent = 'Số liệu tại ngưỡng ' + row.t.toFixed(2) + ' từ đánh giá trên face_embeddings.npz. Ngưỡng lưu trong mô hình: ' + state.info.threshold.toFixed(2) + '.';
}

$('thr').addEventListener('input', (event) => {
  state.threshold = parseFloat(event.target.value);
  renderThreshold();
  sendSettings();
});

$('btn-pause').addEventListener('click', () => {
  state.paused = !state.paused;
  $('btn-pause').textContent = state.paused ? 'Tiếp tục' : 'Tạm dừng';
  $('live-state').replaceChildren(...(state.paused ? ['TẠM DỪNG'] : [el('span', { class: 'dot' }), 'LIVE']));
  if (state.paused) $('chip-fps').textContent = '0 FPS';
});

$('btn-log').addEventListener('click', () => {
  state.logOn = !state.logOn;
  $('btn-log').setAttribute('aria-checked', String(state.logOn));
  sendSettings();
});

$('btn-snap').addEventListener('click', () => {
  const video = $('live-video');
  if (!video.videoWidth) return;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  ctx.lineWidth = 3;
  ctx.font = '600 20px sans-serif';
  for (const f of state.lastFaces) {
    const color = f.known ? KNOWN_COLOR : UNKNOWN_COLOR;
    const [x, y, w, h] = [f.x * canvas.width, f.y * canvas.height, f.w * canvas.width, f.h * canvas.height];
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 28, ctx.measureText(f.name).width + 16, 26);
    ctx.fillStyle = '#0F1417';
    ctx.fillText(f.name, x + 8, y - 8);
  }
  const link = el('a', { href: canvas.toDataURL('image/png'), download: 'faceid_' + Date.now() + '.png' });
  link.click();
});

// ---- 2. Đăng ký người mới ----------------------------------------------------------------

function setStep(step) {
  const enroll = state.enroll;
  if (step >= 2 && !enroll.name) step = 1;
  if (step === 3 && !(enroll.summary && enroll.summary.embeddings)) step = 2;
  enroll.step = step;
  for (const n of [1, 2, 3]) $('enroll-step' + n).hidden = n !== step;
  document.querySelectorAll('.step').forEach((button) => {
    const n = Number(button.dataset.step);
    if (n === step) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
    button.classList.toggle('done', n < step);
  });
  if (step === 3) renderSummary();
}

document.querySelectorAll('.step').forEach((button) => button.addEventListener('click', () => setStep(Number(button.dataset.step))));

$('enroll-next').addEventListener('click', () => {
  const name = $('enroll-name').value.trim();
  const error = !name ? 'Hãy nhập tên.' : /[\\/]/.test(name) ? 'Tên không được chứa / hoặc \\.' : '';
  $('enroll-name-error').textContent = error;
  $('enroll-name-error').hidden = !error;
  if (error) return;
  if (name !== state.enroll.name) Object.assign(state.enroll, { name, session: null, clips: [], summary: null });
  renderClips();
  setStep(2);
});

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

$('enroll-record').addEventListener('click', async () => {
  const video = $('enroll-video');
  if (!video.videoWidth) return;
  const button = $('enroll-record');
  button.disabled = true;
  $('enroll-rec').hidden = false;
  const frames = [];
  for (let i = 0; i < 30; i++) {
    $('enroll-count').textContent = Math.ceil((30 - i) / 5) + 's';
    frames.push(grabCanvas(video).toDataURL('image/jpeg', 0.85));
    await sleep(200);
  }
  $('enroll-count').textContent = 'đang phân tích…';
  try {
    const response = await fetch('/api/enroll/clip', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session: state.enroll.session, name: state.enroll.name, frames }),
    });
    const summary = await response.json();
    if (!response.ok) throw new Error(summary.detail || response.statusText);
    state.enroll.session = summary.session;
    state.enroll.summary = summary;
    state.enroll.clips.push(summary);
    renderClips();
  } catch (error) {
    $('clip-list').prepend(el('p', { class: 'small', text: 'Lỗi: ' + error.message, style: { color: 'var(--warn)' } }));
  } finally {
    button.disabled = false;
    $('enroll-rec').hidden = true;
  }
});

function renderClips() {
  const clips = state.enroll.clips;
  $('enroll-review').disabled = !(state.enroll.summary && state.enroll.summary.embeddings);
  if (!clips.length) {
    $('clip-list').replaceChildren(el('p', { class: 'small muted', text: 'Chưa quay đoạn nào. Chỉ để một người trong khung hình.' }));
    return;
  }
  $('clip-list').replaceChildren(...clips.map((clip) => {
    const s = clip.stats;
    const total = clip.total || 1;
    const poor = s.blurry + s.low_score + s.small + s.pose;
    const missing = s.no_face + s.multiple_faces;
    const tips = [];
    if (s.blurry / total > 0.4) tips.push('Nhiều frame bị mờ — quay nơi sáng hơn, giữ máy yên.');
    if (s.multiple_faces / total > 0.2) tips.push('Có người khác lọt vào khung hình.');
    if (s.no_face / total > 0.3) tips.push('Nhiều frame không thấy mặt — ngồi gần camera hơn.');
    return el('div', { style: { display: 'flex', flexDirection: 'column', gap: '6px', paddingTop: '10px', borderTop: '1px solid var(--line-soft)' } }, [
      el('div', { class: 'card-head', style: { fontSize: '14px' } }, [
        el('span', { text: 'Đoạn ' + clip.clip, style: { fontWeight: 500 } }),
        el('span', { class: 'mono', text: s.ok + '/' + total + ' frame tốt', style: { color: '#B4C1C9' } }),
      ]),
      el('div', { class: 'qbar' }, [
        el('div', { style: { width: s.ok / total * 100 + '%', background: 'var(--known)' } }),
        el('div', { style: { width: poor / total * 100 + '%', background: 'var(--unknown)' } }),
        el('div', { style: { width: missing / total * 100 + '%', background: 'var(--pose)' } }),
      ]),
      ...tips.map((tip) => el('div', { class: 'small', text: tip, style: { color: 'var(--warn)' } })),
    ]);
  }));
}

$('enroll-review').addEventListener('click', () => setStep(3));
$('enroll-more').addEventListener('click', () => setStep(2));

function renderSummary() {
  const enroll = state.enroll;
  const summary = enroll.summary;
  const valid = enroll.clips.filter((clip) => clip.stats.ok > 0).length;
  $('sum-clips').textContent = valid + ' / ' + enroll.clips.length;
  $('sum-emb').textContent = summary.embeddings;
  $('sum-nearest').textContent = summary.nearest ? summary.nearest.name + ' ' + summary.nearest.score.toFixed(2) : '—';
  const threshold = state.info ? state.info.threshold : 0.55;
  const notice = $('sum-notice');
  const lines = [];
  let good = true;
  if (summary.nearest && summary.nearest.score >= threshold) {
    good = false;
    lines.push('Giống ' + summary.nearest.name + ' (' + summary.nearest.score.toFixed(2) + ' ≥ ' + threshold.toFixed(2) + '): có thể người này đã được đăng ký dưới tên khác.');
  } else {
    lines.push('Dưới ngưỡng ' + threshold.toFixed(2) + ' với mọi người đã đăng ký — không trùng hồ sơ cũ.');
  }
  if (summary.exists) lines.push('Tên "' + enroll.name + '" đã có: dữ liệu mới sẽ được thêm vào hồ sơ cũ.');
  if (valid < 2) lines.push('Nên quay ít nhất 2 đoạn để hồ sơ ổn định và đánh giá được.');
  notice.className = 'notice ' + (good ? 'good' : 'bad');
  notice.replaceChildren(el('div', {}, lines.map((line) => el('div', { text: line }))));
  $('enroll-result').hidden = true;
}

$('enroll-save').addEventListener('click', async () => {
  const button = $('enroll-save');
  button.disabled = true;
  try {
    const response = await fetch('/api/enroll/commit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session: state.enroll.session }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || response.statusText);
    $('enroll-result').textContent = 'Đã lưu ' + result.name + ' (' + result.embeddings + ' embedding). Danh sách hiện có ' + result.people + ' người.';
    $('enroll-result').style.color = 'var(--ok)';
    Object.assign(state.enroll, { session: null, clips: [], summary: null });
    await loadInfo();
  } catch (error) {
    $('enroll-result').textContent = 'Lỗi: ' + error.message;
    $('enroll-result').style.color = 'var(--warn)';
  } finally {
    $('enroll-result').hidden = false;
    button.disabled = false;
  }
});

// ---- 3. Đánh giá mô hình -----------------------------------------------------------------

const THRESHOLD_NOTES = {
  '0.45': 'Ngưỡng cũ: một phần frame người lạ bị gọi bằng tên người giống nhất.',
  '0.50': 'Điểm người lạ vừa hết bị nhận nhầm, chưa có khoảng dự phòng.',
  '0.55': 'Ngưỡng đề xuất: thêm 0.05 dự phòng cho frame webcam mờ hơn dữ liệu hiệu chỉnh.',
};

function renderReport() {
  const info = state.info;
  if (!info) return;
  const report = info.report;
  $('kpi-people').textContent = info.people.length;
  $('kpi-videos').textContent = report.videos || '—';
  $('kpi-frames').textContent = report.frames ? report.frames.toLocaleString('vi-VN') : '—';
  $('kpi-top1').textContent = report.top1 != null ? pct(report.top1) : '—';

  const rows = report.calibration || [];
  $('chart').replaceChildren(...rows.map((row) => el('button', {
    type: 'button', 'aria-pressed': String(Math.abs(row.t - state.reportT) < 0.001), 'aria-label': 'Ngưỡng ' + row.t.toFixed(2),
    onclick: () => { state.reportT = row.t; renderReport(); },
  }, [
    el('span', { class: 'bars' }, [
      el('span', { style: { height: row.tar * 100 + '%', background: 'var(--known)' } }),
      el('span', { style: { height: row.far * 100 + '%', background: 'var(--unknown)' } }),
    ]),
    el('span', { class: 't', text: row.t.toFixed(2) }),
  ])));

  const sel = calibrationAt(state.reportT);
  if (sel) {
    $('sel-t').textContent = sel.t.toFixed(2);
    $('sel-tar').textContent = pct(sel.tar);
    $('sel-far').textContent = pct(sel.far);
    $('sel-far').style.color = sel.far > 0 ? UNKNOWN_COLOR : '';
    $('sel-note').textContent = THRESHOLD_NOTES[sel.t.toFixed(2)]
      || (sel.far > 0 ? 'Quá dễ: người lạ thường bị gọi bằng tên người khác.' : 'An toàn với người lạ; ngưỡng càng cao càng dễ bỏ sót người quen.');
  }

  const maxFrames = Math.max(1, ...report.people.map((p) => p.frames));
  $('people-table').replaceChildren(...report.people.map((p) => {
    const close = p.nearestScore >= 0.4;
    return el('tr', {}, [
      el('td', { text: p.name, style: { fontWeight: 500 } }),
      el('td', { class: 'mono', text: String(p.videos) }),
      el('td', {}, el('span', { style: { display: 'flex', alignItems: 'center', gap: '10px' } }, [
        el('span', { class: 'mono', text: String(p.frames), style: { width: '44px' } }),
        el('span', { class: 'mini' }, el('span', { style: { width: p.frames / maxFrames * 100 + '%', background: p.frames < 60 ? UNKNOWN_COLOR : KNOWN_COLOR } })),
      ])),
      el('td', { text: p.nearest ? p.nearest + ' ' + p.nearestScore.toFixed(2) : '—', style: { color: close ? '#F2B544' : '#B4C1C9' } }),
    ]);
  }));
}

// ---- Khởi động ---------------------------------------------------------------------------

showView();
setStep(1);
loadInfo();
startCamera();
connect();
