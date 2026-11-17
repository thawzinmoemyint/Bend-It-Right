import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";
import { FilesetResolver, PoseLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";
import { EXERCISES, YOGA_POSES, WorkoutCounter, YogaTimer, formatSeconds } from "./logic.js";

const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const YOGA_MODEL = "model/yoga.onnx";
const INPUT = 640;          // YOLO input size
const MIN_BOX_CONF = 0.25;  // boxes below this are not drawn or counted (Ultralytics default)

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
if (!self.crossOriginIsolated) ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error"; // hide the harmless "some nodes run on CPU" notice

// COCO keypoint skeleton (0-indexed), as drawn by Ultralytics
const SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
  [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]];
const POSE_COLORS = ["#e6550d", "#3182bd", "#31a354", "#9467bd"];

const EXERCISE_IMAGES = { "Push Up": "pushup", "Curl Up": "curlup", "Jumping Jack": "jumpingjack", "Squat": "squat" };
const POSE_IMAGES = { "Goddess": "goddess", "Plank": "plank", "Tree": "tree", "Warrior II": "warrior2" };

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), ctx = overlay.getContext("2d");
const statusEl = $("status");

const state = {
  mode: "workout",
  exercise: "Squat",
  source: null,        // "webcam" | "file" | null
  stream: null,
  counter: new WorkoutCounter(),
  timer: new YogaTimer(),
  landmarker: null, landmarkerLoading: null,
  session: null, sessionLoading: null, yogaBusy: false,
  frames: 0, fpsStart: performance.now(), fps: 0,
};

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

// ------------------------------------------------------------------ panel UI

function buildCards() {
  const ex = $("exercise-cards");
  for (const name of EXERCISES) {
    const b = document.createElement("button");
    b.className = "card";
    b.dataset.exercise = name;
    b.innerHTML = `<img src="images/${EXERCISE_IMAGES[name]}.jpg" alt=""><span class="name">${name}</span><span class="value">0</span>`;
    b.addEventListener("click", () => { state.exercise = name; renderPanel(); });
    ex.appendChild(b);
  }
  const po = $("pose-cards");
  for (const name of YOGA_POSES) {
    const d = document.createElement("div");
    d.className = "card";
    d.dataset.pose = name;
    d.innerHTML = `<img src="images/${POSE_IMAGES[name]}.jpg" alt=""><span class="name">${name}</span><span class="value">0:00.00</span>`;
    po.appendChild(d);
  }
}

function renderPanel(activePoses = []) {
  for (const b of document.querySelectorAll("[data-exercise]")) {
    const name = b.dataset.exercise;
    b.setAttribute("aria-pressed", String(name === state.exercise));
    b.querySelector(".value").textContent = state.counter.counts[name];
  }
  const totals = state.timer.totals(currentTime());
  for (const d of document.querySelectorAll("[data-pose]")) {
    const name = d.dataset.pose;
    d.classList.toggle("active", activePoses.includes(name));
    d.querySelector(".value").textContent = formatSeconds(totals[name]);
  }
}

function setMode(mode) {
  state.mode = mode;
  for (const t of document.querySelectorAll("[role=tab]")) t.setAttribute("aria-selected", String(t.dataset.mode === mode));
  $("panel-workout").hidden = mode !== "workout";
  $("panel-yoga").hidden = mode !== "yoga";
  clearOverlay();
  ensureModel().catch(() => {});
}

// ------------------------------------------------------------------ models

async function ensureModel() {
  if (state.mode === "workout") return loadLandmarker();
  return loadYoga();
}

function loadLandmarker() {
  if (state.landmarker) return Promise.resolve(state.landmarker);
  state.landmarkerLoading ??= (async () => {
    setStatus("Loading pose model…");
    try {
      const files = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
      const opts = (delegate) => ({ baseOptions: { modelAssetPath: POSE_MODEL, delegate }, runningMode: "VIDEO", numPoses: 1 });
      try {
        state.landmarker = await PoseLandmarker.createFromOptions(files, opts("GPU"));
      } catch {
        state.landmarker = await PoseLandmarker.createFromOptions(files, opts("CPU"));
      }
      setStatus(state.source ? "Counting…" : "Pose model ready.");
      return state.landmarker;
    } catch (err) {
      state.landmarkerLoading = null;
      setStatus(`Could not load the pose model: ${err.message}`, true);
      throw err;
    }
  })();
  return state.landmarkerLoading;
}

function loadYoga() {
  if (state.session) return Promise.resolve(state.session);
  state.sessionLoading ??= (async () => {
    setStatus("Loading yoga model (38 MB, first time only)…");
    try {
      if ("gpu" in navigator) {
        try {
          state.session = await ort.InferenceSession.create(YOGA_MODEL, { executionProviders: ["webgpu"] });
          state.backend = "GPU";
        } catch { /* no usable WebGPU adapter: fall back to CPU */ }
      }
      if (!state.session) {
        state.session = await ort.InferenceSession.create(YOGA_MODEL, { executionProviders: ["wasm"] });
        state.backend = "CPU";
      }
      // The first run compiles GPU shaders and is slow, so do it now rather than on the first video frame.
      const warmup = new ort.Tensor("float32", new Float32Array(3 * INPUT * INPUT), [1, 3, INPUT, INPUT]);
      await state.session.run({ [state.session.inputNames[0]]: warmup });
      setStatus(state.source ? "Detecting poses…" : "Yoga model ready.");
      return state.session;
    } catch (err) {
      state.sessionLoading = null;
      setStatus(`Could not load the yoga model: ${err.message}`, true);
      throw err;
    }
  })();
  return state.sessionLoading;
}

// ------------------------------------------------------------------ sources

async function startWebcam() {
  stopSource();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (err) {
    setStatus(`Camera unavailable: ${err.message}. Allow camera access in your browser and try again.`, true);
    return;
  }
  video.srcObject = state.stream;
  video.controls = false;
  video.classList.add("live");
  state.source = "webcam";
  await startPlayback();
}

async function startFile(url, crossOrigin = false) {
  stopSource();
  if (crossOrigin) video.crossOrigin = "anonymous";
  video.src = url;
  video.controls = true;
  state.source = "file";
  state.timer.reset();
  await startPlayback();
}

async function startPlayback() {
  $("placeholder").hidden = true;
  $("btn-stop").disabled = false;
  const source = state.source;
  // Wait for the model so an uploaded video isn't half over before detection starts.
  try { await ensureModel(); } catch { return; }
  if (state.source !== source) return; // stopped or replaced while loading
  setStatus(state.mode === "workout" ? "Counting…" : "Detecting poses…");
  try { await video.play(); } catch { setStatus("Press play on the video to start."); }
  scheduleFrame();
}

function stopSource() {
  // close any running yoga timers at the last processed moment
  if (state.source) state.timer.update([], state.lastYogaTime ?? currentTime());
  state.lastYogaTime = null;
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  if (video.src?.startsWith("blob:")) URL.revokeObjectURL(video.src);
  video.pause();
  video.removeAttribute("src");
  video.srcObject = null;
  video.removeAttribute("crossorigin");
  video.classList.remove("live");
  video.load();
  state.source = null;
  clearOverlay();
  $("placeholder").hidden = false;
  $("btn-stop").disabled = true;
}

// Seconds used by the yoga timer: video time for uploads, wall-clock time for the webcam.
function currentTime() {
  return state.source === "file" ? video.currentTime : performance.now() / 1000;
}

// ------------------------------------------------------------------ frame loop

function scheduleFrame() {
  if (!state.source) return;
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame(onFrame);
}

function onFrame() {
  if (!state.source) return;
  if (video.readyState >= 2 && video.videoWidth) {
    fitOverlay();
    if (state.mode === "workout") workoutFrame();
    else yogaFrame();
  }
  scheduleFrame();
}

// Place the canvas exactly over the visible video picture (the element may letterbox it).
function fitOverlay() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (overlay.width !== vw || overlay.height !== vh) { overlay.width = vw; overlay.height = vh; }
  const ew = video.clientWidth, eh = video.clientHeight;
  const scale = Math.min(ew / vw, eh / vh);
  const w = vw * scale, h = vh * scale;
  Object.assign(overlay.style, {
    width: `${w}px`, height: `${h}px`,
    left: `${video.offsetLeft + (ew - w) / 2}px`, top: `${video.offsetTop + (eh - h) / 2}px`,
  });
}

function clearOverlay() { ctx.clearRect(0, 0, overlay.width, overlay.height); }

function countFps() {
  state.frames += 1;
  const now = performance.now();
  if (now - state.fpsStart > 1000) {
    state.fps = (state.frames * 1000) / (now - state.fpsStart);
    state.frames = 0;
    state.fpsStart = now;
  }
}

function drawLabelBox(lines, color = "#3fb47d") {
  const size = Math.max(16, Math.round(overlay.width / 40));
  ctx.font = `600 ${size}px system-ui, sans-serif`;
  const pad = size * 0.5, lineH = size * 1.3;
  const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + pad * 2;
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.fillRect(0, 0, w, lineH * lines.length + pad);
  ctx.fillStyle = color;
  lines.forEach((l, i) => ctx.fillText(l, pad, lineH * (i + 1)));
}

// ---- workout

let drawingUtils = null;

function workoutFrame() {
  if (!state.landmarker) return;
  const result = state.landmarker.detectForVideo(video, performance.now());
  clearOverlay();
  const landmarks = result.landmarks?.[0];
  if (landmarks) {
    drawingUtils ??= new DrawingUtils(ctx);
    const r = Math.max(2, overlay.width / 300);
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: "#ffffff", lineWidth: r });
    drawingUtils.drawLandmarks(landmarks, { color: "#e6550d", radius: r });
    if (!video.paused) state.counter.update(landmarks, state.exercise);
  }
  const lines = [`${state.exercise}: ${state.counter.counts[state.exercise]}`];
  if (!landmarks) lines.push("No person detected");
  drawLabelBox(lines);
  renderPanel();
  countFps();
}

// ---- yoga

const prepCanvas = new OffscreenCanvas(INPUT, INPUT);
const prepCtx = prepCanvas.getContext("2d", { willReadFrequently: true });

function preprocess() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const r = Math.min(INPUT / vw, INPUT / vh);
  const nw = Math.round(vw * r), nh = Math.round(vh * r);
  const left = Math.floor((INPUT - nw) / 2), top = Math.floor((INPUT - nh) / 2);
  prepCtx.fillStyle = "rgb(114,114,114)";
  prepCtx.fillRect(0, 0, INPUT, INPUT);
  prepCtx.drawImage(video, left, top, nw, nh);
  const { data } = prepCtx.getImageData(0, 0, INPUT, INPUT);
  const n = INPUT * INPUT, input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    input[i] = data[i * 4] / 255;
    input[n + i] = data[i * 4 + 1] / 255;
    input[2 * n + i] = data[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", input, [1, 3, INPUT, INPUT]), r, left, top };
}

// Output rows: x1, y1, x2, y2, confidence, class, then 17 keypoints × (x, y, visibility), in model input pixels.
function decode(output, { r, left, top }) {
  const [, rows, cols] = output.dims, d = output.data, dets = [];
  for (let i = 0; i < rows; i++) {
    const o = i * cols, conf = d[o + 4];
    if (conf < MIN_BOX_CONF) continue;
    const kpts = [];
    for (let k = 0; k < 17; k++) {
      const b = o + 6 + k * 3;
      kpts.push({ x: (d[b] - left) / r, y: (d[b + 1] - top) / r, v: d[b + 2] });
    }
    dets.push({
      box: [(d[o] - left) / r, (d[o + 1] - top) / r, (d[o + 2] - left) / r, (d[o + 3] - top) / r],
      confidence: conf, cls: Math.round(d[o + 5]), pose: YOGA_POSES[Math.round(d[o + 5])], kpts,
    });
  }
  return dets;
}

function drawDetections(dets) {
  clearOverlay();
  const lw = Math.max(2, overlay.width / 300);
  const size = Math.max(14, Math.round(overlay.width / 45));
  ctx.font = `600 ${size}px system-ui, sans-serif`;
  for (const det of dets) {
    const color = POSE_COLORS[det.cls] ?? "#ffffff";
    const [x1, y1, x2, y2] = det.box;
    ctx.lineWidth = lw;
    ctx.strokeStyle = color;
    for (const [a, b] of SKELETON) {
      const p = det.kpts[a], q = det.kpts[b];
      if (p.v < 0.5 || q.v < 0.5) continue;
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
    }
    ctx.fillStyle = color;
    for (const p of det.kpts) if (p.v >= 0.5) { ctx.beginPath(); ctx.arc(p.x, p.y, lw * 1.5, 0, Math.PI * 2); ctx.fill(); }
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${det.pose} ${det.confidence.toFixed(2)}`;
    const tw = ctx.measureText(label).width + size * 0.6;
    const ly = Math.max(y1, size * 1.4);
    ctx.fillRect(x1, ly - size * 1.4, tw, size * 1.4);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x1 + size * 0.3, ly - size * 0.35);
  }
}

async function yogaFrame() {
  if (!state.session || state.yogaBusy) return;
  state.yogaBusy = true;
  try {
    const prep = preprocess();
    const t = currentTime();
    const out = await state.session.run({ [state.session.inputNames[0]]: prep.tensor });
    if (state.mode !== "yoga" || !state.source) return;
    const dets = decode(out[state.session.outputNames[0]], prep);
    if (!video.paused && !video.seeking) {
      state.timer.update(dets.map(({ pose, confidence }) => ({ pose, confidence })), t);
      state.lastYogaTime = t;
    }
    drawDetections(dets);
    countFps();
    renderPanel(dets.filter((d) => d.confidence > 0.8).map((d) => d.pose));
  } catch (err) {
    setStatus(`Yoga detection failed: ${err.message}`, true);
  } finally {
    state.yogaBusy = false;
  }
}

// ------------------------------------------------------------------ wiring

buildCards();
renderPanel();

for (const t of document.querySelectorAll("[role=tab]")) t.addEventListener("click", () => setMode(t.dataset.mode));
$("btn-webcam").addEventListener("click", startWebcam);
$("btn-stop").addEventListener("click", () => { stopSource(); setStatus("Stopped."); });
$("file-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) startFile(URL.createObjectURL(file));
  e.target.value = "";
});
$("btn-reset-workout").addEventListener("click", () => { state.counter.reset(); renderPanel(); });
$("btn-reset-yoga").addEventListener("click", () => { state.timer.reset(); renderPanel(); });
// Seeking would make video time jump, so close running timers at the last frame we processed.
video.addEventListener("seeking", () => state.lastYogaTime != null && state.timer.update([], state.lastYogaTime));
video.addEventListener("ended", () => { state.timer.update([], video.duration); renderPanel(); setStatus("Video finished."); });
window.addEventListener("resize", () => video.videoWidth && fitOverlay());

// Optional: ?mode=yoga&video=<url> opens a tab and plays a video from a URL (handy for demos).
const params = new URLSearchParams(location.search);
if (params.get("mode") === "yoga") setMode("yoga");
else ensureModel().catch(() => {});
if (params.get("video")) startFile(params.get("video"), true);
setInterval(() => {
  if (state.source && !statusEl.classList.contains("error") && (state.landmarker || state.session)) {
    const which = state.mode === "workout" ? "Counting" : "Detecting poses";
    const backend = state.mode === "yoga" && state.backend ? ` (${state.backend})` : "";
    statusEl.textContent = `${which} · ${state.fps.toFixed(0)} fps${backend}`;
  }
}, 1000);
