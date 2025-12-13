const DB_NAME = 'artifact_sifter_db';
const DB_VERSION = 1;
const STORE_MODELS = 'models';
const STORE_CAPTURES = 'captures';

const PHASE_BROWSE = 'browse';
const PHASE_CAPTURE = 'capture';
const PHASE_PROCESSING = 'processing';

class StorageManager {
  constructor() {
    this.db = null;
  }

  init() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = e => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(STORE_MODELS)) {
          const store = db.createObjectStore(STORE_MODELS, { keyPath: 'id', autoIncrement: true });
          store.createIndex('byName', 'name', { unique: false });
        }
        if (!db.objectStoreNames.contains(STORE_CAPTURES)) {
          const store = db.createObjectStore(STORE_CAPTURES, { keyPath: 'id', autoIncrement: true });
          store.createIndex('byTimestamp', 'timestamp', { unique: false });
        }
      };
      req.onsuccess = e => {
        this.db = e.target.result;
        resolve();
      };
      req.onerror = () => reject(req.error);
    });
  }

  addModel(name, blob, url) {
    const tx = this.db.transaction(STORE_MODELS, 'readwrite');
    const store = tx.objectStore(STORE_MODELS);
    const obj = {
      name,
      createdAt: Date.now(),
      sourceUrl: url || null,
      blob
    };
    return new Promise((resolve, reject) => {
      const req = store.add(obj);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  listModels() {
    const tx = this.db.transaction(STORE_MODELS, 'readonly');
    const store = tx.objectStore(STORE_MODELS);
    return new Promise((resolve, reject) => {
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result.sort((a, b) => a.createdAt - b.createdAt));
      req.onerror = () => reject(req.error);
    });
  }

  getModel(id) {
    const tx = this.db.transaction(STORE_MODELS, 'readonly');
    const store = tx.objectStore(STORE_MODELS);
    return new Promise((resolve, reject) => {
      const req = store.get(id);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  }

  deleteModel(id) {
    const tx = this.db.transaction(STORE_MODELS, 'readwrite');
    const store = tx.objectStore(STORE_MODELS);
    return new Promise((resolve, reject) => {
      const req = store.delete(id);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  }

  addCapture(capture) {
    const tx = this.db.transaction(STORE_CAPTURES, 'readwrite');
    const store = tx.objectStore(STORE_CAPTURES);
    return new Promise((resolve, reject) => {
      const req = store.add(capture);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  listCaptures() {
    const tx = this.db.transaction(STORE_CAPTURES, 'readonly');
    const store = tx.objectStore(STORE_CAPTURES);
    return new Promise((resolve, reject) => {
      const req = store.getAll();
      req.onsuccess = () => {
        const arr = req.result || [];
        arr.sort((a, b) => b.timestamp - a.timestamp);
        resolve(arr);
      };
      req.onerror = () => reject(req.error);
    });
  }

  getCapture(id) {
    const tx = this.db.transaction(STORE_CAPTURES, 'readonly');
    const store = tx.objectStore(STORE_CAPTURES);
    return new Promise((resolve, reject) => {
      const req = store.get(id);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  }
}

const MAGMA_LUT = [
  [0, 0, 0], [1, 0, 1], [2, 1, 3], [4, 1, 6], [6, 2, 9], [8, 3, 12],
  [11, 4, 16], [14, 5, 20], [17, 6, 25], [20, 7, 30], [24, 8, 35],
  [28, 9, 41], [32, 10, 47], [36, 11, 54], [40, 12, 60],
  [44, 13, 67], [48, 14, 74], [52, 15, 81], [56, 16, 88],
  [60, 17, 95], [64, 18, 102], [68, 19, 109], [72, 20, 116],
  [76, 21, 123], [80, 22, 130], [84, 23, 137], [88, 24, 144],
  [92, 25, 151], [96, 26, 158], [100, 27, 165], [104, 28, 172],
  [108, 29, 179], [112, 30, 186], [116, 31, 193], [120, 32, 200],
  [124, 33, 207], [128, 34, 214], [132, 35, 221], [136, 36, 228],
  [140, 37, 235], [144, 38, 242], [148, 39, 249], [252, 252, 252]
];

class ModelManager {
  constructor(storage) {
    this.storage = storage;
    this.activeModelId = null;
    this.session = null;
    this.inputShape = [1, 3, 512, 512]; // adjust to your real model
    this.outputName = null;
  }

  async loadModelsFromDB() {
    return this.storage.listModels();
  }

  async setActiveModel(id) {
    this.activeModelId = id;
    this.session = null;
    if (!id) return;
    const modelObj = await this.storage.getModel(id);
    if (!modelObj) {
      this.activeModelId = null;
      return;
    }
    const arrayBuffer = await modelObj.blob.arrayBuffer();
    this.session = await ort.InferenceSession.create(arrayBuffer, {
      executionProviders: ['webgl', 'wasm']
    });
    if (!this.outputName) {
      this.outputName = this.session.outputNames[0];
    }
  }

  async addModelByUrl(name, url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to download model');
    const blob = await res.blob();
    const id = await this.storage.addModel(name, blob, url);
    return id;
  }

  async runInferenceFromImageData(imageData) {
    if (!this.session) return null;

    const [n, c, h, w] = this.inputShape;

    // Canvas containing the original capture
    const srcCanvas = new OffscreenCanvas(imageData.width, imageData.height);
    const srcCtx = srcCanvas.getContext('2d');
    srcCtx.putImageData(imageData, 0, 0);

    // 1) Center-crop to square
    const srcW = imageData.width;
    const srcH = imageData.height;
    const side = Math.min(srcW, srcH);
    const offsetX = Math.floor((srcW - side) / 2);
    const offsetY = Math.floor((srcH - side) / 2);

    const cropCanvas = new OffscreenCanvas(side, side);
    const cropCtx = cropCanvas.getContext('2d');
    cropCtx.drawImage(srcCanvas, offsetX, offsetY, side, side, 0, 0, side, side);

    // 2) Resize square crop to model input (512x512)
    const resizeCanvas = new OffscreenCanvas(w, h);
    const resizeCtx = resizeCanvas.getContext('2d');
    resizeCtx.drawImage(cropCanvas, 0, 0, w, h);
    const resized = resizeCtx.getImageData(0, 0, w, h);
    const data = resized.data;

    // HWC u8 -> CHW f32
    const floatData = new Float32Array(n * c * h * w);
    let idx = 0;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] / 255;
      const g = data[i + 1] / 255;
      const b = data[i + 2] / 255;

      const mean = [0.485, 0.456, 0.406];
      const std  = [0.229, 0.224, 0.225];

      floatData[idx]             = (r - mean[0]) / std[0];
      floatData[idx + h * w]     = (g - mean[1]) / std[1];
      floatData[idx + 2 * h * w] = (b - mean[2]) / std[2];

      idx++;
    }

    const inputName = this.session.inputNames[0];
    const tensor = new ort.Tensor('float32', floatData, this.inputShape);
    const feeds = { [inputName]: tensor };

    const results = await this.session.run(feeds);
    const output = results[this.outputName];

    // Build a heatmap aligned to the ORIGINAL capture:
    // draw the heatmap into the center-square region, with bars around it.
    return this._heatmapFromOutput(output, {
      targetWidth: srcW,
      targetHeight: srcH,
      cropSide: side,
      cropOffsetX: offsetX,
      cropOffsetY: offsetY
    });
  }

  _heatmapFromOutput(output, align) {
    const dims = output.dims;
    let h, w;
    if (dims.length === 4) {
      h = dims[2];
      w = dims[3];
    } else if (dims.length === 3) {
      h = dims[1];
      w = dims[2];
    } else if (dims.length === 2) {
      h = 1;
      w = dims[1];
    } else {
      throw new Error('Unsupported output shape');
    }

    const data = output.data;

    // Build base heatmap at output resolution
    const baseCanvas = new OffscreenCanvas(w, h);
    const ctx = baseCanvas.getContext('2d');
    const imgData = ctx.createImageData(w, h);
    const arr = imgData.data;

    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      const v = x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
      const idx = Math.floor(v * (MAGMA_LUT.length - 1));
      const [r, g, b] = MAGMA_LUT[idx]

      const k = i * 4;
      arr[k] = r;
      arr[k + 1] = g;
      arr[k + 2] = b;
      arr[k + 3] = 255; // fully opaque
    }

    ctx.putImageData(imgData, 0, 0);

    // If no alignment requested, just return base heatmap blob
    if (!align) {
      return baseCanvas.convertToBlob({ type: 'image/png' });
    }

    const { targetWidth, targetHeight, cropSide, cropOffsetX, cropOffsetY } = align;

    // Create a full-size heatmap matching the original capture.
    // Fill bars with opaque black (or change to any neutral you want).
    const outCanvas = new OffscreenCanvas(targetWidth, targetHeight);
    const outCtx = outCanvas.getContext('2d');
    outCtx.fillStyle = 'rgba(0,0,0,1)';
    outCtx.fillRect(0, 0, targetWidth, targetHeight);

    // Paste the heatmap into the same center-crop square region
    outCtx.imageSmoothingEnabled = false;
    outCtx.drawImage(
      baseCanvas,
      0, 0, w, h,
      cropOffsetX, cropOffsetY, cropSide, cropSide
    );

    return outCanvas.convertToBlob({ type: 'image/png' });
  }
}

class CameraController {
  constructor() {
    this.stream = null;
    this.video = document.getElementById('camera-video');
    this.orientationIndicator = document.getElementById('orientation-indicator');
    this.orientationHandler = this._onOrientation.bind(this);

    this.hasIMU = false;

    // Smoothed sensor values
    this.filteredBeta = null;
    this.filteredGamma = null;
  }

  async start() {
    if (this.stream) return;

    const constraints = {
      video: {
        facingMode: { ideal: 'environment' }
      },
      audio: false
    };

    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.video.srcObject = this.stream;
    await this.video.play();

    this._setupOrientation();
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
      this.video.srcObject = null;
    }
    window.removeEventListener('deviceorientation', this.orientationHandler);
  }

  _setupOrientation() {
    if (typeof DeviceOrientationEvent === 'undefined') {
      this.orientationIndicator.textContent =
        'Align the phone visually over the tray, then tap Capture.';
      this.orientationIndicator.classList.add('good');
      return;
    }

    const requestPermission = async () => {
      if (typeof DeviceOrientationEvent.requestPermission === 'function') {
        try {
          const perm = await DeviceOrientationEvent.requestPermission();
          if (perm !== 'granted') {
            this.orientationIndicator.textContent =
              'Orientation access denied. Align visually and tap Capture.';
            return;
          }
        } catch {
          this.orientationIndicator.textContent =
            'Orientation access blocked. Align visually and tap Capture.';
          return;
        }
      }

      this.hasIMU = true;
      this.orientationIndicator.textContent =
        'Tilt so the phone is flat over the tray, then tap Capture.';
      this.orientationIndicator.classList.remove('good');

      window.addEventListener('deviceorientation', this.orientationHandler);
    };

    requestPermission();
  }

  _onOrientation(e) {
    let beta = e.beta;   // front/back tilt
    let gamma = e.gamma; // left/right tilt

    if (beta == null || gamma == null) {
      this.orientationIndicator.textContent =
        'Align visually over the tray, then tap Capture.';
      this.orientationIndicator.classList.add('good');
      return;
    }

    // Low-pass filter to avoid jitter
    const alpha = 0.2;
    if (this.filteredBeta == null) {
      this.filteredBeta = beta;
      this.filteredGamma = gamma;
    } else {
      this.filteredBeta = this.filteredBeta + alpha * (beta - this.filteredBeta);
      this.filteredGamma = this.filteredGamma + alpha * (gamma - this.filteredGamma);
    }

    beta = this.filteredBeta;
    gamma = this.filteredGamma;

    // For camera facing down, "flat" ≈ beta ≈ 90°, gamma ≈ 0°
    const pitchError = beta - 90; // +: tipped away, -: tipped toward (portrait-ish)
    const roll = gamma;           // + / −: left / right tilt

    const tilt = Math.sqrt(pitchError * pitchError + roll * roll);
    const hints = [];

    // Forward/back hint
    if (pitchError > 6) {
      hints.push('tilt slightly toward you');
    } else if (pitchError < -6) {
      hints.push('tilt slightly away from you');
    }

    // Left/right hint
    if (roll > 6) {
      hints.push('tilt a bit left');
    } else if (roll < -6) {
      hints.push('tilt a bit right');
    }

    let message;
    let good = false;

    if (tilt <= 5) {
      message = 'Phone is flat ✔ Hold steady, then tap Capture.';
      good = true;
    } else if (tilt <= 12) {
      if (hints.length) {
        message = 'Almost flat – ' + hints.join(' and ') + '.';
      } else {
        message = 'Almost flat – tiny adjustment, then Capture.';
      }
      good = true;
    } else {
      if (hints.length) {
        message = 'Adjust: ' + hints.join(' and ') + '.';
      } else {
        message = 'Flatten the phone over the tray before Capture.';
      }
    }

    this.orientationIndicator.textContent = message;
    if (good) {
      this.orientationIndicator.classList.add('good');
    } else {
      this.orientationIndicator.classList.remove('good');
    }
  }

  captureFrame() {
    const canvas = document.createElement('canvas');
    const video = this.video;
    const aspect = 4 / 3;

    let w = video.videoWidth;
    let h = video.videoHeight;

    if (!w || !h) {
      w = 1280;
      h = 720;
    }

    const ctx = canvas.getContext('2d');

    if (w / h > aspect) {
      const targetW = h * aspect;
      const offsetX = (w - targetW) / 2;
      canvas.width = targetW;
      canvas.height = h;
      ctx.drawImage(video, offsetX, 0, targetW, h, 0, 0, targetW, h);
    } else {
      const targetH = w / aspect;
      const offsetY = (h - targetH) / 2;
      canvas.width = w;
      canvas.height = targetH;
      ctx.drawImage(video, 0, offsetY, w, targetH, 0, 0, w, targetH);
    }

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return { canvas, imageData };
  }
}

class UIManager {
  constructor(storage, models, camera) {
    this.storage = storage;
    this.models = models;
    this.camera = camera;

    this.phaseBrowse = document.getElementById('phase-browse');
    this.phaseCapture = document.getElementById('phase-capture');
    this.phaseProcessing = document.getElementById('phase-processing');

    this.btnNewCapture = document.getElementById('btn-new-capture');
    this.btnCancelCapture = document.getElementById('btn-cancel-capture');
    this.btnDoCapture = document.getElementById('btn-do-capture');

    this.modelList = document.getElementById('model-list');
    this.modelNameInput = document.getElementById('model-name-input');
    this.modelUrlInput = document.getElementById('model-url-input');
    this.modelAddForm = document.getElementById('model-add-form');
    this.activeModelName = document.getElementById('active-model-name');

    this.captureList = document.getElementById('capture-list');
    this.captureEmpty = document.getElementById('capture-empty');

    this.viewerModal = document.getElementById('viewer-modal');
    this.viewerClose = document.getElementById('viewer-close');
    this.viewerImage = document.getElementById('viewer-image');
    this.viewerHeatmap = document.getElementById('viewer-heatmap');
    this.viewerHeatmapWrapper = document.getElementById('viewer-heatmap-wrapper');
    this.viewerSlider = document.getElementById('viewer-slider');
    this.viewerTitle = document.getElementById('viewer-title');
    this.viewerTs = document.getElementById('viewer-ts');
    this.viewerModel = document.getElementById('viewer-model');
    this.viewerGps = document.getElementById('viewer-gps');

    this.currentPhase = PHASE_BROWSE;
    this.currentModels = [];
    this.currentCaptures = [];

    this.btnNewCapture.addEventListener('click', () => this.enterCapturePhase());
    this.btnCancelCapture.addEventListener('click', () => this.backToBrowse());
    this.btnDoCapture.addEventListener('click', () => this.captureAndAnalyze());
    this.modelAddForm.addEventListener('submit', e => this.onAddModel(e));
    this.viewerClose.addEventListener('click', () => this.hideViewer());
    this.viewerModal.addEventListener('click', e => {
      if (e.target === this.viewerModal || e.target === this.viewerModal.querySelector('.modal-backdrop')) {
        this.hideViewer();
      }
    });
    this.viewerSlider.addEventListener('input', () => this.updateSliderMask());
  }

  setPhase(phase) {
    this.currentPhase = phase;
    this.phaseBrowse.classList.toggle('active', phase === PHASE_BROWSE);
    this.phaseCapture.classList.toggle('active', phase === PHASE_CAPTURE);
    this.phaseProcessing.classList.toggle('active', phase === PHASE_PROCESSING);
  }

  async refreshModels() {
    this.currentModels = await this.models.loadModelsFromDB();
    this.modelList.innerHTML = '';
    this.currentModels.forEach(model => {
      const li = document.createElement('li');
      li.className = 'list-item';

      const main = document.createElement('div');
      main.className = 'list-item-main';
      const title = document.createElement('div');
      title.className = 'list-item-title';
      title.textContent = model.name;
      const sub = document.createElement('div');
      sub.className = 'list-item-sub';
      sub.textContent = model.sourceUrl || 'Local import';
      main.appendChild(title);
      main.appendChild(sub);

      const actions = document.createElement('div');
      actions.className = 'list-item-actions';

      const btnUse = document.createElement('button');
      btnUse.className = 'btn secondary';
      btnUse.textContent = 'Use';
      btnUse.addEventListener('click', () => this.onSelectModel(model.id));

      const btnDelete = document.createElement('button');
      btnDelete.className = 'btn ghost';
      btnDelete.textContent = 'Delete';
      btnDelete.addEventListener('click', () => this.onDeleteModel(model.id));

      actions.appendChild(btnUse);
      actions.appendChild(btnDelete);

      li.appendChild(main);
      li.appendChild(actions);

      this.modelList.appendChild(li);
    });

    const active = this.currentModels.find(m => m.id === this.models.activeModelId);
    this.activeModelName.textContent = active ? active.name : 'None selected';
  }

  async refreshCaptures() {
    this.currentCaptures = await this.storage.listCaptures();
    this.captureList.innerHTML = '';
    if (!this.currentCaptures.length) {
      this.captureEmpty.style.display = 'block';
      return;
    }
    this.captureEmpty.style.display = 'none';

    this.currentCaptures.forEach(capture => {
      const li = document.createElement('li');
      li.className = 'list-item';

      const main = document.createElement('div');
      main.className = 'list-item-main';
      const title = document.createElement('div');
      title.className = 'list-item-title';
      title.textContent = capture.filename;

      const sub = document.createElement('div');
      sub.className = 'list-item-sub';
      const date = new Date(capture.timestamp);
      const modelName = capture.metadata.modelName || 'None';
      sub.textContent = `${date.toLocaleString()} • ${modelName}`;

      main.appendChild(title);
      main.appendChild(sub);

      li.appendChild(main);
      li.addEventListener('click', () => this.showViewer(capture.id));

      this.captureList.appendChild(li);
    });
  }

  async onAddModel(e) {
    e.preventDefault();
    const name = this.modelNameInput.value.trim();
    const url = this.modelUrlInput.value.trim();
    if (!name || !url) return;

    this.modelAddForm.querySelector('button').disabled = true;
    try {
      const id = await this.models.addModelByUrl(name, url);
      await this.refreshModels();
      this.modelNameInput.value = '';
      this.modelUrlInput.value = '';
      await this.onSelectModel(id);
    } catch (err) {
      alert('Failed to add model: ' + err.message);
    } finally {
      this.modelAddForm.querySelector('button').disabled = false;
    }
  }

  async onSelectModel(id) {
    try {
      await this.models.setActiveModel(id);
      await this.refreshModels();
    } catch (err) {
      alert('Failed to load model: ' + err.message);
    }
  }

  async onDeleteModel(id) {
    if (!confirm('Delete this model permanently?')) return;
    if (this.models.activeModelId === id) {
      this.models.activeModelId = null;
      this.models.session = null;
    }
    await this.storage.deleteModel(id);
    await this.refreshModels();
  }

  async enterCapturePhase() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Camera not supported in this browser.');
      return;
    }
    this.setPhase(PHASE_CAPTURE);
    this.btnDoCapture.disabled = true;
    try {
      await this.camera.start();
      this.btnDoCapture.disabled = false;
    } catch (err) {
      alert('Failed to access camera: ' + err.message);
      this.backToBrowse();
    }
  }

  async backToBrowse() {
    this.camera.stop();
    this.setPhase(PHASE_BROWSE);
    await this.refreshCaptures();
  }

  async captureAndAnalyze() {
    const { canvas, imageData } = this.camera.captureFrame();
    const timestamp = Date.now();
    const dt = new Date(timestamp);

    const yyyy = dt.getFullYear();
    const mm = String(dt.getMonth() + 1).padStart(2, '0');
    const dd = String(dt.getDate()).padStart(2, '0');
    const hh = String(dt.getHours()).padStart(2, '0');
    const mi = String(dt.getMinutes()).padStart(2, '0');
    const ss = String(dt.getSeconds()).padStart(2, '0');
    const filename = `${yyyy}${mm}${dd}_${hh}${mi}${ss}`;

    this.setPhase(PHASE_PROCESSING);
    this.camera.stop();

    let gps = null;
    if ('geolocation' in navigator) {
      try {
        gps = await new Promise((resolve, reject) => {
          navigator.geolocation.getCurrentPosition(
            pos => {
              resolve({
                lat: pos.coords.latitude,
                lon: pos.coords.longitude,
                acc: pos.coords.accuracy
              });
            },
            err => resolve(null),
            { enableHighAccuracy: true, timeout: 4000, maximumAge: 10000 }
          );
        });
      } catch {
        gps = null;
      }
    }

    let imageBlob;
    let heatmapBlob = null;

    imageBlob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));

    const modelMeta = this.currentModels.find(m => m.id === this.models.activeModelId);
    if (this.models.session) {
      try {
        heatmapBlob = await this.models.runInferenceFromImageData(imageData);
      } catch (err) {
        console.error(err);
        alert('Inference failed; storing image only.');
      }
    }

    const metadata = {
      timestamp,
      filename,
      gps,
      modelId: this.models.activeModelId || null,
      modelName: modelMeta ? modelMeta.name : null
    };

    const capture = {
      timestamp,
      filename,
      imageBlob,
      heatmapBlob,
      metadata
    };

    await this.storage.addCapture(capture);
    await this.refreshCaptures();
    this.setPhase(PHASE_BROWSE);
  }

  async showViewer(id) {
    const capture = await this.storage.getCapture(id);
    if (!capture) return;

    const imgUrl = URL.createObjectURL(capture.imageBlob);
    this.viewerImage.src = imgUrl;

    if (capture.heatmapBlob) {
      const hmUrl = URL.createObjectURL(capture.heatmapBlob);
      this.viewerHeatmap.src = hmUrl;
      this.viewerHeatmapWrapper.style.display = 'block';
      this.viewerSlider.disabled = false;
    } else {
      this.viewerHeatmapWrapper.style.display = 'none';
      this.viewerSlider.disabled = true;
    }

    const dt = new Date(capture.timestamp);
    this.viewerTs.textContent = dt.toLocaleString();
    this.viewerModel.textContent = capture.metadata.modelName || 'None';

    if (capture.metadata.gps) {
      const g = capture.metadata.gps;
      this.viewerGps.textContent = `${g.lat.toFixed(6)}, ${g.lon.toFixed(6)} (±${Math.round(g.acc)} m)`;
    } else {
      this.viewerGps.textContent = 'Not recorded';
    }

    this.viewerTitle.textContent = capture.filename;
    this.viewerSlider.value = 50;
    this.updateSliderMask();

    this.viewerModal.classList.remove('hidden');
  }

  hideViewer() {
    if (this.viewerImage.src) URL.revokeObjectURL(this.viewerImage.src);
    if (this.viewerHeatmap.src) URL.revokeObjectURL(this.viewerHeatmap.src);
    this.viewerModal.classList.add('hidden');
  }

  updateSliderMask() {
    const v = this.viewerSlider.value;
    const pct = v / 100;
    const wrapper = this.viewerHeatmapWrapper;
    const width = pct * 100;
    wrapper.style.clipPath = `inset(0 ${100 - width}% 0 0)`;
  }
}

class App {
  constructor() {
    this.storage = new StorageManager();
    this.models = new ModelManager(this.storage);
    this.camera = new CameraController();
    this.ui = new UIManager(this.storage, this.models, this.camera);
  }

  async start() {
    await this.storage.init();

    // Load default Sherd Detector model from relative URL if no models exist yet
    const existingModels = await this.models.loadModelsFromDB();
    if (!existingModels.length) {
      try {
        const id = await this.models.addModelByUrl('Sherd Detector', 'sherd-detector.onnx');
        await this.models.setActiveModel(id);
      } catch (err) {
        console.error('Failed to preload default Sherd Detector model:', err);
      }
    }

    await this.ui.refreshModels();
    await this.ui.refreshCaptures();
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const app = new App();
  app.start().catch(console.error);
});
