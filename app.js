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

class ModelManager {
  constructor(storage) {
    this.storage = storage;
    this.activeModelId = null;
    this.session = null;
    this.inputShape = [1, 3, 224, 224]; // adjust to your real model
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
      executionProviders: ['wasm']
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
    const offscreen = new OffscreenCanvas(w, h);
    const ctx = offscreen.getContext('2d');
    ctx.putImageData(imageData, 0, 0);
    const resized = ctx.getImageData(0, 0, w, h);
    const data = resized.data;

    const floatData = new Float32Array(n * c * h * w);
    let idx = 0;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] / 255;
      const g = data[i + 1] / 255;
      const b = data[i + 2] / 255;
      floatData[idx] = r;
      floatData[idx + h * w] = g;
      floatData[idx + 2 * h * w] = b;
      idx++;
    }

    const inputName = this.session.inputNames[0];
    const tensor = new ort.Tensor('float32', floatData, this.inputShape);
    const feeds = {};
    feeds[inputName] = tensor;

    const results = await this.session.run(feeds);
    const output = results[this.outputName];
    return this._heatmapFromOutput(output);
  }

  _heatmapFromOutput(output) {
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
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(w, h);
    const arr = imgData.data;

    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;

    for (let i = 0; i < data.length; i++) {
      const norm = (data[i] - min) / range;
      const v = Math.max(0, Math.min(1, norm));
      const r = v * 255;
      const g = 0;
      const b = (1 - v) * 255;

      const k = i * 4;
      arr[k] = r;
      arr[k + 1] = g;
      arr[k + 2] = b;
      arr[k + 3] = Math.round(180 + v * 75);
    }

    ctx.putImageData(imgData, 0, 0);
    return canvas.convertToBlob({ type: 'image/png' });
  }
}

class CameraController {
  constructor() {
    this.stream = null;
    this.video = document.getElementById('camera-video');
    this.overlay = document.getElementById('camera-overlay');
    this.orientationIndicator = document.getElementById('orientation-indicator');
    this.allowedToCapture = false;
    this.orientationHandler = this._onOrientation.bind(this);
    this.hasIMU = false;
  }

  async start() {
    if (this.stream) return;
    const constraints = {
      video: {
        facingMode: 'environment'
      },
      audio: false
    };
    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.video.srcObject = this.stream;
    await this.video.play();
    this._setupOverlay();
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

  _setupOverlay() {
    const canvas = this.overlay;
    const ctx = canvas.getContext('2d');
    const resize = () => {
      canvas.width = this.video.clientWidth || this.video.videoWidth || 640;
      canvas.height = this.video.clientHeight || this.video.videoHeight || 480;
      this._drawOverlay();
    };
    resize();
    window.addEventListener('resize', resize);
  }

  _drawOverlay() {
    const canvas = this.overlay;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const size = Math.min(canvas.width, canvas.height) * 0.7;
    const x = (canvas.width - size) / 2;
    const y = (canvas.height - size) / 2;
    ctx.strokeStyle = 'rgba(248, 250, 252, 0.8)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(x, y, size, size);
    ctx.setLineDash([]);
  }

  _setupOrientation() {
    if (typeof DeviceOrientationEvent === 'undefined') {
      this.allowedToCapture = true;
      this.orientationIndicator.textContent = 'Sensor unavailable. Align visually and capture.';
      this.orientationIndicator.classList.add('good');
      return;
    }

    const requestPermission = async () => {
      if (typeof DeviceOrientationEvent.requestPermission === 'function') {
        try {
          const perm = await DeviceOrientationEvent.requestPermission();
          if (perm !== 'granted') return;
        } catch {
          return;
        }
      }
      this.hasIMU = true;
      window.addEventListener('deviceorientation', this.orientationHandler);
    };

    requestPermission();
  }

  _onOrientation(e) {
    const beta = e.beta;
    if (beta == null) {
      this.allowedToCapture = true;
      this.orientationIndicator.textContent = 'Align visually and capture.';
      this.orientationIndicator.classList.add('good');
      return;
    }
    const off = beta - 90;
    const tilt = Math.abs(off);
    const threshold = 8;
    if (tilt <= threshold) {
      this.allowedToCapture = true;
      this.orientationIndicator.textContent = 'Aligned. Tap capture.';
      this.orientationIndicator.classList.add('good');
    } else {
      this.allowedToCapture = false;
      this.orientationIndicator.textContent = 'Tilt device until aligned.';
      this.orientationIndicator.classList.remove('good');
    }
  }

  canCapture() {
    return this.allowedToCapture;
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

    if (w / h > aspect) {
      const targetW = h * aspect;
      const offsetX = (w - targetW) / 2;
      canvas.width = targetW;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, offsetX, 0, targetW, h, 0, 0, targetW, h);
    } else {
      const targetH = w / aspect;
      const offsetY = (h - targetH) / 2;
      canvas.width = w;
      canvas.height = targetH;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, offsetY, w, targetH, 0, 0, w, targetH);
    }

    const ctx = canvas.getContext('2d');
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return { canvas, imageData: imgData };
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

    setInterval(() => {
      this.btnDoCapture.disabled = !this.camera.canCapture();
    }, 200);
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
    try {
      await this.camera.start();
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
    await this.ui.refreshModels();
    await this.ui.refreshCaptures();
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const app = new App();
  app.start().catch(console.error);
});
