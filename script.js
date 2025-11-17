const INPUT_WIDTH = 256;
const INPUT_HEIGHT = 256;

const NORMALIZE = true;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const video = document.getElementById("video");
const captureCanvas = document.getElementById("captureCanvas");
const heatmapCanvas = document.getElementById("heatmapCanvas");
const captureBtn = document.getElementById("captureBtn");
const statusEl = document.getElementById("status");
const modelDropZone = document.getElementById("modelDropZone");
const modelFileInput = document.getElementById("modelFileInput");
const modelInfo = document.getElementById("modelInfo");

const captureCtx = captureCanvas.getContext("2d");
const heatmapCtx = heatmapCanvas.getContext("2d");

let ortSession = null;
let inputName = null;
let outputName = null;
let cameraReady = false;

(async function init() {
  try {
    await initCamera();
    cameraReady = true;
    updateStatus();
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error initializing camera: " + err.message;
  }
})();

function updateStatus() {
  if (!cameraReady) {
    statusEl.textContent = "Initializing camera...";
  } else if (!ortSession) {
    statusEl.textContent = "Camera ready. Please load an ONNX model.";
  } else {
    statusEl.textContent = "Camera and model ready. Capture when you like.";
  }
}

async function initCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("getUserMedia is not supported in this browser.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false
  });

  video.srcObject = stream;

  await new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resolve();
    };
  });
}

modelDropZone.addEventListener("click", () => {
  modelFileInput.click();
});

modelFileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (file) {
    await loadModelFromFile(file);
  }
});

modelDropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  modelDropZone.classList.add("dragover");
});

modelDropZone.addEventListener("dragleave", (e) => {
  e.preventDefault();
  modelDropZone.classList.remove("dragover");
});

modelDropZone.addEventListener("drop", async (e) => {
  e.preventDefault();
  modelDropZone.classList.remove("dragover");

  const file = e.dataTransfer.files[0];
  if (file && file.name.toLowerCase().endsWith(".onnx")) {
    await loadModelFromFile(file);
  } else if (file) {
    alert("Please drop a .onnx file.");
  }
});

async function loadModelFromFile(file) {
  try {
    captureBtn.disabled = true;
    modelInfo.textContent = `Loading model: ${file.name}…`;
    statusEl.textContent = "Loading ONNX model into onnxruntime-web…";

    if (!window.ort) {
      throw new Error("onnxruntime-web (ort) is not loaded.");
    }

    const buffer = await file.arrayBuffer();
    // ort.InferenceSession.create can take an ArrayBuffer / Uint8Array
    ortSession = await ort.InferenceSession.create(buffer);

    inputName = ortSession.inputNames[0];
    outputName = ortSession.outputNames[0];

    console.log("Model loaded. Input:", inputName, "Output:", outputName);

    modelInfo.textContent = `Loaded: ${file.name}`;
    updateStatus();

    if (cameraReady && ortSession) {
      captureBtn.disabled = false;
    }
  } catch (err) {
    console.error(err);
    modelInfo.textContent = "Error loading model.";
    statusEl.textContent = "Error loading model: " + err.message;
  }
}

captureBtn.addEventListener("click", async () => {
  if (!ortSession) {
    alert("Load an ONNX model first.");
    return;
  }
  if (!cameraReady) {
    alert("Camera not ready.");
    return;
  }

  try {
    captureBtn.disabled = true;
    statusEl.textContent = "Capturing frame and running inference…";

    captureCtx.drawImage(video, 0, 0, INPUT_WIDTH, INPUT_HEIGHT);

    const imageData = captureCtx.getImageData(0, 0, INPUT_WIDTH, INPUT_HEIGHT);
    const inputTensor = imageDataToTensor(imageData);

    const feeds = {};
    feeds[inputName] = inputTensor;
    const results = await ortSession.run(feeds);

    const outputTensor = results[outputName];
    const logits = outputTensor.data;       // Float32Array
    const [n, c, h, w] = outputTensor.dims; // Expect [1,1,H,W] or [1,2,H,W]

    if (!(n === 1 && (c === 1 || c === 2))) {
      alert("Unexpected output shape:", outputTensor.dims);
      return;
    }

    let channelData = logits;
    if (c === 2) {
      const size = h * w;
      const foreground = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        foreground[i] = logits[size + i]; // second channel
      }
      channelData = foreground;
    }

    const size = h * w;
    const probs = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      const x = channelData[i];
      probs[i] = 1 / (1 + Math.exp(-x));
    }

    renderHeatmap(probs, h, w);

    statusEl.textContent = "Done. Capture again whenever you like.";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Inference error: " + err.message;
  } finally {
    if (cameraReady && ortSession) {
      captureBtn.disabled = false;
    }
  }
});

function imageDataToTensor(imageData) {
  const { width, height, data } = imageData; // Uint8ClampedArray [R,G,B,A,...]
  const size = width * height;
  const floatData = new Float32Array(3 * size);

  for (let i = 0; i < size; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    if (NORMALIZE) {
      floatData[i] = (r - MEAN[0]) / STD[0];             // R
      floatData[i + size] = (g - MEAN[1]) / STD[1];      // G
      floatData[i + 2 * size] = (b - MEAN[2]) / STD[2];  // B
    } else {
      floatData[i] = r;
      floatData[i + size] = g;
      floatData[i + 2 * size] = b;
    }
  }

  return new ort.Tensor("float32", floatData, [1, 3, height, width]);
}

function renderHeatmap(probs, height, width) {
  if (heatmapCanvas.width !== width || heatmapCanvas.height !== height) {
    heatmapCanvas.width = width;
    heatmapCanvas.height = height;
  }

  const imageData = heatmapCtx.createImageData(width, height);
  const out = imageData.data;

  for (let y = 0; y < height; y++) {

    for (let x = 0; x < width; x++) {

      const idx = y * width + x;
      const p = probs[idx];

      let r, g, b;
      if (p < 0.5) {
        const t = p / 0.5;
        r = 255 * t;
        g = 0;
        b = 255;
      } else {
        const t = (p - 0.5) / 0.5;
        r = 255;
        g = 0;
        b = 255 * (1 - t);
      }

      const outIdx = idx * 4;

      out[outIdx + 0] = r;
      out[outIdx + 1] = g;
      out[outIdx + 2] = b;
      out[outIdx + 3] = 255; // alpha
    }
  }

  heatmapCtx.putImageData(imageData, 0, 0);
}
