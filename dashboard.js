const statusElement = document.querySelector("#connection-status");
const toastElement = document.querySelector("#toast");
const captureList = document.querySelector("#captures");
const modelList = document.querySelector("#models");
const modeButtons = [...document.querySelectorAll(".mode-button")];
const previewCanvas = document.querySelector("#model-preview");

let session;
let socket;
let selectedModelId = null;
let toastTimer;
let rendererState;
let previewRequest = 0;

function showToast(message, isError = false) {
  toastElement.textContent = message;
  toastElement.style.borderColor = isError ? "#ff8c8c88" : "";
  toastElement.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastElement.classList.remove("show"), 3600);
}

function sessionQuery(extra = {}) {
  return new URLSearchParams({ session_id: session.id, token: session.token, ...extra });
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let result;
  try {
    result = await response.json();
  } catch {
    result = {};
  }
  if (!response.ok) {
    throw new Error(result.detail || `İstek başarısız (${response.status}).`);
  }
  return result;
}

function setConnection(connected) {
  statusElement.classList.toggle("online", connected);
  statusElement.lastChild.textContent = connected ? " Canlı bağlantı" : " Bağlantı kesildi";
}

function connectDashboard() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${scheme}//${location.host}/ws/dashboard/${session.id}?token=${encodeURIComponent(session.token)}`);
  socket.addEventListener("open", () => setConnection(true));
  socket.addEventListener("close", () => {
    setConnection(false);
    setTimeout(() => {
      if (session) connectDashboard();
    }, 1800);
  });
  socket.addEventListener("message", (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "capture.accepted") refreshCaptures();
      if (message.type === "capture.processed") {
        refreshCaptures();
        showToast("Çekim kaydedildi ve küçük önizlemesi hazır.");
      }
      if (message.type === "capture.error") showToast(message.message, true);
    } catch {
      showToast("Sunucudan beklenmeyen bir WebSocket yanıtı alındı.", true);
    }
  });
}

function captureImageUrl(capture, thumbnail = true) {
  const resource = thumbnail && capture.thumbnail ? capture.thumbnail_url : capture.image_url;
  const params = sessionQuery();
  return `${resource}?${params}`;
}

function renderCaptures(items) {
  document.querySelector("#capture-count").textContent = `${items.length} kayıt`;
  captureList.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Aramanla eşleşen çekim yok.";
    captureList.append(empty);
    return;
  }
  for (const capture of items) {
    const row = document.createElement("article");
    row.className = "capture-item";
    const image = document.createElement("img");
    image.alt = "Çekim önizlemesi";
    image.src = captureImageUrl(capture);
    image.loading = "lazy";
    const info = document.createElement("div");
    info.className = "capture-meta";
    const name = document.createElement("strong");
    name.textContent = capture.filename;
    const detail = document.createElement("small");
    detail.textContent = `${capture.mode.toUpperCase()} · ${new Date(capture.captured_at).toLocaleString("tr-TR")} · ${capture.width}×${capture.height}`;
    info.append(name, detail);
    const badge = document.createElement("span");
    badge.className = "status-tag";
    badge.textContent = capture.status === "stored" ? "Hazır" : capture.status === "error" ? "Hata" : "İşleniyor";
    row.append(image, info, badge);
    captureList.append(row);
  }
}

async function refreshCaptures() {
  if (!session) return;
  const search = document.querySelector("#capture-search").value.trim();
  const params = sessionQuery({ q: search, mode: "all" });
  const result = await api(`/api/captures?${params}`);
  renderCaptures(result.captures);
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function renderModels(items) {
  modelList.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Henüz STL modeli eklenmedi.";
    modelList.append(empty);
    return;
  }
  for (const model of items) {
    const row = document.createElement("div");
    row.className = `model-item${model.id === selectedModelId ? " active" : ""}`;
    const filename = document.createElement("button");
    filename.className = "filename";
    filename.type = "button";
    filename.textContent = `${model.original_name} · ${formatBytes(model.size_bytes)}`;
    filename.title = "3B önizleme için seç";
    filename.addEventListener("click", () => selectModel(model));
    const slicer = document.createElement("button");
    slicer.className = "button secondary";
    slicer.type = "button";
    slicer.textContent = "Slicer";
    slicer.addEventListener("click", () => openInSlicer(model));
    const tinkercad = document.createElement("button");
    tinkercad.className = "button secondary";
    tinkercad.type = "button";
    tinkercad.textContent = "Tinkercad";
    tinkercad.addEventListener("click", () => openInTinkercad(model));
    row.append(filename, slicer, tinkercad);
    modelList.append(row);
  }
}

async function refreshModels() {
  if (!session) return;
  const result = await api(`/api/models?${sessionQuery()}`);
  renderModels(result.models);
  if (selectedModelId && !result.models.some((model) => model.id === selectedModelId)) {
    selectedModelId = null;
    document.querySelector("#preview-label").textContent = "Model seçilmedi";
  }
}

async function modelData(model) {
  const response = await fetch(`/api/models/${model.id}/download?${sessionQuery()}`);
  if (!response.ok) {
    const result = await response.json().catch(() => ({}));
    throw new Error(result.detail || "STL dosyası alınamadı.");
  }
  return response.arrayBuffer();
}

async function selectModel(model) {
  selectedModelId = model.id;
  const requestId = ++previewRequest;
  renderModels((await api(`/api/models?${sessionQuery()}`)).models);
  document.querySelector("#preview-label").textContent = model.original_name;
  const message = document.querySelector("#preview-message");
  message.textContent = "3B önizleme yükleniyor…";
  try {
    const data = await modelData(model);
    if (requestId !== previewRequest) return;
    await renderStl(data);
    message.textContent = "Sürükleyerek modeli döndürebilirsin.";
  } catch (error) {
    message.textContent = "3B önizleme açılamadı. İnternet bağlantısını kontrol edip yeniden dene.";
    showToast(error.message, true);
  }
}

async function renderStl(data) {
  const THREE = await import("three");
  const [{ STLLoader }, { OrbitControls }] = await Promise.all([
    import("https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/loaders/STLLoader.js"),
    import("https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js"),
  ]);
  if (!rendererState) {
    const renderer = new THREE.WebGLRenderer({ canvas: previewCanvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x0a1120);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100000);
    scene.add(new THREE.HemisphereLight(0xcfe8ff, 0x263044, 2.2));
    const light = new THREE.DirectionalLight(0xffffff, 2.6);
    light.position.set(1, 2, 3);
    scene.add(light);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    rendererState = { renderer, scene, camera, controls, mesh: null, THREE };
    const resize = () => {
      const width = Math.max(1, previewCanvas.clientWidth);
      const height = Math.max(1, previewCanvas.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    new ResizeObserver(resize).observe(previewCanvas);
    resize();
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
  }
  const loader = new STLLoader();
  const geometry = loader.parse(data);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  const bounds = geometry.boundingBox;
  const center = bounds.getCenter(new THREE.Vector3());
  geometry.translate(-center.x, -center.y, -center.z);
  const size = bounds.getSize(new THREE.Vector3());
  const largest = Math.max(size.x, size.y, size.z, 1);
  const material = new THREE.MeshStandardMaterial({ color: 0x6ee7c0, metalness: 0.16, roughness: 0.55 });
  const mesh = new THREE.Mesh(geometry, material);
  if (rendererState.mesh) {
    rendererState.scene.remove(rendererState.mesh);
    rendererState.mesh.geometry.dispose();
    rendererState.mesh.material.dispose();
  }
  rendererState.mesh = mesh;
  rendererState.scene.add(mesh);
  rendererState.camera.position.set(largest * 1.35, largest * 1.1, largest * 1.6);
  rendererState.camera.near = largest / 1000;
  rendererState.camera.far = largest * 100;
  rendererState.camera.updateProjectionMatrix();
  rendererState.controls.target.set(0, 0, 0);
  rendererState.controls.update();
}

async function openInSlicer(model) {
  try {
    const result = await api(`/api/models/${model.id}/open-slicer?${sessionQuery()}`, { method: "POST" });
    showToast(`${result.slicer} açıldı.`);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function openInTinkercad(model) {
  try {
    const data = await modelData(model);
    const blob = new Blob([data], { type: "application/sla" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = model.original_name.toLowerCase().endsWith(".stl") ? model.original_name : `${model.id}.stl`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    const result = await api(`/api/models/${model.id}/open-tinkercad?${sessionQuery()}`, { method: "POST" });
    showToast(result.message);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function setMode(mode) {
  try {
    const query = new URLSearchParams({ token: session.token });
    await api(`/api/sessions/${session.id}/mode?${query}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    modeButtons.forEach((button) => button.classList.toggle("selected", button.dataset.mode === mode));
    showToast(mode === "2d" ? "2D belge tarama modu seçildi." : "3D obje tarama modu seçildi.");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function uploadModel(file) {
  const form = new FormData();
  form.append("file", file);
  const query = sessionQuery();
  try {
    await api(`/api/models?${query}`, { method: "POST", body: form });
    showToast("STL modeli yerel kitaplığa eklendi.");
    await refreshModels();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    document.querySelector("#model-upload").value = "";
  }
}

async function start() {
  try {
    session = await api("/api/sessions", { method: "POST" });
    document.querySelector("#session-qr").src = session.qr_url;
    const url = document.querySelector("#mobile-url");
    url.textContent = session.mobile_url;
    document.querySelector("#copy-url").disabled = false;
    connectDashboard();
    await Promise.all([refreshCaptures(), refreshModels()]);
  } catch (error) {
    setConnection(false);
    showToast(error.message, true);
  }
}

document.querySelector("#copy-url").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(session.mobile_url);
    showToast("Telefon bağlantısı kopyalandı.");
  } catch {
    showToast("Bağlantı kopyalanamadı; QR kodunu okut.", true);
  }
});
document.querySelector("#capture-search").addEventListener("input", () => {
  clearTimeout(window.captureSearchTimer);
  window.captureSearchTimer = setTimeout(() => refreshCaptures().catch((error) => showToast(error.message, true)), 180);
});
modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
document.querySelector("#model-upload").addEventListener("change", (event) => {
  if (event.target.files?.[0]) uploadModel(event.target.files[0]);
});

start();
