const params = new URLSearchParams(location.search);
const sessionId = location.pathname.split("/").filter(Boolean).at(-1);
const token = params.get("token");
const video = document.querySelector("#camera-video");
const placeholder = document.querySelector("#camera-placeholder");
const statusElement = document.querySelector("#mobile-status");
const startButton = document.querySelector("#start-camera");
const captureButton = document.querySelector("#take-photo");
const fileInput = document.querySelector("#photo-file");
let socket;
let cameraStream;
let isSending = false;

function setStatus(message, isError = false) {
  statusElement.textContent = message;
  statusElement.classList.toggle("error", isError);
}

function websocketUrl() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host}/ws/mobile/${encodeURIComponent(sessionId)}?token=${encodeURIComponent(token)}`;
}

function connect() {
  if (!sessionId || !token) {
    setStatus("Oturum bağlantısı eksik. Masaüstündeki QR kodu yeniden okut.", true);
    startButton.disabled = true;
    fileInput.disabled = true;
    return;
  }
  socket = new WebSocket(websocketUrl());
  socket.addEventListener("open", () => setStatus("Masaüstüne bağlandı. Fotoğraf çekebilirsin."));
  socket.addEventListener("close", () => setStatus("Bağlantı kesildi. Sayfayı yenileyip tekrar bağlan.", true));
  socket.addEventListener("error", () => setStatus("Sunucuya bağlanılamadı. Wi-Fi ve oturum bağlantısını kontrol et.", true));
  socket.addEventListener("message", (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "capture.accepted") setStatus("Fotoğraf masaüstüne gönderildi; küçük önizleme hazırlanıyor.");
      if (message.type === "capture.processed") setStatus("Fotoğraf kaydedildi, önizleme hazır.");
      if (message.type === "session.mode") {
        document.querySelector("#mobile-mode").textContent =
          message.mode === "2d" ? "2D belge tarama modu seçili." : "3D obje tarama modu seçili.";
      }
      if (message.type === "error") setStatus(message.message, true);
    } catch {
      setStatus("Sunucudan geçersiz yanıt alındı.", true);
    }
  });
}

async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setStatus("Canlı kamera HTTP bağlantısında kullanılamıyor. Fotoğraf seç düğmesini kullan.", true);
    fileInput.click();
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1440 } },
    });
    video.srcObject = cameraStream;
    video.hidden = false;
    placeholder.hidden = true;
    captureButton.disabled = false;
    setStatus("Kamera hazır. Çekim için fotoğraf düğmesine dokun.");
  } catch (error) {
    setStatus(`Kamera açılamadı: ${error.message}. Fotoğraf seçiciyi deneyebilirsin.`, true);
  }
}

function waitForSocket() {
  if (socket?.readyState === WebSocket.OPEN) return Promise.resolve();
  if (!socket || socket.readyState === WebSocket.CLOSED) connect();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Masaüstü bağlantısı zaman aşımına uğradı.")), 8000);
    socket.addEventListener("open", () => {
      clearTimeout(timeout);
      resolve();
    }, { once: true });
    socket.addEventListener("close", () => {
      clearTimeout(timeout);
      reject(new Error("Masaüstü bağlantısı kesildi."));
    }, { once: true });
  });
}

async function sendImage(blob) {
  if (isSending) return;
  if (blob.size > 12 * 1024 * 1024) {
    setStatus("Fotoğraf 12 MiB sınırını aşıyor.", true);
    return;
  }
  if (!["image/jpeg", "image/png", "image/webp"].includes(blob.type)) {
    setStatus("Yalnızca JPEG, PNG veya WebP fotoğraflar desteklenir.", true);
    return;
  }
  isSending = true;
  captureButton.disabled = true;
  setStatus("Fotoğraf masaüstüne gönderiliyor…");
  try {
    await waitForSocket();
    socket.send(JSON.stringify({ type: "capture.start", mime_type: blob.type }));
    const data = await blob.arrayBuffer();
    const chunkSize = 64 * 1024;
    for (let offset = 0; offset < data.byteLength; offset += chunkSize) {
      while (socket.bufferedAmount > 512 * 1024) {
        await new Promise((resolve) => setTimeout(resolve, 30));
      }
      socket.send(data.slice(offset, Math.min(offset + chunkSize, data.byteLength)));
    }
    setStatus("Fotoğraf gönderildi; kaydetme onayı bekleniyor.");
    socket.send(JSON.stringify({ type: "capture.end" }));
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    isSending = false;
    captureButton.disabled = !cameraStream;
  }
}

startButton.addEventListener("click", startCamera);
captureButton.addEventListener("click", async () => {
  if (!video.videoWidth || !video.videoHeight) return;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    if (blob) sendImage(blob);
    else setStatus("Kamera görüntüsü hazırlanamadı.", true);
  }, "image/jpeg", 0.9);
});
fileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) sendImage(file);
  event.target.value = "";
});
window.addEventListener("pagehide", () => cameraStream?.getTracks().forEach((track) => track.stop()));

if (!window.isSecureContext) {
  document.querySelector(".mobile-card .hint").textContent = "Canlı kamera için HTTPS gerekir. HTTP bağlantısında telefonun kamera dosya seçicisini kullan.";
}
connect();
