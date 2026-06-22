// ─────────────────────────────────────────────
// CROP DISEASE PREDICTION — FRONTEND JS
// Author: Abhishek Gorakh Borade
// ─────────────────────────────────────────────

let selectedFile = null;

// ── DRAG & DROP ──
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");

if (dropZone) {
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
  dropZone.addEventListener("click", () => fileInput.click());
}

if (fileInput) {
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });
}

function handleFile(file) {
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    alert("❌ Please upload a JPG, PNG, or WEBP image.");
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("preview-img").src = e.target.result;
    document.getElementById("drop-zone").style.display = "none";
    document.getElementById("preview-container").style.display = "block";
    document.getElementById("results-section").style.display = "none";
  };
  reader.readAsDataURL(file);
}

// ── PREDICT ──
async function runPrediction() {
  if (!selectedFile) return;

  document.getElementById("preview-container").style.display = "none";
  document.getElementById("loading").style.display = "block";
  document.getElementById("results-section").style.display = "none";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();

    document.getElementById("loading").style.display = "none";

    if (data.error) {
      alert("❌ Error: " + data.error);
      resetAll();
      return;
    }

    showResults(data);
  } catch (err) {
    document.getElementById("loading").style.display = "none";
    alert("❌ Something went wrong. Please try again.");
    console.error(err);
    resetAll();
  }
}

function showResults(data) {
  const top = data.top;

  // Image
  document.getElementById("result-img").src = data.image_url;
  document.getElementById("result-badge").textContent = top.icon + " " + top.label;
  document.getElementById("result-badge").style.color = top.color;

  // Diagnosis
  document.getElementById("diag-icon").textContent = top.icon;
  document.getElementById("diag-name").textContent = top.label;
  document.getElementById("diag-name").style.color = top.color;
  document.getElementById("conf-text").textContent = `Confidence: ${top.confidence}%`;

  setTimeout(() => {
    document.getElementById("conf-bar").style.width = top.confidence + "%";
    document.getElementById("conf-bar").style.background = top.color;
  }, 100);

  // Advice
  document.getElementById("advice-text").textContent = top.advice;

  // Top predictions list
  const list = document.getElementById("predictions-list");
  list.innerHTML = "";
  data.results.forEach((r, i) => {
    const div = document.createElement("div");
    div.className = "pred-item";
    div.innerHTML = `
      <div class="pred-header">
        <span class="pred-name">${i + 1}. ${r.icon} ${r.label}</span>
        <span class="pred-conf">${r.confidence}%</span>
      </div>
      <div class="pred-bar-wrap">
        <div class="pred-bar" style="width:0%; background:${r.color}" data-width="${r.confidence}"></div>
      </div>`;
    list.appendChild(div);
  });
  setTimeout(() => {
    document.querySelectorAll(".pred-bar").forEach(bar => {
      bar.style.width = bar.dataset.width + "%";
    });
  }, 150);

  // Severity
  const sevMap = {
    none:    { label: "None — Plant is Healthy 🌿", color: "#27ae60", desc: "No action needed." },
    medium:  { label: "Medium ⚠️",                 color: "#f39c12", desc: "Take action soon to prevent spread." },
    high:    { label: "High 🚨",                   color: "#e74c3c", desc: "Immediate treatment required!" },
    unknown: { label: "Unknown",                   color: "#95a5a6", desc: "Consult an expert." },
  };
  const sev = sevMap[top.severity] || sevMap.unknown;
  document.getElementById("severity-display").innerHTML = `
    <div class="severity-dot" style="background:${sev.color}"></div>
    <div>
      <div class="severity-label" style="color:${sev.color}">${sev.label}</div>
      <div class="severity-desc">${sev.desc}</div>
    </div>`;

  document.getElementById("results-section").style.display = "block";
  document.getElementById("results-section").scrollIntoView({ behavior: "smooth" });
}

// ── RESET ──
function resetAll() {
  selectedFile = null;
  if (fileInput) fileInput.value = "";
  document.getElementById("drop-zone").style.display = "block";
  document.getElementById("preview-container").style.display = "none";
  document.getElementById("loading").style.display = "none";
  document.getElementById("results-section").style.display = "none";
  document.getElementById("conf-bar").style.width = "0%";
  window.scrollTo({ top: 0, behavior: "smooth" });
}
