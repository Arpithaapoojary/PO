// ── State ──
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const previewBox = document.getElementById("previewBox");
const prevImg = document.getElementById("prevImg");
const prevName = document.getElementById("prevName");

let selectedFile = null;
let lastResult = null;

// ── Upload ──
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("over");

  const f = e.dataTransfer.files[0];

  if (f && f.type.startsWith("image/")) {
    loadFile(f);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) {
    loadFile(e.target.files[0]);
  }
});

function loadFile(file) {
  selectedFile = file;

  const r = new FileReader();

  r.onload = (ev) => {
    prevImg.src = ev.target.result;
    prevName.textContent = file.name;

    dropZone.style.display = "none";
    previewBox.classList.add("show");

    resetResults();

    document.getElementById("errMsg").classList.remove("show");
  };

  r.readAsDataURL(file);
}

function resetUpload() {
  dropZone.style.display = "";
  previewBox.classList.remove("show");

  fileInput.value = "";

  selectedFile = null;
  lastResult = null;

  resetResults();

  document.getElementById("pdfBtn").classList.remove("show");
}

function resetResults() {
  ["rb1", "rb2"].forEach((id) => {
    document.getElementById(id).classList.remove("done");
  });

  document.getElementById("res1").textContent = "Ready to analyse";
  document.getElementById("res2").textContent = "Awaiting stage 1";

  document.getElementById("confFill1").style.width = "0%";
  document.getElementById("confFill2").style.width = "0%";

  document.getElementById("chartBars1").innerHTML = "";
  document.getElementById("chartBars2").innerHTML = "";
}

// ── Analysis ──
async function runAnalysis() {
  if (!selectedFile) {
    alert("Please select an image first.");
    return;
  }

  const btn = document.getElementById("runBtn");

  btn.classList.add("loading");
  btn.disabled = true;

  document.getElementById("res1").textContent = "Analysing...";
  document.getElementById("res2").textContent = "Awaiting stage 1...";

  document.getElementById("errMsg").classList.remove("show");
  document.getElementById("pdfBtn").classList.remove("show");

  try {
    const fd = new FormData();
    fd.append("image", selectedFile);

    const res = await fetch("/api/predict", {
      method: "POST",
      body: fd,
      credentials: "include",
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || "Server error");
    }

    lastResult = data;

    await showResult(
      "rb1",
      "res1",
      "confFill1",
      "confVal1",
      data.stage1.label,
      data.stage1.confidence,
    );

    renderChart("chartBars1", data.stage1.all_labels, data.stage1.all_conf);

    await delay(500);

    await showResult(
      "rb2",
      "res2",
      "confFill2",
      "confVal2",
      data.stage2.label,
      data.stage2.confidence,
    );

    renderChart("chartBars2", data.stage2.all_labels, data.stage2.all_conf);

    document.getElementById("pdfBtn").classList.add("show");
  } catch (err) {
    console.error(err);

    document.getElementById("res1").textContent = "Error";

    document.getElementById("errMsg").textContent =
      err.message || "Server not reachable.";

    document.getElementById("errMsg").classList.add("show");
  }

  btn.classList.remove("loading");
  btn.disabled = false;
}

async function showResult(rbId, resId, fillId, valId, label, conf) {
  await delay(200);

  document.getElementById(resId).textContent = label;

  document.getElementById(rbId).classList.add("done");

  document.getElementById(valId).textContent = conf + "%";

  await delay(50);

  document.getElementById(fillId).style.width = conf + "%";
}

function renderChart(containerId, labels, values) {
  const wrap = document.getElementById(containerId);

  wrap.innerHTML = "";

  labels.forEach((lbl, i) => {
    const pct = values[i] || 0;

    wrap.innerHTML += `
      <div class="chart-bar-row">
        <span class="chart-label" title="${lbl}">${lbl}</span>
        <div class="chart-track">
          <div class="chart-fill" style="width:0%" data-pct="${pct}"></div>
        </div>
        <span class="chart-pct">${pct}%</span>
      </div>
    `;
  });

  setTimeout(() => {
    wrap.querySelectorAll(".chart-fill").forEach((el) => {
      el.style.width = el.dataset.pct + "%";
    });
  }, 100);
}

// ── PDF Report ──
async function downloadReport() {
  if (!lastResult) return;

  const patient = {
    name: document.getElementById("pName").value || "Not provided",
    age: document.getElementById("pAge").value || "Not provided",
    gender: document.getElementById("pGender").value || "Not provided",
    area: document.getElementById("pArea").value || "Not provided",
  };

  try {
    const res = await fetch("/api/report", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      credentials: "include",
      body: JSON.stringify({
        patient,
        stage1: {
          raw: lastResult.stage1.label,
          confidence: lastResult.stage1.confidence,
        },
        stage2: {
          raw: lastResult.stage2.label,
          confidence: lastResult.stage2.confidence,
        },
      }),
    });

    const data = await res.json();

    if (data.error) {
      throw new Error(data.error);
    }

    const link = document.createElement("a");

    link.href = "data:application/pdf;base64," + data.pdf;

    link.download = `DermScan_Report_${patient.name.replace(/\s+/g, "_")}.pdf`;

    link.click();
  } catch (e) {
    alert("Could not generate report: " + e.message);
  }
}

// ── Utilities ──
function delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function filterDis(cat, btn) {
  document.querySelectorAll(".fbtn").forEach((b) => {
    b.classList.remove("active");
  });

  btn.classList.add("active");

  document.querySelectorAll(".dc").forEach((c) => {
    c.style.display =
      cat === "all" || c.dataset.cat === cat ? "" : "none";
  });
}

// ── Intersection Observer (Reveal Animations) ──
const obs = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add("in");
      }
    });
  },
  { threshold: 0.08 },
);

document.querySelectorAll(".reveal").forEach((el) => {
  obs.observe(el);
});
