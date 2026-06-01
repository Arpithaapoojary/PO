/* ─────────────────────────────────────────
   MOBILE SIDEBAR
   ───────────────────────────────────────── */

function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");

  document.getElementById("sidebarOverlay").classList.toggle("open");
}

function closeSidebar() {
  document.getElementById("sidebar").classList.remove("open");

  document.getElementById("sidebarOverlay").classList.remove("open");
}

/* ─────────────────────────────────────────
   GLOBALS
   ───────────────────────────────────────── */

let selectedFile = null;

/* ─────────────────────────────────────────
   LOAD USER
   ───────────────────────────────────────── */

async function loadUser() {
  try {
    const res = await fetch("/api/me", {
      credentials: "include",
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      window.location.href = "/login";

      return;
    }

    // ONLY PATIENTS HERE

    if (data.role !== "patient") {
      window.location.href = "/doctor";

      return;
    }

    document.getElementById("welcomeText").textContent =
      `Welcome back, ${data.name}`;

    document.getElementById("sidebarName").textContent = data.name;

    document.getElementById("sidebarRole").textContent = "Patient";

    document.getElementById("sidebarAvatar").textContent = data.name
      ? data.name.charAt(0).toUpperCase()
      : "?";
  } catch (err) {
    window.location.href = "/login";
  }
}

/* ─────────────────────────────────────────
   LOAD HISTORY
   ───────────────────────────────────────── */

async function loadHistory() {
  try {
    const res = await fetch("/api/history", {
      credentials: "include",
    });

    const data = await res.json();

    if (!Array.isArray(data)) {
      throw new Error("Invalid history");
    }

    // STATS

    document.getElementById("totalPredictions").textContent = data.length;

    if (data.length > 0) {
      document.getElementById("recentDisease").textContent =
        data[0].stage2_label;
    }

    const grid = document.getElementById("historyGrid");

    // EMPTY

    if (data.length === 0) {
      grid.innerHTML = `

        <div class="empty">

          <h3>No analyses yet</h3>

          <p>
            Upload a skin image to begin AI analysis.
          </p>

        </div>
      `;

      return;
    }

    grid.innerHTML = "";

    data.forEach((item) => {
      const status = item.review_status || "Pending";
      const dateStr = new Date(item.created_at).toLocaleDateString();
      const card = document.createElement("div");
      card.className = "card";

      card.innerHTML = `
        <div class="img-wrap">
          <img src="/uploads/${item.image_path}" alt="Scan">
          <div class="img-overlay"></div>
        </div>
        <div class="card-body">
          <div class="card-top">
            <div class="badge">${item.stage1_label}</div>
            <span class="card-id">#${item.id}</span>
          </div>
          <div class="disease">${item.stage2_label}</div>
          <div class="conf-section">
            <div class="conf-header">
              <span class="conf-label">AI Confidence</span>
              <span class="conf-value">${item.stage2_conf}%</span>
            </div>
            <div class="track">
              <div class="fill" style="width:${item.stage2_conf}%"></div>
            </div>
          </div>
          <div class="review-status ${
            status === "Approved"
              ? "review-approved"
              : status === "Rejected"
                ? "review-rejected"
                : "review-pending"
          }">${status}</div>
          ${
            item.doctor_note
              ? `<div class="doctor-note">
                  <div class="doctor-note-label">Doctor Review</div>
                  <div class="doctor-note-text">${item.doctor_note}</div>
                  ${item.medication ? `
                  <div class="doctor-note-label" style="margin-top: 8px;">E-Prescription</div>
                  <div class="doctor-note-text"><strong>${item.medication}</strong> - ${item.dosage} for ${item.duration}</div>
                  ` : ''}
                </div>`
              : `<div class="doctor-note">
                  <div class="doctor-note-label">Review Status</div>
                  <div class="doctor-note-text">Awaiting doctor review.</div>
                </div>`
          }
          <div class="card-footer">
            <div class="card-date">
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
              ${dateStr}
            </div>
            <button class="btn btn-primary btn-sm" onclick='downloadReport(${JSON.stringify(item)})'>
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
              Report
            </button>
          </div>
        </div>
      `;

      grid.appendChild(card);
    });
  } catch (err) {
    console.log(err);
  }
}

/* ─────────────────────────────────────────
   ANALYSIS MODAL
   ───────────────────────────────────────── */

function goHome() {
  document.getElementById("analysisModal").classList.add("show");

  initializeAnalysisUpload();
}

function closeAnalysis() {
  document.getElementById("analysisModal").classList.remove("show");
}

/* ─────────────────────────────────────────
   IMAGE UPLOAD
   ───────────────────────────────────────── */

function initializeAnalysisUpload() {
  const input = document.getElementById("analysisInput");

  const zone = document.getElementById("analysisDropZone");

  if (!input || !zone) {
    return;
  }

  zone.onclick = () => {
    input.click();
  };

  input.onchange = (e) => {
    const file = e.target.files[0];

    if (!file) {
      return;
    }

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = (ev) => {
      document.getElementById("analysisPreview").style.display = "block";

      document.getElementById("analysisImg").src = ev.target.result;
    };

    reader.readAsDataURL(file);
  };
}

/* ─────────────────────────────────────────
   RUN ANALYSIS
   ───────────────────────────────────────── */

async function runAnalysis() {
  if (!selectedFile) {
    alert("Please upload image first");

    return;
  }

  const fd = new FormData();

  fd.append("image", selectedFile);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",

      body: fd,

      credentials: "include",
    });

    const data = await res.json();

    if (data.error) {
      alert(data.error);

      return;
    }

    // RESULTS

    document.getElementById("stage1Result").textContent = data.stage1.label;

    document.getElementById("stage2Result").textContent = data.stage2.label;

    // GRADCAM

    if (data.gradcam_image) {
      document.getElementById("gradcamImg").src = data.gradcam_image;

      document.getElementById("gradcamSection").style.display = "block";
    }

    // REFRESH HISTORY

    loadHistory();
  } catch (err) {
    console.log(err);

    alert("Prediction failed");
  }
}
/* ─────────────────────────────────────────
   DOWNLOAD REPORT
   ───────────────────────────────────────── */

async function downloadReport(item) {
  try {
    // ORIGINAL IMAGE

    const originalRes = await fetch(`/uploads/${item.image_path}`);

    const originalBlob = await originalRes.blob();

    const originalBase64 = await blobToBase64(originalBlob);

    // GRADCAM IMAGE

    let gradcamBase64 = null;

    if (item.gradcam_path) {
      const gradcamRes = await fetch(`/uploads/gradcam/${item.gradcam_path}`);

      const gradcamBlob = await gradcamRes.blob();

      gradcamBase64 = await blobToBase64(gradcamBlob);
    }

    // REQUEST PDF

    const res = await fetch("/api/report", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      credentials: "include",

      body: JSON.stringify({
        patient: {
          name: document.getElementById("sidebarName").textContent,

          age: "N/A",

          gender: "N/A",

          area: "Skin Region",
        },

        stage1: {
          raw: item.stage1_label,

          confidence: item.stage1_conf,
        },

        stage2: {
          raw: item.stage2_label,

          confidence: item.stage2_conf,
        },

        original_image: originalBase64,

        gradcam_image: gradcamBase64,

        doctor_note: item.doctor_note || "",
        medication: item.medication || "",
        dosage: item.dosage || "",
        duration: item.duration || "",

        review_status: item.review_status || "Pending",
      }),
    });

    const data = await res.json();

    if (data.error) {
      alert(data.error);

      return;
    }

    // DOWNLOAD PDF

    const link = document.createElement("a");

    link.href = `data:application/pdf;base64,${data.pdf}`;

    link.download = `DermScan_Report_${item.id}.pdf`;

    link.click();
  } catch (err) {
    console.log(err);

    alert("Failed to generate report");
  }
}

/* ─────────────────────────────────────────
   BLOB TO BASE64
   ───────────────────────────────────────── */

function blobToBase64(blob) {
  return new Promise((resolve) => {
    const reader = new FileReader();

    reader.onloadend = () => resolve(reader.result);

    reader.readAsDataURL(blob);
  });
}

/* ─────────────────────────────────────────
   LOGOUT
   ───────────────────────────────────────── */

async function logout() {
  try {
    await fetch("/api/logout", {
      method: "POST",

      credentials: "include",
    });

    window.location.href = "/login";
  } catch (err) {
    alert("Logout failed");
  }
}

/* ─────────────────────────────────────────
   INIT
   ───────────────────────────────────────── */

loadUser();

loadHistory();
