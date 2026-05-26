/* ─── Sidebar toggle (mobile) ─────────────────── */

function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
  document.getElementById("sidebarOverlay").classList.toggle("open");
}

function closeSidebar() {
  document.getElementById("sidebar").classList.remove("open");
  document.getElementById("sidebarOverlay").classList.remove("open");
}

/* ─── Load user ───────────────────────────────── */

async function loadUser() {
  try {
    const res = await fetch("/api/me", { credentials: "include" });
    const data = await res.json();

    if (!res.ok || data.error) {
      window.location.href = "/login";
      return;
    }

    // Topbar welcome
    document.getElementById("welcomeText").textContent =
      `Welcome back, ${data.name}`;

    // Stat card
    document.getElementById("userRole").textContent = data.role;

    // Sidebar
    document.getElementById("sidebarName").textContent = data.name;
    document.getElementById("sidebarRole").textContent = data.role;
    document.getElementById("sidebarAvatar").textContent = data.name
      ? data.name.charAt(0).toUpperCase()
      : "?";
  } catch (err) {
    window.location.href = "/login";
  }
}

/* ─── Load history ────────────────────────────── */

async function loadHistory() {
  try {
    const res = await fetch("/api/history", { credentials: "include" });
    const data = await res.json();

    if (!Array.isArray(data)) throw new Error("Could not load history");

    document.getElementById("totalPredictions").textContent = data.length;

    if (data.length > 0) {
      document.getElementById("recentDisease").textContent =
        data[0].stage2_label;
    }

    const grid = document.getElementById("historyGrid");

    if (data.length === 0) {
      grid.innerHTML = `
        <div class="empty" style="grid-column:1/-1">
          <div class="empty-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
          </div>
          <div class="empty-title">No predictions yet</div>
          <div class="empty-sub">Start your first skin analysis to see results here.</div>
          <button class="btn btn-primary" onclick="goHome()" style="margin-top:4px">
            Begin Analysis
          </button>
        </div>`;
      return;
    }

    // Show count badge
    const badge = document.getElementById("countBadge");
    badge.textContent = data.length;
    badge.style.display = "inline-flex";

    grid.innerHTML = "";

    data.forEach((item, i) => {
      const card = document.createElement("div");
      card.className = "card";
      card.style.animationDelay = `${i * 0.05}s`;

      const dateStr = new Date(item.created_at).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      });

      card.innerHTML = `
        <div class="img-wrap">
          <img src="/uploads/${item.image_path}" alt="Skin scan image" loading="lazy" />
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
          <div class="card-footer">
            <div class="card-date">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
                <line x1="16" y1="2" x2="16" y2="6"/>
                <line x1="8" y1="2" x2="8" y2="6"/>
                <line x1="3" y1="10" x2="21" y2="10"/>
              </svg>
              ${dateStr}
            </div>
            <span style="font-size:11px;font-weight:600;color:var(--teal-700)">View Details</span>
          </div>
        </div>`;

      grid.appendChild(card);
    });
  } catch (err) {
    console.error(err);
  }
}

/* ─── Auth actions ────────────────────────────── */

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

function goHome() {
  document.getElementById("analysisModal").classList.add("show");

  initializeAnalysisUpload();
}

/* ─── Init ────────────────────────────────────── */

loadUser();
loadHistory();
let selectedFile = null;

function initializeAnalysisUpload() {
  const analysisInput = document.getElementById("analysisInput");

  const analysisDropZone = document.getElementById("analysisDropZone");

  if (!analysisInput || !analysisDropZone) {
    return;
  }

  analysisDropZone.onclick = () => {
    analysisInput.click();
  };

  analysisInput.onchange = (e) => {
    const file = e.target.files[0];

    if (!file) return;

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = (ev) => {
      document.getElementById("analysisPreview").style.display = "block";

      document.getElementById("analysisImg").src = ev.target.result;
    };

    reader.readAsDataURL(file);
  };
}
function closeAnalysis() {
  document.getElementById("analysisModal").classList.remove("show");
}

async function runAnalysis() {
  if (!selectedFile) {
    alert("Upload image first");

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

    document.getElementById("stage1Result").textContent = data.stage1.label;

    document.getElementById("stage2Result").textContent = data.stage2.label;

    /* SHOW GRADCAM */

    if (data.gradcam_image) {
      document.getElementById("gradcamImg").src = data.gradcam_image;

      document.getElementById("gradcamSection").style.display = "block";
    }

    loadHistory();

    loadHistory();
  } catch (err) {
    alert("Prediction failed");
  }
}
