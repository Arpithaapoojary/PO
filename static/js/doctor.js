/* ─────────────────────────────────────────
   GLOBALS
───────────────────────────────────────── */

let currentPredictionId = null;

/* ─────────────────────────────────────────
   LOAD CASES
───────────────────────────────────────── */

async function loadCases() {
  try {
    const res = await fetch("/api/history", {
      credentials: "include",
    });

    const data = await res.json();

    const grid = document.getElementById("doctorGrid");

    document.getElementById("totalCases").textContent = data.length;

    const pending = data.filter((x) => x.review_status !== "Approved").length;

    document.getElementById("pendingCount").textContent = pending;

    grid.innerHTML = "";

    data.forEach((item) => {
      const card = document.createElement("div");

      card.className = "card";

      card.innerHTML = `

        <div class="img-wrap">

          <img
            src="/uploads/${item.image_path}">

        </div>

        <div class="card-body">

          <div class="badge">

            ${item.stage1_label}

          </div>

          <div class="disease">

            ${item.stage2_label}

          </div>

          <div class="patient-chip">

            Patient:
            ${item.patient_name}

          </div>

          <div
            class="
              review-status
              ${
                item.review_status === "Approved"
                  ? "review-approved"
                  : item.review_status === "Rejected"
                    ? "review-rejected"
                    : "review-pending"
              }
            ">

            ${item.review_status}

          </div>

          <div class="card-footer">

            <button
              class="btn btn-primary"
              onclick='openReview(${JSON.stringify(item)})'>

              Review Case

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
   OPEN REVIEW
───────────────────────────────────────── */

function openReview(item) {
  currentPredictionId = item.id;
  document.getElementById("reviewGradcam").src =
    `/uploads/gradcam/${item.gradcam_path}`;
  document.getElementById("reviewModal").classList.add("show");

  document.getElementById("reviewImage").src = `/uploads/${item.image_path}`;

  document.getElementById("reviewDisease").textContent = item.stage2_label;

  document.getElementById("reviewConfidence").textContent =
    item.stage2_conf + "%";

  document.getElementById("doctorNote").value = item.doctor_note || "";
}

/* ─────────────────────────────────────────
   CLOSE REVIEW
───────────────────────────────────────── */

function closeReview() {
  document.getElementById("reviewModal").classList.remove("show");
}

/* ─────────────────────────────────────────
   SUBMIT REVIEW
───────────────────────────────────────── */

async function submitReview(status) {
  const note = document.getElementById("doctorNote").value;

  if (!note) {
    alert("Please write review");

    return;
  }

  try {
    const res = await fetch("/api/add-note", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      credentials: "include",

      body: JSON.stringify({
        prediction_id: currentPredictionId,

        note: note,

        status: status,
      }),
    });

    const data = await res.json();

    if (data.error) {
      alert(data.error);

      return;
    }

    alert("Review saved");

    closeReview();

    loadCases();
  } catch (err) {
    alert("Failed to save review");
  }
}

/* ─────────────────────────────────────────
   LOGOUT
───────────────────────────────────────── */

async function logout() {
  await fetch("/api/logout", {
    method: "POST",

    credentials: "include",
  });

  window.location.href = "/login";
}

/* ─────────────────────────────────────────
   INIT
───────────────────────────────────────── */

loadCases();
