/* ─── TAB SWITCHING ─────────────────────────────────────────────────────────── */
const tabs = document.querySelectorAll(".tab");
const forms = document.querySelectorAll(".form");

function switchTab(type) {
  tabs.forEach((t) => t.classList.remove("active"));
  forms.forEach((f) => f.classList.remove("active"));
  clearAllMessages();

  if (type === "login") {
    tabs[0].classList.add("active");
    document.getElementById("loginForm").classList.add("active");
  } else {
    tabs[1].classList.add("active");
    document.getElementById("registerForm").classList.add("active");
  }
}

/* ─── SHOW/HIDE PASSWORD ────────────────────────────────────────────────────── */
function togglePwd(inputId, btn) {
  const input = document.getElementById(inputId);
  const isHidden = input.type === "password";
  input.type = isHidden ? "text" : "password";

  const eyeOpen = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>`;
  const eyeClosed = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"/><path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"/><line x1="2" y1="2" x2="22" y2="22"/></svg>`;
  btn.innerHTML = isHidden ? eyeClosed : eyeOpen;
}

/* ─── FIELD VALIDATION HELPERS ──────────────────────────────────────────────── */
function setFieldError(inputEl, msgEl, text) {
  inputEl.classList.add("field-error");
  msgEl.querySelector("span").textContent = text;
  msgEl.classList.add("show");
}

function clearFieldError(inputEl, msgEl) {
  inputEl.classList.remove("field-error");
  msgEl.classList.remove("show");
  msgEl.querySelector("span").textContent = "";
}

function isValidEmail(val) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(val.trim());
}

function clearAllMessages() {
  document.querySelectorAll(".msg").forEach((m) => {
    m.className = "msg";
    m.innerHTML = "";
  });
  document
    .querySelectorAll(".field-msg")
    .forEach((m) => m.classList.remove("show"));
  document
    .querySelectorAll("input, select")
    .forEach((el) => el.classList.remove("field-error"));
}

/* ─── FORM MESSAGE ──────────────────────────────────────────────────────────── */
function showMessage(boxId, message, type = "success") {
  const box = document.getElementById(boxId);
  const icon =
    type === "success"
      ? `<span class="msg-icon">✓</span>`
      : `<span class="msg-icon">⚠</span>`;
  box.className = `msg show ${type}`;
  box.innerHTML = `${icon}<span>${message}</span>`;
}

/* ─── LOADING STATE ─────────────────────────────────────────────────────────── */
function setLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  btn.disabled = loading;
  btn.classList.toggle("loading", loading);
}

/* ─── LOGIN VALIDATION & SUBMIT ─────────────────────────────────────────────── */
document.getElementById("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const emailEl = document.getElementById("loginEmail");
  const passEl = document.getElementById("loginPassword");
  const emailMsgEl = document.getElementById("loginEmailMsg");
  const passMsgEl = document.getElementById("loginPasswordMsg");

  clearFieldError(emailEl, emailMsgEl);
  clearFieldError(passEl, passMsgEl);

  let valid = true;

  if (!emailEl.value.trim()) {
    setFieldError(emailEl, emailMsgEl, "Email address is required.");
    valid = false;
  } else if (!isValidEmail(emailEl.value)) {
    setFieldError(emailEl, emailMsgEl, "Please enter a valid email address.");
    valid = false;
  }

  if (!passEl.value) {
    setFieldError(passEl, passMsgEl, "Password is required.");
    valid = false;
  }

  if (!valid) return;

  setLoading("loginBtn", true);

  try {
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        email: emailEl.value,
        password: passEl.value,
      }),
    });

    const data = await res.json();

    if (!res.ok || data.error) throw new Error(data.error || "Login failed");

    showMessage("loginMsg", "Authenticated! Redirecting to your dashboard…");

    setTimeout(() => {
      window.location.href = "/dashboard";
    }, 1000);
  } catch (err) {
    showMessage("loginMsg", err.message, "error");
  } finally {
    setLoading("loginBtn", false);
  }
});

/* ─── LOGIN VALIDATION & SUBMIT ─────────────────────────────────────────────── */

document.getElementById("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const emailEl = document.getElementById("loginEmail");

  const passEl = document.getElementById("loginPassword");

  const emailMsgEl = document.getElementById("loginEmailMsg");

  const passMsgEl = document.getElementById("loginPasswordMsg");

  clearFieldError(emailEl, emailMsgEl);

  clearFieldError(passEl, passMsgEl);

  let valid = true;

  // EMAIL VALIDATION

  if (!emailEl.value.trim()) {
    setFieldError(emailEl, emailMsgEl, "Email address is required.");

    valid = false;
  } else if (!isValidEmail(emailEl.value)) {
    setFieldError(emailEl, emailMsgEl, "Please enter a valid email address.");

    valid = false;
  }

  // PASSWORD VALIDATION

  if (!passEl.value) {
    setFieldError(passEl, passMsgEl, "Password is required.");

    valid = false;
  }

  if (!valid) return;

  setLoading("loginBtn", true);

  try {
    const res = await fetch("/api/login", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      credentials: "include",

      body: JSON.stringify({
        email: emailEl.value,

        password: passEl.value,
      }),
    });

    const data = await res.json();

    // ERROR

    if (!res.ok || data.error) {
      throw new Error(data.error || "Login failed");
    }

    // SUCCESS MESSAGE

    showMessage("loginMsg", "Authenticated! Redirecting...");

    // ROLE BASED REDIRECT

    setTimeout(() => {
      window.location.href = data.redirect;
    }, 1000);
  } catch (err) {
    showMessage("loginMsg", err.message, "error");
  } finally {
    setLoading("loginBtn", false);
  }
});

/* ─── LIVE INLINE VALIDATION ON BLUR ────────────────────────────────────────── */
document.getElementById("loginEmail").addEventListener("blur", function () {
  const msgEl = document.getElementById("loginEmailMsg");
  if (this.value && !isValidEmail(this.value)) {
    setFieldError(this, msgEl, "Please enter a valid email address.");
  } else {
    clearFieldError(this, msgEl);
  }
});

document.getElementById("regEmail").addEventListener("blur", function () {
  const msgEl = document.getElementById("regEmailMsg");
  if (this.value && !isValidEmail(this.value)) {
    setFieldError(this, msgEl, "Please enter a valid email address.");
  } else {
    clearFieldError(this, msgEl);
  }
});

document.getElementById("regPassword").addEventListener("blur", function () {
  const msgEl = document.getElementById("regPasswordMsg");
  if (this.value && this.value.length < 8) {
    setFieldError(this, msgEl, "Password must be at least 8 characters.");
  } else {
    clearFieldError(this, msgEl);
  }
});
