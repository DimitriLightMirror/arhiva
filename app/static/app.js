/* arhivadoc.eu - client logic for the OCR test page (vanilla JS, offline). */

(function () {
  "use strict";

  var REGION_COLORS = {
    TEXT: "#1a5fb4",
    TABLE: "#2f9e44",
    IMAGE: "#e8590c",
    PLAN: "#9c36b5",
  };

  // How far through the pipeline each stage is, used only for the
  // indeterminate-feel progress bar while polling.
  var STAGE_PROGRESS = {
    queued: 5, ingest: 10, preprocess: 25, ocr: 45, layout: 60,
    correction: 72, classification: 84, export: 93, done: 100,
  };

  var dropzone = document.getElementById("dropzone");
  var pickBtn = document.getElementById("pick-btn");
  var fileInput = document.getElementById("file-input");
  var progressWrap = document.getElementById("progress-wrap");
  var progressText = document.getElementById("progress-text");
  var progressFill = document.getElementById("progress-fill");
  var errorMsg = document.getElementById("error-msg");
  var resultSection = document.getElementById("result-section");

  var currentResult = null;
  var currentJobId = null;

  // --- Upload ------------------------------------------------------------

  pickBtn.addEventListener("click", function (e) {
    e.stopPropagation();
    fileInput.click();
  });
  dropzone.addEventListener("click", function () { fileInput.click(); });
  dropzone.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") fileInput.click();
  });
  fileInput.addEventListener("change", function () {
    if (fileInput.files.length) upload(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", function (e) {
    if (e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
  });

  function upload(file) {
    hideError();
    resultSection.hidden = true;
    progressWrap.hidden = false;
    setProgress("Se incarca fisierul...", 3);

    var form = new FormData();
    form.append("file", file, file.name);

    fetch("/api/scan", { method: "POST", body: form })
      .then(function (r) { return r.ok ? r.json() : r.json().then(function (d) { throw new Error(d.detail || "Eroare la incarcare"); }); })
      .then(function (data) {
        currentJobId = data.job_id;
        pollStatus(data.job_id);
      })
      .catch(showError);
  }

  // --- Status polling ------------------------------------------------------

  function pollStatus(jobId) {
    fetch("/api/jobs/" + jobId)
      .then(function (r) { return r.json(); })
      .then(function (job) {
        var pct = STAGE_PROGRESS[job.stage] || 10;
        setProgress("Etapa: " + job.stage, pct);
        if (job.status === "done") {
          loadResult(jobId);
        } else if (job.status === "error") {
          showError(new Error("Procesarea a esuat: " + (job.error || "eroare necunoscuta")));
        } else {
          setTimeout(function () { pollStatus(jobId); }, 1500);
        }
      })
      .catch(showError);
  }

  function setProgress(text, pct) {
    progressText.textContent = text;
    progressFill.style.width = pct + "%";
  }

  function showError(err) {
    progressWrap.hidden = true;
    errorMsg.hidden = false;
    errorMsg.textContent = err.message || String(err);
  }
  function hideError() { errorMsg.hidden = true; }

  // --- Result rendering ----------------------------------------------------

  function loadResult(jobId) {
    fetch("/api/jobs/" + jobId + "/result")
      .then(function (r) { return r.json(); })
      .then(function (result) {
        currentResult = result;
        progressWrap.hidden = true;
        renderResult(result);
      })
      .catch(showError);
  }

  function renderResult(result) {
    resultSection.hidden = false;

    // Classification badges
    var cls = result.classification || {};
    document.getElementById("class-badge").textContent =
      cls.class_label || "Necunoscut";
    var conf = Math.round((cls.confidence || 0) * 100);
    var confBadge = document.getElementById("conf-badge");
    confBadge.textContent = "Incredere: " + conf + "%";
    confBadge.className = "badge badge-conf " +
      (conf >= 70 ? "high" : conf >= 40 ? "mid" : "low");

    // Extracted fields
    var tbody = document.querySelector("#fields-table tbody");
    tbody.innerHTML = "";
    var labels = {
      parti: "Parti", numar_cadastral: "Numar cadastral", data: "Data",
      adresa: "Adresa", notar: "Notar", valoare: "Valoare",
    };
    Object.keys(labels).forEach(function (key) {
      var value = (cls.fields || {})[key];
      if (Array.isArray(value)) value = value.join("; ");
      var tr = document.createElement("tr");
      var th = document.createElement("td");
      th.textContent = labels[key];
      var td = document.createElement("td");
      td.textContent = value || "-";
      tr.appendChild(th); tr.appendChild(td);
      tbody.appendChild(tr);
    });

    // Tags
    var tagsView = document.getElementById("tags-view");
    tagsView.innerHTML = "";
    (cls.tags || []).forEach(function (t) {
      var span = document.createElement("span");
      span.className = "tag";
      span.textContent = t;
      tagsView.appendChild(span);
    });
    document.getElementById("tagged-text").textContent = cls.tagged_text || "";
    document.getElementById("tagged-text-wrap").hidden = !cls.tagged_text;

    // Downloads
    document.getElementById("dl-pdf").href = "/api/jobs/" + result.job_id + "/pdf";
    document.getElementById("dl-json").href = "/api/jobs/" + result.job_id + "/json";

    // Pages
    renderPages(result);
    resultSection.scrollIntoView({ behavior: "smooth" });
  }

  function renderPages(result) {
    var container = document.getElementById("pages");
    container.innerHTML = "";
    result.pages.forEach(function (page) {
      var row = document.createElement("div");
      row.className = "page-row";

      var imgWrap = document.createElement("div");
      imgWrap.className = "page-image-wrap";
      var img = document.createElement("img");
      img.src = "/api/jobs/" + result.job_id + "/pages/" + page.page_number + ".jpg";
      img.alt = "Pagina " + page.page_number;
      var canvas = document.createElement("canvas");
      imgWrap.appendChild(img);
      imgWrap.appendChild(canvas);
      img.addEventListener("load", function () {
        drawRegions(canvas, img, page.regions || []);
      });

      var textDiv = document.createElement("div");
      textDiv.className = "page-text";
      textDiv.textContent = page.corrected_text || "(fara text detectat)";
      var meta = document.createElement("div");
      meta.className = "page-meta";
      meta.textContent = "Pagina " + page.page_number + " - incredere OCR medie: " +
        page.mean_confidence + "% - motor: " + (page.ocr_language || "?");
      textDiv.appendChild(meta);

      row.appendChild(imgWrap);
      row.appendChild(textDiv);
      container.appendChild(row);
    });
  }

  function drawRegions(canvas, img, regions) {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    regions.forEach(function (r) {
      var cb = document.querySelector('.region-toggle[data-type="' + r.type + '"]');
      if (cb && !cb.checked) return;
      var color = REGION_COLORS[r.type] || "#666";
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, canvas.width / 500);
      ctx.globalAlpha = 0.9;
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.globalAlpha = 0.12;
      ctx.fillStyle = color;
      ctx.fillRect(r.x, r.y, r.w, r.h);
      ctx.globalAlpha = 1;
      ctx.font = Math.max(14, canvas.width / 60) + "px sans-serif";
      ctx.fillStyle = color;
      ctx.fillText(r.type, r.x + 4, r.y + Math.max(16, canvas.width / 55));
    });
  }

  // Region overlay toggles: redraw all pages on change.
  document.querySelectorAll(".region-toggle").forEach(function (cb) {
    cb.addEventListener("change", function () {
      if (!currentResult) return;
      var wraps = document.querySelectorAll(".page-image-wrap");
      currentResult.pages.forEach(function (page, i) {
        var wrap = wraps[i];
        if (!wrap) return;
        drawRegions(wrap.querySelector("canvas"), wrap.querySelector("img"),
          page.regions || []);
      });
    });
  });

  document.getElementById("new-doc-btn").addEventListener("click", function () {
    resultSection.hidden = true;
    currentResult = null;
    currentJobId = null;
    fileInput.value = "";
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
})();
