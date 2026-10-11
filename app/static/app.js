// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>
//
// The browser renders server-annotated MJPEG frames. JavaScript polls status.

(function () {
  "use strict";

  var POLL_INTERVAL_MS = 500;

  // Preserve ?token=... so the status poll authenticates the same way the
  // stream does when the server was started with --token.
  var token = new URLSearchParams(window.location.search).get("token");
  var statusUrl =
    "/status" + (token ? "?token=" + encodeURIComponent(token) : "");
  if (token) {
    document.getElementById("stream").src =
      "/stream?token=" + encodeURIComponent(token);
  }

  var els = {
    presence: document.getElementById("presence"),
    detections: document.getElementById("detections"),
    fps: document.getElementById("fps"),
    backend: document.getElementById("backend"),
    simd: document.getElementById("simd"),
    videoSource: document.getElementById("video-source"),
    audioSource: document.getElementById("audio-source"),
    host: document.getElementById("host"),
  };

  // A Pi 4 throttles at its soft temperature limit; warn a little before it.
  var TEMP_WARN_C = 70;
  var TEMP_HOT_C = 80;

  function metricSpan(text, cls) {
    var span = document.createElement("span");
    span.textContent = text;
    if (cls) {
      span.className = cls;
    }
    return span;
  }

  // Omit unavailable metrics rather than displaying them as measured zero.
  function renderHost(data) {
    var parts = [];
    if (typeof data.cpu_percent === "number") {
      parts.push(metricSpan("CPU " + data.cpu_percent.toFixed(0) + "%", ""));
    }
    if (typeof data.rss_mb === "number") {
      parts.push(metricSpan("RAM " + data.rss_mb.toFixed(0) + " MB", ""));
    }
    if (typeof data.temp_c === "number") {
      var cls =
        data.temp_c >= TEMP_HOT_C ? "hot" : data.temp_c >= TEMP_WARN_C ? "warn" : "";
      parts.push(metricSpan(data.temp_c.toFixed(1) + " \u00b0C", cls));
    }
    els.host.textContent = "";
    if (parts.length === 0) {
      els.host.hidden = true;
      return;
    }
    // Text-node separators remain present when metrics are copied.
    parts.forEach(function (p, i) {
      if (i > 0) {
        els.host.appendChild(document.createTextNode(" \u00b7 "));
      }
      els.host.appendChild(p);
    });
    els.host.hidden = false;
  }

  function setPresence(present) {
    els.presence.textContent = present ? "Sound detected" : "Quiet";
    els.presence.className =
      "pill " + (present ? "pill--present" : "pill--absent");
  }

  function setOffline() {
    els.presence.textContent = "offline";
    els.presence.className = "pill pill--unknown";
  }

  function render(data) {
    setPresence(data.person_present);
    els.detections.textContent = data.detections;
    els.fps.textContent = Number(data.fps).toFixed(1);
    els.backend.textContent = data.backend;
    els.simd.textContent = data.simd ? "on" : "off";
    els.videoSource.textContent = data.video_source;
    els.audioSource.textContent = data.audio_source;
    renderHost(data);
  }

  function poll() {
    fetch(statusUrl, { cache: "no-store" })
      .then(function (res) {
        if (!res.ok) {
          throw new Error("status " + res.status);
        }
        return res.json();
      })
      .then(render)
      .catch(setOffline);
  }

  poll();
  window.setInterval(poll, POLL_INTERVAL_MS);
})();
