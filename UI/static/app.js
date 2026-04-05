/* global Plotly */

function loadUiConfig() {
  const el = document.getElementById("mamf-ui-config");
  if (!el) return null;
  try {
    return JSON.parse(el.textContent || "{}");
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error("Failed to parse MAMF UI config", err);
    return null;
  }
}

function setSelectOptions(selectEl, options, preferredValue) {
  if (!selectEl) return;
  const previousValue = preferredValue ?? selectEl.value;
  selectEl.innerHTML = "";

  if (!options || options.length === 0) return;

  for (const value of options) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    selectEl.appendChild(opt);
  }

  if (options.includes(previousValue)) {
    selectEl.value = previousValue;
  } else {
    selectEl.value = options[0];
  }
}

function setMultiSelectOptions(selectEl, options, selectedValues) {
  if (!selectEl) return;
  const selected = new Set(selectedValues || []);
  selectEl.innerHTML = "";
  if (!options || options.length === 0) return;

  for (const value of options) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    opt.selected = selected.has(value);
    selectEl.appendChild(opt);
  }
}

function initSingleHardwareForms(root) {
  const config = loadUiConfig();
  if (!config || !config.hardwareMeta) return;

  const forms = root.querySelectorAll("form[data-mamf-single-hardware-form]");
  for (const form of forms) {
    if (form.dataset.mamfInit === "1") continue;
    form.dataset.mamfInit = "1";

    const hardwareSelect = form.querySelector('select[name="hardware"]');
    if (!hardwareSelect) continue;

    const dtypeSelect = form.querySelector('select[name="dtype"]');
    const torchSelect = form.querySelector('select[name="torch_version"]');
    const torchGroup = form.querySelector("[data-mamf-torch-group]");

    const apply = () => {
      const hw = hardwareSelect.value;
      const meta = config.hardwareMeta[hw] || null;
      const allValue = config.allHardwareValue;

      if (dtypeSelect) {
        let dtypes = config.allDtypes || [];
        if (hw && hw !== allValue && meta && Array.isArray(meta.dtypes) && meta.dtypes.length) {
          dtypes = meta.dtypes;
        }
        setSelectOptions(dtypeSelect, dtypes, dtypeSelect.value);
      }

      if (torchSelect) {
        let torchVersions = [];
        if (config.baselineTorchVersion) {
          torchVersions = [config.baselineTorchVersion];
        }
        if (hw === allValue) {
          torchVersions = config.baselineTorchVersion ? [config.baselineTorchVersion] : [];
        } else if (hw && meta && Array.isArray(meta.torch_versions) && meta.torch_versions.length) {
          torchVersions = meta.torch_versions;
        }

        const previousTorch = torchSelect.value;
        let nextTorch = previousTorch;
        if (!torchVersions.includes(nextTorch)) {
          if (config.baselineTorchVersion && torchVersions.includes(config.baselineTorchVersion)) {
            nextTorch = config.baselineTorchVersion;
          } else if (torchVersions.length) {
            nextTorch = torchVersions[torchVersions.length - 1];
          } else {
            nextTorch = "";
          }
        }

        setSelectOptions(torchSelect, torchVersions, nextTorch);

        const shouldShowTorch = hw !== allValue && torchVersions.length > 1;
        if (torchGroup) {
          torchGroup.classList.toggle("hidden", !shouldShowTorch);
        }
      }
    };

    hardwareSelect.addEventListener("change", apply);
    apply();
  }
}

function initCompareConfigForms(root) {
  const config = loadUiConfig();
  if (!config || !config.hardwareMeta) return;

  const forms = root.querySelectorAll("form[data-mamf-compare-config-form]");
  for (const form of forms) {
    if (form.dataset.mamfCompareConfigInit === "1") continue;
    form.dataset.mamfCompareConfigInit = "1";

    const setup = (suffix) => {
      const hardwareSelect = form.querySelector(`select[name="hardware_${suffix}"]`);
      if (!hardwareSelect) return;

      const dtypeSelect = form.querySelector(`select[name="dtype_${suffix}"]`);
      const torchSelect = form.querySelector(`select[name="torch_version_${suffix}"]`);
      const torchGroup = form.querySelector(`[data-mamf-torch-group="${suffix}"]`);

      const apply = () => {
        const hw = hardwareSelect.value;
        const meta = config.hardwareMeta[hw] || null;

        if (dtypeSelect) {
          let dtypes = config.allDtypes || [];
          if (hw && meta && Array.isArray(meta.dtypes) && meta.dtypes.length) {
            dtypes = meta.dtypes;
          }
          setSelectOptions(dtypeSelect, dtypes, dtypeSelect.value);
        }

        if (torchSelect) {
          let torchVersions = [];
          if (config.baselineTorchVersion) {
            torchVersions = [config.baselineTorchVersion];
          }
          if (hw && meta && Array.isArray(meta.torch_versions) && meta.torch_versions.length) {
            torchVersions = meta.torch_versions;
          }

          const previousTorch = torchSelect.value;
          let nextTorch = previousTorch;
          if (!torchVersions.includes(nextTorch)) {
            if (config.baselineTorchVersion && torchVersions.includes(config.baselineTorchVersion)) {
              nextTorch = config.baselineTorchVersion;
            } else if (torchVersions.length) {
              nextTorch = torchVersions[torchVersions.length - 1];
            } else {
              nextTorch = "";
            }
          }

          setSelectOptions(torchSelect, torchVersions, nextTorch);

          const shouldShowTorch = torchVersions.length > 1;
          if (torchGroup) {
            torchGroup.classList.toggle("hidden", !shouldShowTorch);
          }
        }
      };

      hardwareSelect.addEventListener("change", apply);
      apply();
    };

    setup("a");
    setup("b");
  }
}

function initBrowseForms(root) {
  const config = loadUiConfig();
  if (!config) return;

  const forms = root.querySelectorAll("form[data-mamf-browse-form]");
  for (const form of forms) {
    if (form.dataset.mamfBrowseInit === "1") continue;
    form.dataset.mamfBrowseInit = "1";

    const hardwareSelect = form.querySelector('select[name="hardware_values"]');
    const torchSelect = form.querySelector('select[name="torch_versions"]');
    const torchGroup = form.querySelector("[data-mamf-browse-torch-group]");
    if (!hardwareSelect || !torchSelect) continue;

    const apply = () => {
      const selectedHardware = Array.from(hardwareSelect.selectedOptions).map((o) => o.value);
      const allowMulti =
        selectedHardware.length === 1 &&
        Array.isArray(config.multiTorchHardware) &&
        config.multiTorchHardware.includes(selectedHardware[0]);

      let torchOptions = [];
      if (allowMulti) {
        torchOptions = config.allTorchVersions || [];
      } else if (config.baselineTorchVersion) {
        torchOptions = [config.baselineTorchVersion];
      } else {
        torchOptions = (config.allTorchVersions || []).slice(-1);
      }

      const currentSelected = Array.from(torchSelect.selectedOptions).map((o) => o.value);
      let nextSelected = currentSelected.filter((v) => torchOptions.includes(v));
      if (nextSelected.length === 0 && torchOptions.length) {
        nextSelected = [torchOptions[torchOptions.length - 1]];
      }

      setMultiSelectOptions(torchSelect, torchOptions, nextSelected);

      if (torchGroup) {
        torchGroup.classList.toggle("hidden", torchOptions.length <= 1);
      }
    };

    hardwareSelect.addEventListener("change", apply);
    apply();
  }
}

function initScalingForms(root) {
  const config = loadUiConfig();
  if (!config || !config.hardwareMeta) return;

  const forms = root.querySelectorAll("form[data-mamf-scaling-form]");
  for (const form of forms) {
    if (form.dataset.mamfScalingInit === "1") continue;
    form.dataset.mamfScalingInit = "1";

    const dtypeSelect = form.querySelector('select[name="dtype"]');
    const hardwareSelect = form.querySelector('select[name="hardware"]');
    if (!dtypeSelect || !hardwareSelect) continue;

    const apply = () => {
      const dtype = dtypeSelect.value;
      const allHardware = config.hardwareList || [];
      const allowed = allHardware.filter((hw) => {
        const meta = config.hardwareMeta[hw] || null;
        const dtypes = meta && Array.isArray(meta.dtypes) ? meta.dtypes : [];
        return !dtype || dtypes.includes(dtype);
      });

      const options = allowed.length ? allowed : allHardware;
      const selected = Array.from(hardwareSelect.selectedOptions).map((o) => o.value);
      const nextSelected = selected.filter((v) => options.includes(v));

      setMultiSelectOptions(hardwareSelect, options, nextSelected);

      if (nextSelected.length === 0 && options.length) {
        hardwareSelect.options[0].selected = true;
        if (options.length > 1) hardwareSelect.options[1].selected = true;
      }
    };

    dtypeSelect.addEventListener("change", apply);
    apply();
  }
}

function prefersReducedMotion() {
  if (!window.matchMedia) return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function initTiltCards(root) {
  if (prefersReducedMotion()) return;

  const cards = root.querySelectorAll("[data-mamf-tilt]");
  for (const card of cards) {
    if (card.dataset.mamfTiltInit === "1") continue;
    card.dataset.mamfTiltInit = "1";

    const maxTilt = Number(card.getAttribute("data-mamf-tilt-max") || "8");
    const glare = card.querySelector("[data-mamf-tilt-glare]");

    let rafId = 0;
    let lastEvent = null;

    const update = () => {
      rafId = 0;
      if (!lastEvent) return;
      const rect = card.getBoundingClientRect();
      if (!rect.width || !rect.height) return;

      const x = clamp((lastEvent.clientX - rect.left) / rect.width, 0, 1);
      const y = clamp((lastEvent.clientY - rect.top) / rect.height, 0, 1);

      const rx = (0.5 - y) * maxTilt;
      const ry = (x - 0.5) * maxTilt;

      card.style.setProperty("--mx", `${(x * 100).toFixed(2)}%`);
      card.style.setProperty("--my", `${(y * 100).toFixed(2)}%`);
      card.style.transform = `perspective(900px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg) translateZ(0)`;

      if (glare) {
        glare.style.opacity = "1";
      }
    };

    const onMove = (evt) => {
      if (evt.pointerType === "touch") return;
      lastEvent = evt;
      if (!rafId) rafId = requestAnimationFrame(update);
    };

    const onLeave = () => {
      lastEvent = null;
      card.style.transition = "transform 240ms ease";
      card.style.transform = "perspective(900px) rotateX(0deg) rotateY(0deg) translateZ(0)";
      if (glare) glare.style.opacity = "0";
    };

    const onEnter = () => {
      card.style.willChange = "transform";
      card.style.transition = "transform 80ms ease-out";
    };

    card.addEventListener("pointerenter", onEnter);
    card.addEventListener("pointermove", onMove);
    card.addEventListener("pointerleave", onLeave);
  }
}

function initHeroCanvases(root) {
  const canvases = root.querySelectorAll("canvas[data-mamf-hero-canvas]");
  for (const canvas of canvases) {
    if (canvas.dataset.mamfHeroInit === "1") continue;
    canvas.dataset.mamfHeroInit = "1";

    const ctx = canvas.getContext("2d");
    if (!ctx) continue;
    if (prefersReducedMotion()) continue;

    const container = canvas.closest("[data-mamf-hero]") || canvas.parentElement;
    if (!container) continue;

    const palette = [
      [56, 189, 248], // cyan-400
      [217, 70, 239], // fuchsia-500
      [16, 185, 129], // emerald-500
    ];

    const state = {
      dpr: 1,
      w: 0,
      h: 0,
      cx: 0,
      cy: 0,
      scale: 1,
      particles: [],
      lastT: performance.now(),
      running: true,
    };

    function resize() {
      const rect = container.getBoundingClientRect();
      const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
      const w = Math.max(1, Math.floor(rect.width));
      const h = Math.max(1, Math.floor(rect.height));
      state.dpr = dpr;
      state.w = w * dpr;
      state.h = h * dpr;
      canvas.width = state.w;
      canvas.height = state.h;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      state.cx = state.w * 0.5;
      state.cy = state.h * 0.5;
      state.scale = Math.min(state.w, state.h) * 0.55;
    }

    function newParticle(z) {
      const color = palette[Math.floor(Math.random() * palette.length)];
      return {
        x: (Math.random() * 2 - 1) * 1.25,
        y: (Math.random() * 2 - 1) * 1.25,
        z: z ?? (Math.random() * 2.2 + 0.2),
        s: Math.random() * 1.2 + 0.35,
        kind: Math.random() < 0.22 ? "sq" : "dot",
        c: color,
      };
    }

    function seedParticles() {
      const target = Math.max(140, Math.min(260, Math.floor((state.w * state.h) / (state.dpr * state.dpr * 7200))));
      state.particles = Array.from({ length: target }, () => newParticle());
    }

    function draw(t) {
      if (!state.running) return;
      const dt = Math.min(48, t - state.lastT) * 0.001;
      state.lastT = t;

      ctx.clearRect(0, 0, state.w, state.h);

      const rot = t * 0.00008;
      const cos = Math.cos(rot);
      const sin = Math.sin(rot);

      const speed = 0.42;
      const zMin = 0.18;
      const zMax = 2.4;

      for (const p of state.particles) {
        p.z -= dt * speed;
        if (p.z < zMin) {
          p.x = (Math.random() * 2 - 1) * 1.25;
          p.y = (Math.random() * 2 - 1) * 1.25;
          p.z = zMax;
          p.s = Math.random() * 1.2 + 0.35;
          p.kind = Math.random() < 0.22 ? "sq" : "dot";
          p.c = palette[Math.floor(Math.random() * palette.length)];
        }

        const rx = p.x * cos - p.y * sin;
        const ry = p.x * sin + p.y * cos;

        const invZ = 1 / p.z;
        const sx = state.cx + rx * state.scale * invZ;
        const sy = state.cy + ry * state.scale * invZ;

        if (sx < -20 || sy < -20 || sx > state.w + 20 || sy > state.h + 20) continue;

        const alpha = clamp(1 - (p.z - zMin) / (zMax - zMin), 0, 1) * 0.9;
        const size = p.s * 2.2 * invZ * state.dpr;

        const [r, g, b] = p.c;
        ctx.fillStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`;

        if (p.kind === "sq") {
          const half = size * 0.9;
          ctx.save();
          ctx.translate(sx, sy);
          ctx.rotate(rot * 1.2);
          ctx.fillRect(-half, -half, half * 2, half * 2);
          ctx.restore();
        } else {
          ctx.beginPath();
          ctx.arc(sx, sy, Math.max(0.6, size), 0, Math.PI * 2);
          ctx.fill();
        }
      }

      requestAnimationFrame(draw);
    }

    resize();
    seedParticles();
    if (typeof ResizeObserver !== "undefined") {
      const observer = new ResizeObserver(() => {
        resize();
        seedParticles();
      });
      observer.observe(container);
    } else {
      window.addEventListener(
        "resize",
        () => {
          resize();
          seedParticles();
        },
        { passive: true }
      );
    }

    document.addEventListener("visibilitychange", () => {
      state.running = !document.hidden;
      if (state.running) {
        state.lastT = performance.now();
        requestAnimationFrame(draw);
      }
    });

    requestAnimationFrame(draw);
  }
}

function renderPlotlyCharts(root) {
  if (!window.Plotly) return;

  const wrappers = root.querySelectorAll("[data-plotly]");
  for (const wrapper of wrappers) {
    if (wrapper.dataset.rendered === "1") continue;
    const specScript = wrapper.querySelector("script[data-plotly-spec]");
    const target = wrapper.querySelector("[data-plotly-target]");
    if (!specScript || !target) continue;
    try {
      const spec = JSON.parse(specScript.textContent || "{}");
      const data = spec.data || [];
      const layout = spec.layout || {};
      const config = spec.config || { responsive: true, displaylogo: false };
      Plotly.react(target, data, layout, config);
      wrapper.dataset.rendered = "1";
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Failed to render chart", err);
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initSingleHardwareForms(document);
  initCompareConfigForms(document);
  initBrowseForms(document);
  initScalingForms(document);
  initTiltCards(document);
  initHeroCanvases(document);
  renderPlotlyCharts(document);
});

document.body.addEventListener("htmx:afterSwap", (evt) => {
  initSingleHardwareForms(evt.target);
  initCompareConfigForms(evt.target);
  initBrowseForms(evt.target);
  initScalingForms(evt.target);
  initTiltCards(evt.target);
  initHeroCanvases(evt.target);
  renderPlotlyCharts(evt.target);
});
