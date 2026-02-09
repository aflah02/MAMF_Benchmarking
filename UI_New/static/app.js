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
  initBrowseForms(document);
  renderPlotlyCharts(document);
});

document.body.addEventListener("htmx:afterSwap", (evt) => {
  initSingleHardwareForms(evt.target);
  initBrowseForms(evt.target);
  renderPlotlyCharts(evt.target);
});
