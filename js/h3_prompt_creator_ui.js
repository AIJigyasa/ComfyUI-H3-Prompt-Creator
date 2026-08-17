import { app } from "/scripts/app.js";

const NODE_NAMES = new Set([
  "H3VideoPromptCreator",
  "H3FullReferenceVideoPromptCreator",
]);

const USE_CASE_NODE = "H3UseCasePromptCreator";

// Nodes that grow their reference_image_N inputs one slot at a time.
const DYNAMIC_IMAGE_NODES = new Set([
  "H3FullReferenceVideoPromptCreator",
  USE_CASE_NODE,
]);

// Widgets that stay visible for every use case: the common creative fields plus
// the Ollama plumbing. Everything else is shown only when the selected use case
// declares it.
const ALWAYS_VISIBLE = new Set([
  "use_case", "idea", "duration", "aspect_ratio", "language", "extra",
  "ollama_url", "ollama_model", "temperature", "request_timeout", "num_ctx",
  "max_output_tokens", "refresh_local_models", "ollama_status",
]);

let useCaseFieldsPromise = null;
function loadUseCaseFields() {
  if (!useCaseFieldsPromise) {
    useCaseFieldsPromise = fetch("/h3_prompt_creator/use_case_fields")
      .then((r) => r.json())
      .catch((e) => {
        console.warn("H3 Use Case: could not load field map", e);
        return null;
      });
  }
  return useCaseFieldsPromise;
}

function setWidgetVisible(node, widget, visible) {
  if (!widget) return;
  if (visible) {
    if (widget.__h3Type !== undefined) {
      widget.type = widget.__h3Type;
      widget.computeSize = widget.__h3ComputeSize;
      delete widget.__h3Type;
      delete widget.__h3ComputeSize;
    }
  } else if (widget.__h3Type === undefined) {
    widget.__h3Type = widget.type;
    widget.__h3ComputeSize = widget.computeSize;
    widget.type = "h3hidden";
    widget.computeSize = () => [0, -4];
  }
  // Multiline text widgets are DOM overlays and ignore the litegraph type.
  const el = widget.element ?? widget.inputEl;
  if (el?.style) el.style.display = visible ? "" : "none";
  if ("hidden" in widget) widget.hidden = !visible;
}

async function applyUseCaseVisibility(node) {
  const map = await loadUseCaseFields();
  if (!map || !node.widgets) return;
  const useCase = node.widgets.find((w) => w.name === "use_case")?.value;
  const entry = map.by_use_case?.[useCase];
  // Unknown use case: show everything rather than hide fields the user needs.
  const allowed = entry ? new Set(entry.fields) : null;

  for (const widget of node.widgets) {
    if (ALWAYS_VISIBLE.has(widget.name)) continue;
    setWidgetVisible(node, widget, allowed === null || allowed.has(widget.name));
  }

  const size = node.computeSize();
  node.setSize([Math.max(node.size[0], size[0]), size[1]]);
  node.setDirtyCanvas(true, true);
}

const IMAGE_PREFIX = "reference_image_";
const MAX_IMAGE_SLOTS = 6;

function setStatus(node, text) {
  let w = node.widgets?.find((x) => x.name === "ollama_status");
  if (!w) {
    w = node.addWidget("text", "ollama_status", "");
    w.serialize = false;
    w.computeSize = () => [0, -4];
  }
  w.value = text;
}

function addRefresh(node) {
  if (node.widgets?.some((w) => w.name === "refresh_local_models")) return;
  const button = node.addWidget("button", "refresh_local_models", "Refresh Ollama Models", async () => {
    const urlWidget = node.widgets?.find((w) => w.name === "ollama_url");
    const modelWidget = node.widgets?.find((w) => w.name === "ollama_model");
    const base = String(urlWidget?.value || "http://127.0.0.1:11434");
    try {
      const response = await fetch(`/h3_prompt_creator/ollama_models?url=${encodeURIComponent(base)}`);
      const data = await response.json();
      const names = Array.isArray(data.models) ? data.models : [];
      if (modelWidget && names.length) {
        modelWidget.options = modelWidget.options || {};
        modelWidget.options.values = names;
        if (!names.includes(String(modelWidget.value || ""))) modelWidget.value = names[0];
      }
      setStatus(node, names.length ? `Ollama: ${names.length} local model(s)` : "Ollama: no local models found");
      node.setDirtyCanvas(true, true);
    } catch (e) {
      setStatus(node, "Ollama: unavailable");
      console.warn("H3 Prompt Creator: Ollama refresh failed", e);
    }
  });
  button.serialize = false;
}

/**
 * Show only as many reference_image_N inputs as are needed: every connected
 * slot, plus exactly one free slot at the end. The Python side always declares
 * all six as optional, so hiding the unused ones is purely cosmetic and cannot
 * break execution — ComfyUI passes inputs by name.
 */
function syncImageSlots(node) {
  if (!node.inputs) return;

  let highestConnected = 0;
  for (const input of node.inputs) {
    if (!input?.name?.startsWith(IMAGE_PREFIX)) continue;
    const index = parseInt(input.name.slice(IMAGE_PREFIX.length), 10);
    if (Number.isFinite(index) && input.link != null && index > highestConnected) {
      highestConnected = index;
    }
  }

  // One spare slot after the last connected one, capped at the declared maximum.
  const wanted = Math.min(MAX_IMAGE_SLOTS, highestConnected + 1);

  // Remove trailing unconnected slots, highest first so indices stay valid.
  for (let n = MAX_IMAGE_SLOTS; n > wanted; n--) {
    const at = node.inputs.findIndex((i) => i.name === IMAGE_PREFIX + n);
    if (at !== -1 && node.inputs[at].link == null) node.removeInput(at);
  }

  // Add missing slots in ascending order so they stay in numeric order.
  for (let n = 1; n <= wanted; n++) {
    if (!node.inputs.some((i) => i.name === IMAGE_PREFIX + n)) {
      node.addInput(IMAGE_PREFIX + n, "IMAGE");
    }
  }

  const size = node.computeSize();
  // Never shrink below what the user has dragged out.
  node.setSize([Math.max(node.size[0], size[0]), Math.max(node.size[1], size[1])]);
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "H3PromptCreator.UI",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const isUseCase = nodeData.name === USE_CASE_NODE;
    const hasDynamicImages = DYNAMIC_IMAGE_NODES.has(nodeData.name);
    if (!isUseCase && !NODE_NAMES.has(nodeData.name)) return;

    if (hasDynamicImages) {
      const onConnectionsChange = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo, ioSlot) {
        const r = onConnectionsChange?.apply(this, arguments);
        // Defer: litegraph updates input.link after this callback returns.
        setTimeout(() => syncImageSlots(this), 0);
        return r;
      };
    }

    // Run after a saved workflow is restored, so links are attached first and
    // pruning cannot orphan them.
    const onAfterGraphConfigured = nodeType.prototype.onAfterGraphConfigured;
    nodeType.prototype.onAfterGraphConfigured = function () {
      const r = onAfterGraphConfigured?.apply(this, arguments);
      if (hasDynamicImages) syncImageSlots(this);
      if (isUseCase) applyUseCaseVisibility(this);
      return r;
    };
  },

  async nodeCreated(node) {
    if (node.comfyClass === USE_CASE_NODE) {
      const useCaseWidget = node.widgets?.find((w) => w.name === "use_case");
      if (useCaseWidget) {
        const prevCallback = useCaseWidget.callback;
        useCaseWidget.callback = function (...args) {
          const r = prevCallback?.apply(this, args);
          applyUseCaseVisibility(node);
          return r;
        };
      }
      applyUseCaseVisibility(node);
      syncImageSlots(node);
      return;
    }

    if (!NODE_NAMES.has(node.comfyClass)) return;
    addRefresh(node);

    if (node.comfyClass === "H3VideoPromptCreator") {
      node.size = [430, 520];
    } else {
      node.size = [500, 610];
      // Fresh node: collapse to a single empty reference slot.
      syncImageSlots(node);
    }
  },
});
