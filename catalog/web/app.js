"use strict";

const LEVELS = [
  { key: "domain", label: "数据域" },
  { key: "scene", label: "场景" },
  { key: "dataset", label: "数据集" },
  { key: "task_type", label: "任务类型" },
  { key: "primary_atomic_action", label: "原子动作" },
  { key: "native_task", label: "具体任务" },
];

const CONVERSION_STATUSES = new Set(["converted", "partial", "not_converted", "excluded", "unknown"]);
const CONVERSION_LABELS = {
  converted: "已转换",
  partial: "部分转换",
  not_converted: "未转换",
  excluded: "已排除",
  unknown: "状态未知",
};
const DECISION_LABELS = {
  unreviewed: "未标注",
  keep: "保留",
  exclude: "排除",
  review: "需复查",
};
const MANUAL_REVIEW_LABELS = {
  processed: "已复核",
  unprocessed: "未复核",
};
const VIDEO_PLAYBACK_STATUSES = new Set([
  "ready",
  "unverified",
  "transcode_required",
  "unsupported",
  "missing",
  "blocked",
]);
const VIDEO_WALL_VIEWS = [
  { key: "left", label: "左视角", preferences: ["left_wrist", "left"] },
  { key: "right", label: "右视角", preferences: ["right_wrist", "right"] },
  {
    key: "main",
    label: "主视角",
    preferences: ["head", "front", "ego", "third_person", "external", "overhead", "top"],
  },
];

const state = {
  catalog: null,
  recordsById: new Map(),
  embeddingPoints: [],
  embeddingPointsById: new Map(),
  embeddingError: "",
  sceneOrder: [],
  tree: null,
  nodesById: new Map(),
  expanded: new Set(["root"]),
  selectedNodeId: "root",
  activeLevelKeys: new Set(LEVELS.map((level) => level.key)),
  decisions: {},
  processing: {},
  storageKey: "dataset-semantic-explorer",
  processingStorageKey: "dataset-semantic-explorer:processing:v1",
  activeView: "tree",
  scatterTransform: { x: 0, y: 0, k: 1 },
  scatterDrag: null,
  scatterBounds: null,
  visibleScatterRecords: new Map(),
  selectedPointId: null,
  videoConfig: { enabled: false, apiBaseUrl: "", message: "视频服务尚未配置。" },
  videoSession: null,
  governanceTarget: null,
  lastGovernanceChange: null,
  toastTimer: null,
};

const elements = {};

document.addEventListener("DOMContentLoaded", () => {
  Object.assign(elements, {
    catalogMeta: document.getElementById("catalog-meta"),
    statRecords: document.getElementById("stat-records"),
    statTasks: document.getElementById("stat-tasks"),
    statDatasets: document.getElementById("stat-datasets"),
    statDecisions: document.getElementById("stat-decisions"),
    statConverted: document.getElementById("stat-converted"),
    statPartial: document.getElementById("stat-partial"),
    statReviewed: document.getElementById("stat-reviewed"),
    treeTab: document.getElementById("view-tree-tab"),
    scatterTab: document.getElementById("view-scatter-tab"),
    treeView: document.getElementById("tree-view"),
    scatterView: document.getElementById("scatter-view"),
    search: document.getElementById("search"),
    domain: document.getElementById("filter-domain"),
    dataset: document.getElementById("filter-dataset"),
    scene: document.getElementById("filter-scene"),
    taskType: document.getElementById("filter-task-type"),
    decision: document.getElementById("filter-decision"),
    conversion: document.getElementById("filter-conversion"),
    processing: document.getElementById("filter-processing"),
    resetFilters: document.getElementById("reset-filters"),
    hierarchyLevels: document.getElementById("hierarchy-levels"),
    activeHierarchy: document.getElementById("active-hierarchy"),
    resetHierarchy: document.getElementById("reset-hierarchy"),
    breadcrumbs: document.getElementById("breadcrumbs"),
    visibleSummary: document.getElementById("visible-summary"),
    viewport: document.getElementById("tree-viewport"),
    svg: document.getElementById("tree"),
    canvas: document.getElementById("tree-canvas"),
    empty: document.getElementById("tree-empty"),
    detail: document.getElementById("detail-panel"),
    scatterSummary: document.getElementById("scatter-summary"),
    scatterMethod: document.getElementById("scatter-method"),
    scatterLegend: document.getElementById("scatter-legend"),
    scatterDetail: document.getElementById("scatter-detail"),
    scatterViewport: document.getElementById("scatter-viewport"),
    scatterSvg: document.getElementById("scatter"),
    scatterGrid: document.getElementById("scatter-grid"),
    scatterCanvas: document.getElementById("scatter-canvas"),
    scatterTooltip: document.getElementById("scatter-tooltip"),
    scatterEmpty: document.getElementById("scatter-empty"),
    scatterZoomIn: document.getElementById("scatter-zoom-in"),
    scatterZoomOut: document.getElementById("scatter-zoom-out"),
    scatterResetView: document.getElementById("scatter-reset-view"),
    exportManifest: document.getElementById("export-manifest"),
    unresolvedDialog: document.getElementById("unresolved-dialog"),
    unresolvedList: document.getElementById("unresolved-list"),
    showUnresolved: document.getElementById("show-unresolved"),
    closeUnresolved: document.getElementById("close-unresolved"),
    governanceDialog: document.getElementById("governance-dialog"),
    governanceDialogTitle: document.getElementById("governance-dialog-title"),
    governanceScopeTitle: document.getElementById("governance-scope-title"),
    governanceScopeCount: document.getElementById("governance-scope-count"),
    governanceScopeWarning: document.getElementById("governance-scope-warning"),
    governanceCurrentDecision: document.getElementById("governance-current-decision"),
    governanceCurrentReview: document.getElementById("governance-current-review"),
    governanceExcludeField: document.getElementById("governance-exclude-field"),
    governanceExcludeReason: document.getElementById("governance-exclude-reason"),
    closeGovernance: document.getElementById("close-governance"),
    cancelGovernance: document.getElementById("cancel-governance"),
    applyGovernance: document.getElementById("apply-governance"),
    toast: document.getElementById("toast"),
  });
  bindControls();
  setActiveView(viewFromHash(), { updateHash: false });
  loadCatalog();
});

async function loadCatalog() {
  elements.empty.hidden = true;
  try {
    const [catalog, videoConfig] = await Promise.all([
      loadCatalogData(),
      loadVideoConfig(),
    ]);
    validateCatalog(catalog);
    state.catalog = catalog;
    state.videoConfig = videoConfig;
    state.recordsById = new Map(catalog.records.map((record) => [record.id, record]));
    prepareEmbedding(catalog.embedding);
    state.storageKey = `dataset-semantic-explorer:${catalog.dataset_root}`;
    state.processingStorageKey = `${state.storageKey}:processing:v1`;
    state.decisions = loadDecisions();
    state.processing = loadProcessing();
    populateInterface();
    rebuildTree({ resetExpansion: true });
  } catch (error) {
    elements.catalogMeta.textContent = `目录索引加载失败：${error.message}`;
    elements.empty.hidden = false;
    const message = document.createElement("span");
    message.textContent = `目录数据暂时没有加载成功（${error.message}）。`;
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "button secondary";
    retry.textContent = "重新加载目录";
    retry.addEventListener("click", () => {
      retry.disabled = true;
      retry.textContent = "正在重试…";
      loadCatalog();
    });
    const content = document.createElement("div");
    content.className = "catalog-load-error";
    content.append(message, retry);
    elements.empty.replaceChildren(content);
  }
}

async function loadCatalogData() {
  const embeddedCatalog = globalThis.__DATASET_EXPLORER_CATALOG__;
  if (embeddedCatalog && typeof embeddedCatalog === "object") return embeddedCatalog;
  const response = await fetch(runtimeAssetUrl("./catalog.json"), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function runtimeAssetUrl(path) {
  const url = new URL(path, window.location.href);
  url.searchParams.set("_fresh", `${Date.now()}`);
  return url.href;
}

async function loadVideoConfig() {
  const unavailable = { enabled: false, apiBaseUrl: "", message: "视频服务尚未配置。" };
  try {
    const response = await fetch(runtimeAssetUrl("./video-config.json"), { cache: "no-store" });
    if (response.status === 404) return unavailable;
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    if (!payload || payload.schema_version !== 1 || payload.enabled !== true) {
      return {
        ...unavailable,
        message: typeof payload?.message === "string" && payload.message.trim() ? payload.message.trim() : unavailable.message,
      };
    }
    const apiBaseUrl = resolveVideoApiBaseUrl(payload.api_base_url);
    return {
      enabled: true,
      apiBaseUrl,
      message: typeof payload.message === "string" ? payload.message.trim() : "",
    };
  } catch (error) {
    return { ...unavailable, message: `视频服务配置不可用：${error.message}` };
  }
}

function resolveVideoApiBaseUrl(value) {
  const parsed = new URL(String(value || "").trim(), window.location.href);
  if (
    !/^https?:$/.test(parsed.protocol)
    || parsed.username
    || parsed.password
    || parsed.search
    || parsed.hash
  ) throw new Error("API 地址格式无效");
  if (window.location.protocol === "https:" && parsed.protocol !== "https:") {
    throw new Error("公网 HTTPS 页面只能连接 HTTPS 视频服务");
  }
  return parsed.href.replace(/\/+$/, "");
}

function validateCatalog(catalog) {
  if (!catalog || catalog.schema_version !== 1 || !Array.isArray(catalog.records)) {
    throw new Error("不支持的 catalog schema");
  }
  for (const record of catalog.records) {
    if (!record.id || !record.dataset || !record.native_task || !record.source_path) {
      throw new Error("catalog 中存在缺少必要字段的任务记录");
    }
  }
}

function prepareEmbedding(embedding) {
  state.embeddingPoints = [];
  state.embeddingPointsById = new Map();
  state.embeddingError = "";
  state.sceneOrder = [];
  state.scatterBounds = null;
  if (!embedding) {
    state.embeddingError = "当前目录册尚未生成任务词条语义向量。";
    return;
  }
  if (embedding.schema_version !== 1 || embedding.dimensions !== 2 || !Array.isArray(embedding.points)) {
    state.embeddingError = "目录册中的语义向量格式不受支持。";
    return;
  }

  const seen = new Set();
  for (const rawPoint of embedding.points) {
    const validRecordIds = Array.isArray(rawPoint.record_ids)
      ? rawPoint.record_ids.filter((recordId) => state.recordsById.has(recordId))
      : [];
    if (
      !rawPoint.id ||
      seen.has(rawPoint.id) ||
      !rawPoint.task ||
      !rawPoint.scene ||
      !Number.isFinite(rawPoint.x) ||
      !Number.isFinite(rawPoint.y) ||
      !validRecordIds.length ||
      validRecordIds.length !== rawPoint.record_ids.length
    ) {
      state.embeddingError = "目录册中的语义向量点与任务记录不一致。";
      state.embeddingPoints = [];
      state.embeddingPointsById = new Map();
      return;
    }
    seen.add(rawPoint.id);
    const point = { ...rawPoint, record_ids: validRecordIds, displayDx: 0, displayDy: 0 };
    state.embeddingPoints.push(point);
    state.embeddingPointsById.set(point.id, point);
  }

  const pointsByTask = new Map();
  for (const point of state.embeddingPoints) {
    if (!pointsByTask.has(point.task)) pointsByTask.set(point.task, []);
    pointsByTask.get(point.task).push(point);
  }
  for (const points of pointsByTask.values()) {
    if (points.length < 2) continue;
    points.sort((left, right) => left.id.localeCompare(right.id));
    points.forEach((point, index) => {
      const angle = (Math.PI * 2 * index) / points.length;
      point.displayDx = Math.cos(angle) * 4;
      point.displayDy = Math.sin(angle) * 4;
    });
  }
  state.sceneOrder = [...new Set(state.embeddingPoints.map((point) => point.scene))].sort((a, b) =>
    a.localeCompare(b, "zh-CN"),
  );
  state.scatterBounds = embeddingBounds(state.embeddingPoints);
}

function bindControls() {
  const filterElements = [
    elements.domain,
    elements.dataset,
    elements.scene,
    elements.taskType,
    elements.decision,
    elements.conversion,
    elements.processing,
  ];
  for (const select of filterElements) {
    select.addEventListener("change", () => {
      if (select === elements.domain) synchronizeDomainDatasetFilters("domain");
      if (select === elements.dataset) synchronizeDomainDatasetFilters("dataset");
      rebuildTree({ resetExpansion: true });
    });
  }
  elements.search.addEventListener("input", debounce(() => rebuildTree({ resetExpansion: true }), 180));
  elements.resetFilters.addEventListener("click", () => {
    elements.search.value = "";
    for (const select of filterElements) select.value = "";
    updateDatasetOptionsForDomain();
    rebuildTree({ resetExpansion: true });
  });
  elements.resetHierarchy.addEventListener("click", () => {
    state.activeLevelKeys = new Set(LEVELS.map((level) => level.key));
    for (const input of elements.hierarchyLevels.querySelectorAll("input")) {
      input.checked = true;
      input.closest(".hierarchy-level")?.classList.add("is-active");
    }
    rebuildTree({ resetExpansion: true });
  });
  elements.treeTab.addEventListener("click", () => setActiveView("tree"));
  elements.scatterTab.addEventListener("click", () => setActiveView("scatter"));
  for (const tab of [elements.treeTab, elements.scatterTab]) {
    tab.addEventListener("keydown", handleViewTabKeydown);
  }
  window.addEventListener("hashchange", () => setActiveView(viewFromHash(), { updateHash: false }));
  elements.scatterZoomIn.addEventListener("click", () => zoomScatterBy(1.25));
  elements.scatterZoomOut.addEventListener("click", () => zoomScatterBy(1 / 1.25));
  elements.scatterResetView.addEventListener("click", resetScatterView);
  elements.exportManifest.addEventListener("click", exportManifest);
  elements.showUnresolved.addEventListener("click", () => elements.unresolvedDialog.showModal());
  elements.closeUnresolved.addEventListener("click", () => elements.unresolvedDialog.close());
  elements.unresolvedDialog.addEventListener("click", (event) => {
    if (event.target === elements.unresolvedDialog) elements.unresolvedDialog.close();
  });
  elements.closeGovernance.addEventListener("click", closeGovernanceDialog);
  elements.cancelGovernance.addEventListener("click", closeGovernanceDialog);
  elements.applyGovernance.addEventListener("click", applyGovernanceChanges);
  for (const input of document.querySelectorAll('input[name="governance-decision"]')) {
    input.addEventListener("change", updateGovernanceExcludeField);
  }
  elements.governanceDialog.addEventListener("click", (event) => {
    if (event.target === elements.governanceDialog) closeGovernanceDialog();
  });
  elements.governanceDialog.addEventListener("close", () => {
    state.governanceTarget = null;
  });
  elements.scatterSvg.addEventListener("wheel", handleScatterWheel, { passive: false });
  elements.scatterSvg.addEventListener("pointerdown", handleScatterPointerDown);
  elements.scatterSvg.addEventListener("pointermove", handleScatterHover);
  elements.scatterSvg.addEventListener("pointerleave", hideScatterTooltip);
  elements.scatterSvg.addEventListener("click", handleScatterClick);
  window.addEventListener("pointermove", handleScatterPointerMove);
  window.addEventListener("pointerup", handleScatterPointerUp);
  window.addEventListener(
    "resize",
    debounce(() => {
      if (state.activeView === "scatter") renderScatter(filteredRecords(), { renderDetail: false });
    }, 100),
  );
}

function viewFromHash() {
  return window.location.hash === "#scatter" ? "scatter" : "tree";
}

function setActiveView(view, { updateHash = true } = {}) {
  const nextView = view === "scatter" ? "scatter" : "tree";
  const viewChanged = state.activeView !== nextView;
  if (viewChanged) disposeVideoSession();
  state.activeView = nextView;
  const scatterActive = nextView === "scatter";
  elements.treeView.hidden = scatterActive;
  elements.scatterView.hidden = !scatterActive;
  elements.treeTab.setAttribute("aria-selected", String(!scatterActive));
  elements.scatterTab.setAttribute("aria-selected", String(scatterActive));
  elements.treeTab.tabIndex = scatterActive ? -1 : 0;
  elements.scatterTab.tabIndex = scatterActive ? 0 : -1;
  if (updateHash && window.location.hash !== `#${nextView}`) {
    window.history.pushState(null, "", `#${nextView}`);
  }
  if (scatterActive && state.catalog && viewChanged) {
    window.requestAnimationFrame(() => renderScatter(filteredRecords()));
  } else if (state.catalog && viewChanged) {
    renderDetail();
  }
}

function handleViewTabKeydown(event) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  const nextView = event.key === "ArrowRight" || event.key === "End" ? "scatter" : "tree";
  setActiveView(nextView);
  (nextView === "scatter" ? elements.scatterTab : elements.treeTab).focus();
}

function populateInterface() {
  const { catalog } = state;
  const generated = new Date(catalog.generated_at);
  const generatedText = Number.isNaN(generated.valueOf()) ? catalog.generated_at : generated.toLocaleString("zh-CN");
  const conversionChecked = catalog.conversion?.checked_at;
  const conversionDate = conversionChecked ? new Date(conversionChecked) : null;
  const conversionText = conversionDate && !Number.isNaN(conversionDate.valueOf())
    ? conversionDate.toLocaleString("zh-CN")
    : conversionChecked;
  elements.catalogMeta.textContent = `${catalog.dataset_root} · 目录扫描于 ${generatedText}${conversionText ? ` · 转换状态核验于 ${conversionText}` : ""}`;
  elements.statRecords.textContent = catalog.stats.records.toLocaleString("zh-CN");
  elements.statTasks.textContent = catalog.stats.unique_tasks.toLocaleString("zh-CN");
  elements.statDatasets.textContent = catalog.stats.datasets.toLocaleString("zh-CN");
  populateSelect(
    elements.domain,
    Object.keys(catalog.domain_labels),
    (value) => catalog.domain_labels[value] || value,
  );
  const referenceByDataset = new Map(
    (catalog.reference_datasets || []).map((item) => [item.dataset, item]),
  );
  const extractedDatasets = new Set(uniqueValues(catalog.records, "dataset"));
  const datasetValues = [
    ...new Set([
      ...uniqueValues(catalog.records, "dataset"),
      ...(catalog.reference_datasets || []).map((item) => item.dataset),
    ]),
  ].sort((a, b) => a.localeCompare(b, "zh-CN"));
  populateSelect(elements.dataset, datasetValues, (value) => {
    const reference = referenceByDataset.get(value);
    if (reference?.availability === "not_downloaded") return `${value}（未下载）`;
    if (reference?.availability === "downloaded" && !extractedDatasets.has(value)) {
      return `${value}（任务待提取）`;
    }
    return value;
  });
  updateDatasetOptionsForDomain();
  populateSelect(elements.scene, uniqueValues(catalog.records, "scene"));
  populateSelect(elements.taskType, uniqueValues(catalog.records, "task_type"));
  if (catalog.embedding) {
    const revision = String(catalog.embedding.model_revision || "").slice(0, 8);
    const dimension = Number.isInteger(catalog.embedding.embedding_dimension)
      ? `${catalog.embedding.embedding_dimension} 维 → 2 维`
      : "二维投影";
    elements.scatterMethod.textContent = `${catalog.embedding.method_label} · ${dimension} · 版本 ${revision || "未知"} · ${catalog.embedding.point_count.toLocaleString("zh-CN")} 个词条`;
  } else elements.scatterMethod.textContent = state.embeddingError;
  renderHierarchyControls();
  renderUnresolved();
  updateDecisionCount();
}

function populateSelect(select, values, labeler = (value) => value) {
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labeler(value);
    select.append(option);
  }
}

function uniqueValues(records, key) {
  return [...new Set(records.map((record) => record[key]).filter(Boolean))].sort((a, b) => a.localeCompare(b, "zh-CN"));
}

function datasetDomains(dataset) {
  if (!dataset || !state.catalog) return new Set();
  const domains = new Set(
    state.catalog.records
      .filter((record) => record.dataset === dataset && record.domain)
      .map((record) => record.domain),
  );
  for (const reference of state.catalog.reference_datasets || []) {
    if (reference.dataset === dataset && reference.domain) domains.add(reference.domain);
  }
  return domains;
}

function updateDatasetOptionsForDomain() {
  if (!elements.dataset || !elements.domain) return;
  const domain = elements.domain.value;
  for (const option of elements.dataset.options) {
    if (!option.value) continue;
    const compatible = !domain || datasetDomains(option.value).has(domain);
    option.hidden = !compatible;
    option.disabled = !compatible;
  }
}

function synchronizeDomainDatasetFilters(changedFilter) {
  if (changedFilter === "dataset" && elements.dataset.value) {
    const domains = datasetDomains(elements.dataset.value);
    if (domains.size === 1) elements.domain.value = [...domains][0];
    else if (domains.size > 1) elements.domain.value = "";
  } else if (changedFilter === "domain" && elements.dataset.value) {
    const domains = datasetDomains(elements.dataset.value);
    if (elements.domain.value && !domains.has(elements.domain.value)) elements.dataset.value = "";
  }
  updateDatasetOptionsForDomain();
}

function activeLevels() {
  return LEVELS.filter((level) => state.activeLevelKeys.has(level.key));
}

function levelValue(record, levelKey) {
  if (levelKey === "domain") return state.catalog.domain_labels[record.domain] || record.domain;
  return record[levelKey] || `未提供${LEVELS.find((level) => level.key === levelKey)?.label || "分类"}`;
}

function renderHierarchyControls() {
  elements.hierarchyLevels.replaceChildren();
  LEVELS.forEach((level, index) => {
    const label = document.createElement("label");
    label.className = "hierarchy-level is-active";
    label.htmlFor = `hierarchy-${level.key}`;

    const input = document.createElement("input");
    input.id = `hierarchy-${level.key}`;
    input.type = "checkbox";
    input.value = level.key;
    input.checked = true;
    input.addEventListener("change", () => {
      if (input.checked) state.activeLevelKeys.add(level.key);
      else state.activeLevelKeys.delete(level.key);
      label.classList.toggle("is-active", input.checked);
      rebuildTree({ resetExpansion: true });
    });

    const name = document.createElement("span");
    name.className = "hierarchy-level-name";
    name.textContent = `${index + 1}. ${level.label}`;
    const count = document.createElement("span");
    count.className = "hierarchy-level-count";
    count.dataset.levelCount = level.key;
    count.textContent = "—";
    label.append(input, name, count);
    elements.hierarchyLevels.append(label);
  });
  renderActiveHierarchy();
}

function updateHierarchyCounts(records) {
  for (const level of LEVELS) {
    const count = new Set(records.map((record) => levelValue(record, level.key))).size;
    const target = elements.hierarchyLevels.querySelector(`[data-level-count="${level.key}"]`);
    if (target) target.textContent = count.toLocaleString("zh-CN");
  }
}

function renderActiveHierarchy() {
  elements.activeHierarchy.replaceChildren();
  const levels = activeLevels();
  if (!levels.length) {
    const item = document.createElement("li");
    item.className = "hierarchy-empty";
    item.textContent = "仅显示全部数据";
    elements.activeHierarchy.append(item);
    return;
  }
  for (const level of levels) {
    const item = document.createElement("li");
    item.textContent = level.label;
    elements.activeHierarchy.append(item);
  }
}

function filteredRecords() {
  const query = normalizeSearch(elements.search.value);
  return state.catalog.records.filter((record) => {
    if (elements.domain.value && record.domain !== elements.domain.value) return false;
    if (elements.dataset.value && record.dataset !== elements.dataset.value) return false;
    if (elements.scene.value && record.scene !== elements.scene.value) return false;
    if (elements.taskType.value && record.task_type !== elements.taskType.value) return false;
    const decision = state.decisions[record.id]?.decision || "unreviewed";
    if (elements.decision.value && decision !== elements.decision.value) return false;
    const conversion = conversionStatus(record);
    if (elements.conversion.value && conversion !== elements.conversion.value) return false;
    const manualReview = isReviewed(record.id) ? "processed" : "unprocessed";
    if (elements.processing.value && manualReview !== elements.processing.value) return false;
    if (!query) return true;
    const haystack = normalizeSearch(
      [
      record.native_task,
      record.dataset,
      record.native_group,
      record.native_scene,
      record.scene,
      record.task_type,
      record.primary_atomic_action,
      ...(record.atomic_actions || []),
      record.source_path,
      ]
        .filter(Boolean)
        .join(" "),
    );
    return haystack.includes(query);
  });
}

function currentManifestFilters() {
  const filters = {};
  const search = elements.search.value.trim();
  if (search) filters.search = search;
  if (elements.domain.value) filters.domain = elements.domain.value;
  if (elements.dataset.value) filters.dataset = elements.dataset.value;
  if (elements.scene.value) filters.scene = elements.scene.value;
  if (elements.taskType.value) filters.task_type = elements.taskType.value;
  if (elements.decision.value) filters.decision = elements.decision.value;
  if (elements.conversion.value) filters.conversion_status = elements.conversion.value;
  if (elements.processing.value) filters.manual_review_status = elements.processing.value;
  return filters;
}

function manifestFilterSummary(filters) {
  const parts = [];
  if (filters.search) parts.push(`搜索“${filters.search}”`);
  if (filters.domain) parts.push(`数据域：${state.catalog.domain_labels[filters.domain] || filters.domain}`);
  if (filters.dataset) parts.push(`数据集：${filters.dataset}`);
  if (filters.scene) parts.push(`场景：${filters.scene}`);
  if (filters.task_type) parts.push(`任务类型：${filters.task_type}`);
  if (filters.decision) parts.push(`筛选结论：${DECISION_LABELS[filters.decision] || filters.decision}`);
  if (filters.conversion_status) {
    parts.push(`转换状态：${CONVERSION_LABELS[filters.conversion_status] || filters.conversion_status}`);
  }
  if (filters.manual_review_status) {
    parts.push(`人工复核：${MANUAL_REVIEW_LABELS[filters.manual_review_status] || filters.manual_review_status}`);
  }
  return parts.join(" · ") || "全部数据";
}

function updateExportControl(records) {
  const filters = currentManifestFilters();
  const isFiltered = Object.keys(filters).length > 0;
  const count = records.length.toLocaleString("zh-CN");
  elements.exportManifest.disabled = records.length === 0;
  elements.exportManifest.textContent = isFiltered ? `导出当前筛选（${count}）` : `导出全部数据（${count}）`;
  const summary = manifestFilterSummary(filters);
  elements.exportManifest.title = records.length
    ? `将导出 ${count} 条路径记录 · ${summary}`
    : `当前筛选没有可导出的路径记录 · ${summary}`;
  elements.exportManifest.setAttribute("aria-label", elements.exportManifest.title);
}

function rebuildTree({ resetExpansion = false } = {}) {
  if (!state.catalog) return;
  const records = filteredRecords();
  const levels = activeLevels();
  state.tree = buildTree(records);
  state.nodesById = indexNodes(state.tree);
  if (resetExpansion) {
    state.expanded = new Set(["root"]);
    for (const child of state.tree.children) state.expanded.add(child.id);
    if (records.length <= 30 || elements.search.value.trim()) expandThroughDepth(state.tree, levels.length - 1);
    state.selectedNodeId = "root";
  } else if (!state.nodesById.has(state.selectedNodeId)) {
    state.selectedNodeId = "root";
  }
  elements.empty.hidden = records.length > 0;
  if (!records.length) elements.empty.textContent = emptyStateMessage();
  const terminalLabel = state.activeLevelKeys.has("native_task") ? "具体任务" : "末级分类";
  elements.visibleSummary.textContent = `${records.length.toLocaleString("zh-CN")} 条路径记录 · ${countTerminalNodes(state.tree).toLocaleString("zh-CN")} 个${terminalLabel}`;
  updateExportControl(records);
  elements.svg.setAttribute("aria-label", `${levels.length} 层数据集语义分类目录`);
  updateHierarchyCounts(records);
  renderActiveHierarchy();
  renderTree();
  renderDetail();
  renderBreadcrumbs();
  if (state.activeView === "scatter") renderScatter(records);
}

function emptyStateMessage() {
  const references = state.catalog.reference_datasets || [];
  const selectedDataset = elements.dataset.value;
  const reference = references.find((item) => item.dataset === selectedDataset);
  if (reference?.availability === "not_downloaded") {
    const expected = reference.expected_path ? `，预期目录为 ${datasetPath(reference.expected_path)}` : "";
    return `“${reference.dataset}”已列入文档目录，但服务器上尚未发现对应数据${expected}。`;
  }
  if (reference?.availability === "downloaded") {
    return `“${reference.dataset}”的服务器目录已存在，`
      + "但尚未提取出可用于层级目录的任务 metadata。";
  }
  const hasNarrowingFilter =
    elements.search.value.trim() ||
    selectedDataset ||
    elements.scene.value ||
    elements.taskType.value ||
    elements.decision.value ||
    elements.conversion.value ||
    elements.processing.value;
  if (elements.domain.value === "vqa" && !hasNarrowingFilter) {
    const vqaReferences = references.filter((item) => item.domain === "vqa");
    const downloaded = vqaReferences.filter((item) => item.availability === "downloaded").length;
    return `VQA 目录册已收录 ${vqaReferences.length} 个候选数据集，当前检测到 ${downloaded} 个已下载目录。可打开“数据目录 / 待提取”查看清单。`;
  }
  return "当前筛选条件下没有任务路径记录。";
}

function buildTree(records) {
  const root = makeNode("root", "全部数据", -1, null, null);
  const levels = activeLevels();
  for (const record of records) {
    let node = root;
    node.recordIds.push(record.id);
    levels.forEach((level, depth) => {
      const name = levelValue(record, level.key);
      const id = `${node.id}\u001f${level.key}:${name}`;
      let child = node.childMap.get(id);
      if (!child) {
        child = makeNode(id, name, depth, node, level);
        node.childMap.set(id, child);
        node.children.push(child);
      }
      child.recordIds.push(record.id);
      node = child;
    });
  }
  sortTree(root);
  return root;
}

function makeNode(id, name, depth, parent, level) {
  return {
    id,
    name,
    depth,
    parent,
    levelKey: level?.key || null,
    levelLabel: level?.label || "总览",
    children: [],
    childMap: new Map(),
    recordIds: [],
    x: 0,
    y: 0,
  };
}

function sortTree(node) {
  node.children.sort((a, b) => b.recordIds.length - a.recordIds.length || a.name.localeCompare(b.name, "zh-CN"));
  for (const child of node.children) sortTree(child);
}

function indexNodes(root) {
  const index = new Map();
  const visit = (node) => {
    index.set(node.id, node);
    for (const child of node.children) visit(child);
  };
  visit(root);
  return index;
}

function expandThroughDepth(node, maxDepth) {
  if (node.depth < maxDepth && node.children.length) state.expanded.add(node.id);
  for (const child of node.children) expandThroughDepth(child, maxDepth);
}

function countTerminalNodes(root) {
  let count = 0;
  const visit = (node) => {
    if (node !== root && !node.children.length) count += 1;
    else for (const child of node.children) visit(child);
  };
  visit(root);
  return count;
}

function isTerminalNode(node) {
  return node !== state.tree && !node.children.length;
}

function aggregateEpisodes(recordIds) {
  let sum = 0;
  let knownRecords = 0;
  for (const id of recordIds) {
    const value = state.recordsById.get(id)?.episode_count;
    if (!Number.isInteger(value) || value < 0) continue;
    sum += value;
    knownRecords += 1;
  }
  const totalRecords = recordIds.length;
  const status = knownRecords === 0 ? "unknown" : knownRecords === totalRecords ? "complete" : "partial";
  return { sum, knownRecords, totalRecords, status };
}

function episodeNodeText(summary) {
  if (summary.status === "unknown") return "回合数未知";
  const count = summary.sum.toLocaleString("zh-CN");
  if (summary.status === "partial") return `≥${count} 回合 · ${summary.knownRecords}/${summary.totalRecords} 路径已知`;
  return `${count} 回合`;
}

function episodeAriaText(summary) {
  if (summary.status === "unknown") return "回合数尚未从目录提取";
  const count = summary.sum.toLocaleString("zh-CN");
  if (summary.status === "partial") {
    return `已知至少 ${count} 个回合，${summary.knownRecords}/${summary.totalRecords} 条路径已统计`;
  }
  return `共 ${count} 个回合`;
}

function episodeDetailText(summary) {
  if (summary.status === "unknown") return "未知（尚未从目录提取）";
  const count = summary.sum.toLocaleString("zh-CN");
  if (summary.status === "partial") {
    return `≥${count}（${summary.knownRecords}/${summary.totalRecords} 条路径已统计）`;
  }
  return `${count}（全部路径已统计）`;
}

function conversionStatus(record) {
  return CONVERSION_STATUSES.has(record?.conversion_status) ? record.conversion_status : "unknown";
}

function aggregateConversion(recordIds) {
  const counts = { converted: 0, partial: 0, not_converted: 0, excluded: 0, unknown: 0 };
  for (const id of recordIds) counts[conversionStatus(state.recordsById.get(id))] += 1;
  const total = recordIds.length;
  const exactStatus = total ? Object.entries(counts).find(([, count]) => count === total)?.[0] : "unknown";
  const status = exactStatus || "partial";
  return { counts, total, status };
}

function conversionNodeText(summary) {
  if (summary.status === "converted") return "✓ 已转换";
  if (summary.status === "excluded") return "⊘ 已排除";
  if (summary.status === "not_converted") return "○ 未转换";
  if (summary.status === "unknown") return "? 状态未知";
  if (summary.counts.partial === summary.total) return "◐ 部分转换";
  const pieces = [`◐ ${summary.counts.converted}/${summary.total} 已转换`];
  if (summary.counts.partial) pieces.push(`${summary.counts.partial} 部分`);
  return pieces.join(" · ");
}

function conversionAriaText(summary) {
  if (summary.status !== "partial") return CONVERSION_LABELS[summary.status] || CONVERSION_LABELS.unknown;
  const parts = [
    ["converted", "已转换"],
    ["partial", "部分转换"],
    ["not_converted", "未转换"],
    ["excluded", "已排除"],
    ["unknown", "状态未知"],
  ]
    .filter(([status]) => summary.counts[status] > 0)
    .map(([status, label]) => `${summary.counts[status]} 条${label}`);
  return parts.join("，");
}

function aggregateManualReview(recordIds) {
  const reviewed = recordIds.filter((id) => isReviewed(id)).length;
  const total = recordIds.length;
  const status = reviewed === 0 ? "unreviewed" : reviewed === total ? "reviewed" : "partial";
  return { reviewed, total, status };
}

function manualReviewAriaText(summary) {
  if (summary.status === "reviewed") return "已人工复核";
  if (summary.status === "partial") return `${summary.reviewed}/${summary.total} 条路径已人工复核`;
  return "未人工复核";
}

function findFirstNodeByLevel(root, levelKey) {
  if (root.levelKey === levelKey) return root;
  for (const child of root.children) {
    const match = findFirstNodeByLevel(child, levelKey);
    if (match) return match;
  }
  return null;
}

function visibleTreeNodes() {
  const nodes = [];
  const visit = (node) => {
    nodes.push(node);
    if (state.expanded.has(node.id)) {
      for (const child of node.children) visit(child);
    }
  };
  visit(state.tree);
  return nodes;
}

function renderTree() {
  if (!state.tree || !elements.canvas) return;
  elements.canvas.replaceChildren();
  const fragment = document.createDocumentFragment();
  for (const node of visibleTreeNodes()) fragment.append(createDirectoryNode(node));
  elements.canvas.append(fragment);
}

function createDirectoryNode(node) {
  const status = aggregateDecision(node.recordIds);
  const terminal = isTerminalNode(node);
  const episodes = terminal ? aggregateEpisodes(node.recordIds) : null;
  const conversion = terminal ? aggregateConversion(node.recordIds) : null;
  const conversionClass = conversion
    ? conversion.status === "converted"
      ? "converted"
      : conversion.status === "partial"
        ? "partially-converted"
        : conversion.status === "excluded"
          ? "conversion-excluded"
          : ""
    : "";
  const row = document.createElement("button");
  row.type = "button";
  row.className = [
    "directory-node",
    node.id === state.selectedNodeId ? "selected" : "",
    status,
    conversionClass,
    terminal ? "terminal" : "branch",
  ].filter(Boolean).join(" ");
  row.style.setProperty("--directory-depth", String(Math.max(0, node.depth + 1)));
  row.setAttribute("role", "treeitem");
  row.setAttribute("aria-level", String(node.depth + 2));
  if (node.children.length) row.setAttribute("aria-expanded", String(state.expanded.has(node.id)));
  if (node.id === state.selectedNodeId) row.setAttribute("aria-current", "true");
  row.setAttribute(
    "aria-label",
    `${node.name}，${node.recordIds.length} 条路径记录${conversion ? `，转换状态：${conversionAriaText(conversion)}` : ""}${episodes ? `，${episodeAriaText(episodes)}` : ""}`,
  );
  row.title = `${node.levelLabel} · ${node.name}`;

  const toggle = document.createElement("span");
  toggle.setAttribute("class", "toggle-mark");
  toggle.setAttribute("aria-hidden", "true");
  toggle.textContent = node.children.length ? (state.expanded.has(node.id) ? "▾" : "▸") : "•";

  const label = document.createElement("span");
  label.className = "directory-node-label";
  const level = document.createElement("small");
  level.className = "directory-node-level";
  level.textContent = node.levelLabel;
  const name = document.createElement("strong");
  name.className = "directory-node-name";
  name.textContent = node.name;
  label.append(level, name);
  if (episodes) {
    const meta = document.createElement("small");
    meta.className = "directory-node-meta";
    meta.textContent = `${conversionNodeText(conversion)} · ${episodeNodeText(episodes)}`;
    label.append(meta);
  }

  const count = document.createElement("span");
  count.setAttribute("class", "node-count");
  count.textContent = node.recordIds.length.toLocaleString("zh-CN");
  row.append(toggle, label, count);
  row.addEventListener("click", () => activateNode(node));
  return row;
}

function activateNode(node) {
  state.selectedNodeId = node.id;
  if (node.children.length) {
    if (state.expanded.has(node.id)) state.expanded.delete(node.id);
    else state.expanded.add(node.id);
  }
  renderTree();
  renderDetail();
  renderBreadcrumbs();
}

function expandAncestors(node) {
  for (let current = node; current; current = current.parent) state.expanded.add(current.id);
}

function renderBreadcrumbs() {
  elements.breadcrumbs.replaceChildren();
  const selected = state.nodesById.get(state.selectedNodeId) || state.tree;
  const path = [];
  for (let node = selected; node; node = node.parent) path.unshift(node);
  path.forEach((node, index) => {
    if (index) {
      const separator = document.createElement("span");
      const button = breadcrumbButton(node);
      separator.append(button);
      elements.breadcrumbs.append(separator);
    } else {
      elements.breadcrumbs.append(breadcrumbButton(node));
    }
  });
}

function breadcrumbButton(node) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = node.name;
  button.addEventListener("click", () => {
    state.selectedNodeId = node.id;
    expandAncestors(node);
    renderTree();
    renderDetail();
    renderBreadcrumbs();
  });
  return button;
}

function renderDetail() {
  if (state.activeView === "tree") disposeVideoSession();
  const node = state.nodesById.get(state.selectedNodeId) || state.tree;
  const records = node.recordIds.map((id) => state.recordsById.get(id)).filter(Boolean);
  const counts = decisionCounts(node.recordIds);
  const levelLabel = node.levelLabel;
  const representative = node.levelKey === "native_task" && records.length ? records[0] : null;
  const episodes = aggregateEpisodes(node.recordIds);
  const conversion = aggregateConversion(node.recordIds);
  const manualReview = aggregateManualReview(node.recordIds);
  elements.detail.replaceChildren();

  const level = document.createElement("span");
  level.className = "detail-level";
  level.textContent = levelLabel;
  const heading = document.createElement("h2");
  heading.textContent = node.name;
  heading.title = node.name;
  elements.detail.append(level, heading);

  const governanceCard = createGovernanceCard({
    recordIds: [...node.recordIds],
    scopeLabel: node.name,
    scopeLevel: levelLabel,
    counts,
    conversion,
    manualReview,
  });

  if (representative) {
    if (state.activeView === "tree") elements.detail.append(createVideoBrowser(records, `tree:${node.id}`));
    elements.detail.append(
      createFileInformation(representative, records, episodes, conversion, manualReview),
      governanceCard,
    );
  }
  else {
    elements.detail.append(governanceCard);
    const context = document.createElement("p");
    context.className = "catalog-meta";
    context.textContent = `这个分支包含 ${node.recordIds.length.toLocaleString("zh-CN")} 条任务路径记录；回合：${episodeDetailText(episodes)}。标记操作会应用到当前筛选后的整个分支。`;
    elements.detail.append(context);
  }
}

function createGovernanceCard({ recordIds, scopeLabel, scopeLevel, counts, conversion, manualReview }) {
  const section = document.createElement("section");
  section.className = "governance-card";
  section.setAttribute("aria-label", "数据治理状态");

  const header = document.createElement("div");
  header.className = "governance-card-header";
  const headingGroup = document.createElement("div");
  const heading = document.createElement("h3");
  heading.textContent = "数据治理状态";
  const scope = document.createElement("p");
  scope.className = "governance-card-scope";
  const scopePrefix = scopeLevel === "总览" ? "当前筛选结果" : scopeLevel;
  scope.textContent = `${scopePrefix} · ${recordIds.length.toLocaleString("zh-CN")} 条路径`;
  headingGroup.append(heading, scope);
  const editButton = document.createElement("button");
  editButton.type = "button";
  editButton.className = "button secondary governance-edit-button";
  editButton.textContent = "编辑标记";
  editButton.disabled = recordIds.length === 0;
  editButton.setAttribute("aria-label", `编辑当前范围内 ${recordIds.length} 条路径的数据治理标记`);
  editButton.addEventListener("click", () => {
    openGovernanceDialog({ recordIds: [...recordIds], scopeLabel, scopeLevel });
  });
  header.append(headingGroup, editButton);

  const statuses = document.createElement("div");
  statuses.className = "governance-status-list";
  statuses.append(
    createGovernanceStatusRow("筛选结论", [
      { label: "保留", value: counts.keep, tone: "keep" },
      { label: "排除", value: counts.exclude, tone: "exclude" },
      { label: "需复查", value: counts.review, tone: "review" },
      { label: "未标注", value: counts.unreviewed, tone: "neutral" },
    ]),
    createGovernanceStatusRow("人工复核", [
      { label: "已复核", value: manualReview.reviewed, tone: "keep" },
      { label: "未复核", value: manualReview.total - manualReview.reviewed, tone: "neutral" },
    ]),
    createGovernanceStatusRow(
      "服务器转换",
      [
        { label: "已转换", value: conversion.counts.converted, tone: "converted" },
        { label: "部分", value: conversion.counts.partial, tone: "review" },
        { label: "未转换", value: conversion.counts.not_converted, tone: "neutral" },
        { label: "已排除", value: conversion.counts.excluded, tone: "exclude" },
        { label: "未知", value: conversion.counts.unknown, tone: "neutral" },
      ],
      { readOnly: true },
    ),
  );

  const footer = document.createElement("div");
  footer.className = "governance-card-footer";
  const saved = document.createElement("span");
  saved.className = "governance-save-status";
  const latestUpdate = latestGovernanceUpdatedAt(recordIds);
  saved.textContent = latestUpdate
    ? `本浏览器自动保存 · 最近更新 ${formatGovernanceTimestamp(latestUpdate)}`
    : "本浏览器自动保存 · 尚未标注";
  const undoButton = document.createElement("button");
  undoButton.type = "button";
  undoButton.className = "governance-undo-button";
  undoButton.textContent = "撤销上次操作";
  undoButton.disabled = !state.lastGovernanceChange;
  if (state.lastGovernanceChange) undoButton.title = `撤销：${state.lastGovernanceChange.label}`;
  undoButton.addEventListener("click", undoLastGovernanceChange);
  footer.append(saved, undoButton);

  const help = document.createElement("details");
  help.className = "governance-help";
  const summary = document.createElement("summary");
  summary.textContent = "状态说明";
  const helpText = document.createElement("p");
  helpText.textContent = "筛选结论与人工复核相互独立；服务器转换状态由目录扫描生成，只读且不会被这里的操作修改。";
  help.append(summary, helpText);

  section.append(header, statuses, footer, help);
  return section;
}

function createGovernanceStatusRow(labelText, items, { readOnly = false } = {}) {
  const row = document.createElement("div");
  row.className = "governance-status-row";
  const label = document.createElement("span");
  label.className = "governance-status-label";
  label.textContent = labelText;
  if (readOnly) {
    const marker = document.createElement("small");
    marker.textContent = "只读";
    label.append(marker);
  }
  const values = document.createElement("div");
  values.className = "governance-status-values";
  for (const item of items.filter(({ value }) => value > 0)) {
    const chip = document.createElement("span");
    chip.className = `governance-status-chip ${item.tone}`;
    chip.textContent = `${item.label} ${item.value.toLocaleString("zh-CN")}`;
    values.append(chip);
  }
  if (!values.childElementCount) {
    const empty = document.createElement("span");
    empty.className = "governance-status-chip neutral";
    empty.textContent = "无路径";
    values.append(empty);
  }
  row.append(label, values);
  return row;
}

function latestGovernanceUpdatedAt(recordIds) {
  let latest = 0;
  for (const id of recordIds) {
    for (const value of [state.decisions[id]?.updated_at, state.processing[id]?.updated_at]) {
      const timestamp = Date.parse(value || "");
      if (Number.isFinite(timestamp)) latest = Math.max(latest, timestamp);
    }
  }
  return latest ? new Date(latest) : null;
}

function formatGovernanceTimestamp(date) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function createFileInformation(record, records, episodes, conversion, manualReview) {
  const section = document.createElement("details");
  section.className = "file-information";
  section.setAttribute("aria-label", "文件信息");

  const header = document.createElement("summary");
  header.className = "file-information-header";
  const titleGroup = document.createElement("div");
  const eyebrow = document.createElement("span");
  eyebrow.className = "detail-level";
  eyebrow.textContent = "文件信息";
  const heading = document.createElement("h3");
  heading.textContent = "路径与转换产物";
  titleGroup.append(eyebrow, heading);
  const toggle = document.createElement("span");
  toggle.className = "file-information-toggle";
  toggle.textContent = "展开";
  header.append(titleGroup, toggle);
  section.addEventListener("toggle", () => {
    toggle.textContent = section.open ? "收起" : "展开";
  });

  section.append(header, createRecordFields(record, records, episodes, conversion, manualReview));
  return section;
}

function createRecordFields(record, records, episodes, conversion, manualReview) {
  const list = document.createElement("dl");
  list.className = "detail-fields";
  const fields = [
    ["数据域", state.catalog.domain_labels[record.domain] || record.domain],
    ["数据集", record.dataset],
    ["任务 instruction", record.native_task],
    ["原生分组", record.native_group],
    ["场景", `${record.scene}（${sourceLabel(record.scene_source)}）`],
    ["任务类型", `${record.task_type}（${sourceLabel(record.task_type_source)}）`],
    ["主原子动作", record.primary_atomic_action || "待分类"],
    ["动作序列", Array.isArray(record.atomic_actions) && record.atomic_actions.length ? record.atomic_actions.join(" → ") : "待分类"],
    ["数据角色", record.data_role],
    ["路径数量", records.length.toLocaleString("zh-CN")],
    ["服务器转换状态", conversionAriaText(conversion)],
    ["人工复核", manualReviewAriaText(manualReview)],
    ["回合总数", episodeDetailText(episodes)],
    ["原始路径", record.source_path],
    ["转换产物", joinedRecordValues(records, "conversion_output_path") || "—"],
    ["转换证据", joinedRecordValues(records, "conversion_evidence_path") || "—"],
  ];
  for (const [label, value] of fields) {
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    if (["原始路径", "转换产物", "转换证据"].includes(label) && value !== "—") {
      const code = document.createElement("code");
      code.textContent = String(value);
      description.append(code);
    } else description.textContent = String(value);
    list.append(term, description);
  }
  return list;
}

function createVideoPickerStep(number, title) {
  const section = document.createElement("section");
  section.className = "video-picker-step";
  const heading = document.createElement("div");
  heading.className = "video-picker-heading";
  const index = document.createElement("span");
  index.className = "video-picker-index";
  index.textContent = number;
  const label = document.createElement("strong");
  label.textContent = title;
  heading.append(index, label);
  const body = document.createElement("div");
  body.className = "video-picker-body";
  section.append(heading, body);
  return { section, body };
}

function createVideoBrowser(records, ownerKey) {
  const section = document.createElement("section");
  section.className = "video-browser";
  section.setAttribute("aria-label", "任务回合视频浏览器");

  const header = document.createElement("div");
  header.className = "video-browser-header";
  const titleGroup = document.createElement("div");
  const eyebrow = document.createElement("span");
  eyebrow.className = "detail-level";
  eyebrow.textContent = "转换视频";
  const heading = document.createElement("h3");
  heading.textContent = "任务回合视频";
  titleGroup.append(eyebrow, heading);
  const badge = document.createElement("span");
  badge.className = `video-service-badge ${state.videoConfig.enabled ? "online" : "offline"}`;
  badge.textContent = state.videoConfig.enabled ? "按需加载" : "未接通";
  header.append(titleGroup, badge);

  const status = document.createElement("p");
  status.className = "video-status";
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  section.append(header, status);

  if (!state.videoConfig.enabled) {
    section.classList.add("unavailable");
    status.textContent = state.videoConfig.message || "视频服务尚未配置。";
    return section;
  }

  const playableRecords = records.filter((record) => ["converted", "partial"].includes(conversionStatus(record)));
  if (!playableRecords.length) {
    section.classList.add("unavailable");
    status.textContent = "当前任务尚无已转换或部分转换的视频产物。";
    return section;
  }

  const recordField = document.createElement("label");
  recordField.className = "video-record-field";
  const recordLabel = document.createElement("span");
  recordLabel.textContent = "转换路径";
  const recordSelect = document.createElement("select");
  recordSelect.setAttribute("aria-label", "选择要浏览的视频转换路径");
  playableRecords.forEach((record, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${record.dataset} · ${record.native_group || record.native_task}`;
    option.title = record.source_path;
    recordSelect.append(option);
  });
  recordField.append(recordLabel, recordSelect);
  recordField.hidden = playableRecords.length < 2;

  const picker = document.createElement("div");
  picker.className = "video-picker episode-only";
  const episodeStep = createVideoPickerStep("1", "选择回合");
  const episodeList = document.createElement("div");
  episodeList.className = "video-episode-list";
  episodeList.setAttribute("role", "listbox");
  episodeList.setAttribute("aria-label", "选择视频回合");
  episodeStep.body.append(episodeList);
  picker.append(episodeStep.section);

  const loadMore = document.createElement("button");
  loadMore.type = "button";
  loadMore.className = "button secondary video-load-more";
  loadMore.textContent = "加载更多";
  loadMore.hidden = true;

  const wall = document.createElement("div");
  wall.className = "video-wall";
  wall.setAttribute("aria-label", "三个同步视频视角");
  const players = {};
  for (const definition of VIDEO_WALL_VIEWS) {
    const view = document.createElement("section");
    view.className = `video-view ${definition.key}`;
    const viewHeader = document.createElement("div");
    viewHeader.className = "video-view-header";
    const viewLabel = document.createElement("strong");
    viewLabel.textContent = definition.label;
    const cameraSelect = document.createElement("select");
    cameraSelect.className = "video-view-camera-select";
    cameraSelect.setAttribute("aria-label", `${definition.label}选择其他视角`);
    cameraSelect.disabled = true;
    const pendingOption = document.createElement("option");
    pendingOption.textContent = "待选择";
    cameraSelect.append(pendingOption);
    viewHeader.append(viewLabel, cameraSelect);
    const player = document.createElement("div");
    player.className = "video-player";
    const video = document.createElement("video");
    video.controls = true;
    video.playsInline = true;
    video.preload = "none";
    video.crossOrigin = "anonymous";
    video.muted = true;
    video.hidden = true;
    const placeholder = document.createElement("div");
    placeholder.className = "video-player-placeholder";
    placeholder.textContent = "选择回合后自动加载";
    const meta = document.createElement("p");
    meta.className = "video-player-meta";
    meta.textContent = `${definition.label} · 尚未选择回合`;
    player.append(video, placeholder, meta);
    view.append(viewHeader, player);
    wall.append(view);
    players[definition.key] = { definition, view, cameraSelect, player, video, placeholder, meta, itemIndex: -1 };
  }

  const wallControls = document.createElement("div");
  wallControls.className = "video-wall-controls";
  const playToggle = document.createElement("button");
  playToggle.type = "button";
  playToggle.className = "button secondary";
  playToggle.textContent = "同时播放";
  playToggle.disabled = true;
  const syncButton = document.createElement("button");
  syncButton.type = "button";
  syncButton.className = "button secondary";
  syncButton.textContent = "重新同步进度";
  syncButton.disabled = true;
  wallControls.append(playToggle, syncButton);

  section.append(recordField, picker, loadMore, wall, wallControls);

  const session = {
    ownerKey,
    records: playableRecords,
    recordIndex: 0,
    offset: 0,
    limit: 24,
    total: 0,
    indexedTotal: 0,
    sourceTotal: 0,
    videoStatus: "unmapped",
    videoNote: "",
    truncated: false,
    items: [],
    selectedEpisodeKey: null,
    renderedEpisodeSignature: null,
    synchronizing: false,
    requestGeneration: 0,
    controller: null,
    transcodeGeneration: 0,
    transcodeController: null,
    section,
    status,
    recordSelect,
    episodeList,
    loadMore,
    wall,
    players,
    playToggle,
    syncButton,
  };
  state.videoSession = session;

  recordSelect.addEventListener("change", () => {
    session.recordIndex = Number(recordSelect.value) || 0;
    loadVideoPage(session, { reset: true });
  });
  loadMore.addEventListener("click", () => loadVideoPage(session));
  playToggle.addEventListener("click", () => toggleVideoWallPlayback(session));
  syncButton.addEventListener("click", () => synchronizeVideoWall(session, { announce: true }));
  for (const player of Object.values(players)) {
    bindVideoWallPlayer(session, player);
    player.cameraSelect.addEventListener("change", () => {
      selectVideoWallPlayerItem(session, player, Number(player.cameraSelect.value), { play: true });
    });
  }
  loadVideoPage(session, { reset: true });
  return section;
}

async function loadVideoPage(session, { reset = false } = {}) {
  if (state.videoSession !== session) return;
  session.controller?.abort();
  const controller = new AbortController();
  const generation = session.requestGeneration + 1;
  session.requestGeneration = generation;
  session.controller = controller;
  if (reset) {
    cancelVideoPreparation(session);
    session.items = [];
    session.offset = 0;
    session.total = 0;
    session.indexedTotal = 0;
    session.sourceTotal = 0;
    session.videoStatus = "unmapped";
    session.videoNote = "";
    session.truncated = false;
    session.selectedEpisodeKey = null;
    session.renderedEpisodeSignature = null;
    session.episodeList.replaceChildren();
    resetVideoWall(session, "选择回合后自动加载三个视角。");
  }
  session.status.classList.remove("error");
  session.status.textContent = reset ? "正在读取任务回合视频索引…" : "正在加载更多视频…";
  session.loadMore.disabled = true;

  const record = session.records[session.recordIndex];
  const recordId = record.id;
  const requestedOffset = session.offset;
  const url = new URL(
    `${state.videoConfig.apiBaseUrl}/v1/records/${encodeURIComponent(recordId)}/videos`,
  );
  url.searchParams.set("offset", String(requestedOffset));
  url.searchParams.set("limit", String(session.limit));
  try {
    const response = await fetch(url, {
      cache: "no-store",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    if (!isCurrentVideoRequest(session, controller, generation, recordId)) return;
    if (response.status === 404) {
      if (reset) renderVideoItems(session, [], 0);
      session.status.textContent = "这个任务的转换产物中暂未索引到视频。";
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    if (!isCurrentVideoRequest(session, controller, generation, recordId)) return;
    if (payload.record_id !== recordId || payload.offset !== requestedOffset || !Array.isArray(payload.items)) {
      throw new Error("视频索引响应与当前请求不一致");
    }
    const rawItemCount = payload.items.length;
    const incoming = normalizeVideoItems(payload.items);
    const total = Number.isInteger(payload.total) && payload.total >= 0 ? payload.total : session.items.length + incoming.length;
    session.videoStatus = normalizeVideoStatus(payload.video_status);
    session.videoNote = typeof payload.video_note === "string" ? payload.video_note : "";
    session.indexedTotal = Number.isInteger(payload.indexed_total) && payload.indexed_total >= 0
      ? payload.indexed_total
      : total;
    session.sourceTotal = Number.isInteger(payload.source_total) && payload.source_total >= 0
      ? payload.source_total
      : session.indexedTotal;
    session.truncated = payload.truncated === true;
    session.items.push(...incoming);
    session.total = Math.max(total, session.items.length);
    session.offset = requestedOffset + rawItemCount;
    if (rawItemCount === 0 && session.offset < session.total) session.total = session.offset;
    renderVideoItems(session);
  } catch (error) {
    if (error.name === "AbortError" || !isCurrentVideoRequest(session, controller, generation, recordId)) return;
    session.status.textContent = `视频索引加载失败：${error.message}`;
    session.status.classList.add("error");
    session.loadMore.hidden = false;
    session.loadMore.textContent = "重试";
  } finally {
    if (isCurrentVideoRequest(session, controller, generation, recordId)) session.loadMore.disabled = false;
  }
}

function isCurrentVideoRequest(session, controller, generation, recordId) {
  return state.videoSession === session
    && session.controller === controller
    && session.requestGeneration === generation
    && session.records[session.recordIndex]?.id === recordId;
}

function normalizeVideoItems(items) {
  if (!Array.isArray(items)) throw new Error("视频索引响应格式无效");
  return items.flatMap((item) => {
    if (!item) return [];
    const assetId = typeof item.asset_id === "string" ? item.asset_id : "";
    const streamUrl = typeof item.stream_url === "string" ? item.stream_url : "";
    if (!assetId && !streamUrl) return [];
    const rawPlaybackStatus = typeof item.playback_status === "string" ? item.playback_status : "";
    const playbackStatus = VIDEO_PLAYBACK_STATUSES.has(rawPlaybackStatus) ? rawPlaybackStatus : "unknown";
    return [{
      assetId,
      streamUrl,
      episodeIndex: item.episode_index,
      cameraKey: String(item.camera_key || item.video_key || "camera"),
      codec: item.source_codec ? String(item.source_codec) : "",
      startSeconds: Number.isFinite(item.start_seconds) && item.start_seconds >= 0 ? Number(item.start_seconds) : null,
      endSeconds: Number.isFinite(item.end_seconds) && item.end_seconds > item.start_seconds
        ? Number(item.end_seconds)
        : null,
      playbackStatus,
      playable: ["ready", "unverified"].includes(playbackStatus),
      transcodeState: "idle",
      transcodeError: "",
    }];
  });
}

function cameraViewKeyword(value) {
  const raw = String(value ?? "").trim();
  const normalized = raw
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  const tokens = normalized.split("_").filter(Boolean);
  const tokenSet = new Set(tokens);
  const classifiedViews = [
    ["left_wrist", ["left", "wrist"]],
    ["right_wrist", ["right", "wrist"]],
    ["third_person", ["third", "person"]],
    ["front", ["front"]],
    ["head", ["head"]],
    ["left", ["left"]],
    ["right", ["right"]],
    ["overhead", ["overhead"]],
    ["top", ["top"]],
    ["wrist", ["wrist"]],
    ["external", ["external"]],
    ["ego", ["ego"]],
    ["hand", ["hand"]],
  ];
  for (const [label, requiredTokens] of classifiedViews) {
    if (requiredTokens.every((token) => tokenSet.has(token))) return label;
  }

  const structuralTokens = new Set([
    "observation",
    "observations",
    "image",
    "images",
    "rgb",
    "video",
    "videos",
    "camera",
  ]);
  const fallback = tokens.filter((token) => !structuralTokens.has(token)).at(-1);
  return fallback || normalized || raw || "camera";
}

function videoEpisodeKey(value) {
  return `${typeof value}:${String(value ?? "")}`;
}

function groupedVideoEpisodes(items) {
  const groups = [];
  const byKey = new Map();
  items.forEach((item, index) => {
    const key = videoEpisodeKey(item.episodeIndex);
    let group = byKey.get(key);
    if (!group) {
      group = { key, episodeIndex: item.episodeIndex, itemIndexes: [] };
      byKey.set(key, group);
      groups.push(group);
    }
    group.itemIndexes.push(index);
  });
  return groups;
}

function renderVideoItems(session) {
  if (state.videoSession !== session) return;
  const episodeGroups = groupedVideoEpisodes(session.items);
  if (session.selectedEpisodeKey && !episodeGroups.some((group) => group.key === session.selectedEpisodeKey)) {
    session.selectedEpisodeKey = null;
    session.renderedEpisodeSignature = null;
  }

  session.episodeList.replaceChildren();
  for (const group of episodeGroups) {
    const button = document.createElement("button");
    button.type = "button";
    const selected = group.key === session.selectedEpisodeKey;
    button.className = `video-episode-item ${selected ? "selected" : ""}`.trim();
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(selected));
    const label = document.createElement("strong");
    label.textContent = episodeLabel(group.episodeIndex);
    const count = document.createElement("span");
    count.textContent = `${group.itemIndexes.length.toLocaleString("zh-CN")} 路视频`;
    button.append(label, count);
    button.addEventListener("click", () => {
      if (session.selectedEpisodeKey === group.key) return;
      cancelVideoPreparation(session);
      session.selectedEpisodeKey = group.key;
      session.renderedEpisodeSignature = null;
      renderVideoItems(session);
    });
    session.episodeList.append(button);
  }

  const selectedGroup = episodeGroups.find((group) => group.key === session.selectedEpisodeKey);
  if (selectedGroup) {
    const signature = `${selectedGroup.key}:${selectedGroup.itemIndexes.map((index) => session.items[index]?.assetId).join("|")}`;
    if (signature !== session.renderedEpisodeSignature) renderVideoWallEpisode(session, selectedGroup, signature);
  } else if (session.renderedEpisodeSignature) {
    resetVideoWall(session, episodeGroups.length ? "选择回合后自动加载三个视角。" : "尚未加载到可选回合。");
  }

  session.loadMore.hidden = session.offset >= session.total;
  session.loadMore.textContent = "加载更多回合";
  updateVideoStatusText(session);
}

function preferredVideoIndex(session, group, definition) {
  for (const preferred of definition.preferences) {
    const match = group.itemIndexes.find((index) => cameraViewKeyword(session.items[index]?.cameraKey) === preferred);
    if (Number.isInteger(match)) return match;
  }
  return -1;
}

function renderVideoWallEpisode(session, group, signature) {
  if (state.videoSession !== session) return;
  cancelVideoPreparation(session);
  session.synchronizing = true;
  const pending = [];
  for (const player of Object.values(session.players)) {
    const preferredIndex = preferredVideoIndex(session, group, player.definition);
    populateVideoWallCameraSelect(session, player, group.itemIndexes, preferredIndex);
    if (preferredIndex < 0) {
      showVideoWallPlaceholder(player, `当前回合缺少${player.definition.label}，可在右上角选择其他视角。`);
      continue;
    }
    const item = session.items[preferredIndex];
    if (item.playable) loadVideoWallPlayer(session, player, preferredIndex);
    else if (item.playbackStatus === "transcode_required") {
      showVideoWallPlaceholder(player, `${player.definition.label}正在准备浏览器兼容视频…`, item, preferredIndex);
      pending.push({ player, index: preferredIndex });
    } else showVideoWallPlaceholder(player, `${player.definition.label}：${videoPlaybackLabel(item)}`, item, preferredIndex);
  }
  session.synchronizing = false;
  session.renderedEpisodeSignature = signature;
  updateVideoWallControls(session);
  playVideoWall(session);
  if (pending.length) prepareVideoWallItems(session, pending);
}

function populateVideoWallCameraSelect(session, player, indexes, selectedIndex) {
  player.cameraSelect.replaceChildren();
  if (selectedIndex < 0) {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "选择其他视角";
    placeholder.disabled = true;
    player.cameraSelect.append(placeholder);
  }
  for (const index of indexes) {
    const item = session.items[index];
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = cameraViewKeyword(item.cameraKey);
    option.title = item.cameraKey;
    option.selected = index === selectedIndex;
    player.cameraSelect.append(option);
  }
  player.cameraSelect.disabled = indexes.length === 0;
  if (selectedIndex >= 0) player.cameraSelect.value = String(selectedIndex);
}

function selectVideoWallPlayerItem(session, player, index, { play = false } = {}) {
  if (state.videoSession !== session || !Number.isInteger(index) || !session.items[index]) return;
  cancelVideoPreparation(session);
  const item = session.items[index];
  player.cameraSelect.value = String(index);
  if (item.playable) {
    loadVideoWallPlayer(session, player, index, { play });
    synchronizeVideoWall(session);
  } else if (item.playbackStatus === "transcode_required") {
    showVideoWallPlaceholder(player, `${cameraViewKeyword(item.cameraKey)} 正在准备浏览器兼容视频…`, item, index);
    prepareVideoWallItems(session, [{ player, index }]);
  } else {
    showVideoWallPlaceholder(player, `${cameraViewKeyword(item.cameraKey)}：${videoPlaybackLabel(item)}`, item, index);
    updateVideoWallControls(session);
  }
}

function showVideoWallPlaceholder(player, message, item = null, itemIndex = -1) {
  stopVideo(player.video);
  player.itemIndex = Number.isInteger(itemIndex) ? itemIndex : -1;
  player.video.hidden = true;
  player.placeholder.hidden = false;
  player.placeholder.textContent = message;
  player.meta.textContent = item
    ? `${cameraViewKeyword(item.cameraKey)} · ${videoPlaybackLabel(item)}`
    : `${player.definition.label} · 暂无视频`;
}

function loadVideoWallPlayer(session, player, index, { play = false } = {}) {
  const item = session.items[index];
  if (!item?.playable) return;
  let streamUrl;
  try {
    streamUrl = resolveVideoStreamUrl(item);
  } catch (error) {
    showVideoWallPlaceholder(player, `视频地址无效：${error.message}`, item, index);
    return;
  }
  stopVideo(player.video);
  player.itemIndex = index;
  player.video.setAttribute("aria-label", `${episodeLabel(item.episodeIndex)}，${player.definition.label}，${cameraViewKeyword(item.cameraKey)}`);
  player.video.src = streamUrl;
  player.video.hidden = false;
  player.placeholder.hidden = true;
  player.video.load();
  updateVideoWallPlayerMeta(session, player);
  if (play) player.video.play().catch(() => {});
  updateVideoWallControls(session);
}

async function prepareVideoWallItems(session, requests) {
  if (state.videoSession !== session || !requests.length) return;
  cancelVideoPreparation(session);
  const controller = new AbortController();
  const generation = session.transcodeGeneration + 1;
  session.transcodeGeneration = generation;
  session.transcodeController = controller;

  for (const { player, index } of requests) {
    const item = session.items[index];
    if (!item || state.videoSession !== session || session.transcodeController !== controller) return;
    item.transcodeState = "pending";
    item.transcodeError = "";
    showVideoWallPlaceholder(player, `${cameraViewKeyword(item.cameraKey)} 正在准备浏览器兼容视频…`, item, index);
    let streamUrl;
    try {
      streamUrl = resolveVideoStreamUrl(item);
      const response = await fetch(streamUrl, {
        method: "GET",
        cache: "no-store",
        credentials: "omit",
        headers: { Accept: "video/mp4, video/*;q=0.9, application/json;q=0.8", Range: "bytes=0-0" },
        referrerPolicy: "no-referrer",
        signal: controller.signal,
      });
      if (!isCurrentVideoWallPreparation(session, controller, generation, item, index, player)) return;
      const contentType = String(response.headers.get("Content-Type") || "").toLowerCase();
      const responseIsJson = contentType.includes("application/json") || contentType.includes("+json");
      if (!response.ok || ![200, 206].includes(response.status) || responseIsJson) {
        throw new Error(await transcodeFailureMessage(response, { responseIsJson }));
      }
      await cancelResponseBody(response);
      if (!isCurrentVideoWallPreparation(session, controller, generation, item, index, player)) return;
      item.playbackStatus = "ready";
      item.playable = true;
      item.transcodeState = "ready";
      item.transcodeError = "";
      loadVideoWallPlayer(session, player, index, { play: true });
    } catch (error) {
      if (error.name === "AbortError" || state.videoSession !== session || session.transcodeGeneration !== generation) return;
      item.transcodeState = "error";
      item.transcodeError = error.message.startsWith("转码") || error.message.startsWith("视频")
        ? error.message
        : `按需转码请求失败：${error.message}`;
      showVideoWallPlaceholder(player, item.transcodeError, item, index);
    }
  }
  if (state.videoSession === session && session.transcodeController === controller) session.transcodeController = null;
  updateVideoStatusText(session);
  updateVideoWallControls(session);
}

function isCurrentVideoWallPreparation(session, controller, generation, item, index, player) {
  return state.videoSession === session
    && session.transcodeController === controller
    && session.transcodeGeneration === generation
    && session.items[index] === item
    && player.cameraSelect.value === String(index);
}

function updateVideoStatusText(session) {
  if (state.videoSession !== session) return;
  const pending = session.items.find((item) => item.transcodeState === "pending");
  if (pending) {
    session.status.classList.remove("error");
    session.status.textContent = `${episodeLabel(pending.episodeIndex)} · ${pending.cameraKey} 正在按需准备浏览器兼容视频，请稍候…`;
    return;
  }
  const failed = session.items.find((item) => item.transcodeState === "error" && item.transcodeError);
  if (failed) {
    session.status.classList.add("error");
    session.status.textContent = failed.transcodeError;
    return;
  }
  session.status.classList.remove("error");
  if (!session.items.length) session.status.textContent = videoStatusSummary(session);
  else if (session.selectedEpisodeKey) {
    const visible = activeVideoWallPlayers(session).length;
    session.status.textContent = `${videoStatusSummary(session)} 当前回合已加载 ${visible} / 3 个同步视角。`;
  } else session.status.textContent = `${videoStatusSummary(session)} 请选择回合，将自动播放左腕、右腕和主视角。`;
}

function normalizeVideoStatus(value) {
  return ["available", "linked", "partial", "absent", "unmapped"].includes(value) ? value : "unmapped";
}

function videoStatusSummary(session) {
  const indexed = session.indexedTotal.toLocaleString("zh-CN");
  const source = session.sourceTotal.toLocaleString("zh-CN");
  if (session.videoStatus === "absent") {
    return session.videoNote || "转换元数据存在，但当前未找到可播放 MP4。";
  }
  if (session.videoStatus === "unmapped") return "正式转换产物尚未稳定映射到这个任务词条。";
  if (session.videoStatus === "partial" || session.truncated || session.indexedTotal < session.sourceTotal) {
    return `视频映射不完整：已索引 ${indexed} / 源端 ${source} 个。`;
  }
  if (session.videoStatus === "linked") return `共 ${source} 个视频，媒体来自受控符号链接。`;
  return `共 ${source} 个视频。`;
}

function cancelVideoPreparation(session) {
  let changed = false;
  if (session.transcodeController) {
    session.transcodeController.abort();
    session.transcodeController = null;
    changed = true;
  }
  for (const item of session.items) {
    if (!["pending", "error"].includes(item.transcodeState)) continue;
    item.transcodeState = "idle";
    item.transcodeError = "";
    changed = true;
  }
  if (changed) session.transcodeGeneration += 1;
  return changed;
}

async function transcodeFailureMessage(response, { responseIsJson = false } = {}) {
  let serverMessage = "";
  if (responseIsJson) {
    try {
      const payload = await response.json();
      serverMessage = extractVideoErrorMessage(payload);
    } catch {
      serverMessage = "";
    }
  } else await cancelResponseBody(response);

  if (response.status === 503) {
    const retryAfter = retryAfterText(response.headers.get("Retry-After"));
    const reason = serverMessage ? `：${serverMessage}` : "";
    return `转码服务暂未就绪${reason}；${retryAfter || "请稍后"}点击该视频重试。`;
  }
  const reason = serverMessage ? `：${serverMessage}` : "";
  return `视频准备失败（HTTP ${response.status}）${reason}；请点击该视频重试。`;
}

function extractVideoErrorMessage(payload) {
  const candidates = [payload?.error?.message, payload?.message, payload?.detail];
  const message = candidates.find((value) => typeof value === "string" && value.trim());
  return message ? message.trim().slice(0, 240) : "";
}

function retryAfterText(value) {
  if (!value) return "";
  const normalized = value.trim();
  let seconds = /^\d+$/.test(normalized) ? Number(normalized) : NaN;
  if (!Number.isFinite(seconds)) {
    const retryAt = Date.parse(normalized);
    if (Number.isFinite(retryAt)) seconds = Math.max(0, Math.ceil((retryAt - Date.now()) / 1000));
  }
  if (!Number.isFinite(seconds)) return "";
  if (seconds < 60) return `建议 ${Math.max(1, Math.ceil(seconds))} 秒后再次`;
  return `建议 ${Math.ceil(seconds / 60)} 分钟后再次`;
}

async function cancelResponseBody(response) {
  try {
    await response.body?.cancel();
  } catch {
    // The response may already be closed by the browser; no payload is needed for a one-byte readiness probe.
  }
}

function activeVideoWallPlayers(session) {
  return Object.values(session.players).filter((player) => !player.video.hidden && player.video.hasAttribute("src"));
}

function bindVideoWallPlayer(session, player) {
  player.video.addEventListener("loadedmetadata", () => {
    if (state.videoSession !== session) return;
    const item = session.items[player.itemIndex];
    if (item && item.startSeconds !== null) player.video.currentTime = item.startSeconds;
    updateVideoWallPlayerMeta(session, player);
    synchronizeVideoWall(session);
    updateVideoWallControls(session);
  });
  player.video.addEventListener("play", () => synchronizeVideoWallFrom(session, player, "play"));
  player.video.addEventListener("pause", () => synchronizeVideoWallFrom(session, player, "pause"));
  player.video.addEventListener("seeked", () => synchronizeVideoWallFrom(session, player, "seek"));
  player.video.addEventListener("timeupdate", () => {
    const item = session.items[player.itemIndex];
    if (item?.endSeconds !== null && player.video.currentTime >= item.endSeconds - 0.03) {
      pauseVideoWall(session);
      return;
    }
    correctVideoWallDrift(session, player);
  });
  player.video.addEventListener("ended", () => pauseVideoWall(session));
  player.video.addEventListener("error", () => {
    if (state.videoSession !== session || player.video.hidden) return;
    const item = session.items[player.itemIndex];
    showVideoWallPlaceholder(player, "视频加载失败，请切换视角或稍后重试。", item, player.itemIndex);
    session.status.classList.add("error");
    session.status.textContent = "部分视角加载失败；可能是编码不兼容、文件缺失或媒体服务暂时不可用。";
    updateVideoWallControls(session);
  });
}

function videoWallMaster(session, preferred = null) {
  const active = activeVideoWallPlayers(session);
  if (preferred && active.includes(preferred)) return preferred;
  const main = session.players.main;
  return active.includes(main) ? main : active[0] || null;
}

async function playVideoWall(session, source = null) {
  if (state.videoSession !== session || session.synchronizing) return;
  const players = activeVideoWallPlayers(session);
  const master = videoWallMaster(session, source);
  if (!players.length || !master) return;
  session.synchronizing = true;
  const masterItem = session.items[master.itemIndex];
  const masterStart = masterItem?.startSeconds ?? 0;
  let time = Number.isFinite(master.video.currentTime) ? master.video.currentTime : masterStart;
  if (masterItem?.endSeconds !== null && time >= masterItem.endSeconds - 0.03) time = masterStart;
  const relativeTime = Math.max(0, time - masterStart);
  for (const player of players) {
    const item = session.items[player.itemIndex];
    const target = (item?.startSeconds ?? 0) + relativeTime;
    const upper = item?.endSeconds ?? player.video.duration;
    if (Number.isFinite(player.video.duration) && Math.abs(player.video.currentTime - target) > 0.2) {
      player.video.currentTime = Math.min(target, Math.max(item?.startSeconds ?? 0, upper - 0.05));
    }
  }
  await Promise.allSettled(players.map((player) => player.video.play()));
  session.synchronizing = false;
  updateVideoWallControls(session);
}

function pauseVideoWall(session) {
  if (state.videoSession !== session || session.synchronizing) return;
  session.synchronizing = true;
  for (const player of activeVideoWallPlayers(session)) player.video.pause();
  session.synchronizing = false;
  updateVideoWallControls(session);
}

function toggleVideoWallPlayback(session) {
  const players = activeVideoWallPlayers(session);
  if (!players.length) return;
  if (players.some((player) => !player.video.paused)) pauseVideoWall(session);
  else playVideoWall(session);
}

function synchronizeVideoWall(session, { announce = false } = {}) {
  if (state.videoSession !== session || session.synchronizing) return;
  const players = activeVideoWallPlayers(session);
  const master = videoWallMaster(session);
  if (!master) return;
  session.synchronizing = true;
  const masterItem = session.items[master.itemIndex];
  const relativeTime = Math.max(0, master.video.currentTime - (masterItem?.startSeconds ?? 0));
  for (const player of players) {
    if (player === master || !Number.isFinite(player.video.duration)) continue;
    const item = session.items[player.itemIndex];
    const target = (item?.startSeconds ?? 0) + relativeTime;
    const upper = item?.endSeconds ?? player.video.duration;
    player.video.currentTime = Math.min(target, Math.max(item?.startSeconds ?? 0, upper - 0.05));
  }
  session.synchronizing = false;
  if (announce) {
    session.status.classList.remove("error");
    session.status.textContent = "三个视角的播放进度已重新同步。";
  }
  updateVideoWallControls(session);
}

function synchronizeVideoWallFrom(session, source, action) {
  if (state.videoSession !== session || session.synchronizing || source.video.hidden) return;
  if (action === "play") playVideoWall(session, source);
  else if (action === "pause") pauseVideoWall(session);
  else synchronizeVideoWall(session);
}

function correctVideoWallDrift(session, source) {
  if (state.videoSession !== session || session.synchronizing || videoWallMaster(session) !== source) return;
  const sourceItem = session.items[source.itemIndex];
  const relativeTime = Math.max(0, source.video.currentTime - (sourceItem?.startSeconds ?? 0));
  session.synchronizing = true;
  for (const player of activeVideoWallPlayers(session)) {
    if (player === source || !Number.isFinite(player.video.duration)) continue;
    const item = session.items[player.itemIndex];
    const target = (item?.startSeconds ?? 0) + relativeTime;
    const upper = item?.endSeconds ?? player.video.duration;
    if (Math.abs(player.video.currentTime - target) > 0.4) {
      player.video.currentTime = Math.min(target, Math.max(item?.startSeconds ?? 0, upper - 0.05));
    }
  }
  session.synchronizing = false;
}

function updateVideoWallControls(session) {
  const players = activeVideoWallPlayers(session);
  const hasPlayers = players.length > 0;
  session.playToggle.disabled = !hasPlayers;
  session.syncButton.disabled = players.length < 2;
  session.playToggle.textContent = hasPlayers && players.some((player) => !player.video.paused)
    ? "同时暂停"
    : "同时播放";
}

function resolveVideoStreamUrl(item) {
  const apiBase = new URL(`${state.videoConfig.apiBaseUrl}/`);
  const raw = item.streamUrl || `v1/media/${encodeURIComponent(item.assetId)}`;
  const apiRelativeRaw = typeof raw === "string" && raw.startsWith("/") ? raw.replace(/^\/+/, "") : raw;
  const resolved = new URL(apiRelativeRaw, apiBase);
  if (
    resolved.origin !== apiBase.origin
    || !/^https?:$/.test(resolved.protocol)
    || !resolved.pathname.startsWith(apiBase.pathname)
  ) {
    throw new Error("媒体地址不属于已配置的视频服务");
  }
  return resolved.href;
}

function updateVideoWallPlayerMeta(session, player) {
  const item = session.items[player.itemIndex];
  if (!item) return;
  const segmentDuration = item.startSeconds !== null && item.endSeconds !== null
    ? item.endSeconds - item.startSeconds
    : player.video.duration;
  const duration = Number.isFinite(segmentDuration) ? ` · ${formatDuration(segmentDuration)}` : "";
  const codec = item.codec ? ` · ${item.codec}` : "";
  player.meta.textContent = `${episodeLabel(item.episodeIndex)} · ${cameraViewKeyword(item.cameraKey)}${codec}${duration}`;
  player.meta.title = `原始视角字段：${item.cameraKey}`;
}

function episodeLabel(value) {
  if (Number.isInteger(value)) return `回合 ${value.toLocaleString("zh-CN")}`;
  const normalized = String(value ?? "").trim();
  return normalized ? `回合 ${normalized}` : "回合未知";
}

function videoPlaybackLabel(item) {
  if (item.playbackStatus === "transcode_required") {
    if (item.transcodeState === "pending") return "正在准备浏览器兼容视频…";
    if (item.transcodeState === "error") return "需转码 · 点击重试";
    return "需转码 · 点击后准备";
  }
  const labels = {
    ready: "可播放",
    unverified: "可播放 · 编码待核验",
    unsupported: "编码不兼容",
    missing: "文件缺失",
    blocked: "路径被阻止",
    unknown: "状态未知",
  };
  return labels[item.playbackStatus] || item.playbackStatus;
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function stopVideo(video) {
  if (!video) return;
  video.pause();
  video.removeAttribute("src");
  video.removeAttribute("aria-label");
  video.load();
}

function resetVideoWall(session, message) {
  session.synchronizing = true;
  for (const player of Object.values(session.players)) {
    stopVideo(player.video);
    player.video.hidden = true;
    player.placeholder.hidden = false;
    player.placeholder.textContent = message;
    player.meta.textContent = `${player.definition.label} · 尚未选择回合`;
    player.itemIndex = -1;
    player.cameraSelect.replaceChildren();
    const option = document.createElement("option");
    option.textContent = "待选择";
    player.cameraSelect.append(option);
    player.cameraSelect.disabled = true;
  }
  session.synchronizing = false;
  session.renderedEpisodeSignature = null;
  updateVideoWallControls(session);
}

function disposeVideoSession() {
  const session = state.videoSession;
  if (!session) return;
  session.controller?.abort();
  cancelVideoPreparation(session);
  session.synchronizing = true;
  for (const player of Object.values(session.players)) stopVideo(player.video);
  session.synchronizing = false;
  state.videoSession = null;
}

function openGovernanceDialog({ recordIds, scopeLabel, scopeLevel }) {
  if (!recordIds.length) return;
  const counts = decisionCounts(recordIds);
  const manualReview = aggregateManualReview(recordIds);
  state.governanceTarget = { recordIds: [...recordIds], scopeLabel, scopeLevel };

  elements.governanceDialogTitle.textContent = "编辑数据治理标记";
  elements.governanceScopeTitle.textContent =
    scopeLevel === "总览" ? `当前筛选结果 / ${scopeLabel}` : `${scopeLevel} / ${scopeLabel}`;
  elements.governanceScopeCount.textContent = `${recordIds.length.toLocaleString("zh-CN")} 条任务路径`;
  elements.governanceScopeWarning.textContent =
    recordIds.length > 100
      ? "这是批量操作，应用前会再次确认范围。"
      : "应用后会自动保存到当前浏览器。";
  elements.governanceCurrentDecision.textContent = `当前：${decisionSummaryText(counts)}`;
  elements.governanceCurrentReview.textContent = `当前：${manualReviewAriaText(manualReview)}`;
  elements.governanceExcludeReason.value = commonExcludeReason(recordIds);
  setGovernanceRadioValue("governance-decision", "unchanged");
  setGovernanceRadioValue("governance-review", "unchanged");
  updateGovernanceExcludeField();
  elements.governanceDialog.showModal();
}

function closeGovernanceDialog() {
  if (elements.governanceDialog.open) elements.governanceDialog.close();
}

function setGovernanceRadioValue(name, value) {
  for (const input of document.querySelectorAll(`input[name="${name}"]`)) input.checked = input.value === value;
}

function selectedGovernanceRadioValue(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value || "unchanged";
}

function updateGovernanceExcludeField() {
  const excluded = selectedGovernanceRadioValue("governance-decision") === "exclude";
  elements.governanceExcludeField.hidden = !excluded;
  elements.governanceExcludeReason.disabled = !excluded;
}

function decisionSummaryText(counts) {
  const parts = [
    ["保留", counts.keep],
    ["排除", counts.exclude],
    ["需复查", counts.review],
    ["未标注", counts.unreviewed],
  ]
    .filter(([, count]) => count > 0)
    .map(([label, count]) => `${label} ${count.toLocaleString("zh-CN")}`);
  return parts.join(" · ") || "无路径";
}

function applyGovernanceChanges() {
  const target = state.governanceTarget;
  if (!target) return;
  const decision = selectedGovernanceRadioValue("governance-decision");
  const manualReview = selectedGovernanceRadioValue("governance-review");
  const excludeReason = elements.governanceExcludeReason.value.trim();
  if (decision === "exclude" && !excludeReason) {
    showToast("请先填写排除原因。", true);
    elements.governanceExcludeReason.focus();
    return;
  }

  const changeDecision = decision !== "unchanged" && target.recordIds.some((id) =>
    governanceDecisionDiffers(id, decision, excludeReason),
  );
  const changeManualReview = manualReview !== "unchanged" && target.recordIds.some((id) =>
    governanceReviewDiffers(id, manualReview),
  );
  if (!changeDecision && !changeManualReview) {
    closeGovernanceDialog();
    showToast("当前范围的状态没有变化。");
    return;
  }
  if (
    target.recordIds.length > 100
    && !window.confirm(`确认将本次标记应用到当前筛选范围内的 ${target.recordIds.length} 条路径？`)
  ) return;

  const snapshot = captureGovernanceSnapshot(target);
  const nextDecisions = { ...state.decisions };
  const nextProcessing = { ...state.processing };
  const updatedAt = new Date().toISOString();
  for (const id of target.recordIds) {
    if (changeDecision) {
      if (decision === "unreviewed") delete nextDecisions[id];
      else {
        nextDecisions[id] = {
          decision,
          keep: decision === "keep" ? true : decision === "exclude" ? false : null,
          exclude_reason: decision === "exclude" ? excludeReason : "",
          updated_at: updatedAt,
        };
      }
    }
    if (changeManualReview) {
      nextProcessing[id] = {
        status: manualReview,
        updated_at: updatedAt,
      };
    }
  }

  try {
    persistGovernanceState(nextDecisions, nextProcessing);
  } catch {
    showToast("浏览器未能保存标记，本次修改已取消。", true);
    return;
  }
  state.decisions = nextDecisions;
  state.processing = nextProcessing;
  const changedLabels = [];
  if (changeDecision) changedLabels.push("筛选结论");
  if (changeManualReview) changedLabels.push("人工复核");
  snapshot.label = `${changedLabels.join("和")} · ${target.scopeLabel}`;
  state.lastGovernanceChange = snapshot;
  closeGovernanceDialog();
  updateDecisionCount();
  rebuildTree();
  showToast(`已将${changedLabels.join("和")}应用到 ${target.recordIds.length} 条路径。`);
}

function governanceDecisionDiffers(recordId, decision, excludeReason) {
  const current = state.decisions[recordId];
  if (decision === "unreviewed") return Boolean(current);
  if (current?.decision !== decision) return true;
  return decision === "exclude" && current.exclude_reason !== excludeReason;
}

function governanceReviewDiffers(recordId, manualReview) {
  const current = isReviewed(recordId) ? "processed" : "unprocessed";
  return current !== manualReview;
}

function captureGovernanceSnapshot(target) {
  const decisions = {};
  const processing = {};
  for (const id of target.recordIds) {
    decisions[id] = state.decisions[id] ? { ...state.decisions[id] } : null;
    processing[id] = state.processing[id] ? { ...state.processing[id] } : null;
  }
  return {
    recordIds: [...target.recordIds],
    scopeLabel: target.scopeLabel,
    decisions,
    processing,
    label: target.scopeLabel,
  };
}

function persistGovernanceState(nextDecisions, nextProcessing) {
  const previousDecisions = localStorage.getItem(state.storageKey);
  const previousProcessing = localStorage.getItem(state.processingStorageKey);
  try {
    localStorage.setItem(state.storageKey, JSON.stringify(nextDecisions));
    localStorage.setItem(state.processingStorageKey, JSON.stringify(nextProcessing));
  } catch (error) {
    try {
      restoreStorageValue(state.storageKey, previousDecisions);
      restoreStorageValue(state.processingStorageKey, previousProcessing);
    } catch {
      // The in-memory state remains unchanged even if the browser also rejects the rollback write.
    }
    throw error;
  }
}

function restoreStorageValue(key, value) {
  if (value === null) localStorage.removeItem(key);
  else localStorage.setItem(key, value);
}

function undoLastGovernanceChange() {
  const snapshot = state.lastGovernanceChange;
  if (!snapshot) return;
  if (
    snapshot.recordIds.length > 100
    && !window.confirm(`确认撤销上一次作用于 ${snapshot.recordIds.length} 条路径的操作？`)
  ) return;
  const nextDecisions = { ...state.decisions };
  const nextProcessing = { ...state.processing };
  for (const id of snapshot.recordIds) {
    if (snapshot.decisions[id] === null) delete nextDecisions[id];
    else nextDecisions[id] = { ...snapshot.decisions[id] };
    if (snapshot.processing[id] === null) delete nextProcessing[id];
    else nextProcessing[id] = { ...snapshot.processing[id] };
  }
  try {
    persistGovernanceState(nextDecisions, nextProcessing);
  } catch {
    showToast("浏览器未能保存撤销结果，原状态保持不变。", true);
    return;
  }
  state.decisions = nextDecisions;
  state.processing = nextProcessing;
  state.lastGovernanceChange = null;
  updateDecisionCount();
  rebuildTree();
  showToast(`已撤销：${snapshot.label}。`);
}

function decisionCounts(recordIds) {
  const counts = { keep: 0, exclude: 0, review: 0, unreviewed: 0, reviewed: 0, notReviewed: 0 };
  for (const id of recordIds) {
    counts[state.decisions[id]?.decision || "unreviewed"] += 1;
    counts[isReviewed(id) ? "reviewed" : "notReviewed"] += 1;
  }
  return counts;
}

function isReviewed(recordId) {
  return state.processing[recordId]?.status === "processed";
}

function aggregateDecision(recordIds) {
  if (!recordIds.length) return "";
  const values = new Set(recordIds.map((id) => state.decisions[id]?.decision || "unreviewed"));
  return values.size === 1 && !values.has("unreviewed") ? [...values][0] : "";
}

function commonExcludeReason(recordIds) {
  const reasons = new Set(
    recordIds.map((id) => state.decisions[id]?.exclude_reason).filter((reason) => typeof reason === "string" && reason),
  );
  return reasons.size === 1 ? [...reasons][0] : "";
}

function loadDecisions() {
  try {
    const parsed = JSON.parse(localStorage.getItem(state.storageKey) || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function loadProcessing() {
  try {
    const parsed = JSON.parse(localStorage.getItem(state.processingStorageKey) || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function updateDecisionCount() {
  const decisions = Object.values(state.decisions).filter((item) =>
    ["keep", "exclude", "review"].includes(item?.decision),
  ).length;
  const conversionCounts = { converted: 0, partial: 0, not_converted: 0, excluded: 0, unknown: 0 };
  for (const record of state.catalog.records) conversionCounts[conversionStatus(record)] += 1;
  const reviewed = state.catalog.records.filter((record) => isReviewed(record.id)).length;
  elements.statDecisions.textContent = decisions.toLocaleString("zh-CN");
  elements.statConverted.textContent = conversionCounts.converted.toLocaleString("zh-CN");
  elements.statConverted.parentElement.title = `${conversionCounts.converted.toLocaleString("zh-CN")} / ${state.catalog.records.length.toLocaleString("zh-CN")} 条路径已完成转换`;
  elements.statPartial.textContent = conversionCounts.partial.toLocaleString("zh-CN");
  elements.statPartial.parentElement.title = `未转换 ${conversionCounts.not_converted.toLocaleString("zh-CN")} · 已排除 ${conversionCounts.excluded.toLocaleString("zh-CN")} · 状态未知 ${conversionCounts.unknown.toLocaleString("zh-CN")}`;
  elements.statReviewed.textContent = reviewed.toLocaleString("zh-CN");
}

function exportManifest() {
  const records = filteredRecords();
  const filters = currentManifestFilters();
  if (!records.length) {
    showToast("当前筛选没有可导出的任务路径，请调整数据集或场景。", true);
    return;
  }
  const exportedAt = new Date().toISOString();
  const exportedHierarchy = activeLevels().map((level) => level.key);
  const exportScope = Object.keys(filters).length ? "current_filters" : "all";
  const exportContext = {
    exportedAt,
    exportedHierarchy,
    exportScope,
    filters,
    sourceRecordCount: state.catalog.records.length,
    exportedRecordCount: records.length,
    catalogGeneratedAt: state.catalog.generated_at || null,
  };
  const lines = records.map((record) => JSON.stringify(buildManifestRecord(record, exportContext)));
  const blob = new Blob([`${lines.join("\n")}\n`], { type: "application/x-ndjson;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = manifestFilename(filters, exportedAt.slice(0, 10));
  anchor.click();
  URL.revokeObjectURL(url);
  const scopeText = exportScope === "all" ? "全部数据" : manifestFilterSummary(filters);
  showToast(`已按${scopeText}导出 ${lines.length} 条记录，数据集内容未改变。`);
}

function buildManifestRecord(record, context) {
  const decision = state.decisions[record.id] || {};
  const processing = state.processing[record.id] || {};
  return {
    ...record,
    decision: decision.decision || "unreviewed",
    keep: decision.keep ?? null,
    exclude_reason: decision.exclude_reason || "",
    decision_updated_at: decision.updated_at || null,
    manual_review_status: processing.status === "processed" ? "reviewed" : "unreviewed",
    manual_review_updated_at: processing.updated_at || null,
    processing_status: processing.status === "processed" ? "processed" : "unprocessed",
    processing_updated_at: processing.updated_at || null,
    processing_status_semantics: "manual_review_legacy",
    workflow_schema_version: 3,
    manifest_exported_at: context.exportedAt,
    manifest_hierarchy: context.exportedHierarchy,
    manifest_export_scope: context.exportScope,
    manifest_filters: context.filters,
    manifest_source_record_count: context.sourceRecordCount,
    manifest_exported_record_count: context.exportedRecordCount,
    manifest_catalog_generated_at: context.catalogGeneratedAt,
  };
}

function manifestFilename(filters, exportedDate) {
  const selectedScope = [filters.dataset, filters.scene].filter(Boolean).map(safeManifestFilenamePart);
  const fallbackScope = Object.keys(filters).length ? "filtered" : "all";
  const scope = selectedScope.join("-") || fallbackScope;
  return `dataset-filter-manifest-${scope}-${exportedDate}.jsonl`;
}

function safeManifestFilenamePart(value) {
  return String(value)
    .normalize("NFKC")
    .replace(/[\\/:*?"<>|\u0000-\u001f]/g, "-")
    .replace(/\s+/g, "_")
    .replace(/-+/g, "-")
    .replace(/^[-_.]+|[-_.]+$/g, "")
    .slice(0, 48) || "filter";
}

function embeddingBounds(points) {
  if (!points.length) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  const xValues = points.map((point) => point.x);
  const yValues = points.map((point) => point.y);
  let minX = Math.min(...xValues);
  let maxX = Math.max(...xValues);
  let minY = Math.min(...yValues);
  let maxY = Math.max(...yValues);
  if (minX === maxX) {
    minX -= 1;
    maxX += 1;
  }
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  }
  const xPadding = (maxX - minX) * 0.06;
  const yPadding = (maxY - minY) * 0.06;
  return { minX: minX - xPadding, maxX: maxX + xPadding, minY: minY - yPadding, maxY: maxY + yPadding };
}

function sceneColor(scene) {
  const index = state.sceneOrder.indexOf(scene);
  return index < 0 ? "var(--muted)" : `var(--scene-${(index % 16) + 1})`;
}

function scatterLayout(width, height) {
  const padding = { top: 24, right: 24, bottom: 46, left: 52 };
  const plotWidth = Math.max(1, width - padding.left - padding.right);
  const plotHeight = Math.max(1, height - padding.top - padding.bottom);
  const bounds = state.scatterBounds || { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  return {
    padding,
    plotWidth,
    plotHeight,
    x: (value) => padding.left + ((value - bounds.minX) / (bounds.maxX - bounds.minX)) * plotWidth,
    y: (value) => padding.top + (1 - (value - bounds.minY) / (bounds.maxY - bounds.minY)) * plotHeight,
  };
}

function renderScatter(records = filteredRecords(), { renderDetail = true } = {}) {
  if (!state.catalog || !elements.scatterViewport) return;
  const width = Math.max(320, elements.scatterViewport.clientWidth);
  const height = Math.max(420, elements.scatterViewport.clientHeight);
  elements.scatterSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  elements.scatterGrid.replaceChildren();
  elements.scatterCanvas.replaceChildren();
  state.visibleScatterRecords = new Map();

  if (state.embeddingError || !state.embeddingPoints.length) {
    elements.scatterSummary.textContent = "无可用语义向量";
    elements.scatterEmpty.hidden = false;
    elements.scatterEmpty.textContent = state.embeddingError || "当前目录册没有任务词条语义向量。";
    elements.scatterLegend.replaceChildren();
    if (renderDetail) renderScatterDetail();
    return;
  }

  const visibleRecordIds = new Set(records.map((record) => record.id));
  const visiblePoints = state.embeddingPoints.filter((point) => {
    const pointRecords = point.record_ids
      .filter((recordId) => visibleRecordIds.has(recordId))
      .map((recordId) => state.recordsById.get(recordId))
      .filter(Boolean);
    if (pointRecords.length) state.visibleScatterRecords.set(point.id, pointRecords);
    return pointRecords.length > 0;
  });
  if (state.selectedPointId && !state.visibleScatterRecords.has(state.selectedPointId)) {
    state.selectedPointId = null;
  }

  const layout = scatterLayout(width, height);
  renderScatterGrid(layout, width, height);
  const fragment = document.createDocumentFragment();
  for (const point of visiblePoints) {
    const circle = svgElement("circle");
    circle.setAttribute("class", `scatter-point${point.id === state.selectedPointId ? " selected" : ""}`);
    circle.setAttribute("cx", String(layout.x(point.x) + point.displayDx));
    circle.setAttribute("cy", String(layout.y(point.y) + point.displayDy));
    circle.setAttribute("r", "4.2");
    circle.setAttribute("fill", sceneColor(point.scene));
    circle.dataset.pointId = point.id;
    circle.setAttribute("role", "button");
    circle.setAttribute("aria-label", `${point.task}，场景 ${point.scene}`);
    const title = svgElement("title");
    title.textContent = `${point.task} · ${point.scene}`;
    circle.append(title);
    fragment.append(circle);
  }
  elements.scatterCanvas.append(fragment);
  applyScatterTransform();

  const visibleScenes = new Set(visiblePoints.map((point) => point.scene)).size;
  elements.scatterSummary.textContent = `${visiblePoints.length.toLocaleString("zh-CN")} / ${state.embeddingPoints.length.toLocaleString("zh-CN")} 个任务词条 · ${records.length.toLocaleString("zh-CN")} 条路径 · ${visibleScenes} 个场景`;
  elements.scatterEmpty.hidden = visiblePoints.length > 0;
  if (!visiblePoints.length) elements.scatterEmpty.textContent = "当前筛选条件下没有任务词条。";
  renderScatterLegend(visiblePoints);
  if (renderDetail) renderScatterDetail();
}

function renderScatterGrid(layout, width, height) {
  const fragment = document.createDocumentFragment();
  for (let index = 0; index <= 5; index += 1) {
    const vertical = svgElement("line");
    const x = layout.padding.left + (layout.plotWidth * index) / 5;
    vertical.setAttribute("class", "scatter-grid-line");
    vertical.setAttribute("x1", String(x));
    vertical.setAttribute("x2", String(x));
    vertical.setAttribute("y1", String(layout.padding.top));
    vertical.setAttribute("y2", String(height - layout.padding.bottom));
    fragment.append(vertical);

    const horizontal = svgElement("line");
    const y = layout.padding.top + (layout.plotHeight * index) / 5;
    horizontal.setAttribute("class", "scatter-grid-line");
    horizontal.setAttribute("x1", String(layout.padding.left));
    horizontal.setAttribute("x2", String(width - layout.padding.right));
    horizontal.setAttribute("y1", String(y));
    horizontal.setAttribute("y2", String(y));
    fragment.append(horizontal);
  }
  const xLabel = svgElement("text");
  xLabel.setAttribute("class", "scatter-axis-label");
  xLabel.setAttribute("x", String(layout.padding.left + layout.plotWidth / 2));
  xLabel.setAttribute("y", String(height - 13));
  xLabel.setAttribute("text-anchor", "middle");
  xLabel.textContent = "主成分 1";
  const yLabel = svgElement("text");
  yLabel.setAttribute("class", "scatter-axis-label");
  yLabel.setAttribute("transform", `translate(15 ${layout.padding.top + layout.plotHeight / 2}) rotate(-90)`);
  yLabel.setAttribute("text-anchor", "middle");
  yLabel.textContent = "主成分 2";
  fragment.append(xLabel, yLabel);
  elements.scatterGrid.append(fragment);
}

function renderScatterLegend(visiblePoints) {
  elements.scatterLegend.replaceChildren();
  const counts = new Map(state.sceneOrder.map((scene) => [scene, 0]));
  for (const point of visiblePoints) counts.set(point.scene, (counts.get(point.scene) || 0) + 1);
  const fragment = document.createDocumentFragment();
  for (const scene of state.sceneOrder) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `scatter-legend-item${elements.scene.value === scene ? " active" : ""}`;
    button.title = `只看 ${scene}`;
    button.addEventListener("click", () => {
      elements.scene.value = elements.scene.value === scene ? "" : scene;
      rebuildTree({ resetExpansion: true });
    });
    const swatch = document.createElement("span");
    swatch.className = "scatter-swatch";
    swatch.style.background = sceneColor(scene);
    const label = document.createElement("span");
    label.textContent = scene;
    const count = document.createElement("strong");
    count.textContent = counts.get(scene).toLocaleString("zh-CN");
    button.append(swatch, label, count);
    fragment.append(button);
  }
  elements.scatterLegend.append(fragment);
}

function renderScatterDetail() {
  if (state.activeView === "scatter") disposeVideoSession();
  elements.scatterDetail.replaceChildren();
  const point = state.embeddingPointsById.get(state.selectedPointId);
  const records = point ? state.visibleScatterRecords.get(point.id) || [] : [];
  if (!point || !records.length) {
    const heading = document.createElement("h2");
    heading.textContent = "选择一个任务词条";
    const paragraph = document.createElement("p");
    paragraph.textContent = "悬停查看摘要，点击散点后在这里保留数据集、路径和回合信息。";
    elements.scatterDetail.append(heading, paragraph);
    return;
  }

  const level = document.createElement("span");
  level.className = "detail-level";
  level.textContent = "原生任务词条";
  const heading = document.createElement("h2");
  heading.textContent = point.task;
  const recordIds = records.map((record) => record.id);
  const episodes = aggregateEpisodes(recordIds);
  const conversion = aggregateConversion(recordIds);
  const manualReview = aggregateManualReview(recordIds);
  const counts = decisionCounts(recordIds);
  elements.scatterDetail.append(
    level,
    heading,
    createGovernanceCard({
      recordIds: [...recordIds],
      scopeLabel: `${point.scene} / ${point.task}`,
      scopeLevel: "原生任务词条",
      counts,
      conversion,
      manualReview,
    }),
  );

  const list = document.createElement("dl");
  list.className = "detail-fields scatter-detail-fields";
  const fields = [
    ["场景", point.scene],
    ["数据域", joinedRecordValues(records, "domain", (value) => state.catalog.domain_labels[value] || value)],
    ["数据集", joinedRecordValues(records, "dataset")],
    ["任务类型", joinedRecordValues(records, "task_type")],
    ["原子动作", joinedRecordValues(records, "primary_atomic_action")],
    ["原生分组", joinedRecordValues(records, "native_group")],
    ["路径数量", records.length.toLocaleString("zh-CN")],
    ["服务器转换状态", conversionAriaText(conversion)],
    ["人工复核", manualReviewAriaText(manualReview)],
    ["回合总数", episodeDetailText(episodes)],
  ];
  for (const [labelText, value] of fields) {
    const term = document.createElement("dt");
    term.textContent = labelText;
    const description = document.createElement("dd");
    description.textContent = value;
    list.append(term, description);
  }
  appendPathField(list, "原始路径", records.map((record) => record.source_path));
  appendPathField(list, "转换产物", records.map((record) => record.conversion_output_path).filter(Boolean));
  appendPathField(list, "转换证据", records.map((record) => record.conversion_evidence_path).filter(Boolean));
  elements.scatterDetail.append(list);
  if (state.activeView === "scatter") {
    elements.scatterDetail.append(createVideoBrowser(records, `scatter:${point.id}`));
  }
}

function appendPathField(list, labelText, paths) {
  const term = document.createElement("dt");
  term.textContent = labelText;
  const description = document.createElement("dd");
  description.className = "scatter-path-list";
  const uniquePaths = [...new Set(paths)];
  if (!uniquePaths.length) description.textContent = "—";
  for (const path of uniquePaths.slice(0, 8)) {
    const code = document.createElement("code");
    code.textContent = path;
    description.append(code);
  }
  if (uniquePaths.length > 8) {
    const remainder = document.createElement("span");
    remainder.textContent = `另有 ${uniquePaths.length - 8} 条路径`;
    description.append(remainder);
  }
  list.append(term, description);
}

function joinedRecordValues(records, key, labeler = (value) => value) {
  const values = [...new Set(records.map((record) => record[key]).filter(Boolean).map(labeler))];
  values.sort((left, right) => left.localeCompare(right, "zh-CN"));
  if (values.length <= 4) return values.join("、");
  return `${values.slice(0, 4).join("、")} 等 ${values.length} 项`;
}

function handleScatterHover(event) {
  if (state.scatterDrag) {
    hideScatterTooltip();
    return;
  }
  const circle = event.target.closest?.(".scatter-point");
  const point = circle ? state.embeddingPointsById.get(circle.dataset.pointId) : null;
  const records = point ? state.visibleScatterRecords.get(point.id) || [] : [];
  if (!point || !records.length) {
    hideScatterTooltip();
    return;
  }
  showScatterTooltip(event, point, records);
}

function showScatterTooltip(event, point, records) {
  const episodes = aggregateEpisodes(records.map((record) => record.id));
  const conversion = aggregateConversion(records.map((record) => record.id));
  const heading = document.createElement("strong");
  heading.textContent = point.task;
  const taxonomy = document.createElement("span");
  taxonomy.textContent = `${point.scene} · ${joinedRecordValues(records, "dataset")}`;
  const summary = document.createElement("span");
  summary.textContent = `${records.length.toLocaleString("zh-CN")} 条路径 · ${conversionNodeText(conversion)} · ${episodeNodeText(episodes)}`;
  elements.scatterTooltip.replaceChildren(heading, taxonomy, summary);
  elements.scatterTooltip.hidden = false;
  const viewportRect = elements.scatterViewport.getBoundingClientRect();
  let left = event.clientX - viewportRect.left + 14;
  let top = event.clientY - viewportRect.top + 14;
  if (left + elements.scatterTooltip.offsetWidth > viewportRect.width - 8) {
    left = event.clientX - viewportRect.left - elements.scatterTooltip.offsetWidth - 14;
  }
  if (top + elements.scatterTooltip.offsetHeight > viewportRect.height - 8) {
    top = event.clientY - viewportRect.top - elements.scatterTooltip.offsetHeight - 14;
  }
  elements.scatterTooltip.style.left = `${Math.max(8, left)}px`;
  elements.scatterTooltip.style.top = `${Math.max(8, top)}px`;
}

function hideScatterTooltip() {
  elements.scatterTooltip.hidden = true;
}

function handleScatterClick(event) {
  const circle = event.target.closest?.(".scatter-point");
  if (!circle) return;
  state.selectedPointId = circle.dataset.pointId;
  for (const pointElement of elements.scatterCanvas.querySelectorAll(".scatter-point")) {
    pointElement.classList.toggle("selected", pointElement.dataset.pointId === state.selectedPointId);
  }
  renderScatterDetail();
}

function handleScatterWheel(event) {
  event.preventDefault();
  hideScatterTooltip();
  const rect = elements.scatterSvg.getBoundingClientRect();
  const pointerX = event.clientX - rect.left;
  const pointerY = event.clientY - rect.top;
  const oldScale = state.scatterTransform.k;
  const newScale = clamp(oldScale * Math.exp(-event.deltaY * 0.0012), 0.65, 7);
  const worldX = (pointerX - state.scatterTransform.x) / oldScale;
  const worldY = (pointerY - state.scatterTransform.y) / oldScale;
  state.scatterTransform.x = pointerX - worldX * newScale;
  state.scatterTransform.y = pointerY - worldY * newScale;
  state.scatterTransform.k = newScale;
  applyScatterTransform();
}

function handleScatterPointerDown(event) {
  if (event.button !== 0 || event.target.closest?.(".scatter-point")) return;
  state.scatterDrag = {
    x: event.clientX,
    y: event.clientY,
    startX: state.scatterTransform.x,
    startY: state.scatterTransform.y,
  };
  elements.scatterViewport.classList.add("dragging");
  elements.scatterSvg.setPointerCapture?.(event.pointerId);
}

function handleScatterPointerMove(event) {
  if (!state.scatterDrag) return;
  state.scatterTransform.x = state.scatterDrag.startX + event.clientX - state.scatterDrag.x;
  state.scatterTransform.y = state.scatterDrag.startY + event.clientY - state.scatterDrag.y;
  applyScatterTransform();
}

function handleScatterPointerUp() {
  state.scatterDrag = null;
  elements.scatterViewport?.classList.remove("dragging");
}

function zoomScatterBy(factor) {
  const rect = elements.scatterViewport.getBoundingClientRect();
  const oldScale = state.scatterTransform.k;
  const newScale = clamp(oldScale * factor, 0.65, 7);
  const centerX = rect.width / 2;
  const centerY = rect.height / 2;
  const worldX = (centerX - state.scatterTransform.x) / oldScale;
  const worldY = (centerY - state.scatterTransform.y) / oldScale;
  state.scatterTransform.x = centerX - worldX * newScale;
  state.scatterTransform.y = centerY - worldY * newScale;
  state.scatterTransform.k = newScale;
  applyScatterTransform();
}

function resetScatterView() {
  state.scatterTransform = { x: 0, y: 0, k: 1 };
  applyScatterTransform();
}

function applyScatterTransform() {
  elements.scatterCanvas?.setAttribute(
    "transform",
    `translate(${state.scatterTransform.x},${state.scatterTransform.y}) scale(${state.scatterTransform.k})`,
  );
}

function renderUnresolved() {
  elements.unresolvedList.replaceChildren();
  const unresolved = state.catalog.unresolved || [];
  const references = state.catalog.reference_datasets || [];
  elements.showUnresolved.textContent = `数据目录 / 待提取（${references.length + unresolved.length}）`;
  if (!references.length && !unresolved.length) {
    const paragraph = document.createElement("p");
    paragraph.textContent = "当前没有文档目录或待提取数据源。";
    elements.unresolvedList.append(paragraph);
    return;
  }
  if (references.length) {
    elements.unresolvedList.append(
      createDialogSectionHeading(
        "文档数据目录",
        "已下载状态来自服务器目录检查；未下载条目只作为规划目录展示，不会进入任务树统计。",
      ),
    );
    for (const item of references) elements.unresolvedList.append(createReferenceItem(item));
  }
  if (unresolved.length) {
    elements.unresolvedList.append(
      createDialogSectionHeading("待提取任务", "目录已经存在，但任务名称仍需从 metadata 中提取。"),
    );
  }
  for (const item of unresolved) {
    const section = document.createElement("section");
    section.className = "unresolved-item";
    const heading = document.createElement("h3");
    heading.textContent = item.native_scene ? `${item.dataset} / ${item.native_scene}` : item.dataset;
    const path = document.createElement("code");
    path.textContent = datasetPath(item.source_path);
    const reason = document.createElement("p");
    reason.textContent = `${item.available_level}：${item.reason}`;
    section.append(heading, path, reason);
    elements.unresolvedList.append(section);
  }
}

function createDialogSectionHeading(title, description) {
  const header = document.createElement("header");
  header.className = "dialog-section-heading";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const paragraph = document.createElement("p");
  paragraph.textContent = description;
  header.append(heading, paragraph);
  return header;
}

function createReferenceItem(item) {
  const section = document.createElement("section");
  section.className = "catalog-item";

  const header = document.createElement("div");
  header.className = "catalog-item-heading";
  const heading = document.createElement("h3");
  heading.textContent = item.dataset;
  const status = document.createElement("span");
  status.className = `availability ${item.availability === "downloaded" ? "downloaded" : "pending"}`;
  status.textContent = item.availability === "downloaded" ? "已下载" : "未下载";
  header.append(heading, status);

  const taxonomy = document.createElement("p");
  const domainLabel = state.catalog.domain_labels[item.domain] || item.domain;
  taxonomy.textContent = `${domainLabel} · ${item.category}`;
  section.append(header, taxonomy);

  const relativePath = item.source_path || item.expected_path;
  if (relativePath) {
    const path = document.createElement("code");
    path.textContent = datasetPath(relativePath);
    section.append(path);
  } else {
    const pathNote = document.createElement("p");
    pathNote.textContent = "尚未配置服务器预期目录。";
    section.append(pathNote);
  }

  if (item.source_reference) {
    const source = document.createElement("p");
    source.className = "catalog-source";
    source.append("来源：");
    if (/^https?:\/\//.test(item.source_reference)) {
      const link = document.createElement("a");
      link.href = item.source_reference;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = item.source_reference;
      source.append(link);
    } else source.append(item.source_reference);
    section.append(source);
  }
  return section;
}

function datasetPath(relativePath) {
  const root = state.catalog.dataset_root.replace(/\/$/, "");
  return `${root}/${String(relativePath).replace(/^\//, "")}`;
}

function sourceLabel(source) {
  return source === "native_name"
    ? "原生"
    : source === "manual"
      ? "人工"
      : source === "instruction_classification"
        ? "指令分类"
        : "名称推断";
}

function svgElement(name) {
  return document.createElementNS("http://www.w3.org/2000/svg", name);
}

function truncate(value, maxLength) {
  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}…` : value;
}

function normalizeSearch(value) {
  return value.toLocaleLowerCase().replace(/[_/.-]+/g, " ").replace(/\s+/g, " ").trim();
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function debounce(callback, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => callback(...args), delay);
  };
}

function showToast(message, isWarning = false) {
  clearTimeout(state.toastTimer);
  elements.toast.hidden = false;
  elements.toast.textContent = message;
  elements.toast.style.background = isWarning ? "var(--review)" : "var(--primary)";
  state.toastTimer = setTimeout(() => {
    elements.toast.hidden = true;
  }, 3000);
}
