"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const explorerRoot = path.resolve(__dirname, "..");
const webRoot = path.join(explorerRoot, "web");
const appSource = fs.readFileSync(path.join(webRoot, "app.js"), "utf8");
const indexSource = fs.readFileSync(path.join(webRoot, "index.html"), "utf8");
const context = vm.createContext({
  document: { addEventListener() {} },
  console,
  URL,
  window: {
    location: new URL("https://console.bce.baidu.com/api/aihcbox/proxy/v1/webide/example/bj/vscode/proxy/8788/"),
  },
});
vm.runInContext(appSource, context, { filename: "app.js" });

test("visible module headings use Chinese labels", () => {
  const visibleSource = `${indexSource}\n${appSource}`;
  for (const pattern of [
    />\s*Dataset semantic explorer\s*</,
    />\s*Hierarchy\s*</,
    />\s*Embedding\s*</,
    />\s*Metadata required\s*</,
    />\s*Data governance\s*</,
    /"LeRobot media"/,
    /"Episode 视频"/,
    /"PCA 维度/,
    /随 Manifest 导出/,
  ]) {
    assert.doesNotMatch(visibleSource, pattern);
  }
});

test("runtime data requests preserve a forwarded deployment subpath", () => {
  assert.match(indexSource, /src="catalog\.js\?v=20260903-atomic-action"/);
  assert.match(indexSource, /href="styles\.css\?v=20260904-three-view-compact"/);
  assert.match(indexSource, /src="app\.js\?v=20260911-file-before-governance"/);
  assert.match(appSource, /globalThis\.__DATASET_EXPLORER_CATALOG__/);
  assert.match(appSource, /fetch\(runtimeAssetUrl\("\.\/catalog\.json"\)/);
  assert.match(appSource, /fetch\(runtimeAssetUrl\("\.\/video-config\.json"\)/);
  assert.doesNotMatch(appSource, /fetch\("\/catalog\.json"/);
  assert.doesNotMatch(appSource, /fetch\("\/video-config\.json"/);

  const forwardedBase = new URL(
    "https://console.bce.baidu.com/api/aihcbox/proxy/v1/webide/example/bj/vscode/proxy/8788/",
  );
  assert.equal(
    new URL("./catalog.json", forwardedBase).pathname,
    "/api/aihcbox/proxy/v1/webide/example/bj/vscode/proxy/8788/catalog.json",
  );
  const runtimeCatalogUrl = new URL(context.runtimeAssetUrl("./catalog.json"));
  assert.equal(
    runtimeCatalogUrl.pathname,
    "/api/aihcbox/proxy/v1/webide/example/bj/vscode/proxy/8788/catalog.json",
  );
  assert.match(runtimeCatalogUrl.searchParams.get("_fresh"), /^\d+$/);
  assert.equal(
    context.resolveVideoApiBaseUrl("./video-api"),
    "https://console.bce.baidu.com/api/aihcbox/proxy/v1/webide/example/bj/vscode/proxy/8788/video-api",
  );
  assert.throws(
    () => context.resolveVideoApiBaseUrl("http://127.0.0.1:8790"),
    /公网 HTTPS 页面只能连接 HTTPS 视频服务/,
  );
  vm.runInContext('state.videoConfig.apiBaseUrl = resolveVideoApiBaseUrl("./video-api");', context);
  assert.equal(
    context.resolveVideoStreamUrl({ streamUrl: "/v1/media/test-token" }),
    "https://console.bce.baidu.com/api/aihcbox/proxy/v1/webide/example/bj/vscode/proxy/8788/video-api/v1/media/test-token",
  );
});

test("tree uses a left-aligned directory and videos group by episode before camera", () => {
  assert.match(indexSource, /<div id="tree" class="directory-tree" role="tree"/);
  assert.doesNotMatch(indexSource, /<svg id="tree"/);
  assert.doesNotMatch(indexSource, /id="zoom-(?:in|out)"/);
  assert.match(appSource, /createVideoPickerStep\("1", "选择回合"\)/);
  assert.match(appSource, /VIDEO_WALL_VIEWS/);
  assert.match(appSource, /自动播放左腕、右腕和主视角/);
  const groups = JSON.parse(JSON.stringify(context.groupedVideoEpisodes([
    { episodeIndex: 3, cameraKey: "front" },
    { episodeIndex: 3, cameraKey: "wrist" },
    { episodeIndex: 4, cameraKey: "front" },
  ])));
  assert.deepEqual(groups, [
    { key: "number:3", episodeIndex: 3, itemIndexes: [0, 1] },
    { key: "number:4", episodeIndex: 4, itemIndexes: [2] },
  ]);
});

test("native group is provenance only and not a directory level", () => {
  const levels = JSON.parse(vm.runInContext("JSON.stringify(LEVELS)", context));
  assert.deepEqual(
    levels.map((level) => level.key),
    ["domain", "scene", "dataset", "task_type", "primary_atomic_action", "native_task"],
  );
  assert.match(appSource, /\["原生分组", record\.native_group\]/);
});

test("downloaded reference datasets without task rows are labelled as pending extraction", () => {
  assert.match(appSource, /reference\?\.availability === "downloaded" && !extractedDatasets\.has\(value\)/);
  assert.match(appSource, /任务待提取/);
  assert.match(appSource, /尚未提取出可用于层级目录的任务 metadata/);
});

test("dataset and source filters update each other in both directions", () => {
  const result = vm.runInContext(`(() => {
    state.catalog = {
      records: [
        { dataset: "10Kh-RealOmin", domain: "real" },
        { dataset: "FastUMI-Data", domain: "umi" },
        { dataset: "RoboMINDv2", domain: "real" },
        { dataset: "RoboMINDv2", domain: "sim" }
      ],
      reference_datasets: []
    };
    Object.assign(elements, {
      domain: { value: "umi" },
      dataset: {
        value: "10Kh-RealOmin",
        options: [
          { value: "", hidden: false, disabled: false },
          { value: "10Kh-RealOmin", hidden: false, disabled: false },
          { value: "FastUMI-Data", hidden: false, disabled: false },
          { value: "RoboMINDv2", hidden: false, disabled: false }
        ]
      }
    });
    synchronizeDomainDatasetFilters("dataset");
    const selectedDatasetDomain = elements.domain.value;
    const realSourceOptions = elements.dataset.options.map((option) => ({
      value: option.value,
      hidden: option.hidden,
      disabled: option.disabled,
    }));
    elements.dataset.value = "RoboMINDv2";
    synchronizeDomainDatasetFilters("dataset");
    const multiSourceDatasetDomain = elements.domain.value;
    const allSourceOptionsVisible = elements.dataset.options
      .filter((option) => option.value)
      .every((option) => !option.hidden && !option.disabled);
    elements.domain.value = "umi";
    synchronizeDomainDatasetFilters("domain");
    return {
      selectedDatasetDomain,
      multiSourceDatasetDomain,
      realSourceOptions,
      allSourceOptionsVisible,
      datasetAfterDomainChange: elements.dataset.value,
    };
  })()` , context);
  assert.equal(result.selectedDatasetDomain, "real");
  assert.equal(result.multiSourceDatasetDomain, "");
  assert.equal(result.realSourceOptions.find((option) => option.value === "10Kh-RealOmin").disabled, false);
  assert.equal(result.realSourceOptions.find((option) => option.value === "FastUMI-Data").disabled, true);
  assert.equal(result.allSourceOptionsVisible, true);
  assert.equal(result.datasetAfterDomainChange, "");
});

test("camera fields are reduced to compact semantic view keywords", () => {
  assert.equal(context.cameraViewKeyword("observation.images.camera_left"), "left");
  assert.equal(context.cameraViewKeyword("observation.images.camera_right"), "right");
  assert.equal(context.cameraViewKeyword("observation.images.front"), "front");
  assert.equal(context.cameraViewKeyword("observation.images.head"), "head");
  assert.equal(context.cameraViewKeyword("observation.images.left_wrist"), "left_wrist");
  assert.equal(context.cameraViewKeyword("observation.images.right_wrist"), "right_wrist");
  assert.equal(context.cameraViewKeyword("sensors.third-person.rgb"), "third_person");
  assert.equal(context.cameraViewKeyword("camera_0"), "0");
});

test("inline image datasets can explain why no standalone video exists", () => {
  assert.match(appSource, /session\.videoNote = typeof payload\.video_note/);
  assert.match(appSource, /session\.videoNote \|\| "转换元数据存在/);
});

test("shared LeRobot v3 videos play only the selected episode time range", () => {
  assert.match(appSource, /startSeconds: Number\.isFinite\(item\.start_seconds\)/);
  assert.match(appSource, /player\.video\.currentTime = item\.startSeconds/);
  assert.match(appSource, /player\.video\.currentTime >= item\.endSeconds/);
});

test("task details place file information above governance and show the instruction", () => {
  const renderDetailSource = appSource.slice(
    appSource.indexOf("function renderDetail()"),
    appSource.indexOf("function createGovernanceCard"),
  );
  assert.match(
    renderDetailSource,
    /if \(representative\) \{[\s\S]*?createVideoBrowser[\s\S]*?elements\.detail\.append\(\s*createFileInformation\([\s\S]*?governanceCard/,
  );
  assert.match(appSource, /section\.className = "file-information"/);
  assert.match(appSource, /heading\.textContent = "路径与转换产物"/);
  assert.match(appSource, /\["任务 instruction", record\.native_task\]/);
});

test("governance labels preserve storage values while improving display copy", () => {
  assert.match(indexSource, /value="review"[^>]*>[\s\S]*?需复查/);
  assert.match(indexSource, /value="unreviewed"[^>]*>[\s\S]*?未标注/);
  assert.equal((indexSource.match(/id="governance-dialog"/g) || []).length, 1);
  assert.equal(context.decisionSummaryText({ keep: 2, exclude: 1, review: 3, unreviewed: 4 }),
    "保留 2 · 排除 1 · 需复查 3 · 未标注 4");
});

test("governance comparison detects semantic changes without rewriting unchanged dimensions", () => {
  vm.runInContext(
    `state.decisions = {
      keep: { decision: "keep", keep: true, exclude_reason: "" },
      excluded: { decision: "exclude", keep: false, exclude_reason: "重复" }
    };
    state.processing = { reviewed: { status: "processed" } };`,
    context,
  );

  assert.equal(context.governanceDecisionDiffers("missing", "unreviewed", ""), false);
  assert.equal(context.governanceDecisionDiffers("keep", "keep", ""), false);
  assert.equal(context.governanceDecisionDiffers("keep", "exclude", "重复"), true);
  assert.equal(context.governanceDecisionDiffers("excluded", "exclude", "重复"), false);
  assert.equal(context.governanceDecisionDiffers("excluded", "exclude", "质量问题"), true);
  assert.equal(context.governanceReviewDiffers("reviewed", "processed"), false);
  assert.equal(context.governanceReviewDiffers("reviewed", "unprocessed"), true);
  assert.equal(context.governanceReviewDiffers("missing", "unprocessed"), false);
});

test("manifest export filters intersect dataset and scene and capture the exact scope", () => {
  vm.runInContext(
    `Object.assign(elements, {
      search: { value: "" },
      domain: { value: "" },
      dataset: { value: "Dataset-A" },
      scene: { value: "厨房" },
      taskType: { value: "" },
      decision: { value: "" },
      conversion: { value: "" },
      processing: { value: "" }
    });
    state.catalog = {
      domain_labels: { real: "真机" },
      records: [
        { id: "a", domain: "real", dataset: "Dataset-A", scene: "厨房", task_type: "抓取", native_task: "拿杯子", source_path: "/a", conversion_status: "converted" },
        { id: "b", domain: "real", dataset: "Dataset-A", scene: "客厅", task_type: "放置", native_task: "放遥控器", source_path: "/b", conversion_status: "partial" },
        { id: "c", domain: "real", dataset: "Dataset-B", scene: "厨房", task_type: "抓取", native_task: "拿盘子", source_path: "/c", conversion_status: "converted" }
      ]
    };
    state.decisions = {};
    state.processing = {};`,
    context,
  );

  const recordIds = () => JSON.parse(JSON.stringify(context.filteredRecords().map((record) => record.id)));
  assert.deepEqual(recordIds(), ["a"]);
  assert.deepEqual(JSON.parse(JSON.stringify(context.currentManifestFilters())), {
    dataset: "Dataset-A",
    scene: "厨房",
  });
  assert.equal(context.manifestFilename(context.currentManifestFilters(), "2026-08-31"),
    "dataset-filter-manifest-Dataset-A-厨房-2026-08-31.jsonl");
  vm.runInContext(
    `state.decisions.a = { decision: "keep", keep: true, exclude_reason: "", updated_at: "2026-08-31T09:00:00Z" };
    state.processing.a = { status: "processed", updated_at: "2026-08-31T09:05:00Z" };`,
    context,
  );
  const manifestRecord = JSON.parse(JSON.stringify(context.buildManifestRecord(
    vm.runInContext("state.catalog.records[0]", context),
    {
      exportedAt: "2026-08-31T10:00:00Z",
      exportedHierarchy: ["domain", "scene", "dataset"],
      exportScope: "current_filters",
      filters: context.currentManifestFilters(),
      sourceRecordCount: 3,
      exportedRecordCount: 1,
      catalogGeneratedAt: "2026-08-31T08:00:00Z",
    },
  )));
  assert.equal(manifestRecord.decision, "keep");
  assert.equal(manifestRecord.manual_review_status, "reviewed");
  assert.equal(manifestRecord.manifest_export_scope, "current_filters");
  assert.deepEqual(manifestRecord.manifest_filters, { dataset: "Dataset-A", scene: "厨房" });
  assert.equal(manifestRecord.manifest_source_record_count, 3);
  assert.equal(manifestRecord.manifest_exported_record_count, 1);

  vm.runInContext('elements.dataset.value = "";', context);
  assert.deepEqual(recordIds(), ["a", "c"]);
  vm.runInContext('elements.scene.value = "";', context);
  assert.deepEqual(recordIds(), ["a", "b", "c"]);
  vm.runInContext('elements.dataset.value = "Dataset-B"; elements.scene.value = "客厅";', context);
  assert.deepEqual(recordIds(), []);
});
