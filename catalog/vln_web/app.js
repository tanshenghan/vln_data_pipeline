"use strict";

const LANGUAGE_LABELS = { en: "English", "en-US": "English · US", "en-IN": "English · IN", "hi-IN": "हिन्दी · HI", "te-IN": "తెలుగు · TE", unknown: "语言未标注" };
const ISSUE_LABELS = { missing_instruction: "缺少指令", missing_geometry: "缺少参考路径坐标", missing_goal: "未公开或缺少目标", missing_scene: "缺少场景标识" };
const state = { metadata: null, params: {}, offset: 0, limit: 40, total: 0, items: [], selected: null, listRequest: null, detailRequest: null, videoPoll: null, gtTimer: null, exportBusy: false };
const $ = (selector, parent = document) => parent.querySelector(selector);
const number = value => Number(value).toLocaleString("zh-CN");
const metres = value => value == null ? "未提供" : `${Number(value).toFixed(1)} m`;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function apiUrl(endpoint, params = {}) {
  const url = new URL(`./vln-api/${endpoint}`, window.location.href);
  for (const [key, value] of Object.entries(params)) if (value !== "" && value != null) url.searchParams.set(key, String(value));
  return url;
}

async function api(endpoint, params = {}, signal) {
  const response = await fetch(apiUrl(endpoint, params), { cache: "no-store", signal });
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try { message = (await response.json()).error || message; } catch (_) { /* Keep HTTP error. */ }
    throw new Error(message);
  }
  return response.json();
}

function notify(message, error = false, retry) {
  const box = $("#notice");
  box.replaceChildren(document.createTextNode(message));
  box.className = error ? "notice error" : "notice";
  box.hidden = false;
  if (retry) { const button = element("button", "", "重试"); button.addEventListener("click", retry); box.append(button); }
}

function applyFilters(values) {
  for (const [key, value] of Object.entries(values)) {
    const control = $("#filters").elements.namedItem(key);
    if (control) {
      if (control.tagName === "SELECT" && value && ![...control.options].some(o => o.value === value)) control.add(new Option(value, value));
      control.value = value;
    }
  }
  readFilters();
}

function readFilters() {
  state.params = Object.fromEntries([...new FormData($("#filters"))].filter(([, value]) => value !== ""));
  state.offset = 0;
  state.selected = null;
  const url=new URL(window.location.href);url.search="";for(const [key,value]of Object.entries(state.params))url.searchParams.set(key,value);window.history.replaceState(null,"",url);
  loadList();
}

function populateFacets(facets) {
  for (const [field, choices] of Object.entries(facets)) {
    const select = $(`#filters select[name="${field}"]`);
    if (!select) continue;
    const current = select.value;
    const first = select.options[0].textContent;
    select.replaceChildren(new Option(first, ""));
    for (const choice of choices) {
      let label = field === "language" ? LANGUAGE_LABELS[choice.value] || choice.value : choice.value;
      if (field === "source_id") { const source = state.metadata.sources.find(s => s.id === choice.value); label = source ? `${source.dataset} · ${source.version}` : choice.value; }
      select.add(new Option(`${label} (${number(choice.count)})`, choice.value));
    }
    if (current && !choices.some(item => item.value === current)) select.add(new Option(`${current} (0)`, current));
    select.value = current;
  }
}

function renderDirectory(rows) {
  const root = $("#directory"); root.replaceChildren();
  const datasets = Map.groupBy ? Map.groupBy(rows, r => r.dataset) : groupBy(rows, r => r.dataset);
  for (const [dataset, entries] of datasets) {
    const section = element("details"); section.open = true;
    const summary = element("summary", "", dataset); summary.append(element("b", "", number(entries.reduce((n,r) => n+r.count, 0))));
    section.append(summary);
    const all = element("button", "branch", `查看全部 ${dataset}`);
    all.addEventListener("click", () => applyFilters({ dataset, source_id: "", split: "", scene: "", language: "", role: "" })); section.append(all);
    for (const [split, scenes] of groupBy(entries, r => r.split)) {
      const splitNode = element("details");
      const title = element("summary", "", split); title.append(element("b", "", number(scenes.reduce((n,r) => n+r.count, 0)))); splitNode.append(title);
      const splitButton = element("button", "branch", `查看 ${split}`);
      splitButton.addEventListener("click", () => applyFilters({ dataset, source_id: "", split, scene: "", language: "", role: "" })); splitNode.append(splitButton);
      for (const [scene, entries] of groupBy(scenes, r => r.scene)) {
        const button = element("button", "scene-button", scene);
        button.append(element("span", "", number(entries.reduce((n,r) => n+r.count, 0))));
        button.addEventListener("click", () => applyFilters({ dataset, source_id: "", split, scene, language: "", role: "" })); splitNode.append(button);
      }
      section.append(splitNode);
    }
    root.append(section);
  }
}

function groupBy(rows, key) { const result = new Map(); for (const row of rows) { const value = key(row); if (!result.has(value)) result.set(value, []); result.get(value).push(row); } return result; }

async function loadList() {
  state.listRequest?.abort(); state.detailRequest?.abort(); stopVideoPolling(); stopPlayback();
  const controller = new AbortController(); state.listRequest = controller;
  $("#sample-list").setAttribute("aria-busy", "true");
  $("#previous").disabled = true; $("#next").disabled = true; $("#export").disabled = true;
  try {
    const result = await api("episodes", { ...state.params, offset: state.offset, limit: state.limit }, controller.signal);
    if (controller !== state.listRequest) return;
    state.items = result.items; state.total = result.stats.records;
    for (const field of ["records", "routes", "scenes", "media"]) $(`#stat-${field}`).textContent = number(result.stats[field]);
    $("#stat-length").textContent = metres(result.stats.mean_path_length);
    $("#result-count").textContent = `${number(state.total)} 条`;
    $("#page-info").textContent = state.total ? `${number(state.offset+1)}–${number(state.offset+state.items.length)} / ${number(state.total)}` : "0 条样本";
    $("#previous").disabled = state.offset === 0; $("#next").disabled = state.offset + state.limit >= state.total;
    $("#export").disabled = !state.total || state.exportBusy;
    populateFacets(result.facets); renderList();
    if (state.items.length) selectRecord(state.items.find(r => r.id === state.selected)?.id || state.items.find(r => r.episode_id === state.params.q)?.id || state.items[0].id);
    else { state.selected = null; $("#detail").replaceChildren(empty("没有匹配的样本", "调整筛选条件，或点击重置查看完整目录。")); }
  } catch (error) {
    if (error.name !== "AbortError") notify(error.message, true, loadList);
  } finally {
    if (controller === state.listRequest) $("#sample-list").setAttribute("aria-busy", "false");
  }
}

function empty(title, text) { const box = element("div", "empty"); box.append(element("h2", "", title), element("p", "", text)); return box; }
function chip(text, className = "") { return element("span", `chip ${className}`, text); }

function renderList() {
  const list = $("#sample-list"); list.replaceChildren();
  if (!state.items.length) { list.append(empty("没有匹配结果", "尝试减少筛选条件。")); return; }
  for (const record of state.items) {
    const button = element("button", "sample");
    button.setAttribute("aria-current", String(record.id === state.selected));
    const top = element("div", "sample-top"); const labels = element("div", "chips");
    labels.append(chip(record.dataset, record.dataset.toLowerCase()), chip(record.language));
    labels.append(chip(record.has_gt ? "GT 已接入" : record.split === "test" ? "GT 未公开" : "GT 未接入", record.has_gt ? "keep" : ""));
    top.append(element("span", "sample-id", `Episode ${record.episode_id}`), labels);
    const instruction = element("p", "sample-instruction", record.instruction || "该样本没有公开指令"); instruction.lang = record.language; instruction.dir = "auto";
    const meta = element("div", "sample-meta");
    for (const text of [record.scene, record.split, record.role, metres(record.path_length), record.has_gt_video ? "GT 视频" : ""]) if (text) meta.append(element("span", "", text));
    button.append(top, instruction, meta);
    button.addEventListener("click", () => selectRecord(record.id)); list.append(button);
  }
}

async function selectRecord(id) {
  state.detailRequest?.abort(); stopVideoPolling(); stopPlayback();
  const controller = new AbortController(); state.detailRequest = controller;
  state.selected = id; renderList();
  $("#detail").replaceChildren(empty("正在读取样本…", ""));
  try { const record = await api("episode", { id }, controller.signal); if (controller === state.detailRequest) renderDetail(record); }
  catch (error) { if (error.name !== "AbortError") { $("#detail").replaceChildren(empty("样本加载失败", error.message)); notify(error.message, true, () => selectRecord(id)); } }
}

function section(title) { const node = element("section", "detail-section"); if (title) node.append(element("h3", "", title)); return node; }
function facts(items) { const list = element("dl", "facts"); for (const [key, value] of items) { const entry = element("div"); entry.append(element("dt", "", key), element("dd", "", value)); list.append(entry); } return list; }

function renderDetail(record) {
  const detail = $("#detail"); detail.replaceChildren();
  const header = element("div", "detail-header"); const chips = element("div", "chips");
  chips.append(chip(record.dataset, record.dataset.toLowerCase()), chip(record.split), chip(record.role), chip(LANGUAGE_LABELS[record.language] || record.language));
  header.append(chips, element("h2", "", `Episode ${record.episode_id}`), element("p", "", `${record.scene} · ${record.version}`)); detail.append(header);
  const instructionSection = section(); instructionSection.append(element("p", "instruction-label", `导航指令 · ${record.instruction_id}`));
  const instruction = element("p", "instruction", record.instruction || "该样本没有公开指令。"); instruction.lang = record.language; instruction.dir = "auto"; instructionSection.append(instruction);
  if (record.issues.length) instructionSection.append(element("p", "issues", record.issues.map(x => ISSUE_LABELS[x] || x).join(" · ")));
  detail.append(instructionSection, renderGTVideo(record), renderRoute(record));
  const provenance = section(); const disclosure = element("details", "details-toggle"); disclosure.append(element("summary", "", "样本来源与原始字段"));
  disclosure.append(element("p", "source", `标注文件：${record.source_path}\n记录位置（从 0 计数）：${record.source_row}\n场景文件：${record.scene_path}\n轨迹 ID：${record.trajectory_id}\n起点：${JSON.stringify(record.start_position)}\n目标：${JSON.stringify(record.goals)}\n初始旋转：${JSON.stringify(record.start_rotation ?? record.heading)}\n稳定样本 ID：${record.id}`));
  if (record.viewpoints.length) disclosure.append(element("p", "source", `Viewpoints：${record.viewpoints.join(" → ")}`));
  if (record.timed_instruction?.length) {
    const timings = element("div", "timings"); for (const word of record.timed_instruction) { const item = element("span", "", word.word || ""); item.append(element("small", "", `${word.start_time ?? "?"}–${word.end_time ?? "?"}s`)); timings.append(item); } disclosure.append(timings);
  }
  provenance.append(disclosure); detail.append(provenance);
  window.dispatchEvent(new CustomEvent("vln:record-selected", { detail: record }));
}

function projectRoute(points, axes = [0, 2], width = 620, height = 300) {
  if (!points.length) return [];
  const horizontal = points.map(p => p[axes[0]]), vertical = points.map(p => p[axes[1]]);
  const minX = Math.min(...horizontal), maxX = Math.max(...horizontal), minY = Math.min(...vertical), maxY = Math.max(...vertical);
  const scale = Math.min((width-90)/Math.max(maxX-minX, 1), (height-70)/Math.max(maxY-minY, 1));
  return points.map(p => [width/2+(p[axes[0]]-(minX+maxX)/2)*scale, height/2-(p[axes[1]]-(minY+maxY)/2)*scale]);
}

function svgNode(tag, attributes = {}) { const node = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [key,value] of Object.entries(attributes)) node.setAttribute(key, String(value)); return node; }

function nearestFloor(height, floors) {
  if (!floors.length) return -1;
  return floors.reduce((best, floor, index) => Math.abs(floor.height-height) < Math.abs(floors[best].height-height) ? index : best, 0);
}

function worldToMap(point, floor) {
  return [(point[0]-floor.origin[0])/floor.meters_per_pixel, (point[2]-floor.origin[1])/floor.meters_per_pixel];
}

function floorSegments(points, floors, index) {
  if (!floors.length) return [points];
  const segments = []; let current = [];
  for (const point of points) {
    if (nearestFloor(point[1], floors) === index) current.push(point);
    else if (current.length) { segments.push(current); current = []; }
  }
  if (current.length) segments.push(current);
  return segments;
}

function boundsForPoints(points, full, padding) {
  if (!points.length) return [...full];
  const xs=points.map(p=>p[0]), ys=points.map(p=>p[1]);
  const left=Math.max(full[0],Math.min(...xs)-padding), top=Math.max(full[1],Math.min(...ys)-padding);
  const right=Math.min(full[0]+full[2],Math.max(...xs)+padding), bottom=Math.min(full[1]+full[3],Math.max(...ys)+padding);
  return right>left && bottom>top ? [left,top,right-left,bottom-top] : [...full];
}

function downloadBlob(blob, filename) {
  const url=URL.createObjectURL(blob), anchor=element("a");
  anchor.href=url;anchor.download=filename;document.body.append(anchor);anchor.click();anchor.remove();
  setTimeout(()=>URL.revokeObjectURL(url),30000);
}

function renderRoute(record) {
  const box=section();box.classList.add("gt-section");
  const gt=record.ground_truth || {available:false,reason:"not_indexed"};
  const points=gt.available ? gt.locations : record.reference_path;
  const floors=record.topdown?.floors || [];
  const heading=element("div","section-heading");
  heading.append(element("h3","","GT 轨迹与场景俯视图"),chip(gt.available ? "数据集评测 GT" : "GT 不可用",gt.available?"keep":""));box.append(heading);
  if(!gt.available){
    const reason=gt.reason==="test_unpublished" ? `${record.dataset} test 没有公开 GT 和目标。这里仅展示可用的场景底图与起点。` : gt.reason==="start_mismatch" ? "GT 起点与此样本不匹配，已停止关联；请检查数据版本。" : "此样本尚无已接入的 GT。已有稀疏参考路标会单独显示。";
    box.append(element("p","issues",reason));
  }
  const allPoints=[...points,...record.reference_path,...record.goals];
  if(record.start_position)allPoints.push(record.start_position);
  if(!allPoints.length && !floors.length){box.append(empty("没有可绘制的轨迹", "需要原始 GT 坐标或对应场景的 NavMesh。"));return box;}
  const projected=projectRoute(allPoints,record.projection_axes);
  const fallback=new Map(allPoints.map((p,i)=>[JSON.stringify(p),projected[i]]));
  let floorIndex=nearestFloor((points[0]||record.start_position||[0,0,0])[1],floors);
  let view=[0,0,620,300],cursor=null,markers=[],scaleLayer=null,currentIndex=0,drawnFloor=-2;
  const toolbar=element("div","map-toolbar");
  const floorSelect=element("select");floorSelect.setAttribute("aria-label","俯视图高度层");
  floors.forEach((floor,index)=>floorSelect.add(new Option(`高度层 ${index+1} · Y=${floor.height.toFixed(2)} m`,String(index))));
  if(!floors.length)floorSelect.add(new Option("底图未生成 · 坐标投影","-1"));
  floorSelect.value=String(floorIndex);floorSelect.disabled=!floors.length;
  const fit=element("button","","聚焦轨迹"),whole=element("button","","完整场景"),zoomIn=element("button","","＋"),zoomOut=element("button","","−");
  zoomIn.setAttribute("aria-label","放大俯视图");zoomOut.setAttribute("aria-label","缩小俯视图");
  toolbar.append(floorSelect,fit,whole,zoomOut,zoomIn);box.append(toolbar);
  const toggles=element("div","map-toggles");
  function toggle(label,checked){const node=element("label"),input=element("input");input.type="checkbox";input.checked=checked;node.append(input,document.createTextNode(label));toggles.append(node);return input;}
  const showReference=toggle("叠加稀疏参考路标",!gt.available),follow=toggle("随轨迹切换高度层",true);
  follow.disabled=floors.length<2;box.append(toggles);
  const layout=element("div","gt-layout"),mapPanel=element("div","gt-map-panel"),summary=element("aside","gt-summary");
  summary.setAttribute("aria-label","GT 轨迹统计");layout.append(mapPanel,summary);box.append(layout);
  const svg=svgNode("svg",{class:"route gt-map",role:"img","aria-label":"NavMesh 可通行区域与 GT 轨迹俯视图",viewBox:"0 0 620 300"});
  mapPanel.append(svg);
  const legend=element("div","map-legend");
  for(const [label,color] of [["可通行区域","#fafcff"],["不可通行 / 未覆盖","#dee5ef"],["GT","#4338ca"],["参考路标","#c16d24"],["起点","#087f83"],["目标","#e56b35"]]){
    const item=element("span","",label),swatch=element("i");swatch.style.background=color;item.prepend(swatch);legend.append(item);
  }
  mapPanel.append(legend);
  const controls=element("div","route-controls"),play=element("button","gt-play","播放 GT");
  const slider=element("input");slider.type="range";slider.min="0";slider.value="0";slider.setAttribute("aria-label","GT 轨迹进度");
  const aligned=gt.available && gt.action_location_indices?.length>0;
  slider.max=String(Math.max(0,(aligned?gt.action_location_indices.length:points.length)-1));slider.disabled=!points.length;play.disabled=!gt.available || points.length<2;
  const stepLabel=element("span"),coordinate=element("p","gt-step-info");
  controls.append(play,slider,stepLabel);mapPanel.append(controls,coordinate);
  const cumulative=[0];for(let i=1;i<points.length;i++)cumulative.push(cumulative[i-1]+Math.hypot(...points[i].map((v,j)=>v-points[i-1][j])));
  function floor(){return floors[floorIndex];}
  function pixel(point){return floor()?worldToMap(point,floor()):fallback.get(JSON.stringify(point))||[0,0];}
  function onFloor(point){return !floors.length || nearestFloor(point[1],floors)===floorIndex;}
  function fullView(){return floor()?[0,0,floor().width,floor().height_pixels]:[0,0,620,300];}
  function setView(next){
    view=next;svg.setAttribute("viewBox",view.join(" "));
    const ratio=Math.max(view[2]/620,view[3]/420);
    markers.forEach(([node,size])=>node.setAttribute("r",size*ratio));
    if(scaleLayer){
      scaleLayer.replaceChildren();
      const target=view[2]*floor().meters_per_pixel*.18;
      const power=10**Math.floor(Math.log10(Math.max(target,.001)));
      const length=[1,2,5,10].map(n=>n*power).filter(n=>n<=target).pop()||power;
      const x=view[0]+view[2]*.045,y=view[1]+view[3]*.925,width=length/floor().meters_per_pixel;
      scaleLayer.append(svgNode("rect",{x:x-8*ratio,y:y-24*ratio,width:width+50*ratio,height:36*ratio,fill:"#ffffff",opacity:.92,rx:3*ratio}));
      scaleLayer.append(svgNode("path",{d:`M${x},${y-5*ratio} V${y} H${x+width} V${y-5*ratio}`,stroke:"#26364e","stroke-width":2,fill:"none","vector-effect":"non-scaling-stroke"}));
      const text=svgNode("text",{x,y:y-9*ratio,fill:"#26364e"});text.style.fontSize=`${13*ratio}px`;text.textContent=`${Number(length.toPrecision(3))} m`;scaleLayer.append(text);
    }
  }
  function fitRoute(){
    const visible=[...points,...record.goals,...(record.start_position?[record.start_position]:[])].filter(onFloor).map(pixel);
    setView(boundsForPoints(visible,fullView(),floor()?2/floor().meters_per_pixel:35));
  }
  function addPath(path,color,dashed){
    for(const segment of floorSegments(path,floors,floorIndex)){
      if(segment.length<2)continue;
      svg.append(svgNode("polyline",{points:segment.map(p=>pixel(p).join(",")).join(" "),fill:"none",stroke:color,"stroke-width":dashed?2:3,"stroke-linejoin":"round","stroke-linecap":"round","stroke-dasharray":dashed?"6 5":"none","vector-effect":"non-scaling-stroke"}));
    }
  }
  function addMarker(point,color,size,title,always=false){
    if(!point||(!always&&!onFloor(point)))return null;
    const [cx,cy]=pixel(point),marker=svgNode("circle",{cx,cy,r:size,fill:color,stroke:"#fff","stroke-width":2,"vector-effect":"non-scaling-stroke",visibility:onFloor(point)?"visible":"hidden"});
    const label=svgNode("title");label.textContent=title;marker.append(label);svg.append(marker);markers.push([marker,size]);return marker;
  }
  function draw(reset=true){
    const previous=view;svg.replaceChildren();markers=[];scaleLayer=null;
    const extent=fullView();svg.append(svgNode("rect",{x:0,y:0,width:extent[2],height:extent[3],fill:"#f7f9fc"}));
    if(floor()){
      const background=svgNode("image",{href:mediaUrl(floor().image),x:0,y:0,width:floor().width,height:floor().height_pixels});background.setAttribute("data-map-background","true");
      background.addEventListener("error",()=>notify("场景底图暂时无法加载；GT 坐标仍可查看和下载。",true));svg.append(background);
    }else{for(let x=0;x<620;x+=30)svg.append(svgNode("line",{x1:x,x2:x,y1:0,y2:300,stroke:"#e5eaf3"}));for(let y=0;y<300;y+=30)svg.append(svgNode("line",{x1:0,x2:620,y1:y,y2:y,stroke:"#e5eaf3"}));}
    if(showReference.checked)addPath(record.reference_path,"#c16d24",true);
    if(gt.available)addPath(points,"#4338ca",false);
    addMarker(record.start_position||points[0],"#087f83",8,"起点");record.goals.forEach(point=>addMarker(point,"#e56b35",8,"目标"));
    cursor=addMarker(points[currentIndex],"#4338ca",5,"当前 GT 位置",true);
    if(floor()){scaleLayer=svgNode("g",{"data-scale":"true"});svg.append(scaleLayer);}
    drawnFloor=floorIndex;floorSelect.value=String(floorIndex);
    if(reset)fitRoute();else setView(previous);
  }
  function update(){
    const step=Number(slider.value);currentIndex=aligned?gt.action_location_indices[step]:step;
    const point=points[currentIndex];
    if(point&&follow.checked&&floors.length)floorIndex=nearestFloor(point[1],floors);
    if(drawnFloor!==floorIndex)draw();
    if(cursor&&point){const [cx,cy]=pixel(point);cursor.setAttribute("cx",cx);cursor.setAttribute("cy",cy);cursor.setAttribute("visibility",onFloor(point)?"visible":"hidden");}
    if(!point){stepLabel.textContent="无 GT";coordinate.textContent="当前样本没有公开可播放的 GT 坐标。";return;}
    const action=aligned?(step===0?"初始状态":gt.action_names[String(gt.actions[step-1])]||String(gt.actions[step-1])):"坐标节点";
    stepLabel.textContent=`${step} / ${slider.max}`;
    coordinate.textContent=`${action} · 坐标点 ${currentIndex+1}/${points.length} · 累计 ${cumulative[currentIndex].toFixed(2)} m\nXYZ [${point.map(n=>n.toFixed(4)).join(", ")}] m`;
  }
  floorSelect.addEventListener("change",()=>{floorIndex=Number(floorSelect.value);follow.checked=false;draw();update();});
  showReference.addEventListener("change",()=>draw(false));follow.addEventListener("change",update);
  slider.addEventListener("input",update);fit.addEventListener("click",fitRoute);whole.addEventListener("click",()=>setView(fullView()));
  function zoom(factor,center){const [x,y,w,h]=view;const anchor=center||[x+w/2,y+h/2];const nextWidth=w*factor;if(nextWidth<8||nextWidth>fullView()[2]*5)return;setView([anchor[0]-(anchor[0]-x)*factor,anchor[1]-(anchor[1]-y)*factor,w*factor,h*factor]);}
  zoomIn.addEventListener("click",()=>zoom(.75));zoomOut.addEventListener("click",()=>zoom(1.3));
  function localPoint(event){const p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;return p.matrixTransform(svg.getScreenCTM().inverse());}
  svg.addEventListener("wheel",event=>{event.preventDefault();const p=localPoint(event);zoom(event.deltaY>0?1.15:1/1.15,[p.x,p.y]);},{passive:false});
  let drag=null;
  svg.addEventListener("pointerdown",event=>{if(event.button!==0)return;drag={point:localPoint(event),view:[...view],inverse:svg.getScreenCTM().inverse()};svg.setPointerCapture(event.pointerId);});
  svg.addEventListener("pointermove",event=>{if(!drag)return;const p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;const current=p.matrixTransform(drag.inverse);setView([drag.view[0]+drag.point.x-current.x,drag.view[1]+drag.point.y-current.y,drag.view[2],drag.view[3]]);});
  svg.addEventListener("pointerup",()=>drag=null);svg.addEventListener("pointercancel",()=>drag=null);
  play.addEventListener("click",()=>{
    if(state.gtTimer){stopPlayback();return;}
    stopPlayback();if(Number(slider.value)>=Number(slider.max))slider.value="0";play.textContent="暂停 GT";update();
    state.gtTimer=setInterval(()=>{if(Number(slider.value)>=Number(slider.max)){stopPlayback();return;}slider.value=String(Number(slider.value)+1);update();},125);
  });
  const downloads=element("div","map-downloads");
  const gtLink=element("a","button-link","下载 GT JSON");gtLink.href=apiUrl("ground-truth",{id:record.id}).href;
  if(gt.available)downloads.append(gtLink);
  const saveMap=element("button","","下载俯视图 SVG");downloads.append(saveMap);
  saveMap.addEventListener("click",async()=>{
    saveMap.disabled=true;
    try{
      const clone=svg.cloneNode(true);clone.setAttribute("xmlns","http://www.w3.org/2000/svg");clone.setAttribute("width","1600");clone.setAttribute("height",String(Math.round(1600*view[3]/view[2])));
      const image=clone.querySelector("image");
      if(image){const response=await fetch(image.getAttribute("href"));if(!response.ok)throw new Error("底图下载失败");const blob=await response.blob();const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsDataURL(blob);});image.setAttribute("href",data);}
      const title=svgNode("title");title.textContent=`${record.dataset} ${record.split} episode ${record.episode_id} / ${record.scene} / height ${floor()?.height ?? "unknown"}`;clone.prepend(title);
      downloadBlob(new Blob([new XMLSerializer().serializeToString(clone)],{type:"image/svg+xml;charset=utf-8"}),`vln-topdown-${record.id}-layer-${floorIndex+1}.svg`);
    }catch(error){notify(error.message||"俯视图导出失败",true);}finally{saveMap.disabled=false;}
  });
  if(gt.available){
    summary.append(facts([["GT 路径长度",metres(gt.length)],["GT 坐标点",number(gt.point_count)],["GT 动作数",number(gt.action_count)],["稀疏参考长度",metres(record.path_length)],["标注测地距离",metres(record.geodesic_distance)],["参考路标数",number(record.point_count)]]));
    summary.append(element("p","hint",aligned?"8 Hz · 非采集时间":"按坐标点演示"));
    const details=element("details","gt-data");details.append(element("summary","","查看完整 GT 坐标与原始动作"));
    details.addEventListener("toggle",()=>{if(!details.open||details.dataset.loaded)return;details.dataset.loaded="true";
      details.append(element("p","source",`${gt.source_path}\nGT 键：${gt.source_key} · 与标注起点偏差：${gt.start_error==null?"未核验":gt.start_error.toFixed(4)+" m"}`));
      const scroll=element("div","gt-coordinate-table"),table=element("table"),head=element("thead"),headRow=element("tr");
      for(const name of ["点","X / m","Y / m","Z / m","累计 / m"])headRow.append(element("th","",name));head.append(headRow);table.append(head);
      const body=element("tbody");points.forEach((point,index)=>{const row=element("tr");for(const value of [index,...point.map(n=>n.toFixed(4)),cumulative[index].toFixed(3)])row.append(element("td","",value));body.append(row);});table.append(body);scroll.append(table);details.append(scroll);
      const actions=element("div","gt-actions");gt.actions.forEach((action,index)=>{const span=element("span","chip",`${index+1}: ${gt.action_names[String(action)]||action}`);actions.append(span);});details.append(actions);
    });box.append(details);
  }
  summary.append(downloads);
  draw();update();
  return box;
}

function stopPlayback(){if(state.gtTimer){clearInterval(state.gtTimer);state.gtTimer=null;}document.querySelectorAll(".gt-play").forEach(button=>button.textContent="播放 GT");document.querySelectorAll(".gt-video video").forEach(video=>video.pause());}
function stopVideoPolling(){clearTimeout(state.videoPoll);state.videoPoll=null;}
function mediaUrl(id){return apiUrl("media",{id}).href;}
function gtVideoUrl(id,kind){return apiUrl("gt-video-media",{id,kind}).href;}
function renderGTVideo(record){
  const box=section("LeRobot GT 视频");box.classList.add("gt-video");
  const content=element("div");box.append(content);
  const controller=state.detailRequest;
  function current(){return controller===state.detailRequest && record.id===state.selected;}
  function display(status){
    if(!current())return;
    content.replaceChildren();
    if(status.state==="unavailable"){
      content.append(element("p","hint",status.reason||"暂无对应的 LeRobot RGB 数据。"));return;
    }
    if(status.state==="ready"){
      content.append(element("p","hint",`470×280 · 8 Hz（每秒 8 帧）· ${status.frames} 帧 · ${(status.frames/8).toFixed(3)} 秒`));
      const video=element("video");video.controls=true;video.playsInline=true;video.preload="metadata";
      video.width=470;video.height=280;video.poster=gtVideoUrl(record.id,"poster");video.src=gtVideoUrl(record.id,"video");video.setAttribute("aria-label","LeRobot GT 标注视频");
      const controls=element("div","frame-controls");
      const play=element("button","","播放");play.setAttribute("aria-label","播放 GT 视频");
      play.onclick=()=>{if(video.paused){if(video.ended)video.currentTime=0;video.play().catch(()=>content.append(element("p","hint","视频暂时无法播放，请重试。")));}else video.pause();};
      video.addEventListener("play",()=>{play.textContent="暂停";play.setAttribute("aria-label","暂停 GT 视频");});
      video.addEventListener("pause",()=>{play.textContent="播放";play.setAttribute("aria-label","播放 GT 视频");});
      const previous=element("button","","上一帧"),next=element("button","","下一帧");
      const slider=element("input");slider.type="range";slider.min="0";slider.max=String(status.frames-1);slider.value="0";slider.setAttribute("aria-label","GT 视频帧");
      const count=element("span");
      function update(){slider.value=String(Math.min(status.frames-1,Math.floor(video.currentTime*8+1e-5)));count.textContent=`${Number(slider.value)+1} / ${status.frames}`;}
      function seek(frame){video.pause();video.currentTime=(Math.max(0,Math.min(status.frames-1,frame))+.1)/8;update();}
      previous.onclick=()=>seek(Number(slider.value)-1);next.onclick=()=>seek(Number(slider.value)+1);slider.oninput=()=>seek(Number(slider.value));
      video.addEventListener("timeupdate",update);video.addEventListener("seeked",update);
      video.addEventListener("error",()=>content.append(element("p","hint","视频暂时无法读取，请重新选择该样本。")));
      controls.append(play,previous,next,slider,count);content.append(video,controls,element("p","hint","蓝色：未来 10 步 GT · 绿色：apos · 粉色：opos。仅绘制当前可见且有效的标注。"));
      const downloads=element("div","map-downloads");
      for(const [kind,label] of [["video","下载 GT 视频"],["report","下载逐帧标注"]]){const link=element("a","button-link",label);link.href=gtVideoUrl(record.id,kind);link.download=`episode_${record.episode_id}${kind==="video"?".mp4":".json"}`;downloads.append(link);}
      content.append(downloads);update();return;
    }
    if(status.state==="error"){
      content.append(element("p","hint","该视频生成失败，请重试。"));const retry=element("button","","重新生成");retry.onclick=()=>request(true);content.append(retry);return;
    }
    const labels={pending:"正在准备 GT 视频…",rendering:"正在从 LeRobot 观测生成 GT 视频…",queued:"GT 视频正在排队生成…",busy:"视频队列繁忙，稍后自动重试…"};
    content.append(element("p","hint",labels[status.state]||labels.pending),element("p","hint","正在后台生成 470×280、8 Hz 视频，完成后自动显示；已生成的视频会直接读取。"));
    state.videoPoll=setTimeout(()=>request(),1500);
  }
  async function request(retry=false){
    if(!current())return;
    try{const response=await fetch(apiUrl("gt-video",{id:record.id,retry:retry?"1":""}),{method:"POST",signal:controller.signal});if(!response.ok)throw Error("GT 视频请求失败");display(await response.json());}
    catch(error){if(error.name!=="AbortError"&&current())display({state:"error"});}
  }
  const status=record.gt_video||{state:"unavailable"};
  if(status.state==="pending"){display(status);stopVideoPolling();request();}else display(status);
  return box;
}

function exportPayload(filters, reviews) { return { filters: { ...filters }, reviews: structuredClone(reviews) }; }

function exportManifest() {
  if (state.exportBusy) return;
  state.exportBusy = true;
  const button = $("#export"); button.disabled = true;
  try {
    // Native downloads stream directly to disk; a complete RxR manifest need
    // not be accumulated as JavaScript strings and a second in-memory Blob.
    const target = `vln-export-${Date.now()}`;
    const frame = element("iframe"); frame.name = target; frame.hidden = true;
    frame.title = "样本清单下载";
    frame.addEventListener("load", () => {
      const body = frame.contentDocument?.body?.textContent;
      if (body?.trim()) {
        try { const result = JSON.parse(body); if (result.error) notify(result.error, true); }
        catch (_) { notify("清单下载失败，请重试。", true); }
      }
    });
    const form = element("form"); form.method = "POST"; form.action = apiUrl("export").href;
    form.target = target; form.hidden = true;
    const input = element("input"); input.type = "hidden"; input.name = "payload";
    input.value = JSON.stringify(exportPayload(state.params, {})); form.append(input);
    document.body.append(frame, form); form.submit(); form.remove();
    setTimeout(() => frame.remove(), 3600000);
    notify(`已开始下载 ${number(state.total)} 条样本的清单，包含当前筛选条件。可在浏览器下载列表查看进度。`);
  } catch (error) { notify(error.message, true); }
  finally { state.exportBusy = false; button.disabled = !state.total; }
}

function renderVideoProgress(progress){
  const rows=progress?.coverage||[];$("#video-coverage").hidden=!rows.length;if(!rows.length)return;
  const missing=rows.reduce((sum,row)=>sum+row.missing_observations,0);
  $("#video-progress").textContent=`视频整理：已生成 ${number(progress.ready)} / ${number(progress.records)} · 待生成 ${number(progress.pending)} · 缺少观测 ${number(missing)}${progress.failed?` · 失败 ${number(progress.failed)}`:""}`;
  const table=element("table"),head=element("tr");for(const label of ["数据集 / 划分 / 角色","全部指令","已有观测","已生成视频","缺少观测"])head.append(element("th","",label));const thead=element("thead");thead.append(head);table.append(thead);const body=element("tbody");
  for(const row of rows){const tr=element("tr");for(const value of [`${row.dataset} / ${row.split} / ${row.role}`,row.total,row.observations,row.ready,row.missing_observations])tr.append(element("td","",typeof value==="number"?number(value):value));body.append(tr);}table.append(body);$("#video-coverage-table").replaceChildren(table,element("p","hint","已有观测的轨迹正在后台批量生成；缺少观测的条目保留指令和已公开 GT，暂不提供视频。"));
}
async function initialize(){
  try{
    state.metadata=await api("catalog");$("#catalog-date").textContent=`更新于 ${new Date(state.metadata.generated_at).toLocaleString("zh-CN")}`;
    populateFacets(state.metadata.facets);renderDirectory(state.metadata.tree);renderVideoProgress(state.metadata.gt_videos);
    const query=new URL(window.location.href).searchParams;state.params={};
    for(const key of ["dataset","source_id","split","q","scene","role","language","media","quality","min_length","max_length"]){if(query.has(key))state.params[key]=query.get(key);}
    for(const [key,value]of Object.entries(state.params))$("#filters").elements[key].value=value;
    await loadList();
    setInterval(async()=>{if(document.hidden)return;try{renderVideoProgress(await api("video-progress"));}catch(_){}},10000);
  }
  catch(error){$("#sample-list").setAttribute("aria-busy","false");$("#sample-list").replaceChildren(empty("目录连接失败",error.message));notify(error.message,true,initialize);}
}

document.addEventListener("DOMContentLoaded",()=>{
  let searchTimer;const form=$("#filters");form.addEventListener("submit",event=>{event.preventDefault();clearTimeout(searchTimer);readFilters();});
  form.addEventListener("input",event=>{if(event.target.matches("input")){clearTimeout(searchTimer);searchTimer=setTimeout(readFilters,250);}});
  form.addEventListener("change",event=>{if(event.target.matches("select")){clearTimeout(searchTimer);if(["dataset","source_id","split"].includes(event.target.name))form.elements.scene.value="";readFilters();}});
  $("#reset").addEventListener("click",()=>{clearTimeout(searchTimer);HTMLFormElement.prototype.reset.call(form);readFilters();});
  $("#previous").addEventListener("click",()=>{state.offset=Math.max(0,state.offset-state.limit);loadList();});$("#next").addEventListener("click",()=>{state.offset+=state.limit;loadList();});$("#export").addEventListener("click",exportManifest);
  document.addEventListener("visibilitychange",()=>{if(document.hidden)stopPlayback();});initialize();
});
