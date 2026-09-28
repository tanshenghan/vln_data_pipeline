// Native ObjectNav browsing is intentionally separate from VLN expert replay.
const $ = id => document.getElementById(`objectnav-${id}`);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let sources = [], offset = 0, requestVersion = 0, detailVersion = 0;
const active = () => location.hash === '#objectnav';
async function api(path, params) {
  const response = await fetch(`/api/objectnav/${path}${params ? '?' + new URLSearchParams(params) : ''}`);
  const data = await response.json();
  if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'ObjectNav 数据读取失败');
  return data;
}
function options(id, entries, preferred) {
  $(id).replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
  if (entries.some(([value]) => value === preferred)) $(id).value = preferred;
}
function selected() { return sources.find(source => source.id === $('dataset').value); }
function selectedScene() { return selected()?.splits?.[$('split').value]?.scenes.find(scene => scene.scene === $('scene').value); }
function updateSplits(preferred) {
  const source = selected();
  options('split', Object.entries(source?.splits || {}).map(([name, split]) => [name, `${name} · ${split.episodes.toLocaleString()} 条`]), preferred);
}
function updateScenes(preferred) {
  const scenes = selected()?.splits?.[$('split').value]?.scenes || [];
  options('scene', scenes.map(scene => [scene.scene, `${scene.scene}${scene.geometry_ready ? ' · 场景已就绪' : ' · 场景待下载'}`]), preferred);
}
function updateCategories(preferred = '') {
  options('category', [['', '全部类别'], ...Object.entries(selectedScene()?.categories || {}).map(([category, count]) => [category, `${category} · ${count.toLocaleString()}`])], preferred);
}
async function loadPage() {
  const version = ++requestVersion;
  ++detailVersion;
  $('detail').hidden = true;
  $('prev').disabled = $('next').disabled = true;
  $('table').replaceChildren();
  const source = selected(), scene = selectedScene();
  if (!source?.available || !scene) {
    $('status').textContent = '该数据集尚未完成下载、校验与索引。';
    $('page').textContent = '';
    return;
  }
  $('status').textContent = '正在读取该场景的原始任务…';
  try {
    const page = await api('episodes', {dataset:source.id, split:$('split').value, scene:scene.scene, category:$('category').value, offset, limit:30});
    if (version !== requestVersion) return;
    $('status').textContent = `${source.label} · ${source.semantics} · ${page.geometry_ready ? 'RGB 场景 / navmesh 已就绪' : 'RGB 场景 / navmesh 未就绪'} · ${page.semantic_assets_ready ? '语义资产已就绪' : '语义资产未齐'}。${page.collection_note}`;
    $('table').innerHTML = `<table><thead><tr><th>原始 Episode / 行号</th><th>目标类别</th><th>源测地距离</th><th>原始动作路径</th><th></th></tr></thead><tbody>${page.items.map(row => `<tr><td>${esc(row.source_episode_id)}<small>source_row ${row.source_row}</small></td><td>${esc(row.object_category)}</td><td>${row.geodesic_distance == null ? '未提供' : esc(Number(row.geodesic_distance).toFixed(2)) + ' m'}</td><td>${row.has_native_shortest_paths ? '有 · 未适配为采集 GT' : '未提供'}</td><td><button type="button" data-objectnav-row="${row.source_row}">查看目标与来源</button></td></tr>`).join('')}</tbody></table>`;
    $('page').textContent = `${Math.min(offset + 1, page.total)}–${Math.min(offset + 30, page.total)} / ${page.total.toLocaleString()}`;
    $('prev').disabled = offset === 0;
    $('next').disabled = offset + 30 >= page.total;
    $('table').querySelectorAll('[data-objectnav-row]').forEach(button => button.onclick = () => showDetail(Number(button.dataset.objectnavRow)));
  } catch (error) {
    if (version === requestVersion) $('status').textContent = error.message;
  }
}
async function showDetail(ordinal) {
  const version = ++detailVersion;
  try {
    const row = await api('episode', {dataset:$('dataset').value, split:$('split').value, scene:$('scene').value, source_row:ordinal});
    if (version !== detailVersion) return;
    $('detail').hidden = false;
    $('detail').innerHTML = `<h3>${esc(row.object_category)} · 原始 Episode ${esc(row.source_episode_id)}</h3><p>${esc(row.trajectory_note)}</p><p class="muted">${esc(row.semantic_version)} · ${esc(row.coordinate_frame)}</p><p>起点：${esc(JSON.stringify(row.source_episode.start_position))} · 目标实例 ${row.goal_objects.length} 个 · 候选观测点 ${row.goal_objects.reduce((total, goal) => total + goal.view_point_count, 0)} 个</p><p class="muted">文件 ${esc(row.source_file)} · 行号 ${row.source_row}<br>SHA-256 ${esc(row.source_sha256)}</p><details><summary>原始 episode、目标及观测点预览（每个目标最多 3 点）</summary><pre class="log">${esc(JSON.stringify({episode:row.source_episode, goals_key:row.goals_key, goal_objects:row.goal_objects}, null, 2))}</pre></details><p><button type="button" id="objectnav-export">导出此原始任务及目标预览</button></p>`;
    $('export').onclick = () => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(row, null, 2)], {type:'application/json'}));
      const link = document.createElement('a');
      link.href = url;
      link.download = `${row.dataset}-${row.scene}-${row.source_row}-source-preview.json`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
  } catch (error) { if (version === detailVersion) $('status').textContent = error.message; }
}
async function refresh() {
  if (!active()) return;
  const previous = ['dataset','split','scene','category'].map(id => $(id).value);
  try {
    const result = await api('datasets');
    sources = result.items;
    options('dataset', sources.map(source => [source.id, source.label + (source.available ? '' : ' · 待准备')]), previous[0]);
    updateSplits(previous[1]); updateScenes(previous[2]); updateCategories(previous[3]);
    offset = 0;
    await loadPage();
  } catch (error) { $('status').textContent = error.message; }
}
$('dataset').onchange = () => { updateSplits(); updateScenes(); updateCategories(); offset = 0; loadPage(); };
$('split').onchange = () => { updateScenes(); updateCategories(); offset = 0; loadPage(); };
$('scene').onchange = () => { updateCategories(); offset = 0; loadPage(); };
$('category').onchange = () => { offset = 0; loadPage(); };
$('prev').onclick = () => { offset = Math.max(0, offset - 30); loadPage(); };
$('next').onclick = () => { offset += 30; loadPage(); };
$('refresh').onclick = refresh;
window.addEventListener('hashchange', refresh);
refresh();
