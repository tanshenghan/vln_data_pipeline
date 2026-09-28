const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state={health:null,runs:[],sources:[],offset:0,selected:new Set(),run:null};
const statusName={queued:'等待执行',waiting:'等待空闲',running:'运行中',succeeded:'已完成',failed:'失败',paused:'已暂停',cancelled:'已取消'};
const badge=s=>`<span class="status ${esc(s)}">${statusName[s]||esc(s)}</span>`;
const date=s=>new Date(s).toLocaleString('zh-CN',{hour12:false});
function notice(s){$('#pipeline-notice').textContent=s;$('#pipeline-notice').hidden=false;clearTimeout(notice.timer);notice.timer=setTimeout(()=>$('#pipeline-notice').hidden=true,7000)}
async function api(path,options={}){const r=await fetch('/api'+path,{...options,headers:{'Content-Type':'application/json',...options.headers}});const data=await r.json();if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||data));return data}
function download(data,name){const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function runTable(rows){return rows.length?`<table><thead><tr><th>任务</th><th>管线</th><th>状态 / 阶段</th><th>创建时间</th><th></th></tr></thead><tbody>${rows.map(r=>`<tr><td><a href="#run/${r.id}">${esc(r.name)}</a><small>${esc(r.id.slice(0,8))}</small></td><td>${r.kind==='gt'?'GT 采集':r.config.operation==='preview'?'Prompt 预览':'Seedance'}${r.kind==='gt'?`<small>${esc(runComputeName(r))}</small>`:''}</td><td>${badge(r.status)}<small>${esc(r.stage)}</small></td><td>${date(r.created_at)}</td><td><a href="#run/${r.id}">详情 →</a></td></tr>`).join('')}</tbody></table>`:'<div class="empty">暂无任务 · 从 GT 采集或 Seedance 页面开始</div>'}
async function refresh(){try{const [h,r,c,plans]=await Promise.all([api('/health'),api('/runs'),api('/compute-pools').catch(()=>({pools:[],ready:false})),api('/collection-plans')]);state.health=h;state.runs=r;state.compute=c;state.plans=plans;renderCollectionPlans();updateComputePools();$('#service').textContent=h.ready?'● 工作站采集就绪':'○ 工作站准备中';$('#gt-ready').textContent=h.ready?'采集链路已就绪，点击即可在工作站运行。':'尚未就绪：'+Object.entries(h.checks).filter(([k,v])=>!v).map(([k])=>k).join('、');updateGtReadiness();updateSceneCaptureButtons();$('#checks').innerHTML=Object.entries(h.checks).map(([k,v])=>`<span class="check ${v?'':'no'}">${v?'✓':'○'} ${esc(k)}</span>`).join('');$('#gpu').textContent=`GPU：${h.gpu} · 存储可用 ${h.free_gb} GB`;$('#stats').innerHTML=[['生产任务',r.length],['正在运行',r.filter(x=>x.status==='running').length],['已完成',r.filter(x=>x.status==='succeeded').length],['需要检查',r.filter(x=>x.status==='failed').length],['生成估算费用（非账单）',Number(h.generation_cost?.estimated_cny||0).toFixed(4)+' 元']].map(([k,v])=>`<div class="stat"><span>${k}</span><strong>${v}</strong></div>`).join('');$('#recent').innerHTML=runTable(r.slice(0,5));$('#run-table').innerHTML=runTable(r);if(location.hash.startsWith('#run/'))await loadRun(location.hash.slice(5),true)}catch(e){$('#service').textContent='连接暂不可用';console.error(e)}}
async function loadSources(){const p=new URLSearchParams({q:$('#search').value,dataset:$('#filter-dataset').value,split:$('#filter-split').value,limit:30,offset:state.offset});try{const data=await api('/sources?'+p);state.sources=data.items;state.selected.clear();$('#source-count').textContent=`${data.stats.records.toLocaleString()} 条指令 · ${data.stats.scenes||0} 个场景`;$('#source-table').innerHTML=`<table><thead><tr><th></th><th>来源</th><th>指令</th><th>场景 / 语言</th></tr></thead><tbody>${data.items.map(r=>`<tr><td><input type="checkbox" data-source="${esc(r.id)}" aria-label="选择 episode ${esc(r.episode_id)}"></td><td>${esc(r.dataset)} / ${esc(r.episode_id)}<small>${esc(r.split)} · ${esc(r.role)}${r.dataset==='SRDF'?' · 离散参考路线，非逐帧 GT':''}</small></td><td class="instruction">${esc(r.instruction)}</td><td>${esc(r.scene)}<small>${esc(r.language)}${r.dataset==='ScaleVLN'?(r.scene_ready?' · 场景已就绪':' · 场景未下载'):''}</small></td></tr>`).join('')}</tbody></table>`;$('#page-num').textContent=`第 ${state.offset/30+1} 页`;$('#source-prev').disabled=state.offset===0;$('#source-next').disabled=state.offset+30>=data.stats.records;$$('[data-source]').forEach(e=>e.addEventListener('change',()=>e.checked?state.selected.add(e.dataset.source):state.selected.delete(e.dataset.source)))}catch(e){notice(e.message)}}
function selectedTo(kind){const rows=state.sources.filter(r=>state.selected.has(r.id));if(!rows.length)return notice('请先选择样本');if(new Set(rows.map(r=>[r.dataset,r.split,r.role].join(':'))).size>1)return notice('请选取同一数据集、split 和角色的样本');if(kind==='seedance'&&['VLNVerse','ScaleVLN','SRDF'].includes(rows[0].dataset))return notice('该数据集当前仅支持 GT 采集');if(rows[0].split==='test'&&rows[0].dataset!=='VLNVerse')return notice('test 未公开 GT，不能发起专家回放采集');const f=$('#'+(kind==='gt'?'gt':'seedance')+'-form');f.elements.source_dataset.value=rows[0].dataset.toLowerCase();if(f.elements.split)f.elements.split.value=rows[0].split;if(f.elements.role)f.elements.role.value=rows[0].role;f.elements.episode_ids.value=rows.map(r=>r.episode_id).join(',');f.elements.count.value=rows.length;delete f.dataset.previewId;f.dataset.sourceIds=JSON.stringify(rows.map(r=>r.id));f.dataset.sourceSplit=rows[0].split;f.dataset.sourceRole=rows[0].role;location.hash=kind;if(kind==='gt'){updateGtOptions();updateGtReadiness()}}
function formPayload(f,kind){const d=new FormData(f),target=kind==='gt'?(selectedDevices()[0]||'local'):'local';return {kind,execution:target==='local'?'local':'cloud_idle',compute_pool:target==='local'?'auto':target,name:d.get('name')||'',dataset:d.get('source_dataset'),split:d.get('split')||f.dataset.sourceSplit||'train',role:d.get('role')||f.dataset.sourceRole||'guide',languages:[d.get('language')||'all'],count:Number(d.get('count')||1),episode_ids:String(d.get('episode_ids')||'').split(/[,，\s]+/).filter(Boolean),source_ids:JSON.parse(f.dataset.sourceIds||'[]'),full_dataset:d.has('full_dataset'),routes_per_shard:Number(d.get('routes_per_shard')||8),workers:Number(d.get('workers')||1),lightnav_max_steps:Number(d.get('lightnav_max_steps')||500),duration:Number(d.get('duration')||15),instruction:d.get('instruction')||'',operation:'generate',preview_id:kind==='seedance'?(f.dataset.previewId||null):null}}
function collectionRequest(form,kind){
 const payload=formPayload(form,kind),devices=selectedDevices();
 if(payload.dataset==='srdf')throw new Error(state.health?.srdf?.collection_reason||'SRDF 连续轨迹尚未就绪，当前仅支持源数据浏览。');
 const continuous=kind==='gt'&&form.elements.collection_mode.value==='continuous';
 if(continuous&&!['r2r','rxr','scalevln'].includes(payload.dataset))throw new Error('该数据集暂不支持持续采集');
 if(kind==='gt'&&!continuous&&devices.length>1)throw new Error('自定义数量请只选择一台设备；多设备请使用持续采集');
 if(continuous){
  const execution=devices.includes('local')?(devices.length>1?'mixed':'local'):'cloud_idle';
  const pools=devices.flatMap(id=>id==='volc-h20'?devicePools(id).filter(p=>p.enabled).map(p=>p.id):[id]);
  return {continuous:true,path:'/collection-plans',payload:{...payload,execution,count:1,full_dataset:false,compute_pool:'auto',compute_pools:pools}};
 }
 return {continuous:false,path:'/runs',payload};
}
for(const kind of ['gt','seedance'])$('#'+kind+'-form').addEventListener('submit',async e=>{
 e.preventDefault();const f=e.currentTarget,b=f.querySelector('[type=submit]');b.disabled=true;
 try{
  const request=collectionRequest(f,kind),encoded=JSON.stringify(request.payload);
  if(f.dataset.lastPayload!==encoded){f.dataset.requestKey=Array.from(crypto.getRandomValues(new Uint8Array(16)),b=>b.toString(16).padStart(2,'0')).join('');f.dataset.lastPayload=encoded;}
  const r=await api(request.path,{method:'POST',headers:{'Idempotency-Key':f.dataset.requestKey},body:encoded});
  notice(request.continuous?'持续采集已启用，所选设备空闲后自动领取批次':request.payload.execution==='cloud_idle'?'任务已保存，云端资源空闲后自动采集':'任务已提交，工作站将自动执行');
  location.hash=request.continuous?'gt':'run/'+r.id;await refresh();
  if(request.continuous)$('#collection-plans').scrollIntoView({behavior:'smooth',block:'start'});
 }catch(e){notice(e.message)}finally{if(kind==='gt')updateGtReadiness();else b.disabled=false;}
});
function renderCollectionPlans(){
 const names={waiting:'等待空闲 / 就绪',running:'持续采集中',watching:'本轮采完，等待新增',paused:'已暂停',cancelled:'已停止',needs_attention:'需要检查'};
 $('#collection-plan-list').innerHTML=(state.plans||[]).map(p=>`<article class="collection-plan"><div class="run-head"><div><h4>${esc(p.name)}</h4><p>${esc([...new Set(p.compute_pools.map(computeName))].join(' + '))} · ${esc(names[p.status]||p.status)}</p></div><div class="run-actions">${p.control==='run'?'<button data-plan-action="pause" data-plan="'+esc(p.id)+'">暂停计划</button>':''}${p.control!=='cancel'&&(p.control==='pause'||p.status==='needs_attention')?'<button data-plan-action="retry" data-plan="'+esc(p.id)+'">继续 / 重试</button>':''}${p.control!=='cancel'?'<button data-plan-action="cancel" data-plan="'+esc(p.id)+'">停止计划</button>':''}</div></div><p class="muted">${esc(p.config.dataset.toUpperCase())} / ${esc(p.config.split)} / ${esc(p.config.role)} · ${esc(p.config.languages.join(', '))} · 范围 ${p.progress.total??'—'} 条 · 已完成 ${p.progress.completed??0} 条 · 待领取 ${p.progress.remaining??'—'} 条</p>${p.error?'<p class="error">'+esc(p.error)+'</p>':''}<p class="muted">${[...new Set(p.compute_pools.map(computeDevice))].map(id=>esc(computeName(id)+'：'+computeStatus(id,p.config.dataset,p.compute_pools))).join(' · ')}</p>${p.batches?.length?'<details><summary>最近采集批次</summary>'+p.batches.map(b=>'<p><a href="#run/'+esc(b.id)+'">'+esc(computeName(b.compute_pool))+' · '+esc(b.id.slice(0,8))+'</a> '+badge(b.status)+'</p>').join('')+'</details>':''}</article>`).join('')||'<p class="muted">采集数量选择“持续采集”，再选择 GPU0 或云端设备即可启用。</p>';
}
$('#collection-plan-list').addEventListener('click',async e=>{
 const b=e.target.closest('[data-plan-action]');if(!b)return;b.disabled=true;
 try{await api('/collection-plans/'+b.dataset.plan+'/'+b.dataset.planAction,{method:'POST'});notice(b.dataset.planAction==='pause'?'已暂停计划，正在释放采集资源':b.dataset.planAction==='cancel'?'已停止计划，已有产物保留':'已恢复持续采集');await refresh();}catch(e){notice(e.message);b.disabled=false;}
});
function generationPanel(r){
 if(r.kind!=='seedance')return '';
 const report=r.generation||{},usage=report.usage||{};
 if(r.config.operation==='preview')return `<article style="margin:20px 0"><h3>Prompt 预览</h3><p class="muted">这是生成前的提示词快照。尚未提交视频生成。</p>${(report.preview||[]).map(p=>`<div class="contract-row"><b>${esc(p.instruction)}</b><pre class="prompt-text">${esc(p.prompt)}</pre></div>`).join('')||'<p class="muted">正在准备提示词，请等待任务完成。</p>'}${r.status==='succeeded'?'<button class="primary" id="use-preview">使用此 Prompt 配置生成</button>':''}<p class="muted">本次预览估算费用：${usage.estimated_cost_cny==null?'暂无估算':Number(usage.estimated_cost_cny).toFixed(4)+' 元'}；生成任务将沿用提示词。</p></article>`;
 const quality={accepted:'通过',review:'待复核',rejected_pre_qc:'基础质量未通过',rejected_vlm:'语义检查未通过',rejected_trajectory:'轨迹检查未通过'};
 const cost=usage.estimated_cost_cny==null?'暂无估算':Number(usage.estimated_cost_cny).toFixed(4)+' 元';
 const images=r.artifacts.filter(a=>a.kind==='image'&&['trajectory_plot.png','contact_sheet.jpg'].includes(a.name)&&a.details.relative_path?.startsWith('samples/'));
 return `<article style="margin:20px 0"><h3>生成质量与用量</h3><p class="muted">生成流程的执行状态与样本质量分别记录。质量未通过的原始产物仍保留。</p><div class="form-note"><b>估算费用：${esc(cost)}</b><p>按已配置单价估算；实际账单以服务方为准。文本 tokens：${esc(usage.total_tokens??'—')} · 视频 tokens：${esc(usage.video_provider_total_tokens??'—')}</p></div>${(report.samples||[]).map(s=>`<div class="contract-row"><b>${esc(s.sample_id)} · ${esc(quality[s.status]||s.status)}</b><p>VGGT 估计轨迹，非 GT；尺度使用参考路线先验。</p><details><summary>查看质量检查与缺失说明</summary><pre class="log">${esc(JSON.stringify({quality:s.status,findings:s.errors,trajectory:s.trajectory_report,semantic:s.vlm_report},null,2))}</pre></details><details><summary>查看 Prompt 编译报告</summary><pre class="log">${esc(JSON.stringify(s.prompt_report,null,2))}</pre></details></div>`).join('')}<div class="artifact-list">${images.map(a=>`<div class="artifact"><p>${a.name==='trajectory_plot.png'?'估计轨迹（独立坐标，非 GT）':'生成视频关键帧'}</p><a href="/api/artifacts/${a.id}" target="_blank"><img loading="lazy" src="/api/artifacts/${a.id}" alt="${esc(a.name)}" style="width:100%;margin-top:12px"></a></div>`).join('')}</div></article>`;
}
function trajectoryReviewPanel(r){
 const report=r.trajectory_review;
 if(!report)return r.config.dataset==='vlnverse'?'<article id="trajectory-review"><h3>VLM 自动审核结果</h3><p>尚未产生审核结果。采集结束后会显示结论、审核模型和证据。</p></article>':'';
 const capture=r.artifacts.find(a=>a.kind==='capture');
 const verdict=value=>value===true?'是':value===false?'否':'无法确定';
 return `<article id="trajectory-review"><h3>VLM 自动审核结果</h3><p>允许训练导出 ${report.accepted} 条 · 候选保留 ${report.candidates} 条</p>${report.episodes.map(e=>{
   const decision={pass:'通过',reject:'未通过',uncertain:'不确定'}[e.decision]||'尚无结论';
   const gtPassed=e.accepted&&r.status==='succeeded'&&r.stages.some(stage=>stage.name==='validate_lerobot'&&stage.complete);
   const status={pass:'succeeded',reject:'failed',uncertain:'queued'}[e.decision]||'queued';
   const confidence=typeof e.confidence==='number'&&Number.isFinite(e.confidence)&&e.confidence>=0&&e.confidence<=1?`${Math.round(e.confidence*100)}%`:'未提供';
   const frames=(e.evidence_frame_ids||[]).filter(Number.isInteger);
   const evidence=capture?frames.slice(0,8).map(frame=>{
     const file=`${e.accepted?'episodes':'candidates'}/${e.episode_id}/rgb/${String(frame).padStart(6,'0')}.png`;
     const url='/api/artifacts/'+encodeURIComponent(capture.id)+'?'+new URLSearchParams({file});
     return `<a class="review-frame" href="${esc(url)}" target="_blank" rel="noopener"><img loading="lazy" src="${esc(url)}" alt="审核证据：第 ${frame} 帧"><span>第 ${frame} 帧 · 点击查看原图</span></a>`;
   }).join(''):'';
   const file=r.artifacts.find(a=>a.name==='vlm_review.json'&&a.details.episode_id===e.episode_id);
   return `<section class="contract-row review-result"><div class="review-title"><h4>${esc(e.episode_id)}</h4><span class="status ${status}">${gtPassed?'GT 通过':'审核结论：'+decision}</span></div>
     <dl class="review-facts"><div><dt>审核模型</dt><dd>${e.model==='doubao-seed-2-0-pro-260215'?'豆包 Seed 2.0 Pro · ':''}${esc(e.model||report.model||'未提供')}</dd></div><div><dt>模型自报置信度</dt><dd>${confidence}</dd></div><div><dt>路线符合指令</dt><dd>${verdict(e.route_correct)}</dd></div><div><dt>到达指令目标</dt><dd>${verdict(e.destination_reached)}</dd></div></dl>
     <p><b>处理结果：</b>${gtPassed?'自动审核与 GT 数据校验均已通过':e.accepted?'自动审核通过，等待 GT 数据校验':e.decision==='pass'&&!e.explicit_terminal?'保留为候选：语义审核通过，但缺少明确 STOP':'保留为候选，不进入正式训练导出'}。</p>
     <p>${e.trajectory_source==='public_reference'?'公开参考路线回放':'LightNav 推理轨迹'} · ${e.explicit_terminal?'明确 STOP':'未产生明确 STOP'} · 抽样 ${e.frame_ids?.length??'—'} 帧</p>
     <p><b>审核说明（模型原文）</b><br>${esc(e.reason||'未提供')}</p>
     <p><b>证据帧：</b>${frames.length?esc(frames.join('、'))+'（从 0 编号）':'模型未提供证据帧'}</p>
     ${evidence?`<div class="review-evidence">${evidence}</div>${frames.length>8?'<p class="muted">上方展示前 8 张证据图；完整帧号见上方列表。</p>':''}`:''}
     <details data-review-key="${esc(e.episode_id)}"><summary>原始审核报告、限制与用量</summary><pre class="log">${esc(JSON.stringify(e,null,2))}</pre></details>
     ${file?`<a href="/api/artifacts/${encodeURIComponent(file.id)}?download=true">下载此条审核报告</a>`:''}</section>`;
 }).join('')}</article>`;
}
async function loadRun(id,quiet=false){try{const r=await api('/runs/'+id);state.run=r;const content=$('#run-content');const active=document.activeElement;const editing=quiet&&active&&content.contains(active)&&['INPUT','TEXTAREA','SELECT'].includes(active.tagName);if(editing||(quiet&&[...content.querySelectorAll('video')].some(v=>!v.paused)))return;const expandedReviews=new Set([...content.querySelectorAll('details[data-review-key][open]')].map(d=>d.dataset.reviewKey));const oldVideos=new Map([...content.querySelectorAll('video')].map(v=>[v.src,{time:v.currentTime,paused:v.paused}]));const videos=r.artifacts.filter(a=>a.kind==='video'&&(r.kind!=='seedance'||a.name==='video.mp4'));const files=r.artifacts.filter(a=>a.kind!=='video');content.innerHTML=`<div class="run-head"><div><h2>${esc(r.name)}</h2>${badge(r.status)} <span class="muted">${date(r.created_at)} · ${esc(r.stage)}</span></div><div class="run-actions">${['running','queued','waiting'].includes(r.status)?`<button data-action="pause">暂停</button><button data-action="cancel">${r.kind==='seedance'?'停止本地执行':'取消'}</button>`:''}${['failed','paused'].includes(r.status)?'<button data-action="retry" class="primary">续跑 / 重试</button>':''}<a class="button" href="/api/runs/${r.id}/manifest">下载清单</a></div></div>${r.config.execution==='cloud_idle'?`<p class="form-note">采集方式：${esc(runComputeName(r))} · ${r.cloud?.job_id?'云任务 '+esc(r.cloud.job_id):'等待空闲资源'}<br>自动让位后会续跑；手动暂停后保持暂停。</p>`:r.kind==='gt'?`<p class="muted">采集方式：${esc(runComputeName(r))}</p>`:''}${r.error?`<p class="error">${esc(r.error)}</p>`:''}${r.control==='pause'&&r.status==='running'?`<p>${r.config.execution==='cloud_idle'?'正在停止云任务并释放 GPU，已完成分片保留。':r.config.full_dataset?'暂停已请求，停止派发新分片，等待已运行分片完成。':'暂停已请求，将在当前阶段完成后生效。'}</p>`:''}<article><h3>执行阶段</h3>${r.stages.map(s=>`<span class="stage ${s.complete?'done':''}">${s.complete?'✓':'○'} ${esc(s.name)} <small>×${s.attempts}</small></span>`).join('')||`<p class="muted">${r.kind==='gt'?'等待 '+esc(runComputeName(r))+' 执行':'等待工作站执行'}</p>`}<p class="muted">${esc(JSON.stringify(r.progress))}</p></article><details ${r.status==='running'||r.status==='failed'?'open':''}><summary>查看实际执行日志</summary><pre class="log" id="run-log">加载中…</pre></details>${r.kind==='seedance'?'<div class="form-note"><b>估计轨迹，非 GT</b><p>深度、可信标定、真实机体位姿和四类监督：未提供。保留生成原始产物。</p></div>':''}${r.kind==='seedance'?`<p class="muted">${esc(r.generation?.cancellation||'停止本地执行不会取消已提交的云端生成。')}</p><p class="muted">供应商任务：${(r.generation?.provider_tasks||[]).map(t=>esc(t.task_id||'提交结果待核对')).join('、')||'尚未提交'}</p>`:''}${generationPanel(r)}${trajectoryReviewPanel(r)}<article><h3>产物与校验报告</h3><div class="artifact-list">${videos.map(a=>`<div class="artifact"><a href="/api/artifacts/${a.id}" target="_blank">${esc(a.name)}</a><video controls preload="metadata" src="/api/artifacts/${a.id}"></video></div>`).join('')}</div>${files.length?`<details><summary>查看 ${files.length} 个数据 / 报告文件</summary>${files.map(a=>`<p><a href="/api/artifacts/${a.id}${['dataset','capture'].includes(a.kind)?'?archive=tar':'?download=true'}" target="_blank">${esc(a.name)}${['dataset','capture'].includes(a.kind)?'（打包下载）':''}</a> <span class="muted">${esc(a.kind)}</span></p>`).join('')}</details>`:'<p class="muted">产物将在实际完成并登记后出现。</p>'}</article><article style="margin-top:20px">${r.config.dataset==='vlnverse'?'<h3>数据发布</h3><p class="muted">GT 通过后可直接发布，无需人工审核。</p>':`<h3>人工审核与发布</h3><p class="muted">这里保存人工保留、排除或待复核的决定。VLM 自动结论见上方审核结果。</p><form id="review-form" class="review-form"><select name="decision"><option value="keep">保留</option><option value="review">待复核</option><option value="exclude">排除</option></select><input name="note" placeholder="审核备注"><button>保存人工审核</button></form>${r.reviews.map(x=>`<p class="muted">${date(x.created_at)} · ${esc(x.decision)} · ${esc(x.note)}</p>`).join('')}`}${r.release?`<p>已发布版本：${esc(r.release.id)}</p>`:`<button id="publish" ${r.status!=='succeeded'||(r.trajectory_review&&r.trajectory_review.accepted===0)?'disabled':''}>发布数据版本</button>`}</article>`;
content.querySelectorAll('details[data-review-key]').forEach(d=>{d.open=expandedReviews.has(d.dataset.reviewKey)});
if($('#use-preview'))$('#use-preview').onclick=()=>usePreview(r);
content.querySelectorAll('video').forEach(v=>{const old=oldVideos.get(v.src);if(old){v.currentTime=old.time;if(!old.paused)v.play().catch(()=>{})}});
$$('[data-action]').forEach(b=>b.onclick=async()=>{try{const o=await api('/runs/'+id+'/'+b.dataset.action,{method:'POST'});notice(o.note);await loadRun(id)}catch(e){notice(e.message)}});if($('#review-form'))$('#review-form').onsubmit=async e=>{e.preventDefault();const d=new FormData(e.currentTarget);try{await api('/reviews/'+id,{method:'POST',body:JSON.stringify(Object.fromEntries(d))});notice('审核已保存');await loadRun(id)}catch(e){notice(e.message)}};if($('#publish'))$('#publish').onclick=async()=>{try{await api('/releases/'+id,{method:'POST'});notice('数据版本已发布');await loadRun(id)}catch(e){notice(e.message)}};const logs=await api('/runs/'+id+'/logs');if($('#run-log'))$('#run-log').textContent=logs.text||'暂无执行输出。'}catch(e){if(!quiet)notice(e.message)}}
function usePreview(r){
 const payload=r.config.request_payload,f=$('#seedance-form');
 for(const [name,key] of [['name','name'],['source_dataset','dataset'],['count','count'],['duration','duration'],['instruction','instruction']])f.elements[name].value=payload[key]??'';
 f.elements.episode_ids.value=(payload.episode_ids||[]).join(',');
 f.dataset.sourceSplit=payload.split;f.dataset.sourceRole=payload.role;
 f.dataset.sourceIds=JSON.stringify(payload.source_ids||[]);
 f.dataset.previewId=r.id;
 location.hash='seedance';notice('已载入预览的配置，点击开始生成即可使用这份 Prompt。');
}
$('#preview-prompts').onclick=async()=>{
 const f=$('#seedance-form'),button=$('#preview-prompts');button.disabled=true;
 try{
  const payload={...formPayload(f,'seedance'),operation:'preview',preview_id:null};
  const encoded=JSON.stringify(payload);
  if(f.dataset.previewPayload!==encoded){f.dataset.previewKey=Array.from(crypto.getRandomValues(new Uint8Array(16)),b=>b.toString(16).padStart(2,'0')).join('');f.dataset.previewPayload=encoded}
  const r=await api('/runs',{method:'POST',headers:{'Idempotency-Key':f.dataset.previewKey},body:encoded});
  location.hash='run/'+r.id;await refresh();
 }catch(e){notice(e.message)}finally{button.disabled=false}
};
const titles={objectnav:'ObjectNav 数据浏览',scenes:'场景库',overview:'生产总览',data:'数据中心',gt:'GT 轨迹采集',seedance:'Seedance 视频生成',runs:'任务与审核',atlas:'场景与 GT 浏览','run-detail':'任务详情'};
async function route(){
  const hash=location.hash.slice(1)||'atlas';
  const page=hash.startsWith('run/')?'run-detail':hash.startsWith('scenes/')?'scenes':(titles[hash]?hash:'atlas');
  $$('.page').forEach(e=>e.hidden=e.id!==page);
  $('#pipeline-main').hidden=page==='atlas';
  $('#catalog-date').hidden=page!=='atlas';
  $('#export').hidden=page!=='atlas';
  $$('.platform-nav a').forEach(a=>{const selected=a.dataset.page===page||(page==='run-detail'&&a.dataset.page==='runs');a.classList.toggle('active',selected);if(selected)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current')});
  $('#page-title').textContent=titles[page];
  if(page!=='atlas')$('#atlas').querySelectorAll('video').forEach(v=>v.pause());
  if(page!=='run-detail')$('#run-content').querySelectorAll('video').forEach(v=>v.pause());
  if(page==='data')await loadSources();
  if(page==='scenes'){await loadScenes();if(hash.startsWith('scenes/'))await showScene(decodeURIComponent(hash.slice(7)))}
  document.querySelector('footer').hidden=page!=='atlas';
  if(page==='run-detail')await loadRun(hash.slice(4));
}
// Source selection keeps the exact immutable catalog ID and original instruction.
window.addEventListener('vln:record-selected',e=>{
  const record=e.detail,header=$('#detail .detail-header');
  if(!header)return;
  const actions=document.createElement('div');actions.className='source-pipeline-actions';
  for(const [kind,label] of [['gt','采集此样本 GT'],['seedance','用此样本生成视频']]){
    const button=document.createElement('button');button.type='button';button.textContent=label;
    if(record.dataset==='SRDF'){
      button.disabled=true;button.title='SRDF 当前仅有离散参考路线；连续采集尚未接通。';
    }
    button.addEventListener('click',()=>{state.sources=[record];state.selected=new Set([record.id]);selectedTo(kind)});
    actions.append(button);
  }
  header.append(actions);
  api('/sources/'+encodeURIComponent(record.id)+'/derived').then(rows=>{
    if(!header.isConnected||!rows.length)return;
    const box=document.createElement('section');box.className='detail-section';
    box.innerHTML='<h3>由此样本产生的数据</h3>'+rows.map(r=>`<p><a class="button-link" href="#run/${esc(r.run_id)}">${r.origin==='gt'?'GT 采集':'Seedance · 非 GT'} · ${esc(r.run_name)} · ${esc(r.quality)}</a></p>`).join('');
    $('#detail').append(box);
  }).catch(e=>notice('派生产物查询暂不可用：'+e.message));
});
for(const kind of ['gt','seedance']){
  const form=$('#'+kind+'-form');
  for(const name of ['episode_ids','source_dataset','split','role','language','full_dataset']){
    form.elements[name]?.addEventListener('change',()=>{delete form.dataset.sourceIds;delete form.dataset.sourceSplit;delete form.dataset.sourceRole});
  }
}
$('#search-button').onclick=()=>{state.offset=0;loadSources()};$('#search').onkeydown=e=>{if(e.key==='Enter')$('#search-button').click()};$('#source-prev').onclick=()=>{state.offset=Math.max(0,state.offset-30);loadSources()};$('#source-next').onclick=()=>{state.offset+=30;loadSources()};$('#source-gt').onclick=()=>selectedTo('gt');$('#source-video').onclick=()=>selectedTo('seedance');$('#export-sources').onclick=()=>download(state.sources,'vln-source-manifest.json');$('#refresh-runs').onclick=refresh;window.addEventListener('hashchange',route);route();refresh();setInterval(refresh,5000);

$('#check-capacity').onclick=async()=>{
 const button=$('#check-capacity');button.disabled=true;$('#gt-capacity').textContent='正在检查源数据与所需空间…';
 try{const request=collectionRequest($('#gt-form'),'gt');const p=await api(request.continuous?'/collection-plans/preflight':'/preflight',{method:'POST',body:JSON.stringify(request.payload)});
 $('#gt-capacity').textContent=request.continuous?`${p.episodes} 条源轨迹 · 已完成 ${p.completed} 条 · 待领取 ${p.remaining} 条 · 每批最多 ${p.batch_size} 条。${p.basis}`:`${p.episodes} 个 episode · ${p.frames} 帧 · ${p.scenes} 个场景 · 保守峰值预算 ${p.estimated_peak_gb} GiB / 可用 ${p.free_gb} GiB。${p.basis}`;
 }catch(e){$('#gt-capacity').textContent=e.message}finally{button.disabled=false;}
};

let sceneRows=[];let sceneSync=null;
function renderScenes(){
 const q=$('#scene-search').value.trim().toLowerCase();
 const rows=sceneRows.filter(r=>(r.id+' '+r.room_types.join(' ')).toLowerCase().includes(q));
 $('#scene-count').textContent=`${rows.length} / ${sceneSync?.total_scenes||sceneRows.length} 个场景`+(sceneSync?` · 已校验 ${sceneSync.completed_scenes} 个 · ${(sceneSync.verified_bytes/1024**3).toFixed(1)} / ${sceneSync.total_bytes?(sceneSync.total_bytes/1024**3).toFixed(1):'统计中'} GiB · ${sceneSync.status==='complete'?'同步完成':'后台同步中'}`:'');
 $('#scene-list').innerHTML=rows.map(r=>`<button type="button" class="sample" data-scene-id="${esc(r.id)}" aria-current="${$('#scene-detail').dataset.sceneId===r.id}" ${r.local===false?'disabled':''}><b>${esc(r.id)}</b><p>${r.local===false?'等待资源下载':r.room_count+' 个房间'} · ${r.instructions} 条指令</p><p class="muted">${r.public_gt?`${r.public_gt} 条公开参考路线`:'无公开参考路线'}</p></button>`).join('')||'<div class="empty">未找到匹配场景</div>';
 $('#scene-list').querySelectorAll('[data-scene-id]').forEach(b=>b.onclick=()=>showScene(b.dataset.sceneId));
}
async function loadScenes(){try{const data=await api('/scenes');sceneRows=data.items;sceneSync=data.sync;renderScenes();if(sceneRows.some(r=>r.local!==false)&&!$('#scene-detail').dataset.sceneId)await showScene(sceneRows.find(r=>r.local!==false).id);const current=sceneRows.find(r=>r.id===$('#scene-detail').dataset.sceneId);if(current?.sync_status==='complete'&&$('#scene-detail').dataset.assetReady!=='true')await showScene(current.id)}catch(e){notice(e.message)}}
async function showScene(id){
 $('#scene-detail').dataset.sceneId=id;
 $('#scene-list').querySelectorAll('[data-scene-id]').forEach(b=>b.setAttribute('aria-current',String(b.dataset.sceneId===id)));
 try{const r=await api('/scenes/'+encodeURIComponent(id));if($('#scene-detail').dataset.sceneId!==id)return;
 $('#scene-detail').innerHTML=`<div class="detail-header"><h2>${esc(r.id)}</h2><p>VLNVerse / Kujiale · ${esc(r.format)} · ${r.up_axis?esc(r.up_axis)+' 轴向上':'坐标信息在渲染时校验'}${r.meters_per_unit?' · '+r.meters_per_unit+' 米 / 单位':''}</p></div><div class="detail-section"><h3>场景预览</h3>${r.media.thumbnail?`<img src="${esc(r.media.thumbnail)}" alt="${esc(r.id)} 原始场景缩略图" style="max-width:100%;width:320px">`:'<p>未提供缩略图</p>'}</div><div class="detail-section"><h3>源占据图</h3>${r.media.occupancy?`<img src="${esc(r.media.occupancy)}" alt="${esc(r.id)} 源占据图" style="max-width:100%;image-rendering:pixelated"><p class="muted">场景提供的空间占据图，不代表已采集的 GT 轨迹。</p>`:'<p>未提供占据图</p>'}</div><div class="detail-section"><h3>房间信息</h3><p>${esc(r.room_types.join(' / '))}</p><p>${r.room_count} 个房间</p><a href="${esc(r.source)}" target="_blank" rel="noopener">原始场景来源</a></div><div id="scene-episodes" class="detail-section"></div>`;
 $('#scene-detail').dataset.assetReady=String(r.capture_assets_ready);
 await loadSceneEpisodes(id,0);
 }catch(e){notice(e.message)}
}
$('#scene-search').addEventListener('input',renderScenes);

function selectedDevices(){return $$('#compute-options input:checked').map(e=>e.value);}
function computeDevice(id){return id.startsWith('volc-h20-')?'volc-h20':id;}
function devicePools(id){return (state.compute?.pools||[]).filter(p=>id==='volc-h20'?computeDevice(p.id)===id:p.id===id);}
function computeStatus(id,dataset,poolIds){
 if(id==='local')return state.health?.ready?'已就绪，空闲时自动采集':'等待本机环境就绪，不影响其他设备';
 const pools=devicePools(id).filter(p=>!poolIds||poolIds.includes(p.id)),enabled=pools.filter(p=>p.enabled),supported=enabled.filter(p=>p.datasets.includes(dataset)),ready=supported.filter(p=>p.ready);
 if(!pools.length)return '尚未接入';
 if(!enabled.length)return '未启用';
 if(!supported.length)return '等待环境验证，就绪后自动加入';
 if(ready.length)return ready.every(p=>p.pending_foreground)?'有其他任务排队，自动等待':'已就绪，空闲时自动采集';
 return [...new Set(supported.map(p=>p.reason||'等待采集环境就绪'))].join('；');
}
$('#gt-form').elements.collection_mode.addEventListener('change',()=>{
 updateGtReadiness();$('#gt-capacity').textContent='请重新检查所选采集范围。';
});
function updateComputePools(){
 const previous=new Set(selectedDevices().map(computeDevice)),scale=['scalevln','srdf'].includes($('#gt-form').elements.source_dataset.value);
 const targets=['local','baidu-pro6000','baidu-b200'];
 for(const p of state.compute?.pools||[]){const id=computeDevice(p.id);if(p.enabled&&!targets.includes(id))targets.push(id);}
 $('#compute-options').innerHTML=targets.map(id=>`<label><input type="checkbox" name="compute_device" value="${esc(id)}" ${previous.has(id)?'checked':''} ${scale&&id!=='local'?'disabled':''}>${esc(computeName(id))}</label>`).join('');
}
$('#compute-options').addEventListener('change',e=>{
 if(!e.target.matches('input'))return;
 if(!selectedDevices().length)$('#compute-options input[value="local"]').checked=true;
 updateGtReadiness();$('#gt-capacity').textContent='请重新检查所选采集范围。';
});
function computeName(id){return {local:'GPU0','baidu-pro6000':'PRO6000','baidu-b200':'B200','volc-h20':'H20',auto:'自动分配'}[computeDevice(id)]||(state.compute?.pools||[]).find(p=>p.id===id)?.name||id;}
function runComputeName(run){
 if(run.config.execution!=='cloud_idle')return computeName('local');
 const requested=run.config.compute_pool,assigned=(state.compute?.pools||[]).find(p=>p.name===run.progress?.pool)?.id;
 return computeName(run.cloud?.pool_id||(requested&&requested!=='auto'?requested:assigned)||'auto');
}


function updateGtReadiness(){
 const h=state.health;if(!h)return;
 const f=$('#gt-form'),dataset=f.elements.source_dataset.value,vlnverse=dataset==='vlnverse',scalevln=dataset==='scalevln';
 $('#check-capacity').disabled=dataset==='srdf';
 if(dataset==='srdf'){
  f.elements.collection_mode.value='custom';
  f.elements.collection_mode.querySelector('[value="continuous"]').disabled=true;
  f.elements.full_dataset.checked=false;f.elements.full_dataset.disabled=true;
  f.elements.count.disabled=false;f.elements.workers.disabled=false;
  $('#gt-custom-count').hidden=false;$('#gt-full-field').hidden=true;$('#cloud-policy').hidden=true;
  $('#start-gt').disabled=true;$('#start-gt').textContent='连续轨迹尚未就绪';
  const s=h.srdf||{};
  $('#gt-ready').textContent=(s.source_ready?`SRDF 公开 round-3：${s.total_episodes} 条指令，${s.ready_scenes}/${s.total_scenes} 个共享场景已就绪。`:'SRDF 源数据尚未安装。')+(s.collection_reason||'当前仅支持源数据浏览。');
  return;
 }
 const devices=selectedDevices(),cloud=devices.some(id=>id!=='local');
 f.elements.collection_mode.querySelector('[value="continuous"]').disabled=vlnverse;
 if(vlnverse)f.elements.collection_mode.value='custom';
 const continuous=f.elements.collection_mode.value==='continuous';
 // The two visible modes are bounded custom count and a durable continuous plan.
 // Keep the legacy full-split API, but never let its old checkbox override count.
 f.elements.full_dataset.checked=false;f.elements.full_dataset.disabled=true;
 f.elements.count.disabled=continuous;f.elements.workers.disabled=continuous;
 $('#gt-custom-count').hidden=continuous;$('#gt-full-field').hidden=true;
 $('#cloud-policy').hidden=!continuous;
 $('#start-gt').textContent=continuous?'启用持续采集':'开始采集 ↗';
 if(cloud){
  const pools=devices.filter(id=>id!=='local').map(devicePools);
  const enabled=continuous?!scalevln&&pools.every(group=>group.some(p=>p.enabled)):devices.length===1&&pools[0]?.some(p=>p.enabled&&p.ready&&p.datasets.includes(dataset));
  $('#start-gt').disabled=!enabled;
  $('#gt-ready').textContent=devices.map(id=>computeName(id)+'：'+computeStatus(id,dataset)).join('；');
  if(!continuous&&devices.length>1)$('#gt-ready').textContent='自定义数量请只选择一台设备；多设备请切换持续采集。';
  return;
 }
 const ready=h.ready&&(!vlnverse||h.vlnverse_ready)&&(!scalevln||h.scalevln?.can_collect);
 $('#start-gt').disabled=!ready;
 $('#gt-ready').textContent=ready?'GPU0 已就绪，点击即可开始采集。':vlnverse&&!h.vlnverse_ready?'VLNVerse 渲染环境正在部署与实测，完成后可直接采集。':'GPU0 采集环境尚未就绪。';
 if(scalevln){const s=h.scalevln||{};$('#gt-ready').textContent=(ready?`ScaleVLN：当前 ${s.ready_scenes}/${s.total_scenes} 个场景，覆盖 ${s.ready_episodes}/${s.total_episodes} 条路线。${s.full_scene_coverage?'源场景已全量就绪，可按需采集。':'留空 ID 时仅选择场景已就绪的路线；不是全量场景。'}`:'ScaleVLN 源数据、场景或工作站环境尚未就绪。')+' 当前仅支持 GPU0；云端 HM3D 与源数据尚未部署验收，PRO6000 / B200 暂不可选。';}
}
function updateGtOptions(){
 const f=$('#gt-form'),scale=['scalevln','srdf'].includes(f.elements.source_dataset.value);
 for(const [name,value] of [['split','train'],['role','guide'],['language','all']]){
  for(const option of f.elements[name].options)option.disabled=scale&&option.value!==value;
  if(scale)f.elements[name].value=value;
 }
 for(const input of $$('#compute-options input')){input.disabled=scale&&input.value!=='local';if(scale)input.checked=input.value==='local';}
 if(scale){f.elements.full_dataset.checked=false;f.elements.full_dataset.disabled=true;}
}
$('#gt-form').elements.source_dataset.addEventListener('change',e=>{
 const f=$('#gt-form');const v=e.target.value==='vlnverse';
 f.elements.role.value=v?'coarse':'guide';
 f.elements.split.value=v?'val_seen':'train';
 f.elements.language.value='all';f.elements.full_dataset.checked=false;f.elements.full_dataset.disabled=v;
 updateGtOptions();updateGtReadiness();
});
async function loadSceneEpisodes(id,offset){
 const result=await api('/sources?'+new URLSearchParams({dataset:'VLNVerse',scene:id,limit:10,offset}));
 if($('#scene-detail').dataset.sceneId!==id)return;
 const box=$('#scene-episodes');
 box.innerHTML=`<h3>导航指令 · ${result.stats.records} 条</h3><p class="muted">有公开路线则回放采集；无公开路线则运行 LightNav 推理。两者均由 VLM 审核，未通过的轨迹保留待复核。</p>${result.items.map(r=>`<div class="sample"><b>${esc(r.episode_id)}</b><p class="muted">${esc(r.split)} · ${esc(r.role)} · ${r.has_reference_gt?'有公开参考路线':'未提供参考路线'}</p><p>${esc(r.instruction)}</p><button type="button" data-capture-source="${esc(r.id)}" data-public-gt="${r.has_reference_gt}">${r.has_reference_gt?'开始采集 · 公开路线':'开始采集 · LightNav'}</button></div>`).join('')}<div class="pager"><button id="scene-prev" ${offset===0?'disabled':''}>上一页</button><span>${offset+1}–${Math.min(offset+10,result.stats.records)} / ${result.stats.records}</span><button id="scene-next" ${offset+10>=result.stats.records?'disabled':''}>下一页</button></div>`;
 box.querySelectorAll('[data-capture-source]').forEach(b=>b.onclick=()=>{
   if(!state.health?.ready||!state.health?.vlnverse_ready)return notice('VLNVerse 工作站采集环境尚未就绪');
   const record=result.items.find(r=>r.id===b.dataset.captureSource);
   state.sources=[record];state.selected=new Set([record.id]);
   $('#gt-form').elements.full_dataset.checked=false;
   selectedTo('gt');$('#gt-form').requestSubmit();
 });
 updateSceneCaptureButtons();
 $('#scene-prev').onclick=()=>loadSceneEpisodes(id,Math.max(0,offset-10));
 $('#scene-next').onclick=()=>loadSceneEpisodes(id,offset+10);
}

setInterval(()=>{if(location.hash.startsWith('#scenes'))loadScenes()},10000);

function updateSceneCaptureButtons(){
 const assets=$('#scene-detail')?.dataset.assetReady==='true';
 document.querySelectorAll('[data-capture-source]').forEach(b=>{
   const gt=b.dataset.publicGt==='true';
   const ready=state.health?.ready&&state.health?.vlnverse_ready;
   b.disabled=!assets||!ready||(!gt&&!state.health?.lightnav_ready);
   b.textContent=!assets?'场景资源同步中':!ready?'采集环境准备中':!gt&&!state.health?.lightnav_ready?'LightNav 未就绪':gt?'开始采集 · 公开路线':'开始采集 · LightNav';
 });
}
