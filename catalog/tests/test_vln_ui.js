"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");
const context = vm.createContext({document:{addEventListener(){}},window:{location:new URL("https://host.example/proxy/8788/")},URL,console,structuredClone});
vm.runInContext(fs.readFileSync(path.join(__dirname,"../vln_web/app.js"),"utf8"),context);

test("API and indexed media URLs preserve proxy subpaths and encode queries",()=>{
  const url=context.apiUrl("episodes",{q:"తెలుగు & hello",offset:0});
  assert.equal(url.pathname,"/proxy/8788/vln-api/episodes");
  assert.equal(url.searchParams.get("q"),"తెలుగు & hello");
  assert.equal(new URL(context.mediaUrl("asset-id")).pathname,"/proxy/8788/vln-api/media");
});

test("route projection uses Habitat X/Z and native Matterport X/Y without stretching",()=>{
  const habitat=context.projectRoute([[0,0,0],[3,0,4]],[0,2]);
  const native=context.projectRoute([[0,0,0],[3,4,0]],[0,1]);
  assert.equal(JSON.stringify(habitat),JSON.stringify(native));
  const dx=habitat[1][0]-habitat[0][0],dy=habitat[0][1]-habitat[1][1];
  assert.ok(Math.abs(dx/dy-3/4)<1e-10);
  assert.deepEqual(JSON.parse(JSON.stringify(context.projectRoute([[1,1,1]]))),[[310,150]]);
});

test("export requests snapshot filters and local decisions for a streamed download",()=>{
  const reviews={one:{decision:"keep",note:"clear instruction"}},filters={dataset:"R2R"};
  const result=context.exportPayload(filters,reviews);
  reviews.one.decision="exclude";filters.dataset="RxR";
  assert.equal(result.reviews.one.decision,"keep");
  assert.equal(result.reviews.one.note,"clear instruction");
  assert.equal(result.filters.dataset,"R2R");
});

test("NavMesh projection keeps X right, Z down and the real metric scale",()=>{
  const floor={origin:[-10,-20],meters_per_pixel:.04};
  assert.deepEqual(Array.from(context.worldToMap([-9,3,-18],floor)),[25,50]);
  assert.deepEqual(Array.from(context.worldToMap([-10,99,-20],floor)),[0,0]);
});

test("height slicing never joins paths across a different storey",()=>{
  const floors=[{height:0},{height:3}];
  const points=[[0,0,0],[1,.1,1],[2,3,2],[3,3,3],[4,0,4],[5,0,5]];
  assert.equal(context.nearestFloor(2.9,floors),1);
  assert.deepEqual(JSON.parse(JSON.stringify(context.floorSegments(points,floors,0))),[points.slice(0,2),points.slice(4)]);
  assert.deepEqual(JSON.parse(JSON.stringify(context.floorSegments(points,floors,1))),[points.slice(2,4)]);
  assert.deepEqual(JSON.parse(JSON.stringify(context.floorSegments(points,[], -1))),[points]);
});

test("map fitting includes padding, clips to the map and handles empty layers",()=>{
  assert.deepEqual(Array.from(context.boundsForPoints([[20,30],[60,80]],[0,0,100,100],10)),[10,20,60,70]);
  assert.deepEqual(Array.from(context.boundsForPoints([], [0,0,100,100],10)),[0,0,100,100]);
  assert.deepEqual(Array.from(context.boundsForPoints([[0,0]], [0,0,100,100],10)),[0,0,10,10]);
});
