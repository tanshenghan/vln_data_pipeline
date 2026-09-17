Dataset semantic explorer
=========================

This read-only utility scans native task names beneath a downloaded dataset root,
normalizes them into a six-level semantic catalog, and serves an interactive tree
plus a two-dimensional task-term embedding view. It does not inspect media
payloads, extract archives, move files, or delete data.

Hierarchy
---------

``data domain -> scene -> dataset -> task type -> primary atomic action -> native task``

The left hierarchy directory enables all six display levels by default. Clearing any
checkbox rebuilds the tree immediately from the remaining ordered levels, so the
underlying task-path records stay unchanged while categories are merged into a new
view. The current hierarchy is also embedded in every exported manifest row.
``native_group`` remains available as provenance and for conversion/video path
matching, but it is no longer a display hierarchy level.

Native labels and name-based heuristic labels retain separate provenance fields in
``catalog.json``. Dataset layouts that expose only chunks, tables, scenes, or UUID
episodes are listed under ``unresolved`` instead of being presented as tasks.
Known English native scene names are normalized into Chinese catalog labels:
``bedroom`` becomes ``卧室``, ``living_room`` becomes ``客厅``, ``kitchen``
joins ``厨房与餐饮``, and ``study`` becomes ``书房``. The unmodified source
value remains in ``native_scene`` for provenance and path matching.

Document-defined data domains
-----------------------------

The data-domain filter follows the research document and exposes five peer
domains: ``real`` (真机), ``sim`` (仿真), ``umi`` (UMI 数据集), ``ego`` (Ego),
and ``vqa`` (VQA 数据集). UMI is kept separate from other real-robot data so
that its capture-system lineage can be filtered independently.

The UMI document catalog contains:

* HiFi-UMI-2K
* FastUMI-100K
* FastUMI Dataset (the server adapter name is ``FastUMI-Data``)
* UMI Original Dataset

The VQA catalog cross-references the document's three taxonomy headings with
the P2 processing ledger:

* ``EmbodiedVQA``: RefSpatial, EO-Data1.5M
* ``2/3D Grounding``: Omni3D, CA-1M
* ``EmbodiedPlanning``: ShareRobot, EgoPlanIT
* ``VQA / Grounding``: LocateAnything

Catalog references are shown in the "数据目录 / 待提取" dialog with downloaded
status and expected server paths. A VQA entry creates a task-tree record only
when an exact allow-listed directory exists and is not a symbolic link. Missing
references therefore remain planning entries and never inflate downloaded task
or dataset counts.

For downloaded VQA roots, the second hierarchy level remains the broad scene
``视觉问答与空间理解``. The document category is stored at the task-type level,
preserving the global ``data domain -> scene`` ordering requested by the UI.

Episode counts
--------------

Every terminal node shows both its task-path record count and an aggregated
episode count. Complete coverage is displayed as an exact value; partial
coverage uses ``>=`` with the number of known paths, and layouts that cannot be
counted from shallow directory names display ``episode 未知``.

The scanner currently counts native episode directories for RoboMINDv2 and
RoboMIND failure tasks, LeRobot episode parquet files for FastUMI-100K,
trajectory CSV files for FreeTacMan, numbered MP4 files for exUMI, and numeric
episode directories for 10Kh-RealOmin. Compressed RoboMIND/FastUMI archives,
EgoDex task archives, and Touch-in-the-Wild zarr/zip payloads remain unknown;
the scanner does not open or extract them merely to manufacture a count.

Server conversion status
------------------------

LeRobot conversion state is generated from structured evidence beneath explicit,
allow-listed output roots. It is independent from the browser's manual-review
state. A manifest link or directory name alone is never considered proof of a
completed conversion, and missing output mounts produce ``unknown`` rather than a
false ``not_converted`` result.

Each catalog record carries ``conversion_status``, ``conversion_output_path``,
``conversion_evidence_path``, ``conversion_source``, ``conversion_note``, and
``conversion_checked_at``. The current status vocabulary is:

* ``converted``: a formal structured completion or validation report covers the
  record.
* ``partial``: validated output exists, but the current artifact cannot be mapped
  one-to-one to every native catalog path.
* ``not_converted``: an exact-task formal output root is known, but this task has
  no matching completed artifact.
* ``excluded``: a formal conversion report explicitly excludes this source class.
* ``unknown``: the output mount or sufficient structured evidence is unavailable.

Adapters currently recognize the formal RoboMIND, RoboMINDv2, FastUMI-Data,
FastUMI-100K, FreeTacMan, Touch-in-the-Wild, exUMI, EgoDex, and
10Kh-RealOmin output locations. Smoke tests, old revisions, intermediate roots,
README prose, and log text are not consulted. A 10Kh fragment snapshot remains
``partial``; the published ``10Kh-RealOmin-OpenData-80D-multiroot-v1`` catalog is
``converted`` only when its aggregate report is complete, valid, published, and
internally consistent across source, exclusion, episode, Parquet, video, and
completion-marker counts.

Task-term embedding
-------------------

The ``语义散点`` page plots one point for every unique ``(scene,
native_task)`` pair. This keeps every point tied to exactly one scene color while
merging duplicate storage paths and datasets that use the same task term in the
same scene. Its ``record_ids`` retain the underlying catalog relationships, so
all global filters and workflow states apply to both pages consistently.

Coordinates come from the Hugging Face
``intfloat/multilingual-e5-small`` neural text encoder. Each native task is
encoded with E5's ``query:`` prefix, attention-mask mean pooling, and L2
normalization; centered PCA then reduces the 384-dimensional vectors to two
dimensions. The scene label is never included in the embedding input, so color
remains an independent annotation rather than leaking into semantic distance. If
one native task occurs in several scenes, the browser applies only a four-pixel
display offset so the colored points can be inspected without altering their
stored coordinates.

The method requires NumPy, PyTorch, and Transformers at catalog-build time. The
two PCA axes have no direct business meaning. The generated payload stores the
resolved model revision, pooling parameters, and source digest. The default is
pinned to Hugging Face commit ``614241f622f53c4eeff9890bdc4f31cfecc418b3``.
The same model revision and task corpus reproduce the same projection up to
deterministic PCA sign normalization, while adding terms can move the global
projection.

Generate a catalog
------------------

Run this on the machine that owns the dataset root::

  python scripts/dataset_semantic_explorer/build_catalog.py \
    --root /mnt/cfs/data/datasets \
    --output cache/dataset_semantic_explorer/catalog.json \
    --conversion-pfs-root /mnt/pfs/data/datasets/lerobotv3/high_priority_80d \
    --conversion-cfs-root /mnt/cfs/data/datasets/lerobotv3/high_priority_80d \
    --embedding-cache-dir /mnt/pfs/wangqiongyan/.cache/huggingface

Hourly automatic synchronization
--------------------------------

The server helper performs a shallow catalog and structured conversion-evidence
scan every hour without rebuilding task embeddings. It publishes only substantive
changes, preserves the previous embedding payload, and refuses automatic removal,
mount-loss updates, completed-status regressions, or an increase in unknown
conversion states. The website reads the atomically replaced ``catalog.json`` on
the next page load or refresh.

Start, inspect, run once, or stop the server scheduler with::

  bash scripts/dataset_semantic_explorer/catalog_sync_service.sh start
  bash scripts/dataset_semantic_explorer/catalog_sync_service.sh status
  bash scripts/dataset_semantic_explorer/catalog_sync_service.sh run-once
  bash scripts/dataset_semantic_explorer/catalog_sync_service.sh stop

The deployment-specific wrapper uses a 3,600-second interval and writes status,
lock, PID, and log files beneath
``/mnt/pfs/wangqiongyan/dataset_semantic_explorer``. The server currently has no
active cron or systemd service manager, so the wrapper uses ``setsid`` and
``nohup`` while retaining idempotent PID and command-line checks.

The generated catalog is intentionally written beneath the ignored ``cache/``
directory in this example. It is small metadata; no dataset content is copied.
The first run downloads ``intfloat/multilingual-e5-small`` from Hugging Face into
the persistent cache; later runs can add ``--embedding-local-files-only`` to
guarantee offline loading. By default the command writes the embedding payload
inside ``catalog.json``. Use ``--skip-embedding`` only on a lightweight
environment without the model stack; the classification tree will continue to
work and the scatter page will explain that no embedding was generated.

Open the explorer
-----------------

Start the localhost-only server::

  python scripts/dataset_semantic_explorer/serve.py \
    --catalog cache/dataset_semantic_explorer/catalog.json \
    --host 127.0.0.1 \
    --port 8765

Then open ``http://127.0.0.1:8765``. For a remote server, keep the service bound to
localhost and forward it through SSH::

  ssh -L 8765:127.0.0.1:8765 baidu_beta

When the video API described below is also running, expose its browser-visible
local address through the explorer configuration and forward both ports::

  python scripts/dataset_semantic_explorer/serve.py \
    --catalog cache/dataset_semantic_explorer/catalog.json \
    --video-api-base-url http://127.0.0.1:8790 \
    --host 127.0.0.1 \
    --port 8765

  ssh -L 8765:127.0.0.1:8765 -L 8790:127.0.0.1:8790 baidu_beta

LeRobot video API
-----------------

Video playback uses a separate read-only index and media service. The index is
bounded to 200 episodes per task record by default and stores one episode row plus
shared camera/template metadata, rather than one row per video. Its discovery
adapters inspect only the known formal conversion layouts beneath the explicit
PFS/CFS roots. Converted status is not treated as proof that an MP4 exists:
``video_status`` distinguishes ``available``, ``linked``, ``partial``, ``absent``,
and ``unmapped``. A conversion record with no local MP4 is reported as
``absent``; pinned FastUMI-100K video symlinks are counted only when their
resolved targets exist inside an allow-listed media root. For 10Kh-RealOmin, the
indexer streams the formal ``valid_sources.jsonl`` manifest, maps its exact MCAP
source path to a catalog record, and opens only the catalog-relevant fragment
roots. This avoids walking all 667k one-episode repositories while preserving a
bounded playback index and never falling back to ambiguous task-name matching.

Build the independent index on the dataset server::

  python scripts/dataset_semantic_explorer/build_video_index.py \
    --catalog /mnt/pfs/wangqiongyan/dataset_semantic_explorer/catalog.json \
    --output /mnt/pfs/wangqiongyan/dataset_semantic_explorer/video-index.json \
    --conversion-pfs-root /mnt/pfs/data/datasets/lerobotv3/high_priority_80d \
    --conversion-cfs-root /mnt/cfs/data/datasets/lerobotv3/high_priority_80d

Create a private HMAC secret once (do not publish or commit it), then start the
localhost-only service. Add the local explorer origin while validating through
SSH; omit ``--public-base-url`` so the browser resolves media against the locally
forwarded API origin::

  umask 077
  openssl rand -out /mnt/pfs/wangqiongyan/dataset_semantic_explorer/video-api.secret -hex 32
  python scripts/dataset_semantic_explorer/video_serve.py \
    --index /mnt/pfs/wangqiongyan/dataset_semantic_explorer/video-index.json \
    --secret-file /mnt/pfs/wangqiongyan/dataset_semantic_explorer/video-api.secret \
    --cors-origin http://127.0.0.1:8765 \
    --media-root /mnt/pfs/data/datasets/lerobotv3/high_priority_80d \
    --media-root /mnt/cfs/data/datasets/lerobotv3/high_priority_80d \
    --media-root /mnt/pfs/data/datasets \
    --media-root /mnt/cfs/data/datasets \
    --transcode-cache /mnt/pfs/wangqiongyan/dataset_semantic_explorer/video-transcode-cache \
    --transcode-workers 1 \
    --transcode-threads 4 \
    --transcode-cache-max-gib 100 \
    --transcode-min-free-gib 20 \
    --host 127.0.0.1 \
    --port 8790

For the public site, restart the same service behind the approved stable HTTPS
media origin, replace CORS with the exact public site origin, and set
``--public-base-url https://MEDIA_HOSTNAME``. The static site's
``video-config.json`` must point to that same HTTPS origin. CORS limits browser
origins but is not user authentication; a public media origin should be treated
as publishing the indexed videos unless an authenticated gateway is added.

The native dataset media root is explicit because some validated FreeTacMan
videos are symbolic links into that tree. After resolving a file (including every
symbolic link), the service requires the target to remain inside one of these
allow-listed roots. Browser clients never submit filesystem paths.

The list endpoint is
``GET /v1/records/{record_id}/videos?offset=0&limit=24``. It returns the record's
``video_status``, indexed/source totals, and items whose ``playback_status`` is
one of ``ready``, ``unverified``, ``transcode_required``, ``unsupported``,
``missing``, or ``blocked``. Ready, unverified, and transcode-required items
receive a short-lived opaque HMAC stream URL. ``GET`` and ``HEAD`` support a
single byte range and return ``206``/``Content-Range`` for browser seeking. CORS
accepts only the exact origins configured on the command line.

``ffprobe`` and ``ffmpeg`` are auto-detected. H.264 is streamed directly. When a
cache is configured, a detected MPEG-4/HEVC source is labelled
``transcode_required`` and is converted only after the user clicks that item.
The original video is never modified: a validated H.264 MP4 is atomically placed
under the bounded cache and reused by later requests. Concurrent requests for
the same source share one conversion, while the global worker limit prevents a
click burst from starting many FFmpeg processes. Without a configured cache,
non-H.264 remains ``unsupported``/HTTP 415. The service must remain behind a
stable HTTPS reverse proxy or tunnel when called by the public HTTPS site; do
not bind the raw process directly to a public interface.

Decisions and manifests
-----------------------

The workflow has three independent dimensions:

* filter decision: keep, exclude, review, or unreviewed
* server conversion status: generated from structured output evidence in the
  catalog and read-only in the browser
* manual review: reviewed or unreviewed, stored in browser local storage

The detail panel presents these dimensions in one compact data-governance card.
``Edit marks`` opens a shared drawer for both the tree and embedding views. The
drawer defaults each dimension to ``unchanged`` so a mixed branch cannot be
overwritten accidentally, reveals the exclusion reason only when ``exclude`` is
selected, and confirms batch changes affecting more than 100 paths. The most
recent in-memory batch operation can be undone from the card. User-facing
``needs review`` and ``unmarked`` labels continue to serialize as ``review`` and
``unreviewed`` respectively.

This allows combinations such as ``keep + converted + unreviewed`` or
``review + partial + reviewed``. All three dimensions have independent filters;
filter decisions and manual review also have branch-level batch controls. Existing
decision data stays under its original local-storage key. The historical
``:processing:v1`` key is retained and reinterpreted only in user-facing text as
manual review, so upgrades do not erase prior browser work. Missing manual review
defaults to unreviewed.

Exported JSONL rows include the stable catalog record id, source path, semantic
fields, conversion evidence, decision, keep flag, exclusion reason,
``manual_review_status``, update timestamps, and ``workflow_schema_version=3``.
The export button uses the same active record filters as the tree and embedding
views, including dataset, scene, domain, task type, workflow states, and search
text. With no active filter it exports the full catalog; otherwise it exports
only the current filter intersection and displays the exact record count before
download. Every JSONL row records ``manifest_export_scope``, the filter snapshot,
and both source and exported record counts. Display-only hierarchy checkboxes and
the currently selected tree node do not silently narrow the export.
The legacy ``processing_status`` fields remain for one compatibility version and
are explicitly labelled ``manual_review_legacy``. Applying a decision or review
state to a branch affects all currently visible task-path records beneath it.

Browser local storage is device-local. Different computers do not share workflow
state automatically; use the exported manifest for handoff until a shared backend
or manifest-import workflow is added.

The exported manifest is a review artifact only. No command in this utility applies
the manifest to the filesystem.

Supported native-name layouts
-----------------------------

The initial scanner supports RoboMIND v1, RoboMINDv2, FastUMI-Data,
FastUMI-100K, FreeTacMan, Touch-in-the-Wild task directories,
10Kh-RealOmin, exUMI, EgoDex, and shallow allow-listed VQA dataset roots.
Technical or metadata-only layouts remain visible in the explorer's
"数据目录 / 待提取" dialog until a dedicated metadata adapter is added.
