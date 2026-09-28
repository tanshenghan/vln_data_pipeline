"""Workstation paths are configuration, never copied upstream machine paths."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLN_", env_file=ROOT / ".env", extra="ignore")
    database_url: str = "postgresql+psycopg:///vln_pipeline"
    redis_url: str = "redis://127.0.0.1:6379/5"
    resource_root: Path = Path.home() / ".local/share/vln-pipeline"
    # Inputs may be shared independently of the configured output resource root.
    # Databases, queues and runtime state remain deployment-specific.
    source_root: Path | None = None
    shared_run_permissions: bool = False
    rvq_bundle_root: Path | None = None
    runtime_root: Path = ROOT / "runtime"
    habitat_python: Path = Path.home() / ".local/share/vln-pipeline/habitat/bin/python"
    lerobot_python: Path = ROOT / "data_pipeline/.venv/lerobot/bin/python"
    navgen_python: Path = Path.home() / ".cache/navgen/venv/bin/python"
    vggt_repo: Path = Path.home() / ".cache/navgen/vggt"
    navgen_env: Path = ROOT.parent / ".env"
    atlas_url: str = "http://127.0.0.1:8788"
    gpu_id: int = 0
    cloud_pools_file: Path = ROOT / "configs/cloud_pools.local.json"
    cloud_credentials_file: Path = Path.home() / ".config/vln-pipeline/cloud.env"
    cloud_poll_seconds: int = 15
    # Set only by the cloud container entry point, never by an API request.
    cloud_worker: bool = False
    minimum_free_gb: float = 5
    vlnverse_scene_root: Path | None = None
    vlnverse_managed_scene_root: Path | None = None
    lightnav_endpoint: str = "ws://127.0.0.1:8051"
    lightnav_model_manifest: Path | None = None
    lightnav_model_provenance: str = "LightNav-0 workstation serving deployment; checkpoint identity configured by operator"
    isaac_python: Path = ROOT / "data_pipeline/.venv/isaac/bin/python"

    @property
    def source_resource_root(self):
        return self.source_root if self.source_root is not None else self.resource_root

    @property
    def scenes_root(self):
        return self.source_resource_root / "scene_datasets"

    @property
    def rvq_bundle(self):
        return self.rvq_bundle_root if self.rvq_bundle_root is not None else self.source_resource_root / "action_tokenizer"

    @property
    def vlnverse_hdr(self):
        return self.source_resource_root / "render_assets/limpopo_golf_course_4k.hdr"

    @property
    def vlnverse_scene_roots(self):
        return [self.vlnverse_managed_scene_root, self.vlnverse_scene_root, self.scenes_root / "vlnverse"]

    @property
    def runs_root(self):
        return self.resource_root / "runs"

    @property
    def catalog_db(self):
        return self.runtime_root / "vln.sqlite"

    def dataset_root(self, dataset):
        return self.source_resource_root / "datasets" / {"r2r": "R2R_VLNCE_v1-3_preprocessed", "rxr": "RxR_VLNCE_v0", "scalevln": "ScaleVLN_StreamVLN", "srdf": "SRDF", "vlnverse": "VLNVerse/raw_data/final_splits"}[dataset]


settings = Settings()
