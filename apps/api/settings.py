"""Workstation paths are configuration, never copied upstream machine paths."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLN_", env_file=ROOT / ".env", extra="ignore")
    database_url: str = "postgresql+psycopg:///vln_pipeline"
    redis_url: str = "redis://127.0.0.1:6379/5"
    resource_root: Path = Path.home() / ".local/share/vln-pipeline"
    runtime_root: Path = ROOT / "runtime"
    habitat_python: Path = Path.home() / ".local/share/vln-pipeline/habitat/bin/python"
    lerobot_python: Path = ROOT / "data_pipeline/.venv/lerobot/bin/python"
    navgen_python: Path = Path.home() / ".cache/navgen/venv/bin/python"
    vggt_repo: Path = Path.home() / ".cache/navgen/vggt"
    navgen_env: Path = ROOT.parent / ".env"
    atlas_url: str = "http://127.0.0.1:8788"
    gpu_id: int = 0
    minimum_free_gb: float = 5
    vlnverse_scene_root: Path | None = None
    vlnverse_managed_scene_root: Path | None = None
    lightnav_endpoint: str = "ws://127.0.0.1:8051"
    lightnav_model_manifest: Path | None = None
    lightnav_model_provenance: str = "LightNav-0 workstation serving deployment; checkpoint identity configured by operator"
    isaac_python: Path = ROOT / "data_pipeline/.venv/isaac/bin/python"

    @property
    def vlnverse_scene_roots(self):
        return [self.vlnverse_managed_scene_root, self.vlnverse_scene_root, self.resource_root / "scene_datasets/vlnverse"]

    @property
    def runs_root(self):
        return self.resource_root / "runs"

    @property
    def catalog_db(self):
        return self.runtime_root / "vln.sqlite"

    def dataset_root(self, dataset):
        return self.resource_root / "datasets" / {"r2r": "R2R_VLNCE_v1-3_preprocessed", "rxr": "RxR_VLNCE_v0", "vlnverse": "VLNVerse/raw_data/final_splits"}[dataset]


settings = Settings()
