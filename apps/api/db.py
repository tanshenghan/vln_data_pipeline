from contextlib import contextmanager
from datetime import datetime, timezone
from sqlalchemy import create_engine, String, Text, JSON, Integer, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from .settings import settings


def now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    request_key: Mapped[str] = mapped_column(String, unique=True)
    kind: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    config: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String, default="queued")
    stage: Mapped[str] = mapped_column(String, default="queued")
    control: Mapped[str] = mapped_column(String, default="run")
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    progress: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    heartbeat: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Artifact(Base):
    __tablename__ = "artifacts"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    kind: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    path: Mapped[str] = mapped_column(Text)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class CloudExecution(Base):
    """One durable cloud binding per run; checkpoints never move between pools."""
    __tablename__ = "cloud_executions"
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    pool_id: Mapped[str] = mapped_column(String)
    directory: Mapped[str] = mapped_column(Text)
    pool_digest: Mapped[str] = mapped_column(String)
    pool_spec: Mapped[dict] = mapped_column(JSON)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    job_id: Mapped[str | None] = mapped_column(String, nullable=True)
    job_name: Mapped[str | None] = mapped_column(String, nullable=True)
    state: Mapped[str] = mapped_column(String, default="waiting")
    stop_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    not_before: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failures: Mapped[int] = mapped_column(Integer, default=0)


class DataSample(Base):
    """Searchable produced samples; upstream source episodes remain read-only."""
    __tablename__ = "data_samples"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    origin: Mapped[str] = mapped_column(String, index=True)
    dataset: Mapped[str] = mapped_column(String, index=True)
    split: Mapped[str] = mapped_column(String, index=True)
    role: Mapped[str] = mapped_column(String)
    language: Mapped[str] = mapped_column(String, index=True)
    scene: Mapped[str] = mapped_column(String, index=True)
    episode_id: Mapped[str] = mapped_column(String, index=True)
    instruction: Mapped[str] = mapped_column(Text)
    source_record_id: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    quality: Mapped[str] = mapped_column(String, index=True)
    details: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class CollectionPlan(Base):
    """Persistent demand for fresh source episodes, independent of browser sessions."""
    __tablename__ = "collection_plans"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    request_key: Mapped[str] = mapped_column(String, unique=True)
    name: Mapped[str] = mapped_column(String)
    config: Mapped[dict] = mapped_column(JSON)
    compute_pools: Mapped[list] = mapped_column(JSON)
    control: Mapped[str] = mapped_column(String, default="run")
    status: Mapped[str] = mapped_column(String, default="waiting")
    progress: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class CollectionBatch(Base):
    __tablename__ = "collection_batches"
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    plan_id: Mapped[str] = mapped_column(ForeignKey("collection_plans.id"), index=True)
    paused_by_plan: Mapped[int] = mapped_column(Integer, default=0)


class CollectionClaim(Base):
    """One source can be assigned to only one continuous batch, even after restart."""
    __tablename__ = "collection_claims"
    source_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    subject_id: Mapped[str] = mapped_column(String, index=True)
    decision: Mapped[str] = mapped_column(String)
    note: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Release(Base):
    __tablename__ = "releases"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), unique=True)
    name: Mapped[str] = mapped_column(String)
    manifest: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


engine = create_engine(settings.database_url, pool_pre_ping=True)
Session = sessionmaker(engine, expire_on_commit=False)


@contextmanager
def session():
    with Session() as db:
        with db.begin():
            yield db


def serialize(row):
    return {c.name: (v.isoformat() if isinstance(v := getattr(row, c.name), datetime) else v)
            for c in row.__table__.columns if c.name != "path"}


def update_run(run_id, **changes):
    with session() as db:
        row = db.get(Run, run_id)
        for key, value in changes.items():
            setattr(row, key, value)
        row.updated_at = now()
