from typing import Literal
from pydantic import BaseModel, Field, model_validator


class RunRequest(BaseModel):
    kind: Literal["gt", "seedance"] = "gt"
    name: str = Field(default="", max_length=120)
    dataset: Literal["r2r", "rxr"] = "r2r"
    split: Literal["train", "val_seen", "val_unseen"] = "train"
    role: Literal["guide", "follower"] = "guide"
    languages: list[Literal["all", "en-US", "en-IN", "hi-IN", "te-IN"]] = ["all"]
    count: int = Field(default=1, ge=1, le=1000)
    episode_ids: list[str] = Field(default_factory=list, max_length=1000)
    source_ids: list[str] = Field(default_factory=list, max_length=1000)
    full_dataset: bool = False
    routes_per_shard: int = Field(default=8, ge=1, le=64)
    workers: int = Field(default=1, ge=1, le=2)
    duration: int = Field(default=15, ge=4, le=15)
    instruction: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def check(self):
        if self.dataset == "r2r" and (self.role != "guide" or self.languages != ["all"]):
            raise ValueError("R2R 使用 guide 和 all；语言筛选仅用于 RxR")
        if not self.languages or ("all" in self.languages and len(self.languages) != 1):
            raise ValueError("语言必须为 all 或明确语言列表")
        if self.full_dataset and (self.episode_ids or self.source_ids):
            raise ValueError("全量任务不能同时指定 episode")
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("episode ID 不得重复")
        return self


class ReviewRequest(BaseModel):
    decision: Literal["keep", "exclude", "review"]
    note: str = Field(default="", max_length=4000)
