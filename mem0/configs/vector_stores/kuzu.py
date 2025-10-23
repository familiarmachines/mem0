"""
Configuration schema for the Kùzu vector store provider (Pydantic v2 only).

- Accepts common aliases (collection_name -> table_name, embedding_dims -> embedding_model_dims)
- Validates required keys (db, embedding_model_dims)
- Restricts metric to 'cosine' or 'l2'
- Exposes back-compat "virtual" fields (collection_name, embedding_dims) via computed fields
- Permissive about extra keys

Typical Mem0 config:

config = {
    "vector_store": {
        "provider": "kuzu",
        "config": {
            "db": "/home/cwong/Projects/familiar/db/mem0-test-cwong.kuzu",
            "table_name": "Mem0Memory",         # or use alias: collection_name="mem0"
            "index_name": "mem0_memory_idx",
            "embedding_model_dims": 1536,        # MUST match your embedder (alias: embedding_dims)
            "metric": "cosine"                   # or "l2"
        }
    }
}
"""

from __future__ import annotations

from typing import Any, Optional, Literal, Union, Dict
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ConfigDict,
    AliasChoices,
    computed_field,
)

PROVIDER_NAME: str = "kuzu"


class KuzuConfig(BaseModel):
    # Pydantic v2 config: allow unknown keys so callers can add notes/unused fields
    model_config = ConfigDict(extra="allow")

    # REQUIRED
    db: str = Field(..., description="Path to the Kùzu database (directory/file).")
    embedding_model_dims: int = Field(
        ...,
        gt=0,
        validation_alias=AliasChoices("embedding_model_dims", "embedding_dims"),
        description="Embedding vector dimension (must match your embedder).",
    )

    # OPTIONAL
    table_name: str = Field(
        "Mem0Memory",
        validation_alias=AliasChoices("table_name", "collection_name"),
        description="Kùzu node-table (label) that stores memories.",
    )
    index_name: str = Field(
        "mem0_memory_idx",
        description="Name for the HNSW vector index in Kùzu.",
    )
    metric: Literal["cosine", "l2"] = Field(
        "cosine",
        description="Distance metric for the HNSW index.",
    )

    # Advanced: allow passing a prebuilt kuzu.Connection (usually not needed)
    client: Optional[Any] = Field(
        default=None,
        description="Optional kuzu.Connection to reuse (advanced).",
    )

    # Normalize/guard metric (accepts 'COSINE', 'L2', etc.)
    @field_validator("metric", mode="before")
    @classmethod
    def _normalize_metric_v2(cls, v: Any) -> str:
        if v is None:
            return "cosine"
        s = str(v).lower()
        return s if s in ("cosine", "l2") else "cosine"

    # ---- Back-compat "virtual" fields for readers expecting these names ----
    # Included in dumps via their aliases.
    @computed_field(return_type=str, alias="collection_name")
    @property
    def collection_name(self) -> str:
        return self.table_name

    @computed_field(return_type=int, alias="embedding_dims")
    @property
    def embedding_dims(self) -> int:
        return int(self.embedding_model_dims)


# Many Mem0 loaders look for a symbol named `Config`; export an alias just in case.
Config = KuzuConfig


def build_kwargs(cfg: Union[Dict[str, Any], KuzuConfig]) -> Dict[str, Any]:
    """
    Normalize user config (dict or model) into kwargs for the provider class:
    mem0.vector_stores.kuzu.Kuzu(**kwargs)

    - Resolves aliases (collection_name -> table_name, embedding_dims -> embedding_model_dims)
    - Ensures dims is int > 0 and metric is normalized to lowercase
    """
    model = KuzuConfig(**cfg) if isinstance(cfg, dict) else cfg

    # Return only canonical provider keys
    return {
        "db": model.db,
        "table_name": model.table_name,
        "index_name": model.index_name or "mem0_memory_idx",
        "embedding_model_dims": int(model.embedding_model_dims),
        "metric": model.metric,  # already normalized + validated
        "client": model.client,
    }


__all__ = [
    "PROVIDER_NAME",
    "KuzuConfig",
    "Config",
    "build_kwargs",
]
