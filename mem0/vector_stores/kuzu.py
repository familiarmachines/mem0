"""
Kuzu vector-store provider for Mem0.

Requirements:
  pip install kuzu

Config example:
  config = {
      "vector_store": {
          "provider": "kuzu",
          "config": {
              "db": "/path/to/your/mem0.kuzu",     # path to Kuzu DB directory/file
              "table_name": "Mem0Memory",          # optional (default: Mem0Memory)
              "index_name": "mem0_memory_idx",     # optional
              "embedding_model_dims": 1536,        # REQUIRED: match your embedder output size
              "metric": "cosine"                   # optional: 'cosine' or 'l2' (default: cosine)
          }
      }
  }

Notes:
- Uses Kuzu's native vector index:
    INSTALL vector; LOAD vector;
    CALL CREATE_VECTOR_INDEX('<TABLE>', '<INDEX>', 'embedding', metric := 'cosine'|'l2');
    CALL QUERY_VECTOR_INDEX('<TABLE>', '<INDEX>', $query_vector, $k) RETURN node, distance;
  See: https://kuzudb.github.io/docs/extensions/vector/  (CREATE/QUERY/DROP)  # noqa: E501

- Schema columns kept simple to enable common Mem0 filters:
    id (STRING, PK), user_id, agent_id, app_id, data (STRING),
    embedding (FLOAT[DIM]), labels (LIST[STRING]), meta (STRING JSON blob)

- Returned search items expose .id and .payload["data"], matching Mem0 expectations.

Caveats:
- Ensure embedding dimension matches `embedding_model_dims` exactly; Mem0 often
  embeds once per message and passes vectors=[<list of floats>].
- Kuzu’s Python QueryResult also supports get_as_df/get_as_pl in case you prefer
  dataframes; this implementation sticks to iterable rows to avoid extra deps.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from .base import VectorStoreBase  # type: ignore
except Exception:  # pragma: no cover
    # Fallback minimal base so the file is usable in isolation (e.g., during local testing)
    class VectorStoreBase:  # type: ignore
        pass

logger = logging.getLogger(__name__)

try:
    import kuzu  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Kuzu provider requires the 'kuzu' package. Install with: pip install kuzu"
    ) from e

@dataclass
class VectorHit:
    """Minimal Mem0-compatible search result."""
    id: str
    score: Optional[float]  # similarity or -distance; Mem0 doesn't strictly require it
    payload: Dict[str, Any]

    """
    def __iter__(self):
        # Make this object iterable by yielding itself
        yield self
    """

class Kuzu(VectorStoreBase):
    """
    Mem0 VectorStore provider backed by Kuzu.

    Config keys (Python):
        db: str                           # required, path to Kuzu database
        table_name: str                   # optional, default "Mem0Memory"
        index_name: str                   # optional, default "mem0_memory_idx"
        embedding_model_dims: int         # required (e.g., 1536 for text-embedding-3-large)
        metric: str                       # optional, 'cosine' (default) or 'l2'
        client: Optional[kuzu.Connection] # optional pre-made connection

    The provider will:
      - ensure vector extension is installed/loaded
      - create the table (IF NOT EXISTS)
      - create the vector index (ignore error if exists)
    """

    def __init__(self, **config: Any) -> None:
        self.db_path: str = config.get("db")
        if not self.db_path:
            raise ValueError("Kuzu vector store requires 'db' path in config.")

        self.table_name: str = config.get("table_name") or config.get("collection_name") or "Mem0Memory"
        self.index_name: str = config.get("index_name") or "mem0_memory_idx"
        self.dim: int = int(config.get("embedding_model_dims") or 0)
        if self.dim <= 0:
            raise ValueError("Kuzu vector store requires 'embedding_model_dims' (int > 0).")

        self.metric: str = (config.get("metric") or "cosine").lower()
        if self.metric not in ("cosine", "l2"):
            logger.warning("Unsupported metric '%s' requested; defaulting to 'cosine'.", self.metric)
            self.metric = "cosine"

        # Accept an existing connection or database
        client = config.get("client")
        if client and isinstance(client, kuzu.Connection):
            self._db = None
            self._conn: kuzu.Connection = client
        else:
            self._db = kuzu.Database(self.db_path)
            self._conn = kuzu.Connection(self._db)

        # Install/load vector extension (ignore if already loaded)
        self._install_and_load_vector_extension()

        # Ensure table schema & index
        self._ensure_schema()
        self._ensure_vector_index()

        logger.info("Kuzu vector store ready: table=%s, index=%s, dim=%d, metric=%s",
                    self.table_name, self.index_name, self.dim, self.metric)


    # ---------- Public API expected by Mem0 ----------

    def add(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        **_: Any,
    ) -> List[str]:
        """Insert new memory vectors and payloads."""
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have the same length.")

        out_ids: List[str] = []
        for i, (vec, pld) in enumerate(zip(vectors, payloads)):
            if len(vec) != self.dim:
                raise ValueError(f"Vector at position {i} has dim={len(vec)} but expected {self.dim}.")

            vid = ids[i] if ids and i < len(ids) else uuid.uuid4().hex

            # Normalize payload pieces that Mem0 commonly uses
            user_id = pld.get("user_id")
            agent_id = pld.get("agent_id")
            app_id = pld.get("app_id")
            # Mem0 consistently sets "data" for the textual memory
            data = pld.get("data") or pld.get("text") or pld.get("memory") or ""

            # Labels/categories (any list of tags)
            labels = _coerce_to_str_list(
                pld.get("labels") or pld.get("label") or pld.get("categories") or pld.get("category")
            )

            # Keep the rest as a JSON blob in 'meta'
            meta = dict(pld)  # shallow copy
            # Ensure "data" stays inside payload; Mem0 often expects payload["data"]
            if "data" not in meta:
                meta["data"] = data
            # Remove columns that are already top-level to avoid duplication
            for k in ("user_id", "agent_id", "app_id", "labels", "label", "categories", "category", "embedding"):
                meta.pop(k, None)
            meta_json = json.dumps(meta, ensure_ascii=False)

            # CREATE (m:Table { ... })
            query = f"""
            CREATE (m:{_ident(self.table_name)} {{
                id: $id,
                user_id: $user_id,
                agent_id: $agent_id,
                app_id: $app_id,
                data: $data,
                embedding: $embedding,
                labels: $labels,
                meta: $meta
            }});
            """
            params = {
                "id": vid,
                "user_id": user_id,
                "agent_id": agent_id,
                "app_id": app_id,
                "data": data,
                "embedding": vec,
                "labels": labels,
                "meta": meta_json,
            }
            self._conn.execute(query, params)
            out_ids.append(vid)

        return out_ids

    def search(
        self,
        query: str,
        vectors: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        query_dict: Optional[Dict[str, Any]] = None,  # accepted but not required
        **_: Any,
    ) -> List[VectorHit]:
        """Semantic search using Kuzu's vector index."""
        if not vectors:
            return []
        if len(vectors) != self.dim:
            raise ValueError(f"Search vector dim={len(vectors)} must match index dim={self.dim}.")

        # Compose optional WHERE from Mem0 filters (user_id, agent_id, app_id, ids, labels/categories)
        where_clause, where_params = self._build_where(filters or {})

        # Allow custom query overrides (e.g., advanced filtered graph queries)
        # If provided, expect a dict with "cypher" and "params". Minimal but flexible.
        if query_dict and "cypher" in query_dict:
            cypher = query_dict["cypher"]
            params = dict(query_dict.get("params", {}))
            # ensure required params are present
            params.setdefault("query_vector", vectors)
            params.setdefault("k", int(limit))
        else:
            cypher = f"""
            CALL QUERY_VECTOR_INDEX('{_str(self.table_name)}', '{_str(self.index_name)}', $query_vector, $k)
            WITH node, distance
            {where_clause}
            RETURN
                node.id           AS id,
                node.user_id      AS user_id,
                node.agent_id     AS agent_id,
                node.app_id       AS app_id,
                node.data         AS data,
                node.labels       AS labels,
                node.meta         AS meta,
                distance          AS distance
            ORDER BY distance
            LIMIT $k;
            """
            params = {"query_vector": vectors, "k": int(limit), **where_params}

        result = self._conn.execute(cypher, params)

        # Build VectorHit list from result rows.
        hits: List[VectorHit] = []
        # Iterate result to avoid extra dependencies
        for row in result:
            # row is a tuple in the same order as RETURN columns
            # (id, user_id, agent_id, app_id, data, labels, meta_json, distance)
            rid = row[0]
            r_user = row[1]
            r_agent = row[2]
            r_app = row[3]
            r_data = row[4]
            r_labels = row[5] or []
            r_meta_json = row[6] or "{}"
            r_distance = row[7]

            try:
                meta = json.loads(r_meta_json) if isinstance(r_meta_json, str) else (r_meta_json or {})
            except Exception:
                meta = {}

            # payload with at least "data" for Mem0
            payload = {
                "data": r_data,
                "user_id": r_user,
                "agent_id": r_agent,
                "app_id": r_app,
                "labels": r_labels,
                **meta,
            }

            # Mem0 does not require a specific score scale; keep distance (lower is better).
            # For cosine metric, you could convert to similarity = 1 - distance.
            score = None
            if isinstance(r_distance, (int, float)):
                if self.metric == "cosine":
                    # Convert to a rough similarity in [0..1]
                    score = max(0.0, 1.0 - float(r_distance))
                else:
                    # For L2, lower distance is better; expose negative distance as score
                    score = -float(r_distance)

            hits.append(VectorHit(id=rid, score=score, payload=payload))

        return hits

    def get(self, vector_id: str) -> VectorHit:
        """Fetch one memory by id."""
        q = f"""
        MATCH (m:{_ident(self.table_name)} {{id: $id}})
        RETURN
            m.id, m.user_id, m.agent_id, m.app_id, m.data, m.labels, m.meta
        LIMIT 1;
        """
        res = self._conn.execute(q, {"id": vector_id})
        for row in res:
            meta = {}
            try:
                meta = json.loads(row[6]) if isinstance(row[6], str) else (row[6] or {})
            except Exception:
                pass
            payload = {
                "data": row[4],
                "user_id": row[1],
                "agent_id": row[2],
                "app_id": row[3],
                "labels": row[5] or [],
                **meta,
            }
            return VectorHit(id=row[0], score=None, payload=payload)
        raise ValueError(f"Vector id not found: {vector_id}")

    def get_all(
        self,
        user_id: Optional[str] = None,
        limit: int = 1000,
        filters: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> List[VectorHit]:
        """Return all (optionally filtered) memories without vector search."""
        fdict = dict(filters or {})
        if user_id is not None:
            fdict.setdefault("user_id", user_id)
        where_clause, where_params = self._build_where(fdict)
        q = f"""
        MATCH (m:{_ident(self.table_name)})
        {where_clause.replace('node.', 'm.')}
        RETURN
            m.id, m.user_id, m.agent_id, m.app_id, m.data, m.labels, m.meta
        LIMIT $limit;
        """
        res = self._conn.execute(q, {**where_params, "limit": int(limit)})
        out: List[VectorHit] = []
        for row in res:
            meta = {}
            try:
                meta = json.loads(row[6]) if isinstance(row[6], str) else (row[6] or {})
            except Exception:
                pass
            payload = {
                "data": row[4],
                "user_id": row[1],
                "agent_id": row[2],
                "app_id": row[3],
                "labels": row[5] or [],
                **meta,
            }
            out.append(VectorHit(id=row[0], score=None, payload=payload))
        return out

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> bool:
        """Update memory vector and/or payload."""
        sets: List[str] = []
        params: Dict[str, Any] = {"id": vector_id}

        if vector is not None:
            if len(vector) != self.dim:
                raise ValueError(f"Update vector dim={len(vector)} must match index dim={self.dim}.")
            sets.append("m.embedding = $embedding")
            params["embedding"] = vector

        if payload:
            # Update top-level known fields if present
            for k in ("data", "user_id", "agent_id", "app_id"):
                if k in payload:
                    sets.append(f"m.{k} = ${k}")
                    params[k] = payload[k]
            # Labels
            labels = _coerce_to_str_list(
                payload.get("labels") or payload.get("label") or payload.get("categories") or payload.get("category")
            )
            if labels is not None:
                sets.append("m.labels = $labels")
                params["labels"] = labels

            # Merge/overwrite meta json
            meta = dict(payload)
            for k in ("data", "user_id", "agent_id", "app_id", "labels", "label", "categories", "category", "embedding"):
                meta.pop(k, None)
            if meta:
                # fetch existing meta and merge in cypher (simple overwrite here)
                sets.append("m.meta = $meta")
                params["meta"] = json.dumps(meta, ensure_ascii=False)

        if not sets:
            return True  # nothing to change

        q = f"""
        MATCH (m:{_ident(self.table_name)} {{id: $id}})
        SET {", ".join(sets)}
        RETURN m.id
        LIMIT 1;
        """
        res = self._conn.execute(q, params)
        return any(True for _ in res)

    def delete(self, vector_id: str) -> bool:
        """Delete memory by id."""
        q = f"""
        MATCH (m:{_ident(self.table_name)} {{id: $id}})
        DELETE m;
        """
        self._conn.execute(q, {"id": vector_id})
        return True

    def delete_many(self, ids: List[str]) -> bool:
        if not ids:
            return True
        q = f"""
        MATCH (m:{_ident(self.table_name)})
        WHERE m.id IN $ids
        DELETE m;
        """
        self._conn.execute(q, {"ids": ids})
        return True

    def delete_where(self, filters: Dict[str, Any]) -> int:
        """Delete memories by filter (returns count best-effort)."""
        where_clause, where_params = self._build_where(filters)
        q = f"""
        MATCH (m:{_ident(self.table_name)})
        {where_clause.replace('node.', 'm.')}
        WITH count(m) AS cnt
        MATCH (m:{_ident(self.table_name)})
        {where_clause.replace('node.', 'm.')}
        DELETE m
        RETURN cnt;
        """
        res = self._conn.execute(q, where_params)
        count = 0
        for row in res:
            count = int(row[0])
            break
        return count


    # ---------- Compatibility / Abstract API required by VectorStoreBase ----------

    def insert(
        self,
        vectors: Optional[List[List[float]]] = None,
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        *,
        # common alternates some adapters use:
        embeddings: Optional[List[List[float]]] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection: Optional[str] = None,
        table_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Compatibility alias for `add`. Accepts either:
          - vectors + payloads (+ ids), or
          - embeddings + (documents/metadatas) (+ ids)
        Optionally target a specific collection/table via `collection` or `table_name`.
        """
        target_table = table_name or collection or self.table_name

        vecs = vectors or embeddings
        if vecs is None:
            raise ValueError("insert() requires `vectors`/`embeddings`.")

        pls = payloads
        if pls is None:
            if metadatas is not None or documents is not None:
                # build payloads from docs + metadatas
                md = metadatas or [{} for _ in range(len(vecs))]
                docs = documents or [None] * len(vecs)
                if len(md) != len(vecs) or len(docs) != len(vecs):
                    raise ValueError("Length mismatch among embeddings/documents/metadatas.")
                pls = []
                for i in range(len(vecs)):
                    row = dict(md[i] or {})
                    if docs[i] is not None:
                        row.setdefault("data", docs[i])
                    pls.append(row)
            else:
                raise ValueError("insert() requires `payloads` or (`documents`/`metadatas`).")

        # Same-table fast path
        if target_table == self.table_name:
            return self.add(vecs, pls, ids=ids, **kwargs)

        # Different table: ensure schema/index exists, then insert
        self._ensure_schema_for(target_table, self.dim)
        self._ensure_vector_index_for(target_table, f"{_ident(target_table)}_idx", self.metric)
        return self._insert_rows(target_table, vecs, pls, ids)

    def list(
        self,
        *,
        user_id: Optional[str] = None,
        limit: int = 1000,
        filters: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> List[VectorHit]:
        """Compatibility alias for `get_all`."""
        return (self.get_all(user_id=user_id, limit=limit, filters=filters), None)

    def reset(self, name: Optional[str] = None, **_: Any) -> bool:
        """
        Delete all rows in the given collection/table (default: self.table_name).
        Keeps the table and index in place.
        """
        tbl = name or self.table_name
        q = f"MATCH (m:{_ident(tbl)}) DELETE m;"
        self._conn.execute(q)
        return True

    def create_col(
        self,
        name: Optional[str] = None,
        *,
        dims: Optional[int] = None,
        metric: Optional[str] = None,
        index_name: Optional[str] = None,
        **_: Any,
    ) -> bool:
        """
        Create a collection (node table) and vector index if they don't already exist.
        `dims` defaults to this provider's dimension; `metric` defaults to provider metric.
        """
        tbl = name or self.table_name
        dim = int(dims or self.dim)
        met = (metric or self.metric or "cosine").lower()
        if met not in ("cosine", "l2"):
            met = "cosine"
        idx = index_name or (self.index_name if tbl == self.table_name else f"{_ident(tbl)}_idx")

        # Create schema + index idempotently
        self._ensure_schema_for(tbl, dim)
        self._ensure_vector_index_for(tbl, idx, met)
        return True

    def delete_col(self, name: Optional[str] = None, *, index_name: Optional[str] = None, **_: Any) -> bool:
        """
        Drop the vector index (best-effort) and the node table.
        """
        tbl = name or self.table_name
        idx = index_name or (self.index_name if tbl == self.table_name else f"{_ident(tbl)}_idx")

        # Best-effort: drop index if procedure is available; ignore errors if it doesn't exist.
        try:
            self._conn.execute(f"CALL DROP_VECTOR_INDEX('{_str(tbl)}', '{_str(idx)}');")
        except RuntimeError:
            pass
        except Exception:
            # Older/newer Kùzu versions may vary; dropping table will remove associated index anyway.
            pass

        # Drop the table itself (idempotent drop may raise if missing; ignore).
        try:
            self._conn.execute(f"DROP TABLE {_ident(tbl)};")
        except RuntimeError:
            pass
        except Exception:
            pass
        return True

    def list_cols(self, **_: Any) -> List[str]:
        """
        Return available tables. Falls back to [self.table_name] if SHOW_TABLES is unavailable.
        """
        try:
            # Kùzu exposes table listing via a SHOW_TABLES procedure.
            res = self._conn.execute("CALL SHOW_TABLES();")
            out: List[str] = []
            for row in res:
                # Typical shape: (name, type) -> we take first column
                if isinstance(row, (list, tuple)) and row:
                    out.append(str(row[0]))
                else:
                    # Some versions may return dict-like rows
                    try:
                        out.append(str(row.get("name")))
                    except Exception:
                        pass
            # Deduplicate and keep order
            seen = set()
            uniq = []
            for n in out:
                if n and n not in seen:
                    seen.add(n)
                    uniq.append(n)
            return uniq or [self.table_name]
        except Exception:
            return [self.table_name]

    def col_info(self, name: Optional[str] = None, **_: Any) -> Dict[str, Any]:
        """
        Return basic info about a collection:
          - name, count (rows), index_name (best-effort), dims/metric when known
        """
        tbl = name or self.table_name
        info: Dict[str, Any] = {
            "name": tbl,
            "count": 0,
            "index_name": self.index_name if tbl == self.table_name else f"{_ident(tbl)}_idx",
            "dims": self.dim if tbl == self.table_name else None,
            "metric": self.metric if tbl == self.table_name else None,
        }
        try:
            res = self._conn.execute(f"MATCH (m:{_ident(tbl)}) RETURN COUNT(m);")
            for row in res:
                # COUNT result in first column
                info["count"] = int(row[0])
                break
        except Exception:
            pass
        return info


    # ---------- Internals used by the abstract API above ----------

    def _ensure_schema_for(self, table: str, dim: int) -> None:
        """Create node table with a fixed-length embedding array, if not exists."""
        q = f"""
        CREATE NODE TABLE IF NOT EXISTS {_ident(table)}(
            id STRING PRIMARY KEY,
            user_id STRING,
            agent_id STRING,
            app_id STRING,
            data STRING,
            embedding FLOAT[{int(dim)}],
            labels STRING[],
            meta STRING
        );
        """
        self._conn.execute(q)

    def _ensure_vector_index_for(self, table: str, index_name: str, metric: str) -> None:
        """Create HNSW index on embedding, if not exists (ignore 'already exists')."""
        q = f"""
        CALL CREATE_VECTOR_INDEX(
            '{_str(table)}',
            '{_str(index_name)}',
            'embedding',
            metric := '{_str(metric)}'
        );
        """
        try:
            self._conn.execute(q)
        except RuntimeError:
            pass  # index likely already exists

    def _insert_rows(
        self,
        table: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert rows into an arbitrary table (used by insert() w/ custom collection)."""
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have the same length.")

        out_ids: List[str] = []
        for i, (vec, pld) in enumerate(zip(vectors, payloads)):
            if len(vec) != self.dim:
                raise ValueError(f"Vector at position {i} has dim={len(vec)} but expected {self.dim}.")

            vid = ids[i] if ids and i < len(ids) else uuid.uuid4().hex
            user_id = pld.get("user_id")
            agent_id = pld.get("agent_id")
            app_id = pld.get("app_id")
            data = pld.get("data") or pld.get("text") or pld.get("memory") or ""

            labels = _coerce_to_str_list(
                pld.get("labels") or pld.get("label") or pld.get("categories") or pld.get("category")
            )

            meta = dict(pld)
            if "data" not in meta:
                meta["data"] = data
            for k in ("user_id", "agent_id", "app_id", "labels", "label", "categories", "category", "embedding"):
                meta.pop(k, None)
            meta_json = json.dumps(meta, ensure_ascii=False)

            query = f"""
            CREATE (m:{_ident(table)} {{
                id: $id,
                user_id: $user_id,
                agent_id: $agent_id,
                app_id: $app_id,
                data: $data,
                embedding: $embedding,
                labels: $labels,
                meta: $meta
            }});
            """
            params = {
                "id": vid,
                "user_id": user_id,
                "agent_id": agent_id,
                "app_id": app_id,
                "data": data,
                "embedding": vec,
                "labels": labels,
                "meta": meta_json,
            }
            self._conn.execute(query, params)
            out_ids.append(vid)

        return out_ids
    

    # ---------- Internal helpers ----------

    def _install_and_load_vector_extension(self) -> None:
        """INSTALL/LOAD vector extension (ignore errors if already loaded)."""
        # Official docs show installing & loading before using vector index. :contentReference[oaicite:4]{index=4}
        try:
            self._conn.execute("INSTALL vector;")
        except RuntimeError:
            pass
        try:
            self._conn.execute("LOAD vector;")
        except RuntimeError:
            pass

    def _ensure_schema(self) -> None:
        # Node table with fixed-length embedding array
        q = f"""
        CREATE NODE TABLE IF NOT EXISTS {_ident(self.table_name)}(
            id STRING PRIMARY KEY,
            user_id STRING,
            agent_id STRING,
            app_id STRING,
            data STRING,
            embedding FLOAT[{self.dim}],
            labels STRING[],
            meta STRING
        );
        """
        self._conn.execute(q)

    def _ensure_vector_index(self) -> None:
        # Create HNSW index on embedding (idempotent via try/except)
        # Kuzu supports metric := 'cosine' or 'l2'. :contentReference[oaicite:5]{index=5}
        q = f"""
        CALL CREATE_VECTOR_INDEX(
            '{_str(self.table_name)}',
            '{_str(self.index_name)}',
            'embedding',
            metric := '{_str(self.metric)}'
        );
        """
        try:
            self._conn.execute(q)
        except RuntimeError:
            # Likely "index already exists" — safe to ignore
            pass

    def _build_where(self, filters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Translate Mem0-style filters into a Cypher WHERE clause over 'node'.
        Supported keys: user_id, agent_id, app_id, ids, labels/categories.
        """
        clauses: List[str] = []
        params: Dict[str, Any] = {}

        if not filters:
            return "", {}

        # ids (list)
        ids = filters.get("ids") or filters.get("id_list")
        if ids:
            clauses.append("node.id IN $ids")
            params["ids"] = list(ids)

        # simple equality or IN for ids/agents/apps
        for key in ("user_id", "agent_id", "app_id"):
            if key in filters and filters[key] is not None:
                val = filters[key]
                if isinstance(val, (list, tuple, set)):
                    clauses.append(f"node.{key} IN ${key}")
                else:
                    clauses.append(f"node.{key} = ${key}")
                params[key] = val

        # labels / categories overlap
        lbls = filters.get("labels") or filters.get("label") or filters.get("categories") or filters.get("category")
        lbls = _coerce_to_str_list(lbls)
        if lbls:
            clauses.append("ANY(x IN node.labels WHERE x IN $labels)")
            params["labels"] = lbls

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params


# ---------- small utilities ----------

def _ident(name: str) -> str:
    """Return a safe Cypher identifier for labels/tables (letters, digits, underscore)."""
    # Kuzu/Cypher labels are identifiers; keep conservative sanitization.
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    if not safe:
        raise ValueError("Identifier cannot be empty.")
    return safe


def _str(s: Any) -> str:
    """Coerce to str for use inside single-quoted Cypher string literals."""
    return str(s)


def _coerce_to_str_list(v: Any) -> Optional[List[str]]:
    if v is None:
        return None
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v]
    return [str(v)]
