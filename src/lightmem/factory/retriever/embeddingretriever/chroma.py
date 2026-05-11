import logging
from typing import Any, Optional

from lightmem.configs.retriever.embeddingretriever.chroma import ChromaConfig

logger = logging.getLogger(__name__)


class _CollectionsResponse:
    """Wraps Chroma's collection list to match the Qdrant list_cols() return shape."""

    def __init__(self, collections):
        self.collections = collections


class ChromaPoint:
    """Minimal point wrapper so get_all() can call .model_dump() on scroll results."""

    def __init__(self, id: str, payload: dict, vector: Optional[list] = None):
        self.id = id
        self.payload = payload
        self.vector = vector

    def model_dump(self) -> dict:
        result = {"id": self.id, "payload": self.payload}
        if self.vector is not None:
            result["vector"] = self.vector
        return result


class Chroma:
    def __init__(self, config: Optional[ChromaConfig] = None):
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is not installed. Install it with: pip install 'lightmem[chroma]'"
            )

        if config.host:
            params = {"host": config.host, "port": config.port or 8000}
            if config.ssl:
                params["ssl"] = config.ssl
            if config.headers:
                params["headers"] = config.headers
            self.client = chromadb.HttpClient(**params)
        elif config.path:
            self.client = chromadb.PersistentClient(path=config.path)
        else:
            self.client = chromadb.Client()

        self.collection_name = config.collection_name
        self.embedding_model_dims = config.embedding_model_dims
        self._collection = None
        self.create_col(config.embedding_model_dims)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_str_id(id: Any) -> str:
        return str(id)

    def _create_filter(self, filters: dict) -> Optional[dict]:
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict):
                chroma_cond = {}
                if "gte" in value:
                    chroma_cond["$gte"] = value["gte"]
                if "lte" in value:
                    chroma_cond["$lte"] = value["lte"]
                conditions.append({key: chroma_cond})
            else:
                conditions.append({key: {"$eq": value}})
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_col(self, vector_size: int):
        # get_or_create_collection is idempotent — no need to check first
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(f"Collection '{self.collection_name}' ready.")

    def list_cols(self) -> _CollectionsResponse:
        return _CollectionsResponse(self.client.list_collections())

    def delete_col(self):
        self.client.delete_collection(name=self.collection_name)
        self._collection = None

    def col_info(self) -> dict:
        return {
            "name": self._collection.name,
            "count": self._collection.count(),
            "metadata": self._collection.metadata,
        }

    def reset(self):
        logger.warning(f"Resetting collection '{self.collection_name}'...")
        self.delete_col()
        self.create_col(self.embedding_model_dims)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def insert(self, vectors: list, payloads: list = None, ids: list = None):
        logger.info(f"Inserting {len(vectors)} vectors into '{self.collection_name}'")
        str_ids = [
            self._to_str_id(ids[i] if ids is not None else i)
            for i in range(len(vectors))
        ]
        metadatas = [payloads[i] if payloads else {} for i in range(len(vectors))]
        self._collection.upsert(
            ids=str_ids,
            embeddings=vectors,
            metadatas=metadatas,
        )

    def get(self, vector_id: Any) -> Optional[dict]:
        result = self._collection.get(
            ids=[self._to_str_id(vector_id)],
            include=["embeddings", "metadatas"],
        )
        if not result["ids"]:
            return None
        return {
            "id": result["ids"][0],
            "payload": result["metadatas"][0] if result["metadatas"] else {},
            "vector": result["embeddings"][0] if result["embeddings"] else None,
        }

    def delete(self, vector_id: Any):
        self._collection.delete(ids=[self._to_str_id(vector_id)])

    def update(self, vector_id: Any, vector: list = None, payload: dict = None):
        if vector is None and payload is None:
            logger.debug(f"Update called for ID {vector_id} with no data. Skipping.")
            return

        str_id = self._to_str_id(vector_id)

        if vector is None and payload is not None:
            # Merge new fields into existing metadata (matches Qdrant set_payload behaviour)
            existing = self._collection.get(ids=[str_id], include=["metadatas"])
            existing_meta = existing["metadatas"][0] if existing["metadatas"] else {}
            merged = {**existing_meta, **payload}
            self._collection.update(ids=[str_id], metadatas=[merged])
            return

        if vector is not None and payload is None:
            self._collection.update(ids=[str_id], embeddings=[vector])
            return

        # Both vector and payload — full replace
        self._collection.update(ids=[str_id], embeddings=[vector], metadatas=[payload])

    def exists(self, vector_id: Any) -> bool:
        try:
            result = self._collection.get(ids=[self._to_str_id(vector_id)], include=[])
            return len(result["ids"]) > 0
        except Exception as e:
            logger.error(f"Error checking existence of ID {vector_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list,
        limit: int = 5,
        filters: dict = None,
        exclude_ids: list = None,
        return_full: bool = False,
    ) -> list:
        where = self._create_filter(filters) if filters else None

        # Fetch extra results to cover post-filtering of excluded IDs
        fetch_limit = limit + len(exclude_ids) if exclude_ids else limit
        # Chroma requires n_results <= collection size
        fetch_limit = min(fetch_limit, max(self._collection.count(), 1))

        kwargs = dict(
            query_embeddings=[query_vector],
            n_results=fetch_limit,
            include=["distances", "metadatas"] + (["embeddings"] if return_full else []),
        )
        if where:
            kwargs["where"] = where

        hits = self._collection.query(**kwargs)

        ids = hits["ids"][0]
        distances = hits["distances"][0]
        metadatas = hits["metadatas"][0]
        embeddings = hits.get("embeddings", [[]])[0] if return_full else []

        exclude_set = set(self._to_str_id(eid) for eid in exclude_ids) if exclude_ids else set()

        results = []
        for i, doc_id in enumerate(ids):
            if doc_id in exclude_set:
                continue
            # Chroma cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
            score = 1.0 - distances[i]
            if return_full:
                results.append({
                    "id": doc_id,
                    "score": score,
                    "payload": metadatas[i] if metadatas else {},
                })
            else:
                results.append({"id": doc_id, "score": score})
            if len(results) == limit:
                break

        return results

    # ------------------------------------------------------------------
    # List / scroll
    # ------------------------------------------------------------------

    def list(self, filters: dict = None, limit: int = 100) -> tuple:
        return self.scroll(scroll_filter=filters, limit=limit)

    def scroll(
        self,
        scroll_filter=None,
        limit: int = 100,
        offset: Any = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple:
        where = None
        if isinstance(scroll_filter, dict):
            where = self._create_filter(scroll_filter)
        elif scroll_filter is not None:
            where = scroll_filter

        offset_val = int(offset) if offset is not None else 0
        include = []
        if with_payload:
            include.append("metadatas")
        if with_vectors:
            include.append("embeddings")

        kwargs = dict(
            limit=limit + 1,  # fetch one extra to detect whether more pages exist
            offset=offset_val,
            include=include,
        )
        if where:
            kwargs["where"] = where

        raw = self._collection.get(**kwargs)
        raw_ids = raw["ids"]
        has_more = len(raw_ids) > limit
        raw_ids = raw_ids[:limit]

        metadatas = raw.get("metadatas") or []
        embeddings = raw.get("embeddings") or []

        points = []
        for i, doc_id in enumerate(raw_ids):
            payload = metadatas[i] if metadatas and i < len(metadatas) else {}
            vector = embeddings[i] if embeddings and i < len(embeddings) else None
            points.append(ChromaPoint(id=doc_id, payload=payload, vector=vector))

        next_offset = offset_val + limit if has_more else None
        return points, next_offset

    def get_all(self, with_vectors: bool = True, with_payload: bool = True) -> list:
        all_points = []
        offset = 0
        while True:
            points, next_offset = self.scroll(
                limit=100,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            all_points.extend([p.model_dump() for p in points])
            if next_offset is None:
                break
            offset = next_offset
        return all_points
