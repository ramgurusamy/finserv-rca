"""Qdrant wrapper with bitemporal version filtering.

Approach: amendments never delete. A superseded chunk gets `effective_to`, so a
question can be answered 'as of' any date and past decisions stay reproducible.
"""
from __future__ import annotations

import uuid
from datetime import date

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as q

from ..models import Chunk


def date_int(value: date) -> int:
    return value.year * 10000 + value.month * 100 + value.day


class VectorStore:
    def __init__(self, path: str, collection: str, dim: int) -> None:
        self.client = QdrantClient(":memory:") if path == ":memory:" else QdrantClient(path=path)
        self.collection = collection
        self.dim = dim
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config=q.VectorParams(size=dim, distance=q.Distance.COSINE),
            )

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.UUID(chunk_id[:32]))

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = chunk.model_dump(mode="json")
            payload["from_int"] = date_int(chunk.effective_from)
            payload["to_int"] = date_int(chunk.effective_to) if chunk.effective_to else 99991231
            points.append(
                q.PointStruct(
                    id=self._point_id(chunk.chunk_id), vector=vector.tolist(), payload=payload
                )
            )
        if points:
            self.client.upsert(self.collection, points=points, wait=True)

    def close_version(self, chunk: Chunk, effective_to: date) -> None:
        self.client.set_payload(
            self.collection,
            payload={"effective_to": effective_to.isoformat(), "to_int": date_int(effective_to)},
            points=[self._point_id(chunk.chunk_id)],
        )

    @staticmethod
    def _filter(as_of: date | None, frameworks: list[str] | None) -> q.Filter:
        day = date_int(as_of or date.today())
        must: list = [
            q.FieldCondition(key="from_int", range=q.Range(lte=day)),
            q.FieldCondition(key="to_int", range=q.Range(gt=day)),
        ]
        if frameworks:
            must.append(q.FieldCondition(key="framework", match=q.MatchAny(any=frameworks)))
        return q.Filter(must=must)

    def search(
        self,
        vector: np.ndarray,
        limit: int,
        as_of: date | None = None,
        frameworks: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        response = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=limit,
            query_filter=self._filter(as_of, frameworks),
            with_payload=True,
        )
        return [(self._to_chunk(p.payload), float(p.score)) for p in response.points]

    def all_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=True
            )
            chunks.extend(self._to_chunk(p.payload) for p in points)
            if offset is None:
                return chunks

    @staticmethod
    def _to_chunk(payload: dict) -> Chunk:
        return Chunk.model_validate(
            {k: v for k, v in payload.items() if k not in {"from_int", "to_int"}}
        )
