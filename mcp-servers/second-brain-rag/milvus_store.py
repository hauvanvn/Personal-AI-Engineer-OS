from pymilvus import MilvusClient, DataType
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path
import logging
import os
import re
import sys
import time
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

class SecondBrainStore:
    def __init__(self, uri: str = "http://localhost:19530", token: str = ""):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Not found GEMINI_API_KEY in .env!")

        self.genai_client = genai.Client(api_key=api_key)
        self.embedding_model = "gemini-embedding-001"
        self.dim = 768

        self.client = MilvusClient(uri=uri, token=token)
        self.collection_name = "personal_knowledge_v2"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8000)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="created_ts", datatype=DataType.INT64)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Check whether the exception indicates a Gemini API rate-limit / quota error."""
        msg = str(exc).lower()
        return "429" in msg or "resource_exhausted" in msg or "quota" in msg

    @staticmethod
    def _parse_retry_delay(exc: Exception) -> Optional[float]:
        """Try to extract a suggested retry delay (seconds) from the error message."""
        pattern = r"retry in ([\d\.]+)s|retryDelay['\"]?\s*:\s*['\"]?(\d+)s"
        match = re.search(pattern, str(exc), re.IGNORECASE)
        if match:
            raw = match.group(1) or match.group(2)
            return float(raw)
        return None

    def _estimate_tokens(self, text: str) -> int:
        """Conservatively estimate token count for a text string.

        Vietnamese Unicode and mixed code/text average roughly 1.8-2.2 chars
        per Gemini token. We use 1.8 (the lower bound) plus a small overhead
        to stay on the safe side.
        """
        return max(1, int(len(text) / 1.8) + 15)

    def _get_embedding(self, text: str) -> List[float]:
        """Call the Gemini Embedding API for free to obtain a 768-dimensional vector.

        Includes retry with exponential backoff for rate-limit (429) errors.
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.genai_client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dim
                    )
                )
                vec = np.array(response.embeddings[0].values)
                norm_vec = vec / np.linalg.norm(vec)
                return norm_vec.tolist()
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < max_retries - 1:
                    wait_time = min(15 * (2 ** attempt), 90)
                    logger.warning(
                        "Rate limit hit in _get_embedding (attempt %d/%d). "
                        "Waiting %.1f seconds before retry.",
                        attempt + 1, max_retries, wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    raise

    def add_document(self, text: str, source_file: str, doc_type: str = "note") -> int:
        vector = self._get_embedding(text)
        data = [{
            "text": text,
            "dense_vector": vector,
            "doc_type": doc_type,
            "source_file": source_file,
            "created_ts": int(datetime.now().timestamp())
        }]
        res = self.client.insert(collection_name=self.collection_name, data=data)
        return res["insert_count"]

    def search_with_time_decay(
        self,
        query: str,
        top_k: int = 4,
        doc_type: Optional[str] = None,
        decay_rate: float = 0.01,
        min_score: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """
        Execute semantic search and rerank results by combining Cosine Similarity (85%)
        with an exponential Time-Decay score (15%), filtering out chunks below min_score.
        """
        query_vector = self._get_embedding(query)

        filter_expr = f'doc_type == "{doc_type}"' if doc_type else ""
        candidate_limit = max(top_k * 3, 15)

        search_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            filter=filter_expr,
            limit=candidate_limit,
            output_fields=[
                "text",
                "source_file",
                "doc_type",
                "page",
                "created_at",
            ],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )

        if not search_results or not search_results[0]:
            return []

        now_ts = datetime.now().timestamp()
        scored_results: List[Dict[str, Any]] = []

        for hit in search_results[0]:
            entity = hit.get("entity", {})
            semantic_score = float(hit.get("distance", 0.0))

            created_at = int(entity.get("created_at", now_ts))
            age_hours = max(0.0, (now_ts - created_at) / 3600.0)
            recency_score = np.exp(-decay_rate * age_hours)

            final_score = (0.85 * semantic_score) + (0.15 * recency_score)

            if final_score < min_score:
                continue

            scored_results.append(
                {
                    "id": hit.get("id"),
                    "text": entity.get("text", ""),
                    "source_file": entity.get("source_file", "unknown"),
                    "doc_type": entity.get("doc_type", "document"),
                    "page": entity.get("page", 1),
                    "semantic_score": round(semantic_score, 4),
                    "recency_score": round(recency_score, 4),
                    "score": round(final_score, 4),
                }
            )

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def _get_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 10,
        max_batch_tokens: int = 6000,
    ) -> List[List[float]]:
        """Generate normalized 768-dim embeddings with token-aware batching.

        Dynamically groups texts into sub-batches bounded by both item count
        (batch_size, default 10) and estimated token budget
        (max_batch_tokens, default 6000). A 60-second rolling window
        throttles total tokens sent to stay safely under the Free Tier
        30,000 TPM limit (budget: 22,000 estimated tokens per window).

        Each sub-batch is retried independently on 429 / RESOURCE_EXHAUSTED
        errors so that large PDF ingestions survive rate limits without
        losing progress.
        """
        if not texts:
            return []

        # -- 1. Build token-aware sub-batches --------------------------------
        max_single_chunk_chars = 12000  # hard cap so one chunk never blows TPM
        sub_batches: List[List[str]] = []
        current_batch: List[str] = []
        current_tokens = 0

        for raw_text in texts:
            text = (
                raw_text[:max_single_chunk_chars]
                if len(raw_text) > max_single_chunk_chars
                else raw_text
            )
            t_tokens = self._estimate_tokens(text)

            # If a single chunk already exceeds the token budget, isolate it
            if t_tokens >= max_batch_tokens:
                if current_batch:
                    sub_batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                sub_batches.append([text])
                continue

            # Adding this chunk would exceed either limit -> flush
            if (
                len(current_batch) >= batch_size
                or current_tokens + t_tokens > max_batch_tokens
            ):
                if current_batch:
                    sub_batches.append(current_batch)
                current_batch = [text]
                current_tokens = t_tokens
            else:
                current_batch.append(text)
                current_tokens += t_tokens

        if current_batch:
            sub_batches.append(current_batch)

        total_batches = len(sub_batches)
        total_chunks = len(texts)

        # -- 2. Rolling TPM window state -------------------------------------
        tpm_budget = 30000
        tokens_used_in_window = 0
        window_start_time = time.time()

        # -- 3. Process each sub-batch with retry ----------------------------
        max_retries = 8
        all_vectors: List[List[float]] = []
        processed_chunks = 0

        for batch_idx, sub_batch in enumerate(sub_batches):
            sub_batch_tokens = sum(self._estimate_tokens(t) for t in sub_batch)

            # Reset rolling window if 62 seconds have naturally elapsed
            elapsed_in_window = time.time() - window_start_time
            if elapsed_in_window >= 62.0:
                tokens_used_in_window = 0
                window_start_time = time.time()
                elapsed_in_window = 0.0

            # Proactive TPM throttle: if adding this batch would exceed the
            # safe budget, wait for the current 60-second window to expire.
            if tokens_used_in_window + sub_batch_tokens > tpm_budget:
                sleep_time = max(5.0, 62.0 - (time.time() - window_start_time))
                logger.info(
                    f"TPM budget reached ({tokens_used_in_window} tokens). "
                    f"Pausing {sleep_time:.1f}s for quota window reset..."
                )
                time.sleep(sleep_time)
                tokens_used_in_window = 0
                window_start_time = time.time()

            for attempt in range(max_retries):
                try:
                    response = self.genai_client.models.embed_content(
                        model=self.embedding_model,
                        contents=sub_batch,
                        config=types.EmbedContentConfig(
                            output_dimensionality=self.dim
                        ),
                    )
                    for emb in response.embeddings:
                        vec = np.array(emb.values, dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec = vec / norm
                        all_vectors.append(vec.tolist())

                    # Track tokens and log progress
                    tokens_used_in_window += sub_batch_tokens
                    processed_chunks += len(sub_batch)
                    idx = batch_idx + 1
                    logger.info(
                        f"Embedded sub-batch {idx}/{total_batches} "
                        f"({processed_chunks}/{total_chunks} chunks)"
                    )
                    break  # success - move to next sub-batch
                except Exception as e:
                    err_msg = str(e)
                    is_rate_limit = (
                        "429" in err_msg
                        or "resource_exhausted" in err_msg.lower()
                        or "quota" in err_msg.lower()
                    )
                    if is_rate_limit and attempt < max_retries - 1:
                        match = re.search(
                            r"retry in ([\d\.]+)s|retryDelay['\"]?\s*:\s*['\"]?(\d+)s",
                            err_msg,
                            re.IGNORECASE,
                        )
                        if match:
                            parsed_seconds = float(match.group(1) or match.group(2))
                            wait_time = parsed_seconds + 6.0
                        else:
                            wait_time = 65.0

                        logger.warning(
                            f"Rate limit hit on sub-batch {batch_idx + 1}/{total_batches} "
                            f"(attempt {attempt + 1}/{max_retries}). "
                            f"Waiting {wait_time:.1f}s before retry."
                        )
                        time.sleep(wait_time)
                        tokens_used_in_window = 0
                        window_start_time = time.time()
                    else:
                        raise

            # Brief pause between sub-batches to spread requests
            if batch_idx < total_batches - 1:
                time.sleep(2.0)

        return all_vectors

    def delete_by_source(self, source_file: str):
        """Remove existing chunks of a file before re-indexing updated content."""
        self.client.delete(
            collection_name=self.collection_name,
            filter=f'source_file == "{source_file}"',
        )

    def count_by_source(self, source_file: str) -> int:
        """Return the count of vector chunks currently stored in Milvus for source_file."""
        try:
            res = self.client.query(
                collection_name=self.collection_name,
                filter=f'source_file == "{source_file}"',
                output_fields=["count(*)"],
            )
            if res and isinstance(res, list) and "count(*)" in res[0]:
                return int(res[0]["count(*)"])
            return 0
        except Exception as e:
            logger.warning("Failed to query chunk count for '%s': %s", source_file, e)
            return 0

    def add_documents_batch(
        self, chunks: List[str], source_file: str, doc_type: str = "pdf"
    ) -> int:
        """Batch embed and insert multiple text chunks into Milvus.

        Embeddings are generated via token-aware sub-batches (max 10 items /
        6000 tokens per API call) with TPM throttling.  Milvus inserts are
        performed in safe chunks of up to 200 rows per call.
        """
        if not chunks:
            return 0

        vectors = self._get_embeddings_batch(
            chunks, batch_size=10, max_batch_tokens=6000
        )

        total_inserted = 0
        now_ts = int(datetime.now().timestamp())
        milvus_insert_size = 200

        for i in range(0, len(chunks), milvus_insert_size):
            batch_texts = chunks[i : i + milvus_insert_size]
            batch_vecs = vectors[i : i + milvus_insert_size]
            data = [
                {
                    "text": text,
                    "dense_vector": vec,
                    "doc_type": doc_type,
                    "source_file": source_file,
                    "created_ts": now_ts,
                }
                for text, vec in zip(batch_texts, batch_vecs)
            ]
            res = self.client.insert(collection_name=self.collection_name, data=data)
            total_inserted += res["insert_count"]

        return total_inserted