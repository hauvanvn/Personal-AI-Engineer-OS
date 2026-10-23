from pymilvus import MilvusClient, DataType
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path
import os
import numpy as np

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
        self.dim = 768 
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

    def _get_embedding(self, text: str) -> List[float]:
        """Call the Gemini Embedding API for free to obtain a 768-dimensional vector"""
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

    def search_with_time_decay(self, query: str, doc_type: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self._get_embedding(query)

        filter_expr = f'doc_type == "{doc_type}"' if doc_type else ""

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            filter=filter_expr,
            limit=top_k * 2,
            output_fields=["text", "source_file", "doc_type", "created_ts"]
        )[0]

        now_ts = datetime.now().timestamp()
        scored_results = []

        for hit in results:
            entity = hit["entity"]
            semantic_score = hit["distance"] # Cosine similarity

            age_days = max(0, (now_ts - entity.get("created_ts", now_ts)) / 86400)
            recency_score = 1.0 / (1.0 + 0.01 * age_days)

            final_score = 0.85 * semantic_score + 0.15 * recency_score
            scored_results.append({
                "score": round(final_score, 4),
                "text": entity["text"],
                "source_file": entity["source_file"],
                "doc_type": entity["doc_type"]
            })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]