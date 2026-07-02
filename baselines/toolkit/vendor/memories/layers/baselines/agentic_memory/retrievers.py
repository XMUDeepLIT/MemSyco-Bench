from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
import pickle
from nltk.tokenize import word_tokenize
import os
import json
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
    OpenAIEmbeddingFunction
)
import logging
import time


logger = logging.getLogger(__name__)


def _method_logging_enabled() -> bool:
    return os.getenv("AMEM_LOG", "").lower() in {"1", "true", "yes", "on"} or os.getenv(
        "MEMORY_METHOD_LOG", ""
    ).lower() in {"1", "true", "yes", "on"}


def _configure_method_logging() -> None:
    if not _method_logging_enabled():
        return
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.setLevel(logging.INFO)


def simple_tokenize(text):
    return word_tokenize(text)


class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""

    def __init__(
        self,
        collection_name: str = "memories",
        model_name: str = "all-MiniLM-L6-v2",
        embedder_provider: str = "sentence-transformers",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        persist_path: Optional[str] = None,
    ):
        """Initialize ChromaDB retriever.
        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Name of the embedding model
            embedder_provider: Provider for embeddings ("sentence-transformers" or "openai")
            api_key: API key for OpenAI (required if embedder_provider is "openai")
            base_url: Base URL for OpenAI API (optional, defaults to official OpenAI endpoint)
                     Useful for using proxies or OpenAI-compatible services
        """
        _configure_method_logging()
        self.collection_name = collection_name
        self.model_name = model_name
        if persist_path:
            os.makedirs(persist_path, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_path,
                settings=Settings(allow_reset=True,
                                  anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.Client(
                Settings(allow_reset=True, anonymized_telemetry=False))
        self.embedder_provider = embedder_provider
        if embedder_provider == "openai":
            if api_key is None:
                api_key = os.environ.get("OPENAI_API_KEY")
                base_url = os.getenv('OPENAI_API_BASE')
                if api_key is None:
                    raise ValueError(
                        "API key is required for OpenAI embeddings. "
                        "Please provide it via api_key parameter or OPENAI_API_KEY environment variable."
                    )
            openai_params = {
                "api_key": api_key,
                "model_name": model_name,
                "api_base": base_url
            }
            self.embedding_function = OpenAIEmbeddingFunction(**openai_params)
        else:
            self.embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function)
        logger.info(
            "[amem.chroma.init] collection=%s embedder=%s model=%s persist_path=%s",
            collection_name,
            embedder_provider,
            model_name,
            persist_path,
        )

    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB with enhanced embedding using metadata.

        Args:
            document: Text content to add
            metadata: Dictionary of metadata including keywords, tags, context
            doc_id: Unique identifier for the document
        """
        # Build enhanced document content including semantic metadata
        enhanced_document = document

        # Add context information
        if 'context' in metadata and metadata['context'] != "General":
            enhanced_document += f" context: {metadata['context']}"

        # Add keywords information
        if 'keywords' in metadata and metadata['keywords']:
            keywords = metadata['keywords'] if isinstance(
                metadata['keywords'], list) else json.loads(metadata['keywords'])
            if keywords:
                enhanced_document += f" keywords: {', '.join(keywords)}"

        # Add tags information
        if 'tags' in metadata and metadata['tags']:
            tags = metadata['tags'] if isinstance(
                metadata['tags'], list) else json.loads(metadata['tags'])
            if tags:
                enhanced_document += f" tags: {', '.join(tags)}"

        # Convert MemoryNote object to serializable format
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)

        # Store enhanced document content for better embedding
        processed_metadata['enhanced_content'] = enhanced_document

        # Use enhanced document content for embedding generation
        started_at = time.perf_counter()
        self.collection.add(
            documents=[enhanced_document],
            metadatas=[processed_metadata],
            ids=[doc_id]
        )
        logger.info(
            "[amem.chroma.add] collection=%s doc_id=%s document_chars=%d enhanced_chars=%d elapsed_ms=%.1f",
            self.collection_name,
            doc_id,
            len(document),
            len(enhanced_document),
            (time.perf_counter() - started_at) * 1000,
        )

    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.

        Args:
            doc_id: ID of document to delete
        """
        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5):
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            Dict with documents, metadatas, ids, and distances
        """
        started_at = time.perf_counter()
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )

        # Convert string metadata back to original types
        if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0:
            # First level is a list with one item per query
            for i in range(len(results['metadatas'])):
                # Second level is a list of metadata dicts for each result
                if isinstance(results['metadatas'][i], list):
                    for j in range(len(results['metadatas'][i])):
                        # Process each metadata dict
                        if isinstance(results['metadatas'][i][j], dict):
                            metadata = results['metadatas'][i][j]
                            for key, value in metadata.items():
                                try:
                                    # Try to parse JSON for lists and dicts
                                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                        metadata[key] = json.loads(value)
                                    # Convert numeric strings back to numbers
                                    elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                        if '.' in value:
                                            metadata[key] = float(value)
                                        else:
                                            metadata[key] = int(value)
                                except (json.JSONDecodeError, ValueError):
                                    # If parsing fails, keep the original string
                                    pass

        count = len(results.get("ids", [[]])[0]) if isinstance(
            results.get("ids"), list) and results.get("ids") else 0
        logger.info(
            "[amem.chroma.search] collection=%s k=%s query_chars=%d results=%d elapsed_ms=%.1f",
            self.collection_name,
            k,
            len(query),
            count,
            (time.perf_counter() - started_at) * 1000,
        )
        return results
