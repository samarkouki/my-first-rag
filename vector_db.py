

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

from config import SENTENCE_EMBEDDING_MODEL_NAME, COLLECTION_NAME


class VectorDB:

    def __init__(self, persist_path, csv_path=None):
        self.persist_path = persist_path
        self.client = chromadb.PersistentClient(path=persist_path)

        existing_collections = [c.name for c in self.client.list_collections()]

        if COLLECTION_NAME in existing_collections:
            self._reload()
        elif csv_path is not None:
            self._create(csv_path)
        else:
            raise ValueError("No existing database and no CSV provided.")

    def _create(self, csv_path):
        df = pd.read_csv(csv_path)
        ids = df["id"].astype(str).tolist()
        documents = df["text"].tolist()
        metadatas = [
            {"source": source, "categorie": categorie}
            for source, categorie in zip(df["source"], df["categorie"])
        ]

        self.model = SentenceTransformer(SENTENCE_EMBEDDING_MODEL_NAME)
        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"embedding_model": SENTENCE_EMBEDDING_MODEL_NAME},
        )

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    
    def _reload(self):
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )
        model_name = self.collection.metadata["embedding_model"]
        self.model = SentenceTransformer(model_name)

    def retrieve(self, question: str, n_results: int = 3):
        query_embedding = self._embed(question)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        return [
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
                for i in range(len(results["ids"][0]))
            ]

    def _embed(self, text: str):
        return self.model.encode(
            text,
            normalize_embeddings=True,
        )