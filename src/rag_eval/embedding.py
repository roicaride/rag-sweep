import os

from langchain_core.embeddings import Embeddings


# Instrución para Qwen3-Embedding — necesaria nas queries, non nos documentos.
# Os modelos Qwen3 usan last-token pooling con instruction prefix para retrieval asimétrico.
_QWEN3_TASK = (
    "Given a question about EU copyright law, "
    "retrieve relevant legal passages that answer the question"
)


class _Qwen3APIEmbeddings(Embeddings):
    """HF Inference API para modelos Qwen3-Embedding.

    Engade o instruction prefix ás queries (embed_query) pero non aos
    documentos (embed_documents), tal e como require a arquitectura Qwen3.
    """

    def __init__(self, model_id: str, provider: str):
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        self._api = HuggingFaceEndpointEmbeddings(
            model=model_id,
            task="feature-extraction",
            provider=provider,
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._api.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._api.embed_query(f"Instruct: {_QWEN3_TASK}\nQuery: {text}")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def get_embeddings(model_name: str) -> Embeddings:
    # ── Modelos dispoñibles en HF Inference API ────────────────────────────────
    if model_name == "BAAI/bge-base-en-v1.5":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        return HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-base-en-v1.5",
            task="feature-extraction",
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        )

    if model_name == "Qwen/Qwen3-Embedding-8B":
        # Só dispoñible vía Scaleway (único provider que o aloxa)
        return _Qwen3APIEmbeddings(model_id=model_name, provider="scaleway")

    if model_name == "BAAI/bge-m3":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        return HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-m3",
            task="feature-extraction",
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        )

    if model_name == "Snowflake/snowflake-arctic-embed-l-v2.0":
        # Rexistrado como sentence-similarity en HF pero devolve embeddings
        # correctamente con task=feature-extraction vía backend text-embeddings-inference
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        return HuggingFaceEndpointEmbeddings(
            model="Snowflake/snowflake-arctic-embed-l-v2.0",
            task="feature-extraction",
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        )

    raise NotImplementedError(f"Embedding model '{model_name}' non implementado.")
