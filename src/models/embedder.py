from sentence_transformers import SentenceTransformer


class EmbeddingModel(SentenceTransformer):
    """
    Wrapper over SentenceTransformer providing a default embedding model.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda",
    ):
        """
        Initializes the embedding model.

        Args:
            model_name (str): HuggingFace model name for embeddings.
            device (str): Computation device ("cuda" or "cpu").
        """
        super().__init__(model_name, device=device)
