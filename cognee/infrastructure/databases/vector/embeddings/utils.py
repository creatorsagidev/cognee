from typing import List, Union
from cognee.shared.logging_utils import setup_logging

logger = setup_logging()


def is_embeddable(s: str) -> bool:
    """
    Check if input string is embeddable, if not it will be replaced with a dummy value to prevent API errors.
    Empty strings and a string with only a space character are not embeddable.
    If input string contains at least one alphanumeric character, it is considered embeddable.
    """
    if not isinstance(s, str):
        return False
    # Strip whitespace to check if the string is empty or only contains spaces
    s = s.strip()
    if len(s) >= 1:
        return True
    logger.debug(
        "Input string was not embeddable. Skipping embedding and using dummy value instead."
    )
    return False


def _defuse_multimodal_prefix(s: str) -> str:
    """
    Prepend a single space to inputs whose leading characters would trigger
    litellm's Vertex AI / Gemini embedding "multimodal input" heuristic in
    ``litellm/llms/vertex_ai/gemini_embeddings/batch_embed_content_transformation.py``.

    That heuristic auto-routes the entire batch through a multimodal codepath
    that requires file URIs to end in a recognised media extension
    (.png/.jpg/.pdf/etc.). Plain text from a knowledge base often contains
    raw `gs://...` paths, `data:...;base64,...` blobs, or `files/...` Gemini
    File API references that are not actual media URIs — they're just text
    that happens to start with one of those prefixes. Litellm then raises
    ``ValueError: Unable to infer MIME type from GCS URL: ...`` (or the
    file-reference equivalent), aborting the embed call for the whole batch.

    Defusing with a leading space breaks all three startswith() checks while
    leaving the embedding semantically unchanged — the embedding model
    treats leading whitespace as a no-op.
    """
    if not isinstance(s, str):
        return s
    if (
        s.startswith("gs://")
        or s.startswith("files/")
        or (s.startswith("data:") and ";base64," in s)
    ):
        return " " + s
    return s


def sanitize_embedding_text_inputs(text: Union[str, List[str]]) -> List[str]:
    """
    Transform invalid/empty inputs into a safe dummy to prevent API 422 embedding errors while
    keeping list length consistent. Also defuse litellm's multimodal-input
    heuristic for plain-text strings that happen to start with `gs://`,
    `files/`, or `data:...;base64,...`.
    """
    # Ensure we are working with a list
    text_list = [text] if isinstance(text, str) else text
    dummy_value = "."

    return [
        _defuse_multimodal_prefix(t) if is_embeddable(t) else dummy_value
        for t in text_list
    ]


def handle_embedding_response(
    original_texts: Union[List[str], str], embeddings: List[List[float]], dimensions: int
) -> List[List[float]]:
    """
    Compare the original input strings against the results.
    If the original string was 'junk' that was not embeddable, overwrite its vector with zeros.
    """
    if isinstance(original_texts, str):
        original_texts = [original_texts]

    zero_vector = [0.0] * dimensions
    return [
        embeddings[i] if is_embeddable(original_texts[i]) else zero_vector
        for i in range(len(original_texts))
    ]
