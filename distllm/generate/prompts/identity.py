"""Default query prompt: optional user text unchanged, plus RAG context when given."""

from __future__ import annotations

from typing import Literal

from distllm.utils import BaseConfig


class IdentityPromptTemplateConfig(BaseConfig):
    """Configuration for the IdentityPromptTemplate."""

    name: Literal['identity'] = 'identity'  # type: ignore[assignment]


class IdentityPromptTemplate:
    """Pass-through prompt, with optional retrieval context prepended.

    When ``contexts`` is provided and non-empty (RAG), chunks are prepended so
    the generator actually sees retrieved documents. Without this, callers
    using the default template would run retrieval but send only the raw query.
    """

    def __init__(self, config: IdentityPromptTemplateConfig) -> None:
        """Initialize the IdentityPromptTemplate."""
        self.config = config

    def preprocess(
        self,
        text: str | list[str],
        contexts: list[list[str]] | None = None,
        scores: list[list[float]] | None = None,
    ) -> list[str]:
        """Preprocess the text into prompts.

        Parameters
        ----------
        text : str
            The text to format.
        contexts : list[list[str]], optional
            The contexts to include for each text, by default None.
        scores : list[list[float]], optional
            The scores for each context, by default None.

        Returns
        -------
        list[str]
            The formatted prompts.
        """
        if isinstance(text, str):
            text = [text]

        if not contexts:
            return text

        out: list[str] = []
        for i, user_text in enumerate(text):
            docs = contexts[i] if i < len(contexts) else []
            if not docs:
                out.append(user_text)
                continue
            ctx_block = '\n\n'.join(docs)
            out.append(
                '[Context from retrieval]\n\n'
                f'{ctx_block}\n\n---\n\n'
                f'{user_text}',
            )
        return out

    def postprocess(self, responses: list[str]) -> list[str]:
        """Postprocess the responses.

        Parameters
        ----------
        responses : list[str]
            The responses to postprocess.

        Returns
        -------
        list[str]
            The postprocessed responses.
        """
        return responses
