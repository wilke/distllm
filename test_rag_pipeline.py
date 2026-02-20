"""
Diagnostic script to verify the RAG pipeline end-to-end.

Tests each stage independently:
  (a) Query encoding + retrieval from FAISS
  (b) Prompt construction with retrieved context
  (c) LLM generation using the full prompt

Usage:
  export DISTLLM_CHAT_CONFIG=examples/chat/argo-proxy-chat/enzyme_arvind.yaml
  python test_rag_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

import numpy as np


def separator(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80)


def main() -> None:
    config_path = os.getenv("DISTLLM_CHAT_CONFIG")
    if not config_path:
        sys.exit("ERROR: Set DISTLLM_CHAT_CONFIG to point to your YAML config.")

    # ------------------------------------------------------------------
    # 0. Load the RAG model (same path as the server)
    # ------------------------------------------------------------------
    separator("STEP 0 — Loading RAG model from config")
    from distllm.chat_argoproxy import ChatAppConfig, ConversationPromptTemplate

    config = ChatAppConfig.from_yaml(Path(config_path))
    rag_model = config.rag_configs.get_rag_model()

    print(f"  Generator type : {type(rag_model.generator).__name__}")
    print(f"  Generator model: {getattr(rag_model.generator, 'model', 'N/A')}")
    print(f"  Retriever      : {'YES' if rag_model.retriever else 'NONE'}")

    if rag_model.retriever is None:
        sys.exit("ERROR: No retriever configured — cannot test retrieval.")

    retriever = rag_model.retriever

    # ------------------------------------------------------------------
    # Test queries — pick something specific to your enzyme dataset
    # and something deliberately off-topic to verify retrieval filtering.
    # ------------------------------------------------------------------
    QUERIES = [
        "What enzymes are involved in the degradation of cellulose?",
        "Tell me about the weather in Chicago tomorrow.",   # off-topic control
    ]

    for query in QUERIES:
        separator(f"QUERY: {query}")

        # ------------------------------------------------------------------
        # (a) Query encoding + FAISS retrieval
        # ------------------------------------------------------------------
        separator("STEP A — Retrieval (encode query → search FAISS)")

        top_k = 5
        score_threshold = 0.1

        t0 = time.perf_counter()
        results, query_embeddings = retriever.search(
            query=[query],
            top_k=top_k,
            score_threshold=score_threshold,
        )
        retrieval_time = time.perf_counter() - t0

        n_retrieved = len(results.total_indices[0])
        print(f"\n  Query embedding shape: {query_embeddings.shape}")
        print(f"  Retrieval time       : {retrieval_time:.4f}s")
        print(f"  Documents returned   : {n_retrieved}  (top_k={top_k}, threshold={score_threshold})")

        if n_retrieved == 0:
            print("  ⚠  No documents passed the score threshold — the LLM will rely on its own knowledge.")
        else:
            # Show top retrieved documents
            for i, (idx, score) in enumerate(
                zip(results.total_indices[0], results.total_scores[0])
            ):
                text = retriever.get_texts([idx])[0]
                preview = textwrap.shorten(text, width=200, placeholder=" …")
                print(f"\n  [{i+1}] score={score:.4f}  idx={idx}")
                print(f"      {preview}")

        # ------------------------------------------------------------------
        # (b) Prompt construction — show the actual string sent to the LLM
        # ------------------------------------------------------------------
        separator("STEP B — Prompt construction (conversation + context)")

        # Build contexts the same way RagGenerator.generate() does
        contexts = [
            retriever.get_texts(indices)
            for indices in results.total_indices
        ]
        scores_list = results.total_scores

        conversation_history = [("User", query)]
        prompt_template = ConversationPromptTemplate(conversation_history)
        prompts = prompt_template.preprocess([query], contexts, scores_list)

        full_prompt = prompts[0]

        # Verify that retrieved context actually appears in the prompt
        has_context_marker = "[Context from retrieval]" in full_prompt
        print(f"  Prompt length          : {len(full_prompt)} chars")
        print(f"  Contains context block : {has_context_marker}")

        # Print a truncated preview of the prompt
        if len(full_prompt) > 1500:
            print(f"\n  --- Prompt preview (first 600 chars) ---")
            print(textwrap.indent(full_prompt[:600], "  "))
            print("  ...")
            print(f"  --- Prompt preview (last 400 chars) ---")
            print(textwrap.indent(full_prompt[-400:], "  "))
        else:
            print(f"\n  --- Full prompt ---")
            print(textwrap.indent(full_prompt, "  "))

        # ------------------------------------------------------------------
        # (c) LLM generation — send the prompt and get the answer
        # ------------------------------------------------------------------
        separator("STEP C — LLM generation")

        t0 = time.perf_counter()
        answer = rag_model.generator.generate(
            prompt=full_prompt,
            temperature=0.0,
            max_tokens=512,
        )
        gen_time = time.perf_counter() - t0

        print(f"  Generation time: {gen_time:.2f}s")
        print(f"\n  --- LLM Response ---")
        print(textwrap.indent(answer, "  "))

        # ------------------------------------------------------------------
        # Sanity check: does the answer echo content from retrieved docs?
        # ------------------------------------------------------------------
        if n_retrieved > 0 and contexts[0]:
            first_doc_words = set(contexts[0][0].lower().split()[:30])
            answer_words = set(answer.lower().split())
            overlap = first_doc_words & answer_words
            frac = len(overlap) / max(len(first_doc_words), 1)
            print(f"\n  Word-overlap with top retrieved doc (rough check): {frac:.0%}")

    separator("DONE — Pipeline verification complete")


if __name__ == "__main__":
    main()
