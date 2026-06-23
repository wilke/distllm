"""Expose the distllm RAG chat experience through an OpenAI compatible API."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Iterable
from typing import Iterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic import Field

from distllm.chat_argoproxy import ChatAppConfig
from distllm.chat_argoproxy import ConversationPromptTemplate
from distllm.chat_argoproxy import inspect_retrieval_results

# Load environment variables upfront so config paths/API keys can come from .env
load_dotenv()

CONFIG_ENV_VAR = 'DISTLLM_CHAT_CONFIG'
MODEL_NAME = os.getenv('OPENAI_MODEL_NAME', 'distllm-rag')
DEFAULT_TOP_K = int(os.getenv('DISTLLM_CHAT_RETRIEVAL_TOP_K', '20'))
DEFAULT_SCORE_THRESHOLD = float(
    os.getenv('DISTLLM_CHAT_SCORE_THRESHOLD', '0.1'),
)
DEFAULT_DEBUG_RETRIEVAL = os.getenv(
    'DISTLLM_CHAT_DEBUG_RETRIEVAL', '0'
).lower() in {
    '1',
    'true',
    'yes',
}

app = FastAPI(
    title='distllm-rag-server',
    description='OpenAI-compatible wrapper over distllm.chat_argoproxy',
    version='0.1.0',
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

_rag_model = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, str]]


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    top_k: int | None = Field(default=None, gt=0)
    score_threshold: float | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: dict[str, str]
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = 'chat.completion'
    created: int
    model: str
    choices: list[ChatCompletionChoice]


def _load_rag_model():
    """Instantiate (or memoize) the RagGenerator."""
    global _rag_model  # noqa: PLW0603
    if _rag_model is not None:
        return _rag_model

    config_path = os.getenv(CONFIG_ENV_VAR)
    if not config_path:
        raise RuntimeError(
            f'Set {CONFIG_ENV_VAR} to the YAML config used by chat_argoproxy.py',
        )

    config = ChatAppConfig.from_yaml(Path(config_path))
    _rag_model = config.rag_configs.get_rag_model()
    return _rag_model


def _normalize_content(content: str | list[dict[str, str]]) -> str:
    """Support OpenAI multi-part message payloads by coalescing text segments."""
    if isinstance(content, str):
        return content

    texts: list[str] = []
    for block in content:
        if block.get('type') == 'text':
            texts.append(block.get('text', ''))
    return ''.join(texts)


def _build_conversation(
    messages: Iterable[ChatMessage],
) -> tuple[list[tuple[str, str]], str]:
    """Convert OpenAI-style messages into the format ChatArgoproxy expects."""
    conversation_history: list[tuple[str, str]] = []
    latest_user_message = ''

    for msg in messages:
        content = _normalize_content(msg.content).strip()
        if not content:
            # Skip empty content blocks entirely to avoid confusing retrieval
            continue

        role = msg.role.lower()
        if role == 'user':
            speaker = 'User'
            latest_user_message = content
        elif role == 'assistant':
            speaker = 'Assistant'
        elif role == 'system':
            speaker = 'System'
        else:
            speaker = role.title()
        conversation_history.append((speaker, content))

    if not latest_user_message:
        raise HTTPException(
            status_code=400,
            detail='At least one user message is required to run retrieval.',
        )

    return conversation_history, latest_user_message


def _build_response_payload(
    *,
    model: str,
    assistant_message: str,
) -> ChatCompletionResponse:
    choice = ChatCompletionChoice(
        index=0,
        message={'role': 'assistant', 'content': assistant_message},
        finish_reason='stop',
    )
    return ChatCompletionResponse(
        id=f'chatcmpl-{uuid.uuid4()}',
        created=int(time.time()),
        model=model,
        choices=[choice],
    )


def _stream_response(
    *,
    response_id: str,
    created: int,
    model: str,
    text_deltas: Iterable[str],
) -> Iterator[str]:
    """Convert text deltas into OpenAI-compatible SSE chat chunks."""
    role_chunk = {
        'id': response_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': model,
        'choices': [
            {
                'index': 0,
                'delta': {'role': 'assistant'},
                'finish_reason': None,
            },
        ],
    }
    yield f'data: {json.dumps(role_chunk)}\n\n'

    for delta_text in text_deltas:
        chunk = {
            'id': response_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': model,
            'choices': [
                {
                    'index': 0,
                    'delta': {'content': delta_text},
                    'finish_reason': None,
                },
            ],
        }
        yield f'data: {json.dumps(chunk)}\n\n'

    final_chunk = {
        'id': response_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': model,
        'choices': [
            {
                'index': 0,
                'delta': {},
                'finish_reason': 'stop',
            },
        ],
    }
    yield f'data: {json.dumps(final_chunk)}\n\n'
    yield 'data: [DONE]\n\n'


@app.on_event('startup')
def _startup() -> None:
    """Fail fast if the config is missing."""
    _load_rag_model()


@app.get('/health')
def health() -> dict[str, str]:
    """Simple readiness probe."""
    return {'status': 'ok'}


@app.get('/v1/models')
def list_models():
    """OpenAI-compatible model listing so Open WebUI can discover this server."""
    return {
        'object': 'list',
        'data': [
            {
                'id': MODEL_NAME,
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'distllm',
            },
        ],
    }


class DebugQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, gt=0)
    score_threshold: float = 0.1


@app.post('/v1/debug/query')
async def debug_query(request: DebugQueryRequest):
    """Return retrieval results and the assembled prompt without calling the LLM.

    Useful for verifying that (a) retrieval finds relevant documents and
    (b) the prompt is correctly constructed before generation.
    """
    rag_model = _load_rag_model()
    retriever = rag_model.retriever
    if retriever is None:
        raise HTTPException(status_code=500, detail='No retriever configured.')

    detailed = await run_in_threadpool(
        inspect_retrieval_results,
        retriever,
        request.query,
        request.top_k,
        request.score_threshold,
    )

    contexts = [
        [
            doc['attributes'].get('text', '')
            for doc in detailed['retrieved_documents']
        ],
    ]
    scores = [[doc['score'] for doc in detailed['retrieved_documents']]]
    prompt_template = ConversationPromptTemplate([('User', request.query)])
    prompts = prompt_template.preprocess([request.query], contexts, scores)

    return JSONResponse({
        'query': request.query,
        'query_embedding_shape': list(detailed['query_embedding_shape']),
        'num_retrieved': detailed['num_results'],
        'retrieved_documents': [
            {
                'rank': doc['rank'],
                'score': float(doc['score']),
                'text_preview': doc['attributes'].get('text', '')[:500],
            }
            for doc in detailed['retrieved_documents']
        ],
        'assembled_prompt_preview': prompts[0][:2000],
        'prompt_contains_context': '[Context from retrieval]' in prompts[0],
        'prompt_length_chars': len(prompts[0]),
    })


@app.post('/v1/chat/completions')
async def chat_completions(
    request: ChatCompletionRequest,
):
    """OpenAI-compatible chat completions endpoint."""
    rag_model = _load_rag_model()
    conversation_history, latest_user_message = _build_conversation(
        request.messages,
    )
    prompt_template = ConversationPromptTemplate(conversation_history)

    temperature = (
        request.temperature
        if request.temperature is not None
        else getattr(rag_model.generator, 'temperature', 0.0)
    )
    max_tokens = (
        request.max_tokens
        if request.max_tokens is not None
        else getattr(rag_model.generator, 'max_tokens', 1024)
    )

    retrieval_top_k = request.top_k or DEFAULT_TOP_K
    retrieval_score_threshold = request.score_threshold
    if retrieval_score_threshold is None:
        retrieval_score_threshold = DEFAULT_SCORE_THRESHOLD

    if request.stream:
        model_name = request.model or getattr(rag_model.generator, 'model', 'unknown')
        response_id = f'chatcmpl-{uuid.uuid4()}'
        created = int(time.time())

        text_deltas = await run_in_threadpool(
            rag_model.generate_stream,
            [latest_user_message],
            prompt_template,
            retrieval_top_k,
            retrieval_score_threshold,
            max_tokens,
            temperature,
            DEFAULT_DEBUG_RETRIEVAL,
        )
        return StreamingResponse(
            _stream_response(
                response_id=response_id,
                created=created,
                model=model_name,
                text_deltas=text_deltas,
            ),
            media_type='text/event-stream',
        )

    response_list = await run_in_threadpool(
        rag_model.generate,
        [latest_user_message],
        prompt_template,
        retrieval_top_k,
        retrieval_score_threshold,
        max_tokens,
        temperature,
        DEFAULT_DEBUG_RETRIEVAL,
    )
    assistant_response = response_list[0]

    payload = _build_response_payload(
        model=request.model
        or getattr(rag_model.generator, 'model', 'unknown'),
        assistant_message=assistant_response,
    )

    return JSONResponse(payload.model_dump())
