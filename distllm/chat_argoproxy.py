"""Serves as a chat interface to the RAG datasets built with distllm."""

from __future__ import annotations

import json
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Iterator

import numpy as np
import openai
import requests
from dotenv import load_dotenv
from pydantic import Field
from pydantic import model_validator

from distllm.generate.prompts import IdentityPromptTemplate
from distllm.generate.prompts import IdentityPromptTemplateConfig
from distllm.rag.search import Retriever
from distllm.rag.search import RetrieverConfig
from distllm.utils import BaseConfig

# Load environment variables
load_dotenv()


# -----------------------------------------------------------------------------
# Prompt Templates
# -----------------------------------------------------------------------------
class PromptTemplate:
    """Base class for prompt templates."""

    def preprocess(
        self,
        texts: list[str],
        contexts: list[list[str]],
        scores: list[list[float]],
    ) -> list[str]:
        """Preprocess the texts before sending to the model."""
        raise NotImplementedError('Subclasses should implement this method')


class ConversationPromptTemplate(PromptTemplate):
    """Conversation prompt template for RAG.

    Includes the entire conversation history plus the new user question,
    and optionally the retrieved context.
    """

    def __init__(self, conversation_history: list[tuple[str, str]]):
        # conversation_history is a list of (role, text)
        self.conversation_history = conversation_history

    def preprocess(
        self,
        texts: list[str],
        contexts: list[list[str]] | None = None,
        scores: list[list[float]] | None = None,
    ) -> list[str]:
        """
        Preprocess the texts before sending to the model.

        We assume `texts` has exactly one element: the latest user query.
        We build a single string that contains the entire conversation plus
        the new question. If any retrieval contexts are found, we append them.
        """
        if not texts:
            return ['']  # No user input, return empty prompt.

        # The latest user query:
        user_input = texts[0]

        # Build the conversation string
        conversation_str = ''
        for speaker, text in self.conversation_history:
            conversation_str += f'{speaker}: {text}\n'
        # Add the new user question
        conversation_str += f'User: {user_input}\nAssistant:'

        # Optionally, append retrieved context if it exists
        if contexts and len(contexts) > 0 and len(contexts[0]) > 0:
            # contexts[0] is the top-k retrieval results for this query
            conversation_str += '\n\n[Context from retrieval]\n'
            for doc in contexts[0]:
                conversation_str += f'{doc}\n'

        return [conversation_str]


# -----------------------------------------------------------------------------
# RAG Generator
# -----------------------------------------------------------------------------
class VLLMGeneratorConfig(BaseConfig):
    """Configuration for the vLLM generator."""

    server: str = Field(
        ...,
        description='Cels machine you are running on, e.g, rbdgx1',
    )
    port: int = Field(
        ...,
        description='The port vLLM is listening to.',
    )
    api_key: str = Field(
        ...,
        description='The API key for vLLM server, e.g., CELS',
    )
    model: str = Field(
        ...,
        description='The model that vLLM server is running.',
    )
    temperature: float = Field(
        0.0,
        description='Freeze off the temperature to the keep model grounded.',
    )
    max_tokens: int = Field(
        16384,
        description='The maximum number of tokens to generate.',
    )

    def get_generator(self) -> VLLMGenerator:
        """Get the vLLM generator."""
        generator = VLLMGenerator(
            config=self,
        )
        return generator


class VLLMGenerator:
    """A generator that calls a local or remote vLLM server."""

    def __init__(self, config: VLLMGeneratorConfig) -> None:
        self.server = config.server
        self.port = config.port
        self.api_key = config.api_key
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Send a prompt to the local vLLM server and return the completion.

        Returns a dict with keys:
            text: the generated text
            prompt_tokens: number of tokens in the prompt
            completion_tokens: number of tokens in the completion
            total_latency_s: wall-clock seconds for the request
        """
        temp_to_use = self.temperature if temperature is None else temperature
        tokens_to_use = self.max_tokens if max_tokens is None else max_tokens

        url = f'http://{self.server}.cels.anl.gov:{self.port}/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': temp_to_use,
            'max_completion_tokens': tokens_to_use,
        }

        t0 = time.perf_counter()
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
        )
        total_latency_s = time.perf_counter() - t0

        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        if response.status_code == 200:  # noqa: PLR2004
            resp_json = response.json()
            result = resp_json['choices'][0]['message']['content']
            usage = resp_json.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens')
            completion_tokens = usage.get('completion_tokens')
        else:
            print(f'Error: {response.status_code}')
            result = response.text

        return {
            'text': result,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_latency_s': round(total_latency_s, 4),
        }

    def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream text deltas from the vLLM-compatible OpenAI endpoint."""
        temp_to_use = self.temperature if temperature is None else temperature
        tokens_to_use = self.max_tokens if max_tokens is None else max_tokens

        url = f'http://{self.server}.cels.anl.gov:{self.port}/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': temp_to_use,
            'max_completion_tokens': tokens_to_use,
            'stream': True,
        }

        with requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith('data: '):
                    continue
                data = line[6:]
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get('choices', [])
                if not choices:
                    continue
                delta = choices[0].get('delta', {})
                text_delta = delta.get('content')
                if text_delta:
                    yield text_delta


class ArgoGeneratorConfig(BaseConfig):
    """Configuration for the Argo generator using OpenAI client."""

    model: str = Field(
        default_factory=lambda: os.getenv('MODEL', 'argo:gpt-4o'),
        description='The model name for Argo proxy.',
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            'BASE_URL',
            'http://localhost:56267',
        ),
        description='The base URL for the Argo proxy server.',
    )
    api_key: str = Field(
        'whatever+random',
        description='The API key for Argo proxy (can be any string).',
    )
    temperature: float = Field(
        0.0,
        description='Freeze off the temperature to keep model grounded.',
    )
    max_tokens: int = Field(
        16384,
        description='The maximum number of tokens to generate.',
    )

    def get_generator(self) -> ArgoGenerator:
        """Get the Argo generator."""
        generator = ArgoGenerator(
            config=self,
        )
        return generator


class ArgoGenerator:
    """A generator that calls the Argo proxy using OpenAI client."""

    def __init__(self, config: ArgoGeneratorConfig) -> None:
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        # Initialize OpenAI client with Argo proxy settings
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=f'{config.base_url}/v1',
        )

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Send a prompt to the Argo proxy and return the completion.

        Returns a dict with keys:
            text: the generated text
            prompt_tokens: number of tokens in the prompt
            completion_tokens: number of tokens in the completion
            total_latency_s: wall-clock seconds for the request
        """
        temp_to_use = self.temperature if temperature is None else temperature
        tokens_to_use = self.max_tokens if max_tokens is None else max_tokens

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]

        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp_to_use,
                max_completion_tokens=tokens_to_use,
            )
            total_latency_s = time.perf_counter() - t0
            result = response.choices[0].message.content
            if response.usage is not None:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
        except Exception as e:
            total_latency_s = time.perf_counter() - t0
            print(f'Error calling Argo proxy: {e}')
            result = f'Error: {e!s}'

        return {
            'text': result,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_latency_s': round(total_latency_s, 4),
        }

    def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream text deltas from Argo proxy in OpenAI format."""
        temp_to_use = self.temperature if temperature is None else temperature
        tokens_to_use = self.max_tokens if max_tokens is None else max_tokens

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp_to_use,
                max_completion_tokens=tokens_to_use,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text_delta = delta.content if delta else None
                if text_delta:
                    yield text_delta
        except Exception as e:
            print(f'Error calling Argo proxy (stream): {e}')
            yield f'Error: {e!s}'


# Directly use the OpenAI API, instead of the argo-proxy models.
class OpenAIAPIGeneratorConfig(BaseConfig):
    """
    Configuration for directly calling the OpenAI API (no proxy).
    """

    model: str = Field(
        default_factory=lambda: os.getenv('OPENAI_MODEL', 'gpt-4.1'),
        description='OpenAI model name',
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv('OPENAI_API_KEY'),
        description='OpenAI API key',
    )
    base_url: str | None = Field(
        default_factory=lambda: os.getenv('OPENAI_BASE_URL', None),
        description='Optional: override OpenAI base URL (e.g., Azure)',
    )
    temperature: float = Field(0.0)
    max_tokens: int = Field(16384)

    def get_generator(self) -> 'OpenAIAPIGenerator':
        return OpenAIAPIGenerator(config=self)


class OpenAIAPIGenerator:
    """Generator that hits the public OpenAI API directly."""

    def __init__(self, config: OpenAIAPIGeneratorConfig) -> None:
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        # Validate API key
        if not config.api_key:
            raise ValueError(
                'OpenAI API key is required. Set OPENAI_API_KEY environment variable '
                'or provide it in the config file.',
            )

        # Initialize OpenAI client
        if config.base_url:
            self.client = openai.OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
            )
        else:
            self.client = openai.OpenAI(
                api_key=config.api_key,
            )

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Send a prompt to the OpenAI API and return the completion.

        Returns a dict with keys:
            text: the generated text
            prompt_tokens: number of tokens in the prompt
            completion_tokens: number of tokens in the completion
            total_latency_s: wall-clock seconds for the request
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]

        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            total_latency_s = time.perf_counter() - t0

            if response.usage is not None:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens

            content = response.choices[0].message.content

            # Handle case where content might be None
            if content is None:
                # Check finish reason to understand why content is None
                finish_reason = response.choices[0].finish_reason
                # Debug: print full response structure for diagnosis
                print(
                    f'DEBUG: Response content is None. Finish reason: {finish_reason}'
                )
                print(f'DEBUG: Full response structure: {response}')
                content = (
                    f'[No content returned. Finish reason: {finish_reason}]'
                )

            # Debug: check if content is empty string
            if content == '':
                finish_reason = response.choices[0].finish_reason
                print(
                    f'DEBUG: Response content is empty string. Finish reason: {finish_reason}'
                )

        except Exception as e:
            total_latency_s = time.perf_counter() - t0
            print(f'Error calling OpenAI API: {e}')
            content = f'Error: {e}'

        return {
            'text': content,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_latency_s': round(total_latency_s, 4),
        }

    def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream text deltas directly from the OpenAI API."""
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text_delta = delta.content if delta else None
                if text_delta:
                    yield text_delta
        except Exception as e:
            print(f'Error calling OpenAI API (stream): {e}')
            yield f'Error: {e}'


class RagGenerator:
    """RAG generator for generating responses to queries."""

    def __init__(
        self,
        generator: VLLMGenerator,
        retriever: Retriever | None = None,
        verbose: bool = False,
    ) -> None:
        self.generator = generator
        self.retriever = retriever
        self.verbose = verbose

    def generate(  # noqa: PLR0913
        self,
        texts: str | list[str],
        prompt_template: PromptTemplate = None,
        retrieval_top_k: int = 5,
        retrieval_score_threshold: float = 0.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        debug_retrieval: bool = False,  # New parameter for debugging
    ) -> list[str]:
        """
        Generate responses to the given queries.

        If a retriever is present,
        the retrieved context is appended to the prompt.
        """
        if isinstance(texts, str):
            texts = [texts]  # unify type

        # Use the identity prompt template if none is provided
        if prompt_template is None:
            prompt_template = IdentityPromptTemplate(
                IdentityPromptTemplateConfig(),
            )

        # Default: no context
        contexts, scores = None, None

        # Only retrieve using the new user questions
        if self.retriever is not None:
            results, _ = self.retriever.search(
                texts,  # retrieve on just the latest user query
                top_k=retrieval_top_k,
                score_threshold=retrieval_score_threshold,
            )

            # Debug: Print detailed retrieval information
            if debug_retrieval:
                print('=' * 80)
                print('🔍 RETRIEVAL DEBUG INFORMATION')
                print('=' * 80)
                print(f'Query: {texts[0]}')
                print(f'Retrieved {len(results.total_indices[0])} documents')
                print()

                # Show results structure
                print('📊 Results structure:')
                print(
                    f'  - results.total_indices: {type(results.total_indices)}'
                    f' (length: {len(results.total_indices)})',
                )
                print(
                    f'  - results.total_scores: {type(results.total_scores)}(length: {len(results.total_scores)})',
                )
                print(f'  - First query indices: {results.total_indices[0]}')
                print(f'  - First query scores: {results.total_scores[0]}')
                print()

                # Show what columns are available in the dataset
                print('🗂️ Available dataset columns:')
                dataset_columns = list(
                    self.retriever.faiss_index.dataset.column_names,
                )
                print(f'  - Columns: {dataset_columns}')
                print()

                # Show detailed information for each retrieved document
                for i, (idx, score) in enumerate(
                    zip(results.total_indices[0], results.total_scores[0]),
                ):
                    print(
                        f'📄 Document {i + 1} (Index: {idx}, Score: {score:.4f}):',
                    )

                    # Get all available attributes for this document
                    for column in dataset_columns:
                        value = self.retriever.get([idx], column)[0]
                        if column == 'text':
                            # Show truncated text for readability
                            text_preview = (
                                value[:200] + '...'
                                if len(value) > 200
                                else value
                            )
                            print(f'  - {column}: {text_preview}')
                        elif column == 'embeddings':
                            # Show embedding info without printing the full array
                            print(
                                f'  - {column}: array shape {np.array(value).shape}, dtype {np.array(value).dtype}',
                            )
                        else:
                            print(f'  - {column}: {value}')
                    print()

                print('=' * 80)
                print()

            contexts = [
                self.retriever.get_texts(indices)  # top docs for each query
                for indices in results.total_indices
            ]

            scores = results.total_scores

        # Build the final prompts
        prompts = prompt_template.preprocess(texts, contexts, scores)

        # If the verbose is true in config, print contexts.
        if self.verbose:
            print(contexts[0][0] + '\n\n')

        # We only expect one output per query for now
        # (If multiple texts were passed, we would loop.)
        gen_result = self.generator.generate(
            prompt=prompts[0],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # gen_result is a dict with text, prompt_tokens, completion_tokens,
        # total_latency_s
        metrics = {
            'prompt_tokens': gen_result['prompt_tokens'],
            'completion_tokens': gen_result['completion_tokens'],
            'total_latency_s': gen_result['total_latency_s'],
        }
        # Return response list and metrics
        return [gen_result['text']], metrics

    def generate_stream(  # noqa: PLR0913
        self,
        texts: str | list[str],
        prompt_template: PromptTemplate = None,
        retrieval_top_k: int = 5,
        retrieval_score_threshold: float = 0.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        debug_retrieval: bool = False,
    ) -> Iterator[str]:
        """Generate a streaming response to the given query."""
        if isinstance(texts, str):
            texts = [texts]

        if prompt_template is None:
            prompt_template = IdentityPromptTemplate(
                IdentityPromptTemplateConfig(),
            )

        contexts, scores = None, None
        if self.retriever is not None:
            results, _ = self.retriever.search(
                texts,
                top_k=retrieval_top_k,
                score_threshold=retrieval_score_threshold,
            )

            if debug_retrieval:
                print('=' * 80)
                print('🔍 RETRIEVAL DEBUG INFORMATION')
                print('=' * 80)
                print(f'Query: {texts[0]}')
                print(f'Retrieved {len(results.total_indices[0])} documents')
                print()

            contexts = [
                self.retriever.get_texts(indices)
                for indices in results.total_indices
            ]
            scores = results.total_scores

        prompts = prompt_template.preprocess(texts, contexts, scores)
        if self.verbose and contexts:
            print(contexts[0][0] + '\n\n')

        yield from self.generator.generate_stream(
            prompt=prompts[0],
            temperature=temperature,
            max_tokens=max_tokens,
        )


# -----------------------------------------------------------------------------
# Config Classes
# -----------------------------------------------------------------------------
class RetrievalAugmentedGenerationConfig(BaseConfig):
    """Configuration for the retrieval-augmented generation model."""

    generator_config: (
        VLLMGeneratorConfig | ArgoGeneratorConfig | OpenAIAPIGeneratorConfig
    ) = Field(
        ...,
        description='Settings for the generator (VLLM or Argo)',
    )
    retriever_config: RetrieverConfig | None = Field(
        None,
        description='Settings for the retriever',
    )
    verbose: bool = Field(
        default=False,
        description='Whether to print retrieved contexts in chat.',
    )

    @model_validator(mode='before')
    @classmethod
    def handle_target_field(cls, data: dict) -> dict:
        """Handle _target_ field to instantiate the correct config class."""
        if isinstance(data, dict) and 'generator_config' in data:
            gen_config_data = data['generator_config']

            # If generator_config is a dict with _target_ field, instantiate the correct class
            if (
                isinstance(gen_config_data, dict)
                and '_target_' in gen_config_data
            ):
                target_class_name = gen_config_data.pop('_target_')

                # Map class names to config classes
                config_class_map = {
                    'VLLMGeneratorConfig': VLLMGeneratorConfig,
                    'ArgoGeneratorConfig': ArgoGeneratorConfig,
                    'OpenAIAPIGeneratorConfig': OpenAIAPIGeneratorConfig,
                }

                if target_class_name not in config_class_map:
                    raise ValueError(
                        f'Unknown generator config class: {target_class_name}. '
                        f'Available: {list(config_class_map.keys())}',
                    )

                config_class = config_class_map[target_class_name]

                # Handle environment variable substitution (${env:VAR_NAME})
                processed_data = {}
                for key, value in gen_config_data.items():
                    if (
                        isinstance(value, str)
                        and value.startswith('${env:')
                        and value.endswith('}')
                    ):
                        env_var = value[
                            6:-1
                        ]  # Extract VAR_NAME from ${env:VAR_NAME}
                        processed_data[key] = os.getenv(env_var, '')
                    else:
                        processed_data[key] = value

                # Instantiate the config class
                data['generator_config'] = config_class(**processed_data)

        return data

    def get_rag_model(self) -> RagGenerator:
        """Instantiate the RAG model."""
        # Initialize the generator (either VLLM or Argo)
        if isinstance(self.generator_config, VLLMGeneratorConfig):
            generator = VLLMGenerator(self.generator_config)
        elif isinstance(self.generator_config, ArgoGeneratorConfig):
            generator = ArgoGenerator(self.generator_config)
        elif isinstance(self.generator_config, OpenAIAPIGeneratorConfig):
            generator = OpenAIAPIGenerator(self.generator_config)
        else:
            raise ValueError(
                f'Unsupported generator config type: {type(self.generator_config)}',
            )

        # Initialize the retriever
        retriever = None
        if self.retriever_config is not None:
            retriever = self.retriever_config.get_retriever()

        # Initialize the RAG model
        rag_model = RagGenerator(
            generator=generator,
            retriever=retriever,
            verbose=self.verbose,
        )
        return rag_model


class ChatAppConfig(BaseConfig):
    """Configuration for the evaluation suite."""

    rag_configs: RetrievalAugmentedGenerationConfig = Field(
        ...,
        description='Settings for this RAG application.',
    )
    save_conversation_path: Path = Field(
        ...,
        description='Directory to save the output files.',
    )


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def inspect_retrieval_results(
    retriever: Retriever,
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> dict:
    """
    Utility function to inspect retrieval results without generating responses.

    Args:
        retriever: The retriever instance
        query: The query string
        top_k: Number of documents to retrieve
        score_threshold: Minimum score threshold

    Returns
    -------
        Dictionary containing detailed retrieval information
    """
    results, query_embeddings = retriever.search(
        query=[query],
        top_k=top_k,
        score_threshold=score_threshold,
    )

    # Get dataset columns
    dataset_columns = list(retriever.faiss_index.dataset.column_names)

    # Build detailed results
    detailed_results = {
        'query': query,
        'query_embedding_shape': query_embeddings.shape,
        'num_results': len(results.total_indices[0]),
        'dataset_columns': dataset_columns,
        'retrieved_documents': [],
    }

    # Get detailed info for each retrieved document
    for i, (idx, score) in enumerate(
        zip(results.total_indices[0], results.total_scores[0]),
    ):
        doc_info = {
            'rank': i + 1,
            'dataset_index': idx,
            'score': score,
            'attributes': {},
        }

        # Get all available attributes for this document
        for column in dataset_columns:
            value = retriever.get([idx], column)[0]
            if column == 'embeddings':
                # For embeddings, store shape and dtype info
                doc_info['attributes'][column] = {
                    'shape': np.array(value).shape,
                    'dtype': str(np.array(value).dtype),
                }
            else:
                doc_info['attributes'][column] = value

        detailed_results['retrieved_documents'].append(doc_info)

    return detailed_results


def print_retrieval_inspection(results: dict) -> None:
    """Pretty print the retrieval inspection results."""
    print('=' * 80)
    print('🔍 RETRIEVAL INSPECTION')
    print('=' * 80)
    print(f'Query: {results["query"]}')
    print(f'Query embedding shape: {results["query_embedding_shape"]}')
    print(f'Number of results: {results["num_results"]}')
    print(f'Dataset columns: {results["dataset_columns"]}')
    print()

    for doc in results['retrieved_documents']:
        print(
            f'📄 Document {doc["rank"]} (Index: {doc["dataset_index"]}, Score: {doc["score"]:.4f})',
        )
        for attr_name, attr_value in doc['attributes'].items():
            if attr_name == 'text':
                # Truncate text for readability
                text_preview = (
                    attr_value[:200] + '...'
                    if len(attr_value) > 200
                    else attr_value
                )
                print(f'  - {attr_name}: {text_preview}')
            elif attr_name == 'embeddings':
                print(f'  - {attr_name}: {attr_value}')
            else:
                print(f'  - {attr_name}: {attr_value}')
        print()

    print('=' * 80)


# -----------------------------------------------------------------------------
# Main Chat Function
# -----------------------------------------------------------------------------
def chat_with_model(config: ChatAppConfig) -> None:
    """
    Driver function for the chat application.

    Start an interactive chat session:
    1) Keep track of the conversation history.
    2) If user types 'quit', exit the loop.
    3) Upon exit, save the conversation to a local text file with timestamp.
    4) Use only the latest user input for retrieval, but preserve full context
    in the prompt generation so the assistant can handle follow-up queries.
    """
    rag_model = config.rag_configs.get_rag_model()

    # Keep the conversation as list of (role, text)
    conversation_history: list[tuple[str, str]] = []

    # Print welcome message and available commands
    print('🤖 RAG Chat Interface Started!')
    print('Available commands:')
    print('  - Type your questions normally for chat responses')
    print(
        '  - /inspect <query> - Inspect retrieval results without generating a response',
    )
    print('  - quit - Exit the chat')
    print('=' * 60)

    while True:
        user_input = input('You: ')

        # Check for 'quit' to exit
        if user_input.strip().lower() == 'quit':
            print('Exiting the chat...')
            break

        # Check for inspect command
        if user_input.strip().startswith('/inspect'):
            # Extract the query after /inspect
            query = user_input.strip()[8:].strip()  # Remove '/inspect' prefix
            if not query:
                print('Usage: /inspect <your query>')
                continue

            # Get the retriever from RAG model
            if rag_model.retriever is None:
                print('❌ No retriever configured in this RAG model.')
                continue

            # Inspect retrieval results
            print(
                '🔍 Inspecting retrieval results (no response generation)...',
            )
            results = inspect_retrieval_results(
                retriever=rag_model.retriever,
                query=query,
                top_k=5,
                score_threshold=0.1,
            )
            print_retrieval_inspection(results)
            continue  # Don't add to conversation history

        # Add the user's turn to the conversation
        conversation_history.append(('User', user_input))

        # We create a custom prompt template that includes
        # the entire conversation so far plus the newly retrieved context.
        conversation_template = ConversationPromptTemplate(
            conversation_history,
        )

        # Ask the RAG model to generate a response
        response_list, metrics = rag_model.generate(
            texts=[user_input],  # retrieve only on the new user input
            prompt_template=conversation_template,
            retrieval_top_k=100,
            retrieval_score_threshold=0.2,
            debug_retrieval=True,  # Enable debug mode to see retrieval details
        )
        # There's only one element in response_list
        response = response_list[0]

        # Add the model's response to the conversation
        conversation_history.append(('Assistant', response))

        # Print the model's response
        print(
            f'Model: {response} \n --------------------------------------- \n',
        )

    # -------------------------------------------------------------------------
    # Write conversation history to a file with timestamp.
    # -------------------------------------------------------------------------
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(config.save_conversation_path, exist_ok=True)
    filename = (
        f'{config.save_conversation_path}/conversation_{timestamp_str}.txt'
    )
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for speaker, text in conversation_history:
                f.write(f'{speaker}: {text}\n')
        print(f'Conversation saved to {filename}')
    except Exception as e:
        print(f'Error writing conversation to file: {e}')


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()

    # Load the configuration
    config = ChatAppConfig.from_yaml(args.config)

    # Start the interactive chat
    chat_with_model(config)
