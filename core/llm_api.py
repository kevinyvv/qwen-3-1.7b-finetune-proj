"""One small interface over hosted LLM APIs (Claude, OpenAI / Azure OpenAI).

Used by the no-e rewrite script, and meant for the judge_fn hook in scripts/evaluate.py too:

    client = make_client("claude")
    result = client.complete(system="You grade answers...", messages=[{"role": "user", "content": "..."}])
    result.text

SDKs are imported lazily, so nothing here needs `anthropic` or `openai` installed until a client is built.
"""

import os
from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    model: str # the model that actually answered (can differ from the requested one after a fallback)
    stop_reason: str # the provider's raw reason
    truncated: bool # hit the max_tokens limit
    refused: bool # declined by a safety filter
    input_tokens: int = 0 # uncached input
    cached_input_tokens: int = 0 # input read from the prompt cache (much cheaper)
    cache_write_tokens: int = 0 # input written to the prompt cache (a bit more expensive, Claude only)
    output_tokens: int = 0 # includes thinking / reasoning tokens
    request_id: str | None = None


class FatalAPIError(Exception):
    """Errors that will fail every request (bad key, bad model name, malformed request), so stop the run."""


class ClaudeClient:
    name = "claude"

    def __init__(self, model="claude-opus-5-5", effort="high", max_retries=5):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=max_retries) # key from ANTHROPIC_API_KEY
        self.model = model
        self.effort = effort # low | medium | high | xhigh | max, thinking is always on for opus 5.5

    def complete(self, system, messages, max_tokens=16000):
        anthropic = self._anthropic
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                # the system prompt is the same for every request, so cache it
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                output_config={"effort": self.effort},
                # if a safety filter declines, the API retries on another model instead of failing
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError, anthropic.BadRequestError) as error:
            raise FatalAPIError(str(error)) from error

        usage = response.usage
        return Completion(
            text="".join(block.text for block in response.content if block.type == "text"),
            model=response.model,
            stop_reason=response.stop_reason,
            truncated=response.stop_reason == "max_tokens",
            refused=response.stop_reason == "refusal",
            input_tokens=usage.input_tokens,
            cached_input_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            output_tokens=usage.output_tokens,
            request_id=response._request_id,
        )


class OpenAIClient:
    """OpenAI, or Azure OpenAI when an endpoint is set. Not tested against a live key yet."""

    name = "openai"

    def __init__(self, model, effort=None, max_retries=5, azure_endpoint=None, api_version=None):
        import openai

        self._openai = openai
        azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint:
            # on azure, model is your deployment name, and the api version comes from your azure portal
            self.client = openai.AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_KEY"],
                azure_endpoint=azure_endpoint,
                api_version=api_version or os.environ["AZURE_OPENAI_API_VERSION"],
                max_retries=max_retries,
            )
        else:
            self.client = openai.OpenAI(max_retries=max_retries) # key from OPENAI_API_KEY
        self.model = model
        self.effort = effort # low | medium | high, only for reasoning models, None to leave it out

    def complete(self, system, messages, max_tokens=16000):
        openai = self._openai
        extra = {"reasoning_effort": self.effort} if self.effort else {}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}] + messages,
                max_completion_tokens=max_tokens,
                **extra,
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError,
                openai.NotFoundError, openai.BadRequestError) as error:
            raise FatalAPIError(str(error)) from error

        choice, usage = response.choices[0], response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        return Completion(
            text=choice.message.content or "",
            model=response.model,
            stop_reason=choice.finish_reason,
            truncated=choice.finish_reason == "length",
            refused=choice.finish_reason == "content_filter" or bool(getattr(choice.message, "refusal", None)),
            input_tokens=usage.prompt_tokens - cached,
            cached_input_tokens=cached,
            output_tokens=usage.completion_tokens,
            request_id=getattr(response, "_request_id", None),
        )


CLIENTS = {"claude": ClaudeClient, "openai": OpenAIClient}


def make_client(provider="claude", model=None, effort=None, **kwargs):
    cls = CLIENTS[provider]
    if model is not None:
        kwargs["model"] = model
    if effort is not None:
        kwargs["effort"] = effort
    return cls(**kwargs)
