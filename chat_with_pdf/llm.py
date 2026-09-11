"""A local Hugging Face chat model (PyTorch) that can call tools.

LangChain's ChatHuggingFace only supports tool calling through hosted endpoints, so this
small wrapper does it for a local model: tool schemas go into the model's chat template,
and the <tool_call>{...}</tool_call> blocks the model writes are parsed back into
LangChain tool calls. Works with Qwen2.5 Instruct models.
"""
import json
import re
import uuid
from threading import Thread
from typing import Any, Iterator, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

import config

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def to_chat_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to the {"role": ..., "content": ...} format of chat templates."""
    if isinstance(message, ToolMessage):
        return {"role": "tool", "content": message.content}
    if isinstance(message, AIMessage):
        chat = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            chat["tool_calls"] = [
                {"type": "function", "function": {"name": call["name"], "arguments": call["args"]}}
                for call in message.tool_calls
            ]
        return chat
    role = "system" if isinstance(message, SystemMessage) else "user"
    return {"role": role, "content": message.content}


def parse_tool_calls(text: str) -> AIMessage:
    """Turn raw model output into an AIMessage, extracting any <tool_call> JSON blocks."""
    tool_calls = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        try:
            call = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict):
            tool_calls.append(
                {"name": call["name"], "args": call["arguments"], "id": f"call_{uuid.uuid4().hex[:12]}"}
            )
    return AIMessage(content=TOOL_CALL_PATTERN.sub("", text).strip(), tool_calls=tool_calls)


class LocalChatModel(BaseChatModel):
    model_id: str
    max_new_tokens: int = config.MAX_NEW_TOKENS
    # Tool calls are parsed from the complete output, so only plain-text replies are streamed
    disable_streaming: bool | Literal["tool_calling"] = "tool_calling"

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        device = config.get_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=config.get_dtype(device)
        ).to(device)
        self._model.eval()

    @property
    def _llm_type(self) -> str:
        return "local-huggingface"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        inputs = self._tokenize(messages, kwargs.get("tools"))
        output_ids = self._model.generate(**inputs, **self._generation_args(kwargs))
        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        return ChatResult(generations=[ChatGeneration(message=parse_tool_calls(text))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        inputs = self._tokenize(messages, kwargs.get("tools"))
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation = {**inputs, **self._generation_args(kwargs), "streamer": streamer}
        thread = Thread(target=self._model.generate, kwargs=generation)
        thread.start()
        for text in streamer:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk
        thread.join()

    def _tokenize(self, messages: list[BaseMessage], tools: list[dict] | None):
        prompt = self._tokenizer.apply_chat_template(
            [to_chat_dict(m) for m in messages], tools=tools, add_generation_prompt=True, tokenize=False
        )
        return self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

    def _generation_args(self, kwargs: dict) -> dict:
        return {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            # Greedy decoding keeps tool calls and yes/no grades stable
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
