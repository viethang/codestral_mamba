import json
import logging
import os
from pathlib import Path
from typing import List, Type, Union

from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.base import Tokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer, SpecialTokenPolicy
from mistral_common.tokens.tokenizers.sentencepiece import is_sentencepiece
from mistral_common.tokens.tokenizers.tekken import is_tekken

from mistral_inference.generate import generate_mamba
from mistral_inference.mamba import Mamba
from mistral_inference.transformer import Transformer

MODEL_PATH = "/Data/Mistral_models/Mamba-Codestral-7B-v0.1"
NUM_PIPELINE_RANKS = 1
MAX_TOKENS = 2000
TEMPERATURE = 0


def load_tokenizer(model_path: Path) -> MistralTokenizer:
    tokenizer = [f for f in os.listdir(model_path) if is_tekken(
        model_path / f) or is_sentencepiece(model_path / f)]
    assert (
        len(tokenizer) > 0
    ), f"No tokenizer in {model_path}, place a `tokenizer.model.[v1,v2,v3]` or `tekken.json` file in {model_path}."
    assert (
        len(tokenizer) == 1
    ), f"Multiple tokenizers {', '.join(tokenizer)} found in `model_path`, make sure to only have one tokenizer"

    mistral_tokenizer = MistralTokenizer.from_file(
        str(model_path / tokenizer[0]))

    if isinstance(mistral_tokenizer.instruct_tokenizer.tokenizer, Tekkenizer):
        mistral_tokenizer.instruct_tokenizer.tokenizer.special_token_policy = SpecialTokenPolicy.KEEP

    logging.info(
        f"Loaded tokenizer of type {mistral_tokenizer.instruct_tokenizer.__class__}")

    return mistral_tokenizer


def get_model_cls(model_path: str) -> Union[Type[Mamba], Type[Transformer]]:
    with open(Path(model_path) / "params.json", "r") as f:
        args_dict = json.load(f)

    # type: ignore[return-value]
    return {"mamba": Mamba, "transformer": Transformer}[args_dict.get("model_type", "transformer")]


def generate(prompt: str, history: list[list[str]] = None, model_path=MODEL_PATH, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, num_pipeline_ranks=NUM_PIPELINE_RANKS):
    mistral_tokenizer: MistralTokenizer = load_tokenizer(Path(model_path))
    tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

    model_cls = get_model_cls(model_path)
    model = model_cls.from_folder(
        Path(model_path), max_batch_size=3, num_pipeline_ranks=num_pipeline_ranks)
    messages: List[UserMessage | AssistantMessage] = []
    if history:
        for qa in history:
            messages.extend([UserMessage(content=qa[0]),
                            AssistantMessage(content=qa[1], tool_calls=None)])

    messages += [UserMessage(content=prompt)]
    chat_completion_request = ChatCompletionRequest(messages=messages)

    tokens = mistral_tokenizer.encode_chat_completion(
        chat_completion_request).tokens
    generated_tokens, _ = generate_mamba(  # type: ignore[operator]
        [tokens],
        model,
        max_tokens=max_tokens,
        temperature=temperature,
        eos_id=tokenizer.eos_id,
    )

    answer = tokenizer.decode(generated_tokens[0])

    print(answer)
    return answer
