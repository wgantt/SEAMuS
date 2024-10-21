import asyncio
import click
import json
import os

from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

# We have this to ensure we're not accidentally prompting
# prompting models we don't want to be using (e.g. o1)
OPENAI_MODELS = frozenset({"gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"})

# NOTE: As with Claude, you'll need to set these environment variable
# to use this script. That means you must have an OpenAI API key.
CLIENT = AsyncOpenAI(
    organization=os.environ.get("OPENAI_ORG_ID"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)


@click.command()
@click.argument("model", type=str)
@click.argument("prompt_file", type=str)
@click.argument("output_file", type=str)
@click.option(
    "--max-tokens",
    type=int,
    default=256,
    help="Max tokens to generate (should be set to the same value as the fine-tuned models)",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature (higher values are more creative, lower values are more conservative)",
)
def prompt_all(model, prompt_file, output_file, max_tokens, temperature) -> None:
    assert model in OPENAI_MODELS, f"Unsupported model: {model}"

    # load examples from JSONL-formatted file
    # (see test_prompt_file.jsonl for an example of the expected fields)
    examples = []
    with open(prompt_file, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    # prompt all examples
    for ex in tqdm(examples, "Prompting..."):
        response = asyncio.run(
            prompt_gpt(
                model,
                ex["user_prompt"],
                max_tokens,
                temperature=temperature,
                system_prompt=ex["system_prompt"],
            )
        )
        o = {
            "instance_id": ex["instance_id"],
            "user_prompt": ex["user_prompt"],
            "system_prompt": ex["system_prompt"],
            "response": response,
        }
        # write result to output
        with open(output_file, "a") as f:
            f.write(json.dumps(o) + "\n")


# retry is for avoiding rate limiting errors
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
async def prompt_gpt(
    model: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
) -> str:
    # prompt the model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = await CLIENT.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    prompt_all()
