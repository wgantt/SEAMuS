import asyncio
import click
import json
import os

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm


# We have this just to ensure we're not accidentally
# using a model we don't want to be using (e.g. Opus)
ANTHROPIC_MODELS = frozenset(
    {
        # Opus is supported but we didn't use it for cost reasons
        "claude-3-5-sonnet-20240620",
        "claude-3-sonnet-20240307",
        "claude-3-haiku-20240307",
    }
)

# NOTE: You must set the ANTHROPIC_API_KEY environment variable for this to work.
CLIENT = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=2)


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
def prompt_all(
    model: str,
    prompt_file,
    output_file: str,
    max_tokens: int,
    temperature: float,
) -> None:
    assert model in ANTHROPIC_MODELS, f"Unsupported model: {model}"

    # load examples from JSONL-formatted file
    # (see test_prompt_file.jsonl for an example of the expected fields)
    examples = []
    with open(prompt_file, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    # prompt all examples
    for ex in tqdm(examples, "Prompting..."):
        response = asyncio.run(
            prompt_claude(
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

        # write results to output
        with open(output_file, "a") as f:
            f.write(json.dumps(o) + "\n")


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
async def prompt_claude(
    model: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float = 0.7,
    system_prompt: str = "",
) -> str:
    message = await CLIENT.messages.create(
        # Per the docs, manipulating top-p is not recommended,
        # so we don't have it as a parameter here.
        temperature=temperature,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        model=model,
    )
    answer = message.content[0].text
    return answer


if __name__ == "__main__":
    prompt_all()
