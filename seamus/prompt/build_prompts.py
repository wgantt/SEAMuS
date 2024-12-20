import click
import json
import os

from seamus.constants import SPLIT_TO_PATH, SAVED_CONTEXTS_PATH, SAVED_PROMPTS_PATH
from seamus.prompt.prompt_utils import (
    build_report_only_event_only_prompt,
    build_report_only_text_only_prompt,
    build_report_only_text_with_schema_prompt,
    build_report_only_text_with_event_prompt,
    build_report_only_text_with_event_few_shot_prompt,
    build_combined_event_only_prompt,
    build_combined_text_only_prompt,
    build_combined_text_with_schema_prompt,
    build_combined_text_with_event_prompt,
    build_combined_text_with_event_few_shot_prompt,
)


DEFAULT_SOURCE_CONTEXT_OVERRIDES = {
    "train": os.path.join(SAVED_CONTEXTS_PATH, "bm25_train_concat_7.json"),
    "dev": os.path.join(SAVED_CONTEXTS_PATH, "bm25_dev_concat_7.json"),
    "test": os.path.join(SAVED_CONTEXTS_PATH, "bm25_test_concat_7.json"),
}

# Must unzip saved_prompts.zip to use this!
DEFAULT_OUTPUT_DIR = SAVED_PROMPTS_PATH

# We carried out our corrupted experiments only in the unablated setting
SUPPORTED_FEW_SHOT_FORMATS = frozenset({"text-with-event"})
SUPPORTED_CORRUPTED_FORMATS = frozenset({"text-with-event"})


@click.command()
@click.option("--input-file", "-i", default=SPLIT_TO_PATH["test"], type=str)
@click.option("--output-file", "-o", default=None, type=str)
@click.option(
    "--task",
    "-t",
    type=click.Choice(["report-only", "combined"]),
    default="report-only",
    help="which summarization task we're doing",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(
        ["text-only", "event-only", "text-with-event", "text-with-schema"]
    ),
    default="text-only",
    help="format of the prompt",
)
@click.option(
    "--split",
    "-s",
    type=str,
    default="test",
    help="which split we're generating prompts for",
)
@click.option(
    "--use-source-overrides",
    is_flag=True,
    default=False,
    help="whether to use source context overrides",
)
@click.option(
    "--source-override-path",
    "-o",
    type=str,
    default=None,
    help="path to source overrides file",
)
@click.option(
    "--few-shot-source-override-path",
    type=str,
    default=None,
    help="path to source overrides file for few-shot examples",
)
@click.option(
    "--few-shot-data-file",
    type=str,
    default=SPLIT_TO_PATH["train"],
    help="path to file to be used for few-shot example selection",
)
@click.option(
    "--do-few-shot",
    is_flag=True,
    default=False,
    help="whether to generate few-shot prompts",
)
@click.option(
    "--corrupted",
    is_flag=True,
    default=False,
    help="whether to generate prompts for corrupted data",
)
def build_prompts(
    input_file,
    output_file,
    task,
    format,
    split,
    use_source_overrides,
    source_override_path,
    few_shot_source_override_path,
    few_shot_data_file,
    do_few_shot,
    corrupted,
) -> None:

    # Are we building few-shot prompts?
    # We always use the train split to obtain our examples
    if do_few_shot:
        assert format in SUPPORTED_FEW_SHOT_FORMATS, "Unsupported few-shot format"
        print(f"Using the following file for few-shot examples: {few_shot_data_file}")
        with open(few_shot_data_file) as f:
            few_shot_data = json.load(f)
    else:
        few_shot_data = []

    if corrupted:
        assert format in SUPPORTED_CORRUPTED_FORMATS, "Unsupported corrupted format"
        assert do_few_shot, "Using corrupted data currently requires few-shot examples"

    # Figure out which prompt building function to use
    if task == "report-only":
        if format == "event-only":
            prompt_builder = build_report_only_event_only_prompt
        elif format == "text-only":
            prompt_builder = build_report_only_text_only_prompt
        elif format == "text-with-event":
            if do_few_shot:
                prompt_builder = build_report_only_text_with_event_few_shot_prompt
            else:
                prompt_builder = build_report_only_text_with_event_prompt
        else:  # text-with-schema
            prompt_builder = build_report_only_text_with_schema_prompt
    else:  # combined
        if format == "event-only":
            prompt_builder = build_combined_event_only_prompt
        elif format == "text-only":
            prompt_builder = build_combined_text_only_prompt
        elif format == "text-with-event":
            if do_few_shot:
                prompt_builder = build_combined_text_with_event_few_shot_prompt
            else:
                prompt_builder = build_combined_text_with_event_prompt
        else:  # text-with-schema
            prompt_builder = build_combined_text_with_schema_prompt

    # Override the full source context with the retrieved context (if necessary)
    if use_source_overrides:
        if source_override_path is None:
            print("Using default source context overrides")
            source_override_path = DEFAULT_SOURCE_CONTEXT_OVERRIDES[split]

        with open(source_override_path) as f:
            source_contexts = json.load(f)
            for k, v in source_contexts.items():
                # the v's here are just lists of sentences
                # (strings) which we need to concatenate here
                source_contexts[k] = " ".join(v)

        if do_few_shot and few_shot_source_override_path is None:
            print("Using default source context overrides for few-shot examples")
            few_shot_source_override_path = DEFAULT_SOURCE_CONTEXT_OVERRIDES["train"]

        if few_shot_source_override_path is not None:
            with open(few_shot_source_override_path) as f:
                few_shot_source_contexts = json.load(f)
                for k, v in few_shot_source_contexts.items():
                    few_shot_source_contexts[k] = " ".join(v)

    else:
        source_contexts = None
        few_shot_source_contexts = None

    # Load the data
    with open(input_file) as f:
        data = json.load(f)

    # Construct prompts
    all_prompts = []
    for ex in data:

        # the prompt building function can take
        # some additional arguments (see below)
        extra_args = {}
        if corrupted:
            extra_args["corrupted"] = True

        # add few-shot examples if applicable
        if do_few_shot:
            # right now, we just select as examples all those
            # from the train split with matching frame type
            extra_args["few_shot_exs"] = [
                few_shot_ex
                for few_shot_ex in few_shot_data
                if few_shot_ex["trigger"]["frame"] == ex["trigger"]["frame"]
            ]

        # add source overrides
        if task == "combined" and use_source_overrides and source_contexts is not None:
            extra_args["source_override"] = source_contexts[ex["instance_id"]]

            # add source overrides for few-shot examples as well, if applicable
            if do_few_shot:
                extra_args["few_shot_exs_source_overrides"] = [
                    few_shot_source_contexts[few_shot_ex["instance_id"]]
                    for few_shot_ex in extra_args["few_shot_exs"]
                ]

        prompts = prompt_builder(ex, **extra_args)
        all_prompts.append(
            {
                "instance_id": ex["instance_id"],
                "task": task,
                "format": format,
                "system_prompt": prompts["system_prompt"],
                "user_prompt": prompts["user_prompt"],
            }
        )

    if output_file is None:
        if (
            task == "combined"
            and source_override_path is not None
            and use_source_overrides
        ):
            source_override_name = os.path.basename(source_override_path).split(".")[0]
            if do_few_shot:
                output_file = (
                    DEFAULT_OUTPUT_DIR
                    + f"{task}_{format}_{split}_{source_override_name}_few-shot.jsonl"
                )
            else:
                output_file = (
                    DEFAULT_OUTPUT_DIR
                    + f"{task}_{format}_{split}_{source_override_name}_0-shot.jsonl"
                )
        else:
            if do_few_shot:
                output_file = (
                    DEFAULT_OUTPUT_DIR + f"{task}_{format}_{split}_few-shot.jsonl"
                )
            else:
                output_file = (
                    DEFAULT_OUTPUT_DIR + f"{task}_{format}_{split}_0-shot.jsonl"
                )

    # Dump to output (jsonlines)
    with open(output_file, "w") as f:
        for p in all_prompts:
            f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    build_prompts()
