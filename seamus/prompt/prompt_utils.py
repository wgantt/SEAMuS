import json

from typing import Any, Dict, List, Literal, Optional

from seamus.constants import DETOKENIZER, ONTOLOGY_PATH


TEST_PROMPT = "Tell me a joke."
TEST_PROMPTS = ("Tell me a joke.", "Tell me a story.", "Tell me a secret.")

PERSONA = (
    "You are an expert intelligence briefer. Your task is to analyze "
    "a specific, important event based ONLY on certain information, "
    "and to compile a concise summary of that event "
    "to be presented to a high-ranked decision-maker."
)
REPORT_TASK_ZERO_SHOT = (
    "Please write a short, accurate summary that is "
    "one sentence long and that is based ONLY on the "
    "provided information. "
    "DO NOT include any extraneous details. "
    "DO NOT use more than one sentence."
)

# Making these the same for now, but could be different in the future
REPORT_TASK_FEW_SHOT = REPORT_TASK_ZERO_SHOT
REPORT_TASK_FEW_SHOT_CORRUPTED = (
    "Please write a short, accurate summary that is "
    "one sentence long and that is focused ONLY on the "
    "event described in the Report Template, and that corrects "
    "any inaccuracies in the Report Template. "
    "DO NOT include any extraneous details. "
    "DO NOT use more than one sentence."
)

COMBINED_TASK_ZERO_SHOT = (
    "Please write a short, accurate summary that is "
    "preferably one sentence long (and no more than two sentences long) "
    "based ONLY on the provided information. "
    "DO NOT include any extraneous details. "
    "TRY to use one sentence and DO NOT use more than two."
)

COMBINED_TASK_FEW_SHOT = COMBINED_TASK_ZERO_SHOT
COMBINED_TASK_FEW_SHOT_CORRUPTED = (
    "Please write a short, accurate summary that is "
    "preferably one sentence long (and no more than two sentences long), "
    "that is focused ONLY on the event described in the Report Template and the Source Template, "
    "and that corrects any inaccuracies in the Report Template and the Source Template. "
    "DO NOT include any extraneous details. "
    "TRY to use one sentence and DO NOT use more than two."
)

EXAMPLES_PREFIX = "Here are a few examples to show you how to complete the task:"
TARGET_PREFIX = "Now here is the target example for you to complete:"


CODA = "Summary:"

EXAMPLE = Dict[str, Any]
PROMPT = Dict[str, str]
TEMPLATE_ORIGIN = Literal["report_template", "source_template"] | None


with open(ONTOLOGY_PATH) as f:
    ONTOLOGY = json.load(f)


def get_frame_type(ex: EXAMPLE) -> tuple[str, str]:
    frame_name = ex["trigger"]["frame"]
    frame_definition = ONTOLOGY[frame_name]["definition"]

    frame_type = f"Situation Type: {frame_name} ({frame_definition})"

    return frame_name, frame_type


def get_schema(schema_origin: TEMPLATE_ORIGIN, frame_name: str) -> str:
    schema_info = {
        role_name: value["definition"]
        for role_name, value in ONTOLOGY[frame_name]["core roles"].items()
    }

    schema_text = format_schema_text(schema_info, schema_origin)

    return schema_text


def format_schema_text(schema_info: dict, schema_origin: TEMPLATE_ORIGIN) -> str:
    match schema_origin:
        case "report_template":
            origin = "Report "
        case "source_template":
            origin = "Source "
        case _:
            origin = ""

    schema = f"{origin}Schema:\n"
    roles = [
        f" - {role_name} ({definition})"
        for role_name, definition in schema_info.items()
    ]
    roles = "\n".join(roles)

    return "\n".join([schema, roles])


def get_template(
    ex: EXAMPLE,
    template_origin: TEMPLATE_ORIGIN,
    frame_name: str,
    include_empty_roles: bool = False,
) -> str:
    template_info = {}

    for role_name, value in ex[template_origin].items():
        role_definition = ONTOLOGY[frame_name]["roles"][role_name]["definition"]
        if value["arguments"]:
            role_fillers = [
                DETOKENIZER.detokenize(arg["tokens"]) for arg in value["arguments"]
            ]
            role_fillers = "; ".join(role_fillers)

            template_info[role_name] = {
                "definition": role_definition,
                "filler": role_fillers,
            }
        elif include_empty_roles:
            template_info[role_name] = {
                "definition": role_definition,
                "filler": "None",
            }

    template_text = format_template_text(template_info, template_origin)

    return template_text


def format_template_text(template_info: dict, template_origin: TEMPLATE_ORIGIN) -> str:
    origin = "Source" if template_origin == "source_template" else "Report"

    template = f"{origin} Template:"
    roles = [
        f" - {role_name} ({value['definition']}): {value['filler']}"
        for role_name, value in template_info.items()
    ]
    roles = "\n".join(roles)

    return "\n".join([template, roles])


def build_report_only_text_only_prompt(ex: EXAMPLE) -> PROMPT:
    instruction = "The Report text below describes a situation."
    text = "Report: " + DETOKENIZER.detokenize(ex["report"])

    return {
        "system_prompt": PERSONA,
        "user_prompt": "\n\n".join([instruction, REPORT_TASK_ZERO_SHOT, text, CODA]),
    }


def build_report_only_event_only_prompt(ex: EXAMPLE) -> PROMPT:
    instruction = (
        "The Report Template below provides specific details about a situation. "
        "Focus ONLY on information relevant to the Situation Type."
    )

    frame_name, frame_type = get_frame_type(ex)
    template = get_template(ex, "report_template", frame_name)

    return {
        "system_prompt": PERSONA,
        "user_prompt": "\n\n".join(
            [instruction, REPORT_TASK_ZERO_SHOT, frame_type, template, CODA]
        ),
    }


def build_report_only_text_with_schema_prompt(ex: EXAMPLE) -> PROMPT:
    instruction = (
        "The Report text below describes a situation. "
        "The Report Schema provides information on what should be included "
        "in the summary if the relevant details are found in the text."
        "Focus ONLY on information relevant to the Situation Type."
    )
    text = "Report: " + DETOKENIZER.detokenize(ex["report"])

    frame_name, frame_type = get_frame_type(ex)
    schema = get_schema("report_template", frame_name)

    return {
        "system_prompt": PERSONA,
        "user_prompt": "\n\n".join(
            [instruction, REPORT_TASK_ZERO_SHOT, frame_type, text, schema, CODA]
        ),
    }


def build_report_only_text_with_event_prompt(ex: EXAMPLE) -> PROMPT:
    instruction = (
        "The Report text below describes a situation. "
        "The Report Template provides specific details about "
        "the same situation. "
        "Focus ONLY on information relevant to the Situation Type."
    )
    text = "Report: " + DETOKENIZER.detokenize(ex["report"])

    frame_name, frame_type = get_frame_type(ex)
    template = get_template(ex, "report_template", frame_name)

    return {
        "system_prompt": PERSONA,
        "user_prompt": "\n\n".join(
            [instruction, REPORT_TASK_ZERO_SHOT, frame_type, text, template, CODA]
        ),
    }


def build_report_only_text_with_event_few_shot_prompt(
    target_ex: EXAMPLE, few_shot_exs: List[EXAMPLE], corrupted: bool = False
) -> PROMPT:
    if corrupted:
        format_instruction = (
            "The Report text below describes a situation. "
            "The Report Template provides specific details about "
            "the same situation. "
            "However, some of the information in the Report Template "
            "could be inaccurate: the list of semantic roles in the Report Template is correct, "
            "but some of the arguments (following the ':') may be incorrect, missing, or duplicated. "
            "Focus ONLY on the CORRECT information that is relevant to the Situation Type. "
        )
        task_instruction = REPORT_TASK_FEW_SHOT_CORRUPTED
    else:
        format_instruction = (
            "The Report text below describes a situation. "
            "The Report Template provides specific details about "
            "the same situation. "
            "Focus ONLY on information relevant to the Situation Type. "
        )
        task_instruction = REPORT_TASK_FEW_SHOT

    user_prompt = [format_instruction, task_instruction, EXAMPLES_PREFIX]

    for i, few_shot_ex in enumerate(few_shot_exs):
        example_id = f"Example {i + 1}\n---------"
        text = "Report: " + DETOKENIZER.detokenize(few_shot_ex["report"])
        frame_name, frame_type = get_frame_type(few_shot_ex)
        template = get_template(few_shot_ex, "report_template", frame_name)
        summary = "Summary: " + DETOKENIZER.detokenize(few_shot_ex["report_summary"])
        user_prompt.extend([example_id, text, frame_type, template, summary])
    few_shot_prompt = "\n\n".join(user_prompt)

    example_id = "Target\n------"
    text = "Report: " + DETOKENIZER.detokenize(target_ex["report"])
    frame_name, frame_type = get_frame_type(target_ex)
    template = get_template(target_ex, "report_template", frame_name)
    target_prompt = "\n\n".join(
        [TARGET_PREFIX, example_id, text, frame_type, template, CODA]
    )

    return {
        "system_prompt": PERSONA,
        "user_prompt": "\n\n\n".join([few_shot_prompt, target_prompt]),
    }


def build_combined_text_only_prompt(
    ex: EXAMPLE, source_override: Optional[str] = None
) -> PROMPT:
    instruction = (
        "The Report text below describes a situation. "
        "The Source text provides additional context about "
        "the same situation."
    )
    report_text = "Report: " + DETOKENIZER.detokenize(ex["report"])
    if source_override is not None:
        source_text = "Source: " + source_override
    else:
        source_text = "Source: " + DETOKENIZER.detokenize(ex["source"])

    return {
        "system_prompt": PERSONA,
        "user_prompt": "\n\n".join(
            [instruction, COMBINED_TASK_ZERO_SHOT, report_text, source_text, CODA]
        ),
    }


def build_combined_event_only_prompt(ex: EXAMPLE) -> PROMPT:
    instruction = (
        "The Report Template below provides specific details about a situation. "
        "The Source Template provides additional details about "
        "the same situation. "
        "Focus ONLY on information relevant to the Situation Type."
    )

    frame_name, frame_type = get_frame_type(ex)
    report_template = get_template(ex, "report_template", frame_name)
    source_template = get_template(ex, "source_template", frame_name)

    return {
        "system_prompt": PERSONA,
        "user_prompt": "\n\n".join(
            [
                instruction,
                COMBINED_TASK_ZERO_SHOT,
                frame_type,
                report_template,
                source_template,
                CODA,
            ]
        ),
    }


def build_combined_text_with_schema_prompt(
    ex: EXAMPLE, source_override: Optional[str] = None
) -> PROMPT:
    instruction = (
        "The Report text below describes a situation. "
        "The Source text provides additional context about the same "
        "situation. The Schema provides information on what should be included "
        "in the summary if the relevant details are found in the text. "
        "Focus ONLY on information relevant to the Situation Type."
    )
    report_text = "Report: " + DETOKENIZER.detokenize(ex["report"])
    if source_override is not None:
        source_text = "Source: " + source_override
    else:
        source_text = "Source: " + DETOKENIZER.detokenize(ex["source"])

    frame_name, frame_type = get_frame_type(ex)
    # both source and report schemas are the same, so we only supply one
    schema = get_schema(None, frame_name)

    return {
        "system_prompt": PERSONA,
        "user_prompt": "\n\n".join(
            [
                instruction,
                COMBINED_TASK_ZERO_SHOT,
                frame_type,
                report_text,
                source_text,
                schema,
                CODA,
            ]
        ),
    }


def build_combined_text_with_event_prompt(
    ex: EXAMPLE, source_override: Optional[str] = None
) -> PROMPT:
    instruction = (
        "The Report text below describes a situation, and "
        "the Report Template provides specific details about "
        "the same situation. "
        "The Source text provides additional context about this situation, "
        "and the Source Template provides additional details. "
        "Focus ONLY on information relevant to the Situation Type."
    )
    report_text = "Report: " + DETOKENIZER.detokenize(ex["report"])
    if source_override is not None:
        source_text = "Source: " + source_override
    else:
        source_text = "Source: " + DETOKENIZER.detokenize(ex["source"])

    frame_name, frame_type = get_frame_type(ex)
    report_template = get_template(ex, "report_template", frame_name)
    source_template = get_template(ex, "source_template", frame_name)

    return {
        "system_prompt": PERSONA,
        "user_prompt": "\n\n".join(
            [
                instruction,
                COMBINED_TASK_ZERO_SHOT,
                frame_type,
                report_text,
                report_template,
                source_text,
                source_template,
                CODA,
            ]
        ),
    }


def build_combined_text_with_event_few_shot_prompt(
    target_ex: EXAMPLE,
    few_shot_exs: List[EXAMPLE],
    source_override: Optional[str] = None,
    few_shot_exs_source_overrides: Optional[List[str]] = None,
    corrupted: bool = False,
) -> None:
    if few_shot_exs_source_overrides is not None:
        assert len(few_shot_exs) == len(
            few_shot_exs_source_overrides
        ), "Few-shot examples and source overrides must be the same length"
    else:
        few_shot_exs_source_overrides = [None] * len(few_shot_exs)

    if corrupted:
        format_instruction = (
            "The Report text below describes a situation, and "
            "the Report Template provides specific details about "
            "the same situation. "
            "The Source text provides additional context about this situation, "
            "and the Source Template provides additional details. "
            "However, some of the information in the Report Template or the Source Template "
            "could be inaccurate: the lists of semantic roles in the Report Template and in the Source Template are correct, "
            "but some of the arguments (following the ':') may be incorrect, missing, or duplicated. "
            "Focus ONLY on the CORRECT information that is relevant to the Situation Type."
        )
        task_instruction = COMBINED_TASK_FEW_SHOT_CORRUPTED
    else:
        format_instruction = (
            "The Report text below describes a situation, and "
            "the Report Template provides specific details about "
            "the same situation. "
            "The Source text provides additional context about this situation, "
            "and the Source Template provides additional details. "
            "Focus ONLY on information relevant to the Situation Type."
        )
        task_instruction = COMBINED_TASK_FEW_SHOT

    EXAMPLES_PREFIX = "Here are a few examples to show you how to complete the task:"
    user_prompt = [format_instruction, task_instruction, EXAMPLES_PREFIX]

    for i, (few_shot_ex, source_override) in enumerate(
        zip(few_shot_exs, few_shot_exs_source_overrides)
    ):
        example_id = f"Example {i + 1}\n---------"
        report_text = "Report: " + DETOKENIZER.detokenize(few_shot_ex["report"])
        if source_override is not None:
            source_text = "Source: " + source_override
        else:
            source_text = "Source: " + DETOKENIZER.detokenize(few_shot_ex["source"])
        frame_name, frame_type = get_frame_type(few_shot_ex)
        report_template = get_template(few_shot_ex, "report_template", frame_name)
        source_template = get_template(few_shot_ex, "source_template", frame_name)
        summary = "Summary: " + DETOKENIZER.detokenize(few_shot_ex["combined_summary"])
        user_prompt.extend(
            [
                example_id,
                report_text,
                source_text,
                frame_type,
                report_template,
                source_template,
                summary,
            ]
        )
    few_shot_prompt = "\n\n".join(user_prompt)

    example_id = "Target\n------"
    report_text = "Report: " + DETOKENIZER.detokenize(target_ex["report"])
    if source_override is not None:
        source_text = "Source: " + source_override
    else:
        source_text = "Source: " + DETOKENIZER.detokenize(target_ex["source"])
    frame_name, frame_type = get_frame_type(target_ex)
    report_template = get_template(target_ex, "report_template", frame_name)
    source_template = get_template(target_ex, "source_template", frame_name)
    target_prompt = "\n\n".join(
        [
            TARGET_PREFIX,
            example_id,
            report_text,
            source_text,
            frame_type,
            report_template,
            source_template,
            CODA,
        ]
    )

    return {
        "system_prompt": PERSONA,
        "user_prompt": "\n\n\n".join([few_shot_prompt, target_prompt]),
    }


if __name__ == "__main__":
    import os
    from seamus.constants import TRAIN_PATH, DEV_PATH, SAVED_CONTEXTS_PATH

    SOURCE_OVERRIDES_TRAIN_PATH = os.path.join(
        SAVED_CONTEXTS_PATH, "bm25_train_concat_7.json"
    )
    SOURCE_OVERRIDES_DEV_PATH = os.path.join(
        SAVED_CONTEXTS_PATH, "bm25_dev_concat_7.json"
    )

    with open(TRAIN_PATH) as f:
        train = json.load(f)
    with open(DEV_PATH) as f:
        dev = json.load(f)
    with open(SOURCE_OVERRIDES_DEV_PATH) as f:
        source_overrides = json.load(f)
    with open(SOURCE_OVERRIDES_TRAIN_PATH) as f:
        source_overrides.update(json.load(f))

    target_ex = dev[0]
    target_ex_override = " ".join(source_overrides[target_ex["instance_id"]])
    few_shot_exs = [
        ex for ex in train if ex["trigger"]["frame"] == target_ex["trigger"]["frame"]
    ]
    few_shot_exs_overrides = [
        " ".join(source_overrides[ex["instance_id"]]) for ex in few_shot_exs
    ]
    ret = build_combined_text_with_event_few_shot_prompt(
        target_ex, few_shot_exs, target_ex_override, few_shot_exs_overrides
    )
    with open("test_prompt_combined.txt", "w") as f:
        f.write(ret["user_prompt"])
