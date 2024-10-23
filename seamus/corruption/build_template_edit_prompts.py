import click
import json

from sacremoses import MosesDetokenizer
from typing import Any, Dict

from seamus.constants import SPLIT_TO_PATH, ONTOLOGY_PATH

DETOKENIZER = MosesDetokenizer(lang="en")

with open(ONTOLOGY_PATH, "r") as f:
    ONTOLOGY = json.load(f)

# No system prompt for now
SYSTEM_PROMPT = ""

NUM_TO_ORDINAL = {
    0: "first",
    1: "second",
    2: "third",
    3: "fourth",
    4: "fifth",
    5: "sixth",
    6: "seventh",
    7: "eighth",
    8: "ninth",
    9: "tenth",
    10: "eleventh",
    11: "twelfth",
    12: "thirteenth",
}

INSTRUCTIONS = (
    "Instructions:\n\nBelow, I will show you a Document, followed by a JSON dictionary "
    "containing some information about an event that is described in the Document. "
    "The JSON dictionary has semantic roles as keys and includes a role definition and a list of event arguments as values. "
    "I will give you a list of edits that need to be made to the event arguments in the JSON dictionary. "
    "Please make ALL edits to the JSON dictionary EXACTLY as instructed. "
    "Please return the edited JSON dictionary as your response."
)


def format_single_edit(edit: Dict[str, str | int]) -> str:
    if edit["op"] == "ADD":
        return f"Add a new span to the {edit['role']} argument list that is copied EXACTLY from the Document (DO NOT change the spacing, capitalization, or punctuation of the copied span), then randomly shuffle the new argument list."
    elif edit["op"] == "DELETE":
        return f"Delete the {NUM_TO_ORDINAL[edit['arg_num']]} span from the {edit['role']} argument list."
    elif edit["op"] == "REPLACE":
        return f"Replace the {NUM_TO_ORDINAL[edit['arg_num']]} span in the {edit['role']} argument list with a DIFFERENT span copied EXACTLY from the Document. DO NOT change the spacing, capitalization, or punctuation of the copied span."


def format_all_edits(edits: list[Dict[str, str | int]]) -> str:
    header = "Edits:\n\n"
    edits = "\n".join([f"- {format_single_edit(edit)}" for edit in edits])
    return header + edits


def format_template(template: Dict[str, Any], frame: str) -> str:
    for role, role_data in template.items():
        # template[role]["definition"] = ONTOLOGY[frame]["roles"][role]["definition"]
        template[role]["arguments"] = [
            DETOKENIZER.detokenize(arg["tokens"]) for arg in role_data["arguments"]
        ]
    return json.dumps(template, indent=4)


@click.command()
@click.argument("split", type=click.Choice(["train", "dev", "test"]))
@click.argument("edit_file", type=str)
@click.argument("output_file", type=str)
@click.option(
    "--doc",
    type=click.Choice(["report", "source"]),
    help="Document type to build prompts for",
)
def build_prompts(split, edit_file, output_file, doc) -> None:
    with open(SPLIT_TO_PATH[split], "r") as f:
        data = json.load(f)
        data_keys = set([ex["instance_id"] for ex in data])
    with open(edit_file, "r") as f:
        edits = json.load(f)

    assert data_keys == edits.keys(), "Data and edits must have the same keys"

    template_key = "report_template" if doc == "report" else "source_template"
    text_key = "report" if doc == "report" else "source"
    prompts = []
    for ex in data:
        # No edits to be made for this example
        if len(edits[ex["instance_id"]]) == 0:
            continue

        example_id = ex["instance_id"]
        frame = ex["trigger"]["frame"]
        text = f"Document:\n\n{DETOKENIZER.detokenize(ex[text_key])}"
        template = f"JSON dictionary:\n\n{format_template(ex[template_key], frame)}"
        ex_edits = format_all_edits(edits[example_id])
        user_prompt = "\n\n".join([INSTRUCTIONS, text, template, ex_edits])
        prompts.append(
            {
                "instance_id": example_id,
                "user_prompt": user_prompt,
                "system_prompt": SYSTEM_PROMPT,
            }
        )

    with open(output_file, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    build_prompts()
