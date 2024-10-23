import click
import json
import random

from enum import Enum
from seamus.constants import SPLIT_TO_PATH


class EditType(Enum):
    ADD = 0
    DELETE = 1
    REPLACE = 2


EDIT_TYPES = [EditType.ADD, EditType.DELETE, EditType.REPLACE]


@click.command()
@click.argument("output_file", type=str)
@click.option("--doc", "-d", type=click.Choice(["report", "source"]))
@click.option(
    "--split",
    "-s",
    type=str,
    default="test",
    help="The split to corrupt annotations for",
)
@click.option(
    "--p",
    "-p",
    type=float,
    default=0.1,
    help="The probability of making an edit to a particular role",
)
@click.option(
    "--seed",
    type=int,
    default=14620,
    help="The seed to use for the random number generator",
)
def corrupt_annotations(output_file, doc, split, p, seed) -> None:
    random.seed(seed)

    # Load the data
    with open(SPLIT_TO_PATH[split], "r") as f:
        data = json.load(f)

    # Are we editing report templates or source templates?
    template_key = "report_template" if doc == "report" else "source_template"

    # Bookkeeping
    total_roles = 0
    total_role_edits = 0

    # Corrupt the data
    edit_instructions = {}
    for ex in data:
        template = ex[template_key]
        edits = []
        for role, role_data in sorted(template.items()):
            # make an edit to this role with probability p
            rand_val = random.random()
            if rand_val < p:
                # we're making an edit; we now
                # randomly select what kind of edit
                edit_type = random.choice(EDIT_TYPES)

                # for roles without arguments, only additions are possible
                if (
                    edit_type in {EditType.DELETE, EditType.REPLACE}
                    and len(role_data["arguments"]) == 0
                ):
                    edit_type = EditType.ADD

                # take the appropriate edit action based on the edit type
                if edit_type == EditType.ADD:
                    # add a new argument
                    edits.append({"role": role, "op": "ADD"})
                elif edit_type == EditType.DELETE:
                    # delete a randomly selected argument
                    arg_to_delete = random.choice(range(len(role_data["arguments"])))
                    edits.append(
                        {"role": role, "op": "DELETE", "arg_num": arg_to_delete}
                    )
                elif edit_type == EditType.REPLACE:
                    # replace a randomly selected argument
                    arg_to_replace = random.choice(range(len(role_data["arguments"])))
                    edits.append(
                        {"role": role, "op": "REPLACE", "arg_num": arg_to_replace}
                    )
                total_role_edits += 1
            total_roles += 1
        edit_instructions[ex["instance_id"]] = edits

    print(f"Total roles: {total_roles}")
    print(f"Total role edits: {total_role_edits}")
    print(f"Expected edit probability: {p}")
    print(f"Actual edit probability: {total_role_edits / total_roles}")
    with open(output_file, "w") as f:
        json.dump(edit_instructions, f, indent=2)


if __name__ == "__main__":
    corrupt_annotations()
