"""CEAF-REE scoring

Existing implementations of mentions in the Metametric library rely on offset-based
mentions, whereas here we want to use lexical matching. As such, we have to define new
dataclasses based on a lexical notion of a mention.
"""

import click
import editdistance
import json
import metametric.dsl as mm
import os
import re

from collections import defaultdict
from dataclasses import dataclass
from metametric.core.decorator import metametric
from metametric.core.metric_suite import MetricFamily
from metametric.core.reduction import MicroAverage
from metametric.core.normalizers import Precision, Recall, FScore
from pprint import pprint
from spacy.lang.en import English
from typing import Any, Collection, Dict, List, Set, Tuple

from seamus.constants import SPLIT_TO_PATH, DETOKENIZER

nlp = English()
tokenizer = nlp.tokenizer

SINGLE_QUOTE_RE = re.compile(r"' (.*?) '")
DOUBLE_QUOTE_RE = re.compile(r"\" (.*?) \"")


@metametric()
@dataclass(frozen=True, eq=True)
class Mention:
    """A mention, consisting of a single string

    :param mention: the string that defines the mention
    """

    mention: str


@metametric()
@dataclass
class Entity:
    """An entity, consisting of multiple mentions

    :param mentions: the set of mentions that define the entity
    """

    mentions: Collection[Mention]


@metametric()
@dataclass(frozen=True, eq=True)
class EntityArgument:
    """An argument entity commonly used in event extraction.

    :param role: The role satisfied by the argument.
    :param entity: The entity participant satisfying the role.
    """

    role: str
    entity: Entity


@metametric()
@dataclass
class Event:
    """An event commonly used in event extraction.

    :param trigger: The lexical trigger of the event.
    :param args: The arguments of the event.
    """

    trigger: Mention
    args: Collection[EntityArgument]


# Trigger precision, recall, and F1
trigger = mm.normalize["none"](
    mm.from_func(lambda e1, e2: 1.0 if e1.trigger == e2.trigger else 0.0)
)
trigger_metrics = MetricFamily(
    mm.set_matching[Event, "<->", "none"](trigger),
    MicroAverage([Precision(), Recall(), FScore()]),
)


def _mention_edit_distance(e1: Entity, e2: Entity) -> float:
    if len(e1.mentions) != 1 or len(e2.mentions) != 1:
        breakpoint()
    assert len(e1.mentions) == 1 and len(e2.mentions) == 1
    # normalize
    edit_dist = editdistance.eval(e1.mentions[0].mention, e2.mentions[0].mention)
    max_len = max(len(e1.mentions[0].mention), len(e2.mentions[0].mention))
    return (max_len - edit_dist) / max_len


# edit distance-based matching for mentions
phi_edit_distance = mm.normalize["f1"](mm.from_func(_mention_edit_distance))

# the phi-subset similarity function for entities
phi_subset = mm.normalize["none"](
    mm.from_func(
        lambda e1, e2: 1.0 if set(e1.mentions).issubset(set(e2.mentions)) else 0.0
    )
)
# the phi-subset similarity function, generalized to arguments
entity_argument_phi_subset = mm.normalize["none"](
    mm.dataclass[EntityArgument]({"entity": phi_subset, "role": mm.auto[str]})
)

# a phi4 similarity function for entities based on edit distance
entity_argument_phi_edit_distance = mm.normalize["none"](
    mm.dataclass[EntityArgument]({"entity": phi_edit_distance, "role": mm.auto[str]})
)

# The CEAF-REE similarity metric, using phi-subset as the entity/argument similarity
ceaf_ree_phi_subset = mm.dataclass[Event](
    {
        "trigger": mm.auto[Mention],
        "args": mm.set_matching[EntityArgument, "<->", "none"](
            entity_argument_phi_subset
        ),
    }
)


# CEAF-REE w/ subset similarity for *multiple* events
ceaf_ree_phi_subset_set_match = mm.set_matching[Event, "<->", "none"](
    ceaf_ree_phi_subset
)
ceaf_ree_phi_subset_metrics = MetricFamily(
    ceaf_ree_phi_subset_set_match, MicroAverage([Precision(), Recall(), FScore()])
)

# CEAF-REE w/ edit distance similarity on entities
ceaf_ree_phi_edit_distance = mm.dataclass[Event](
    {
        "trigger": mm.auto[Mention],
        "args": mm.set_matching[EntityArgument, "<->", "none"](
            entity_argument_phi_edit_distance
        ),
    }
)
ceaf_ree_phi_edit_distance_set_match = mm.set_matching[Event, "<->", "none"](
    ceaf_ree_phi_edit_distance
)
ceaf_ree_phi_edit_distance_metrics = MetricFamily(
    ceaf_ree_phi_edit_distance_set_match,
    MicroAverage([Precision(), Recall(), FScore()]),
)

# The CEAF-REE similarity metric, using phi4 as the entity similarity
ceaf_ree_phi4_set_match = mm.set_matching[Event, "<->", "none"](mm.auto[Event])
ceaf_ree_phi4_metrics = MetricFamily(
    ceaf_ree_phi4_set_match, MicroAverage([Precision(), Recall(), FScore()])
)


def to_dataclass(trigger: str, args_by_role: Dict[str, List[str]]) -> Event:
    return Event(
        trigger=Mention(trigger),
        args=[
            # we assume a single mention is predicted per argument
            EntityArgument(
                role=role.lower(),
                entity=Entity(mentions=[Mention(arg)]),
            )
            for role, args in args_by_role.items()
            for arg in args
            if arg  # ignore empty args
        ],
    )


def extract_gold_event_structures(
    split: str = "test", task: str = "report_only"
) -> List[List[Event]]:
    with open(SPLIT_TO_PATH[split]) as f:
        data = json.load(f)
    structures = []
    for ex in data:
        event = extract_gold_event_structure(ex, task)
        structures.append([event])
    return structures


def extract_gold_event_structure(ex: Dict[str, Any], task: str) -> Event:
    template_key = (
        "report_summary_template"
        if task == "report_only"
        else "combined_summary_template"
    )
    trigger = ex["trigger"]["frame"]
    args_by_role = {
        role: [DETOKENIZER.detokenize(arg["tokens"]) for arg in role_data["arguments"]]
        for role, role_data in ex[template_key].items()
    }
    return to_dataclass(trigger, args_by_role)


def extract_predicted_event_structures(file: str) -> List[List[Event]]:
    structures = []
    with open(file) as f:
        for line in f:
            ex = json.loads(line)
            event = extract_predicted_event_structure(ex)
            structures.append([event])
    return structures


def extract_predicted_event_structure(ex: Dict[str, Any]) -> Event:
    spans = ex["prediction"]["children"]
    assert len(spans) == 1, ex["meta"]["instance_id"]
    trigger = spans[0]["label"]  # frame
    text = ex["inputs"]["sentence"]
    args_by_role = defaultdict(list)
    for arg in spans[0]["children"]:
        role = arg["label"]
        arg_start_tok, arg_end_tok = arg["span"]
        arg_text = DETOKENIZER.detokenize(text[arg_start_tok : arg_end_tok + 1])
        args_by_role[role].append(arg_text)
    return to_dataclass(trigger, args_by_role)


@click.command()
@click.argument("pred_file", type=str)
@click.option("--output-file", type=str, default=None)
@click.option("--split", type=str, default="test")
@click.option(
    "--task", type=click.Choice(["report_only", "combined"]), default="report_only"
)
@click.option(
    "--metric", type=click.Choice(["subset", "edit_distance"]), default="subset"
)
def score(pred_file: str, output_file, split: str, task: str, metric: str):
    gold = extract_gold_event_structures(split, task)
    pred = extract_predicted_event_structures(pred_file)
    # N.B. 'subset' gives the same results as 'phi4' on FAMuS
    #       because each entity is associated with only one span
    if metric == "subset":
        ceaf_ree = ceaf_ree_phi_subset_metrics.new()
    else:
        ceaf_ree = ceaf_ree_phi_edit_distance_metrics.new()
    ceaf_ree.update_batch(pred, gold)
    scores = ceaf_ree.compute()
    pprint(scores)

    if output_file is not None:
        # if the output file already exists, we assume
        # it already contains some existing metrics, which
        # we want to add to
        prefix = f"{split}_ceaf_ree_{metric}"
        key_to_abbrev = {"precision": "p", "recall": "r", "f1": "f1"}
        out_scores = {}
        for k, v in scores.items():
            out_scores[f"{prefix}_{key_to_abbrev[k]}"] = v

        if os.path.exists(output_file):
            with open(output_file) as f:
                existing_scores = json.load(f)
            existing_scores.update(out_scores)
            with open(output_file, "w") as f:
                json.dump(existing_scores, f, indent=2)
        # otherwise, no existing scores
        else:
            with open(output_file, "w") as f:
                json.dump(out_scores, f, indent=2)


if __name__ == "__main__":
    score()
    # Testing the implementation of the CEAF-REE (subset) metric
    # on examples from the appendix of the following paper:
    # https://aclanthology.org/2021.eacl-main.52/
    # m1 = Mention("Pilmai telephone company building")
    # m2 = Mention("telephone company building")
    # m3 = Mention("telephone company offices")
    # m4 = Mention("water pipes")
    # m5 = Mention("public telephone booth")
    # e1 = Entity([m1, m2, m3])
    # e2 = Entity([m1, m2, m4])

    # g_arg1 = EntityArgument("r", Entity([m1, m2, m3]))
    # g_arg2 = EntityArgument("r", Entity([m4]))
    # g_arg3 = EntityArgument("r", Entity([m5]))
    # g_event = Event(Mention("foo"), [g_arg1, g_arg2, g_arg3])

    # p_arg1 = EntityArgument("r", Entity([m4]))
    # p_arg2 = EntityArgument("r", Entity([m1]))
    # p_arg3 = EntityArgument("r", Entity([m5]))
    # p_arg4 = EntityArgument("r", Entity([m3]))

    # subset = ceaf_ree_phi_subset_metrics.new()
    # phi4 = ceaf_ree_phi4_metrics.new()

    # # case 1
    # p_event_case1 = Event(Mention("foo"), [p_arg1, p_arg2, p_arg3, p_arg4])
    # subset.update_single([p_event_case1], [g_event])
    # phi4.update_single([p_event_case1], [g_event])

    # print("-------")
    # print("case 1:")
    # print("-------")
    # print("-----------")
    # print("phi-subset:")
    # print("-----------")
    # print(subset.compute())
    # print("------")
    # print("phi-4:")
    # print("------")
    # print(phi4.compute())

    # # case 2:
    # p_event_case2 = Event(Mention("foo"), [p_arg2, p_arg1, p_arg3])
    # subset.reset()
    # subset.update_single([p_event_case2], [g_event])
    # phi4.reset()
    # phi4.update_single([p_event_case2], [g_event])

    # print("-------")
    # print("case 2:")
    # print("-------")
    # print("-----------")
    # print("phi-subset:")
    # print("-----------")
    # print(subset.compute())
    # print("------")
    # print("phi-4:")
    # print("------")
    # print(phi4.compute())

    # # case 3:
    # p_event_case3 = Event(Mention("foo"), [p_arg2, p_arg3])
    # subset.reset()
    # subset.update_single([p_event_case3], [g_event])
    # phi4.reset()
    # phi4.update_single([p_event_case3], [g_event])

    # print("-------")
    # print("case 3:")
    # print("-------")
    # print("-----------")
    # print("phi-subset:")
    # print("-----------")
    # print(subset.compute())
    # print("------")
    # print("phi-4:")
    # print("------")
    # print(phi4.compute())
