import click
import datasets
import evaluate
import json
import logging
import numpy as np
import os
import sys
import transformers

from seamus.constants import DETOKENIZER
from seamus.datasets.dataset import SEAMUS_TRAIN, SEAMUS_DEV, SEAMUS_TEST, gen
from datasets import Dataset
from datasets.formatting.formatting import LazyBatch
from functools import partial
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import PredictionOutput
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

SUMMARIZATION_MODELS = {
    "facebook/bart-large": "seq2seq",
    "facebook/bart-large-cnn": "seq2seq",
    "google/pegasus-large": "seq2seq",
    "google/pegasus-cnn_dailymail": "seq2seq",
    "t5-large": "seq2seq",
}
DEFAULT_MODEL = "facebook/bart-large"

ROUGE = evaluate.load("rouge")
METEOR = evaluate.load("meteor")
BERT_SCORE = evaluate.load("bertscore")

# This is quite a bit longer than
# almost summaries in the dataset
MAX_SUMMARY_LENGTH = 256

# Special role separator token
ROLE_SEP = "<role_sep>"
FALLBACK_ROLE_SEP = "^^"

# A special separator token to use when the model doesn't have one
FALLBACK_SEP = "@@"


@click.command()
@click.argument(
    "output_dir",
    type=str,
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(SUMMARIZATION_MODELS),
    default=DEFAULT_MODEL,
    help="the summarization model to train",
)
@click.option(
    "--task",
    "-t",
    type=click.Choice(["report-only", "combined"]),
    default="report-only",
    help="the summarization task",
)
@click.option(
    "--source-override-path-train",
    type=str,
    default=None,
)
@click.option(
    "--source-override-path-dev",
    type=str,
    default=None,
)
@click.option(
    "--source-override-path-test",
    type=str,
    default=None,
)
@click.option(
    "--num-epochs", type=click.INT, default=30, help="maximum training epochs"
)
@click.option("--patience", type=click.INT, default=5, help="early stopping patience")
@click.option(
    "--per-device-train-batch-size",
    type=click.INT,
    default=2,
    help="training batch size",
)
@click.option(
    "--per-device-eval-batch-size",
    type=click.INT,
    default=2,
    help="evaluation batch size",
)
@click.option(
    "--gradient-accumulation-steps",
    type=click.INT,
    default=4,
    help="gradient accumulation steps",
)
@click.option(
    "--max-doc-len",
    type=click.INT,
    default=None,
    help="maximum number of tokens in the input document (all longer docs will be truncated)",
)
@click.option(
    "--min-new-tokens",
    type=click.INT,
    default=15,
    help="the minimum number of tokens to generate in the summary (for evaluation)",
)
@click.option(
    "--max-new-tokens",
    type=click.INT,
    default=256,
    help="maximum number of tokens to generate in the summary (for evaluation)",
)
@click.option(
    "--num-beams",
    type=click.INT,
    default=5,
    help="number of beams to use for beam search (eval loop only)",
)
@click.option(
    "--gradient-checkpointing",
    is_flag=True,
    default=False,
    help="whether to use gradient checkpointing for training",
)
@click.option(
    "--input-format",
    "-f",
    type=click.Choice(
        [
            "event_only",
            "text_only",
            "text_with_schema",
            "text_with_event",
            "text_with_report_event",
        ]
    ),
    default="text_with_event",
    help="the input format",
)
@click.option(
    "--fp16",
    is_flag=True,
    default=False,
    help="whether to use 16-bit floating point precision",
)
@click.option(
    "--bf16",
    is_flag=True,
    default=False,
    help="whether to use 16-bit bfloat precision",
)
@click.option("--seed", type=int, default=1337, help="the random seed for training")
def train(
    output_dir,
    model,
    task,
    source_override_path_train,
    source_override_path_dev,
    source_override_path_test,
    num_epochs,
    patience,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    gradient_accumulation_steps,
    max_doc_len,
    min_new_tokens,
    max_new_tokens,
    num_beams,
    gradient_checkpointing,
    input_format,
    fp16,
    bf16,
    seed,
) -> None:
    """Train a summarization model on FAMUSSUM

    :param output_dir: the directory where checkpoints will be saved
    :param model: a string indicating the HuggingFace base model to be fine-tuned
    :param task: the summarization task to be performed (`report-only` or `combined`)
    :param source_override_path_train: a path to a JSON file containing text to use
        for the source document in place of the original source document (train split)
    :param num_epochs: the number of epochs for which training will be run
    :param per_device_train_batch_size: the batch size for training
    :param per_device_eval_batch_size: the batch size for evaluation
    :param gradient_accumulation_steps: the number of steps to accumulate gradients
    :param max_doc_len: the maximum length of an input document (documents longer
        than this will be truncated)
    :param min_new_tokens: minimum number of tokens to generate in the summary (for evaluation)
    :param max_new_tokens: maximum number of tokens to generate in the summary (for evaluation)
    :param num_beams: number of beams to use for beam search (eval loop only)
    :param gradient_checkpointing: whether to use gradient checkpointing for training
    :param input_format: how the input should be formatted
    :param fp16: whether to use 16-bit floating point precision
    :param bf16: whether to use 16-bit bfloat precision
    :param seed: the random seed to use
    :return: None
    """
    assert model in SUMMARIZATION_MODELS, f"Unsupported model '{model}'"
    if SUMMARIZATION_MODELS[model] != "seq2seq":
        raise NotImplementedError(
            f"Model type {SUMMARIZATION_MODELS[model]} is not supported!"
        )
    else:
        m = AutoModelForSeq2SeqLM.from_pretrained(model)

    if "t5" in model:
        # default behavior of `from_pretrained` here is apparently incorrect for T5; see below:
        if model in {"t5-small", "t5-base"}:
            model_max_length = 512
        else:
            model_max_length = 1024
        tokenizer = AutoTokenizer.from_pretrained(
            model, model_max_length=model_max_length
        )
        # required by T5
        prefix = ["summarize", ":"]
    elif "bart" in model:
        tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True)
        prefix = []
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)
        prefix = []

    # Load alternative source contexts if provided
    have_source_overrides_train = source_override_path_train is not None
    have_source_overrides_dev = source_override_path_dev is not None
    have_source_overrides_test = source_override_path_test is not None
    assert all(
        [
            have_source_overrides_train,
            have_source_overrides_dev,
            have_source_overrides_test,
        ]
    ) or not any(
        [
            have_source_overrides_train,
            have_source_overrides_dev,
            have_source_overrides_test,
        ]
    ), "If you provide source overrides for one split, you must provide them for all splits"

    if source_override_path_train is not None:
        logger.warning("Using source overrides:")
        logger.warning(f"  - Train: {source_override_path_train}")
        logger.warning(f"  - Dev: {source_override_path_train}")
        logger.warning(f"  - Test: {source_override_path_train}")

        train_data = Dataset.from_generator(
            partial(
                gen,
                split="train",
                source_context_override_path=source_override_path_train,
            )
        )
        dev_data = Dataset.from_generator(
            partial(
                gen, split="dev", source_context_override_path=source_override_path_dev
            )
        )
        test_data = Dataset.from_generator(
            partial(
                gen,
                split="test",
                source_context_override_path=source_override_path_test,
            )
        )
    else:
        # Use original source contexts
        train_data = SEAMUS_TRAIN
        dev_data = SEAMUS_DEV
        test_data = SEAMUS_TEST

    preprocess_fn = partial(
        preprocess,
        tokenizer=tokenizer,
        task=task,
        max_doc_len=max_doc_len,
        input_format=input_format,
        prefix=prefix,
    )
    train_dataset = train_data.map(preprocess_fn, batched=True)
    eval_dataset = dev_data.map(preprocess_fn, batched=True)
    test_dataset = test_data.map(preprocess_fn, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=m)

    # Load model's default generation config, but
    # override with user-provided parameters
    assert m.generation_config is not None
    generation_config = m.generation_config
    generation_config.min_new_tokens = min_new_tokens
    generation_config.max_new_tokens = max_new_tokens
    generation_config.num_beams = num_beams

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=num_epochs,
        output_dir=output_dir,
        metric_for_best_model="rouge1",  # TODO: change this to something better?
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=gradient_checkpointing,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_config=generation_config,
        seed=seed,
        fp16=fp16,
        bf16=bf16,
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    logger.warning(f"(Min, Max) summary length: ({min_new_tokens}, {max_new_tokens})")
    logger.warning(f"Using beam size = {num_beams}")
    metrics = partial(compute_metrics, tokenizer=tokenizer)
    trainer = Seq2SeqTrainer(
        model=m,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    trainer.train()
    prediction_output = trainer.predict(test_dataset)
    save_predictions(
        prediction_output,
        test_dataset,
        tokenizer,
        output_dir,
        task,
        input_format,
        source_override_path_test,
    )


def save_predictions(
    prediction_output: PredictionOutput,
    test_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    model_path: str,
    task: str,
    input_format: str,
    source_override_path: Optional[str] = None,
) -> None:
    preds = np.where(
        prediction_output.predictions != -100,
        prediction_output.predictions,
        tokenizer.pad_token_id,
    )
    decoded_preds = [
        p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)
    ]
    refs = np.where(
        prediction_output.label_ids != -100,
        prediction_output.label_ids,
        tokenizer.pad_token_id,
    )
    decoded_refs = [
        r.strip() for r in tokenizer.batch_decode(refs, skip_special_tokens=True)
    ]
    predictions = []
    assert len(decoded_preds) == len(test_dataset)
    for i, (pred, ref) in enumerate(zip(decoded_preds, decoded_refs)):
        out = {
            "instance_id": test_dataset[i]["instance_id"],
            "task": task,
            "format": input_format,
            "input_str": test_dataset[i]["input_str"],
            "prediction": pred,
            "reference": ref,
        }
        if (
            task == "combined"
            and input_format in {"text_only", "text_with_event", "text_with_schema"}
            and source_override_path is not None
        ):
            out["source_override_path"] = source_override_path

        predictions.append(out)
    with open(os.path.join(model_path, "test_preds.jsonl"), "w") as f:
        f.write("\n".join(list(map(json.dumps, predictions))))

    with open(os.path.join(model_path, "test_preds_pretty.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    metrics = prediction_output.metrics
    with open(os.path.join(model_path, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def preprocess(
    examples: LazyBatch,
    tokenizer: PreTrainedTokenizerBase,
    task: str,
    input_format: str,
    prefix: Optional[str] = [],
    max_doc_len: Optional[int] = None,
) -> BatchEncoding:
    """Preprocess FAMUSSUM data

    Code is taken with light adaptation from the example at the following URL:
    https://huggingface.co/docs/transformers/tasks/summarization#preprocess

    :param examples: the examples to be preprocessed
    :param tokenizer: the tokenizer that will be used to tokenize each example
    :param task: whether to do report-only or combined summarization
    :param input_format: how the input should be formatted
    :param prefix: an optional prefix to prepend to the document text (necessary for
        certain pretrained models, like T5)
    :param max_doc_len: the maximum length of an input document (defaults to the
        maximum model length)
    :param event_only: whether to include only the template in the prompt
    :return: the preprocessed data
    """

    def get_role_sep_token() -> str:
        # a special token for separating different roles
        # we try to use a token that does not overload the sep_token,
        # since we use the sep_token to separate the document and
        # the template. sometimes this isn't possible.
        if tokenizer.additional_special_tokens:
            role_sep_token = tokenizer.additional_special_tokens[0]
        elif tokenizer.sep_token:
            # BART: just fall back to regular sep_token
            role_sep_token = tokenizer.sep_token
        else:
            assert FALLBACK_ROLE_SEP in tokenizer.get_vocab()
            role_sep_token = FALLBACK_ROLE_SEP

        return role_sep_token

    def get_sep_token() -> str:
        # a special token for separating different parts of the input
        if tokenizer.sep_token:
            return tokenizer.sep_token
        elif len(tokenizer.additional_special_tokens) > 1:
            # use one of the additional special tokens that's
            # different from the additional special token we
            # may have used for separating roles
            return tokenizer.additional_special_tokens[1]
        else:
            assert FALLBACK_SEP in tokenizer.get_vocab()
            return FALLBACK_SEP

    def format_doc(doc: List[str], prefix: Optional[List[str]] = []) -> List[str]:
        return prefix + doc

    def format_schema(
        trigger: Dict[str, Any],
        template: Dict[str, List[str]],
        prefix: Optional[List[str]] = None,
    ) -> List[str]:
        role_sep_token = get_role_sep_token()
        # each schema is formatted as follows:
        # <role_sep> frame name <role_sep> role1_name <role_sep> role2_name <role_sep> ...
        schema_str = []
        if prefix:
            schema_str.extend(prefix)
            schema_str.append(role_sep_token)
        schema_str.append(trigger["frame"])
        for role, role_data in sorted(template.items()):
            if role_data is None:
                continue
            schema_str.append(role_sep_token)
            schema_str.append(role)
        return schema_str

    def format_template(
        trigger: Dict[str, Any],
        template: Dict[str, List[str]],
        prefix: Optional[List[str]] = [],
        is_report: bool = True,
    ) -> List[str]:
        role_sep_token = get_role_sep_token()

        # each template is formatted as follows:
        # frame <sep> role1 <sep> role1_arg1 ; role1_arg2 ; ... <sep> roleN <sep> roleN_arg1 ; roleN_arg2 ; ...
        if prefix:
            template_str = prefix
        else:
            template_str = []

        # include frame type information
        template_str.append("Frame")
        template_str.append(role_sep_token)
        template_str.append(trigger["frame"])
        template_str.append(role_sep_token)
        # include information about the trigger (report template only)
        if is_report:
            template_str.append("Trigger")
            template_str.append(role_sep_token)
            template_str.extend(trigger["tokens"])

        # include information about arguments
        for role, role_data in sorted(template.items()):
            if role_data is None:
                continue
            template_str.append(role_sep_token)
            template_str.append(role)
            template_str.append(role_sep_token)
            for i, arg in enumerate(role_data["arguments"]):
                template_str.extend(arg["tokens"])
                # arguments are separated by semicolons
                if i < len(role_data["arguments"]) - 1:
                    template_str.append(";")
        return template_str

    model_input_dim = tokenizer.model_max_length
    assert (
        not max_doc_len or max_doc_len < model_input_dim
    ), f"Maximum document length ({max_doc_len}) > model input dimension ({model_input_dim})"
    if max_doc_len:
        logger.warning(f"Maximum document length: {max_doc_len}")
    else:
        logger.warning(f"Maximum document length: {model_input_dim}")

    # Only the report (and source) templates are provided as input
    if input_format == "event_only":
        # report-only: <sep> frame_name <sep> Trigger <sep> report_trigger <sep> role1 <sep> role1_report_args <sep> role2 ...
        if task == "report-only":
            report_events = [
                prefix + format_template(f, t, ["Event", ":"])
                for f, t in zip(examples["trigger"], examples["report_template"])
            ]
            model_inputs = tokenizer(
                report_events,
                max_length=max_doc_len,
                padding="max_length",
                truncation=True,
                is_split_into_words=True,
            )
        # combined: frame_name <sep> Trigger <sep> report_trigger <sep> role1 <sep> role1_report_args <sep> role2 ...
        #           frame_name <sep> Trigger <sep> report_trigger <sep> role1 <sep> role1_source_args <sep> role2 ...
        else:
            report_events = [
                prefix + format_template(f, t, ["Report", "Event", ":"])
                for f, t in zip(examples["trigger"], examples["report_template"])
            ]
            source_events = [
                format_template(f, t, ["Source", "Event", ":"], is_report=False)
                for f, t in zip(examples["trigger"], examples["source_template"])
            ]
            model_inputs = tokenizer(
                text=report_events,
                text_pair=source_events,
                max_length=max_doc_len,
                padding="max_length",
                # we should never have to truncate, but if
                # we do, truncate only the source template
                truncation="only_second",
                is_split_into_words=True,
            )
    elif input_format == "text_only":
        # report-only: report_text
        if task == "report-only":
            reports = [
                prefix + format_doc(doc, ["Report", ":"]) for doc in examples["report"]
            ]
            model_inputs = tokenizer(
                reports,
                padding="max_length",
                truncation=True,
                max_length=max_doc_len,
                is_split_into_words=True,
            )
        # combined: report_text <sep> source_text
        else:
            reports = [
                prefix + format_doc(doc, ["Report", ":"]) for doc in examples["report"]
            ]
            sources = [format_doc(doc, ["Source", ":"]) for doc in examples["source"]]
            model_inputs = tokenizer(
                text=reports,
                text_pair=sources,
                padding="max_length",
                # Similar to above, prioritize truncating the source over the report
                truncation="only_second",
                max_length=max_doc_len,
                is_split_into_words=True,
            )

    elif input_format == "text_with_schema":
        # report-only: report_text <sep> frame_name <sep> role1 <sep> role2 ...
        if task == "report-only":
            reports = [
                prefix + format_doc(doc, ["Report", ":"]) for doc in examples["report"]
            ]
            schemas = [
                format_schema(f, t, ["Schema", ":"])
                for f, t in zip(examples["trigger"], examples["report_template"])
            ]
            # TODO: verify model has sep token?
            model_inputs = tokenizer(
                text=reports,
                text_pair=schemas,
                padding="max_length",
                # We should never need to truncate in this setting,
                # but if we do, we start with the report text
                truncation="only_first",
                is_split_into_words=True,
            )
        # combined: report_text <sep> source_text <sep> frame_name <sep> role1 <sep> role2 ...
        else:
            reports = [format_doc(doc) for doc in examples["report"]]
            sources = examples["source"]
            texts = [
                prefix + ["Report", ":"] + r + [get_sep_token()] + ["Source", ":"] + s
                for r, s in zip(reports, sources)
            ]

            # Schema is the same for both the report and the source
            schemas = [
                format_schema(f, t, ["Schema", ":"])
                for f, t in zip(examples["trigger"], examples["report_template"])
            ]
            model_inputs = tokenizer(
                text=texts,
                text_pair=schemas,
                padding="max_length",
                # Truncate the texts first (starting with the source), not the schema
                truncation="only_first",
                is_split_into_words=True,
            )
    elif input_format == "text_with_event":
        # report-only: report_text <sep> frame_name <sep> Trigger <sep> trigger <sep> role1 <sep> role1_args <sep> role2 <sep> role2_args ...
        if task == "report-only":
            reports = [
                prefix + format_doc(doc, ["Report", ":"]) for doc in examples["report"]
            ]
            events = [
                format_template(f, t, ["Event", ":"])
                for f, t in zip(examples["trigger"], examples["report_template"])
            ]
            model_inputs = tokenizer(
                text=reports,
                text_pair=events,
                padding="max_length",
                # We should never need to truncate in this setting,
                # but if we do, we start with the report text
                truncation="only_first",
                is_split_into_words=True,
            )
        # combined: report_text <sep> source_text <sep>
        #           frame_name <sep> Trigger <sep> trigger <sep> role1 <sep> role1_args <sep> role2 <sep> role2_args ...
        else:
            reports = [format_doc(doc, ["Report", ":"]) for doc in examples["report"]]
            sources = [format_doc(doc, ["Source", ":"]) for doc in examples["source"]]
            texts = [
                prefix + r + [get_sep_token()] + s for r, s in zip(reports, sources)
            ]

            # Schema is the same for both the report and the source
            report_events = [
                format_template(f, t)
                for f, t in zip(examples["trigger"], examples["report_template"])
            ]
            source_events = [
                format_template(f, t, is_report=False)
                for f, t in zip(examples["trigger"], examples["source_template"])
            ]
            events = [
                ["Report", "Event", ":"]
                + re
                + [get_sep_token()]
                + ["Source", "Event", ":"]
                + se
                for re, se in zip(report_events, source_events)
            ]
            model_inputs = tokenizer(
                text=texts,
                text_pair=events,
                padding="max_length",
                # Truncate the texts first (starting with the source), not the event annotations
                truncation="only_first",
                is_split_into_words=True,
            )
    elif input_format == "text_with_report_event":
        if task == "report-only":
            raise ValueError(
                "Input format 'text_with_report_event' is supported only for the 'combined' summarization task."
            )
        # combined: report <sep> source <sep> frame_name <sep> Trigger <sep> trigger <sep> role1 <sep> role1_report_args <sep> role2 <sep> role2_report_args ...
        # Both report and source texts are included in input...
        reports = [format_doc(doc, ["Report", ":"]) for doc in examples["report"]]
        sources = [format_doc(doc, ["Source", ":"]) for doc in examples["source"]]
        texts = [prefix + r + [get_sep_token()] + s for r, s in zip(reports, sources)]

        # ...But for the events, only the report annotations are included
        events = [
            format_template(f, t)
            for f, t in zip(examples["trigger"], examples["report_template"])
        ]
        model_inputs = tokenizer(
            text=texts,
            text_pair=events,
            padding="max_length",
            truncation="only_first",
            is_split_into_words=True,
        )
    else:
        raise ValueError(f"Unrecognized input format '{input_format}'")

    # save the raw input format to a role in the dataset (used during inference)
    input_str = tokenizer.batch_decode(model_inputs["input_ids"])
    if task == "report-only":
        targets = examples["report_summary"]
    else:
        targets = examples["combined_summary"]

    labels = tokenizer(text_target=targets, truncation=True, is_split_into_words=True)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["input_str"] = input_str
    return model_inputs


def compute_metrics(
    eval_pred: EvalPrediction, tokenizer: PreTrainedTokenizerBase
) -> Dict[str, float]:
    """Compute summarization metrics

    Taken from the example at the following URL:
    https://huggingface.co/docs/transformers/tasks/summarization#evaluate

    :param eval_pred: the prediction to be evaluated
    :param tokenizer: the tokenizer associated with the model
    :return: a dictionary containing various metrics, including loss, rouge-1
        rouge-2, rouge-L, and the mean length of the generated summaries
    """
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # TODO: should we de-case here?
    decoded_preds = [
        p.strip() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)
    ]
    decoded_labels = [
        l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)
    ]

    # automatic metrics include ROUGE-{1,2,L}, METEOR, and BERTscore
    result = ROUGE.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result["meteor"] = METEOR.compute(
        predictions=decoded_preds, references=decoded_labels
    )["meteor"]
    bertscore = BERT_SCORE.compute(
        predictions=decoded_preds, references=decoded_labels, lang="en"
    )
    for m, metric in zip(["p", "r", "f1"], ["precision", "recall", "f1"]):
        result["bertscore_" + m] = np.mean(bertscore[metric])

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["mean_summary_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


if __name__ == "__main__":
    train()
