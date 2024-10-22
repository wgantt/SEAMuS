# SEAMuS

This is the official repository for the **S**ummaries of **E**vents **A**cross **Mu**ltiple **S**entences (SEAMuS) dataset, introduced in the paper *Cross-Document Event-Keyed Summarization*. preprint. (Walden et al., 2024).

## Getting Started

If you are interested only in the SEAMuS dataset, you can find it in `data/seamus.zip` &mdash; no need to install any dependencies. You also do not need to install any dependencies if you just want to access our model predictions and scores, which can be found in `model_outputs/`. However, if you would like to try to replicate or adapt results from the paper, please follow the instructions below.

This package uses [Poetry](https://python-poetry.org/) for dependency management. Poetry strongly encourages installing all dependencies inside a virtual environment. We used [Conda](https://www.anaconda.com/download/) for this purpose, with Python version 3.11.9. To do the same, start by creating your environment:

```
conda create --name seamus python=3.11.9
```

If you do not already have Poetry installed, please follow the Poetry installation instructions [here](https://python-poetry.org/docs/#installation). There are multiple ways to install Poetry, but one easy way is to use `pip` (with your virtual environment activated) to first install `pipx`:

```
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

and then use `pipx` to install Poetry:

```
pipx install poetry
```

**NOTE**: this project relies on PyTorch for model training and inference, and PyTorch can unfortunately be rather annoying to install through Poetry. In the `pyproject.toml` file, we have specified the wheel for PyTorch 2.0.1 corresponding to the CUDA version we used during development (11.7). While we strongly recommend you use PyTorch 2.0.1, you should change the CUDA version if necessary to match what is running on your machine.

Once you have done this, and (again) with your virtual environment active, you can run `poetry install` from the root of this project to install all necessary dependencies.

Next, run the following commands, which is required for summary evaluation code to run properly:
```
python -m spacy download en_core_web_sm
python -c 'import nltk; nltk.download("punkt_tab")'
```

Finally, unzip the resources in the `data/` directory:

```
cd data/
unzip seamus.zip
cd ../resources/
unzip saved_contexts.zip
unzip saved_prompts.zip
unzip supplementary_materials.zip
```

## Small Model Fine-Tuning

To replicate the fine-tuning runs for BART, T5, and PEGASUS, please see the example bash script for the setting(s) you are interested in under `scripts/training`. You will of course have to adapt these to your compute environment, but the commands given in these scripts are exactly those used to fine-tune the models reported on in the paper.

## Small Model Inference

Reference scripts for inference with any of the above models can be found in `scripts/inference`. Note that there are distinct scripts for the report and combined summarization tasks. Note also that running inference on a given file will automatically compute ROUGE-1, ROUGE-2, ROUGE-LCS, and BERTScore results on the outputs. For results with the other metrics, see `seamus/evaluation/`.

## Issues

Please do not hesitate to create an issue if you encounter any difficulties with, or have any questions about, this repository.

## TODOs

These are some things we're still working on:

- [x] Add annotation interface examples and supplementary materials
- [ ] Add FActScore results and instructions for evaluation
- [ ] Add instructions for AlignScore evaluation
- [ ] Add analysis code