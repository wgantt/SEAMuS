import os
from sacremoses import MosesDetokenizer

# Determine the project root programmatically
# (NOTE: feel free to set this manually as well)
PROJECT_ROOT = os.path.dirname(os.path.abspath("pyproject.toml"))

# Then we can set data paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
TRAIN_PATH = os.path.join(DATA_PATH, "train.json")
DEV_PATH = os.path.join(DATA_PATH, "dev.json")
TEST_PATH = os.path.join(DATA_PATH, "test.json")
ONTOLOGY_PATH = os.path.join(DATA_PATH, "ontology.json")

# Must unzip saved_contexts.zip to use this
SAVED_CONTEXTS_PATH = os.path.join(PROJECT_ROOT, "resources", "saved_contexts")
# Must unzip saved_prompts.zip to use this
SAVED_PROMPTS_PATH = os.path.join(PROJECT_ROOT, "resources", "saved_prompts")

# Some other helpful constants
SPLIT_TO_PATH = {
    "train": TRAIN_PATH,
    "dev": DEV_PATH,
    "test": TEST_PATH,
}
SPLITS = frozenset({"train", "dev", "test"})
TEXT_FIELDS = frozenset({"report", "source", "report_summary", "combined_summary"})

# We use this a lot
DETOKENIZER = MosesDetokenizer(lang="en")
