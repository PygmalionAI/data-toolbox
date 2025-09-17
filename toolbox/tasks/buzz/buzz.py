import os
import re

from datasets import load_dataset

from ...core import ShareGptHuggingFaceData

class BuzzData(ShareGptHuggingFaceData):
    def __init__(self, split: str = "train") -> None:
        """
        The Buzz-V1.2 dataset from Hive Digital Technologies. This is an extremely large instruction dataset (3M+ examples)
        collated from various sources and somewhat cleaned. It is available on HuggingFace at https://huggingface.co/datasets/H-D-T/Buzz-V1.2
        Args:
            split (str): The split of the dataset to use (e.g. "train", "test", "validation"). Default is "train".
        """
        super().__init__("H-D-T/Buzz-V1.2", split=split)
        # Flags for whether certain types of data are currently excluded from the dataset.
        # This is necessary because we must keep one Buzz task may want to exclude certain sources, while another may not.
        # Because we only have one TrainingData object per set of Tasks, we need to have a mechanism to reload the full
        # dataset if the exclusion criteria change.
        self.excluded_synthetic_data = True
        self.sources_excluded = []

        self.first_load = True # Don't reload on the first load.

    def reload_buzz(
        self,
        exclude_synthetic_data: bool = True,
        sources_to_exclude: list[str] | None = None
    ) -> None:
        """
        Reload the Buzz dataset, applying the specified exclusion criteria.
        Args:
            exclude_synthetic_data (bool): Whether to exclude examples that were synthetically generated. Default is True.
            sources_to_exclude (list[str] | None): A list of RegEx patterns to match against the `source` field. If a match is found,
            examples from that source will be excluded. Note that `exclude_synthetic_data` will stack with whatever patterns are in this field.
        """
        # Combine all RegEx patterns to exclude.
        self.excluded_synthetic_data = exclude_synthetic_data
        combined_patterns = [re.compile(p) for p in sources_to_exclude] or []
        if exclude_synthetic_data:
            combined_patterns.extend(SYNTHETIC_PATTERNS)
        
        # If the exclusion criteria have changed, or if this isn't the first load, reload the dataset.
        if ((self.excluded_synthetic_data != exclude_synthetic_data) or (self.sources_excluded != combined_patterns)) and (not self.first_load):
            self.dataset = load_dataset("H-D-T/Buzz-V1.2", split=self.split)
            self.to_hf_dataset()

        # Filter the dataset based on the combined patterns.
        if combined_patterns:
            self.dataset = self.dataset.filter(
                lambda x: any(p.search(x['source']) for p in combined_patterns),
                num_proc=os.cpu_count(),
                description="Excluding specified sources from Buzz dataset."
            )
        self.sources_excluded = combined_patterns
        self.first_load = False

# RegEx patterns to identify synthetic data.
SYNTHETIC_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"arithmelogic",
    r"[0-9]{1,3}b"
    r"^gpt",
    r"math(?!ematica)",
    r"^open(?!(?:cl|scad))",
    r"^claude",
    r"^gemini",
    r"^mistral",
    r"^palm",
    r"quanta",
    r"synthia",
    r"coder",
    r"-nectar$",
    r"^hotdog",
    r"^know_logic",
    r"^riddler",
    r"^saraswati",
    r"^sodey",
    r"^extractor",
]]
