import logging
import os
import re

from datasets import Dataset

from ..core import Filter

LOG = logging.getLogger("NaiveRefusalFilter")

# Adapted from https://github.com/PygmalionAI/data-toolbox/blob/580459d4750733c3c32c65370259e302f96d260f/toolbox/tasks/openorca_instruction_following.py#L76
# but in RegEx for a cleaner implementation.
REFUSAL_PATTERNS = (re.compile(p, re.IGNORECASE) for p in [
    r"\blanguage model\b",
    r"\bas an ai,?\b",
    r"\bdomestic violence\b",
    r"(derogatory|inappropriate|offensive)",
    r"\braci(s[mt]|al)\b",
    r"\bdiscriminat(e|ory|ion)",
    r"\b(extremis|sexis|feminis)[mt]\b",
    r"\bun(acceptable|comfortable|safe)\b",
    r"\binclusiv(e workplace|ity)\b",
    r"\bviolence against\b",
    r"\blgbt\b",
    r"\bmorals\b",
    r"ethic(al|s)\b",
    r"(not |il)?legal(ity)?\b",
    r"\b(hate|harm|hurt)ful\b",
    r"\bit( is|'s) (not|never) (appropriate|acceptable|safe|ok|okay)\b",
    r"\bit( is|'s) important to\b",
    r"real(-| )?world consequences",
    r"\bhate speech",
    r"(not be|is not|isn't) appropriate\b",
    r"\b(can ?not|can't|will not|won't|unable to|not able to) (assist|comply|fulfill|provide|support or promote|engage in)\b",
    r"\bglorify(ing)?\b",
    r"\bsupremac(ist|y|ism)\b",
    r"\bai (model|assistant|chatbot|principles)\b",
    r"(gender|racial|ethnic) (identity|expression|stereotypes?)\b",
    r"\bunderrepresentation\b",
    r"\bsafe spaces?\b",
    r"\bmarginalized\b",
    r"\bstereotyp(es?|ing|ical)\b",
    r"\bconsensual\b",
    r"\b(jeopard|priorit)ize (the safety|user well-being)\b",
    r"\badhere(nce)? to (the law|safety guidelines)\b",
    r"\bpromoting safe(ty)?\b",
    r"\b(un)?(safe|lawful) information\b",
    r"\bpose a (risk|danger) to others\b",
    r"\bactivit(y|ies) that (could|can|will|would) harm",
    r"\bpotentially dangerous\b",
    r"\bagainst my (guidelines|programming|rules)",
    r"\bmaintain user safety\b",
    r"\b(openai|chatgpt)\b",
    r"\bi'm sorry,? but\b",
])

class NaiveRefusalFilter(Filter):
    FILTER_SHORTHAND = "naive_refusal"

    def __init__(self) -> None:
        """
        A Filter which attempts to identify and remove examples where the LLM may refuse to answer a controversial or sensitive question.
        It also tends to cover some basic LLM slop as a byproduct.
        This is a naive keyword search; it makes no effort to understand the context of the keywords. There will likely be a decent chunk of false positives.
        """
        super().__init__()
        
    def _pattern_check(self, example: dict) -> bool:
        """
        Check if any of the refusal patterns match any of the conversation turns in the example.
        Returns True if any pattern matches, False otherwise.
        """
        for c in example['conversations']:
            if c['from'] == 'gpt':
                for pattern in REFUSAL_PATTERNS:
                    if pattern.search(c['value']):
                        return False
        return True

    def __call__(self, dataset: Dataset) -> Dataset:
        """
        Apply the test filter to the dataset.
        """
        orig_dataset_len = len(dataset)
        dataset = dataset.filter(
            self._pattern_check,
            num_proc=os.cpu_count(),
        )
        LOG.info(f"Removed {orig_dataset_len - len(dataset)} examples from dataset.")

        return dataset
