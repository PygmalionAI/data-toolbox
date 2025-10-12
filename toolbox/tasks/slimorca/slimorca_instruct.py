import os
import re

from datasets import Dataset

from .slimorca import SlimOrcaData
from ...core import Task
from ...utils import generate_sysprompt, gen_dynamic_prompt

class SlimOrcaInstructTask(Task):
    TASK_SHORTHAND = "slimorca_instruct"

    def __init__(
        self,
        dataset: SlimOrcaData,
        use_toolbox_prompts: bool = True
    ) -> None:
        """
        Takes the SlimOrca dataset and does a rather straightforward passthrough on it,
        with the option of using the toolbox-generated system prompts instead of the default "you are an AI assistant"
        prompts.'
        Args:
            dataset (SlimOrcaData): A SlimOrcaData object.
            use_toolbox_prompts (bool): Whether to use the toolbox-generated system prompts instead of the default "you are an AI assistant" prompts. Defaults to True.
        """
        super().__init__(dataset, task_type="instruct")

        self.use_toolbox_prompts = use_toolbox_prompts

    def _process_example(self, example: dict) -> dict:
        conversations = example['conversations']

        if self.use_toolbox_prompts:
            new_sys_prompt = generate_sysprompt(
                conversations=conversations,
                task_name="SlimOrcaInstructTask",
                generic_prompt_type="assistant"
            )
            # If no system prompt, add one.
            if conversations[0]['from'] != 'system':
                conversations.insert(0, {'from': 'system', 'value': new_sys_prompt, 'name': '', 'loss': False})
            else:
                # Otherwise, extract the instructions contained within the system prompt
                # and add it to the original prompt.
                system_prompt = conversations[0]['value']
                system_prompt = ASSISTANT_PATTERN.sub("", system_prompt).strip()
                segue = gen_dynamic_prompt(SEGUES)
                conversations[0]['value'] = f"{new_sys_prompt}\n{segue}\n{system_prompt}".strip()

        return {'conversations': conversations, 'identifier': self._generate_identifier(example)}
    
    def generate_examples(self) -> Dataset:
        # Get rid of the 
        return self.dataset.map(
            self._process_example,
            num_proc=os.cpu_count(),
        )
    
# Should handle most instances of "You are a(n)... assistant"
ASSISTANT_PATTERN = re.compile(r"^You are a.*?\.\s*")
# Segues between the main prompt and additional instructions.
SEGUES = [
    "%{Further|More|Additional|Also take into consideration these|Some more|Here's more|Here are some further} %{instructions|directions|orders for you|things to keep in mind}%{:|.}",
    "%{Also|In addition|Furthermore}%{, | }here %{are|be} some %{more|additional|further} instructions%{:|.|that must be followed}",
]
