import os

from datasets import Dataset

from .buzz import BuzzData
from ...core import Task
from ...utils import gen_dynamic_prompt

class BuzzInstructTask(Task):
    TASK_SHORTHAND = "buzz_instruct"

    def __init__(
        self,
        dataset: BuzzData,
        exclude_synthetic_data: bool = True,
        sources_to_exclude: list[str] | None = None
    ) -> None:
        """
        Takes the Buzz instruct dataset and rather straightforwardly passes it through as an instruction-following task.
        Args:
            dataset (BuzzData): A BuzzData object.
            task
            exclude_synthetic_data (bool): Whether to exclude examples that were synthetically generated. Default is True.
            sources_to_exclude (list[str] | None): A list of RegEx patterns to match against the `source` field. If a match is found,
            examples from that source will be excluded. Note that `exclude_synthetic_data` and `exclude_flan` will stack with whatever patterns are in this list.
        """
        super().__init__(dataset, task_type="instruct")

        self.exclude_synthetic_data = exclude_synthetic_data
        self.sources_to_exclude = sources_to_exclude

    def _process_example(self, example: dict) -> dict:
        # Different sources require different treatments.
        source = example['source'].lower()
        conversations = example['conversations']

        if "text-to-sql-v1" in source:
            # Text-to-SQL-V1 lacks the actual answer in the response,
            # so this needs to be reframed as a "write a question" task.
            sql_context = gen_dynamic_prompt(SQL_TABLE_PROMPTS)
            sql_question = gen_dynamic_prompt(SQL_TABLE_GEN_QUESTIONS)

            system_prompt = conversations[0]['value']
            # Vary placement of the additional context
            if "above" in system_prompt.lower():
                system_prompt = f"{system_prompt}\n{sql_context}"
            else:
                system_prompt = f"{sql_context}\n{system_prompt}"
            # Handle the situation where no additional context is added
            system_prompt = system_prompt.strip()

            conversations = [
                {'from': 'system', 'value': system_prompt, 'name': '', 'loss': False},
                {'from': 'user', 'value': f"{sql_question}", 'name': '', 'loss': False},
                {'from': 'gpt', 'value': conversations[1]['value'], 'name': '', 'loss': True}
            ]
        elif "cogstack-opengpt-sharegpt" in source:
            # Cogstack requires additional context to indicate that it is Bri'ish.
            conversations.insert(0, {'from': 'system', 'value': gen_dynamic_prompt(COGSTACK_SYS_PROMPT), 'name': '', 'loss': False})
        # TODO(TG): Parse the question and answer from extractor-00000-of-00001 properly.
        # Right now it's just a big mess, and it's most likely synthetic so I'm not gonna touch it right now.
        elif "know_sql" in source:
            current_text = conversations[1]['value']
            conversations[1]['value'] = gen_dynamic_prompt(KNOW_SQL_USER_PROMPT) + "\n" + current_text
    
        for c in conversations:
            # Replace any raw "\n" (they're in there) with actual newlines.
            c['value'] = c['value'].replace("\\n", "\n").strip()
            # Add name and loss fields if they don't exist.
            if 'name' not in c:
                c['name'] = ""
            if 'loss' not in c:
                c['loss'] = c['from'] not in ['human', 'system']

        # Build identifier and return.
        example = self._generate_identifier({'conversations': conversations})
        return example

    def generate_examples(self) -> Dataset:
        """
        Generate training examples from the Buzz dataset.
        """
        # Reload the Buzz dataset with the specified exclusion criteria.
        self.dataset.reload_buzz(exclude_synthetic_data=self.exclude_synthetic_data, sources_to_exclude=self.sources_to_exclude)

        return self.dataset.map(
            self._process_example,
            num_proc=os.cpu_count(),
            remove_columns=['source', 'stack'],
        )

SQL_TABLE_PROMPTS = [
    "%{Below|Below this|below|below this|Above|Above this|above|above this} is %{an|a} %{SQL|sql} %{table creation|table-making|table-creation|command which makes an SQL table|command which makes an sql table}.",
    "%{The text below is|Here below is|Here above is|The text above is} a command that %{makes|creates|generates|initializes} a table %{in SQL|in sql}. This %{Q&A session|conversation} will %{revolve around|be about|focus on|be based on|have to do with} %{this|that|the table|this table|that table}.",
    "A %{question|query} is to follow %{in|with} regards to %{the|an} %{SQL|sql} %{query|command} %{above|below}.",
    "",
]

SQL_TABLE_GEN_QUESTIONS = [
    "What %{is|could be|would be|might be} a %{good|suitable|useful|relevant} %{question to ask|problem that is at hand} %{which|that} %{could|can} be %{answered|solved} %{using|with} %{this|the|that|the provided} %{table|table in the system prompt}?",
    "%{Generate|Craft|Give} %{a question|something to ask|your own question|a relevant question} %{which|that} %{could|can|has the ability to} be %{answered|solved|properly answered} %{using|with} %{this|the|that|the provided} %{table|table in the system prompt|sql table|SQL table}.",
]

COGSTACK_SYS_PROMPT = [
    "%{You are|You're a|Take on the role of|Act as|Become} a %{British|UK|United Kingdom|English} NHS %{doctor|physician|medical professional|practitioner|specialist}. You %{must|will|have to} answer the %{patient|user}'s %{question|questions} and%{, most importantly| also| must} provide an appropriate %{URL|link} to https://nhs.uk as a %{reference|source|citation}.",
]

KNOW_SQL_USER_PROMPT = [
    "%{Answer|Now you will answer|Hi, answer} %{the following|this} question with %{an SQL|a valid SQL} query. {Do not|Don't} %{include|provide} any explanations or additional text, %{just|only} the %{SQL query|query} %{itself|by itself|and nothing else}:",
]
