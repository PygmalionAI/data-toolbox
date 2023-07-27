# Much of this taken from dataprepare.py in the LIMARP, thanks anon
# If it ain't broke, don't fix it!
import logging
import random
import re
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.limarp import LimaRpDataset, LimaRpEntry
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)

class LimaRpRoleplayTask(BaseTask):
    def __iter__(self) -> t.Generator[Episode, None, None]:
        for entry in LimaRpDataset():
            turns: list[Turn] = []
            # Format the system prompt first.
            system_prompt = random.choice(SYSTEM_PROMPTS)
            # Fix it up and append it as the first turn
            system_prompt = _fix_punctuation(_substitute_elements(system_prompt))
            turns.append(Turn(
                utterance=system_prompt,
                kind=TurnKind.SYSTEM
            ))

            # Now for the rest
            for msg in entry.conversation:
                cleaned_msg = _fix_punctuation(_substitute_elements(msg['text']))
                turns.append(Turn(
                    utterance=cleaned_msg,
                    kind=TurnKind.MODEL if msg['name'] == "<FIRST>" else TurnKind.USER
                ))

            # TODO(TG): Run some numbers here like in the original LIMARP script
            # to deal with chats above token limit. For now, they get caught by a TurnTooLargeError
            # in build_data.py, so it's not too urgent of a priority.

            # Yield the episode
            yield Episode(
                turns=turns,
                identifier=f"limarp-{entry.forum}-{entry.thread_id}"
            )

def _substitute_elements(input_string: str, entry: LimaRpEntry) -> str:
    '''
    Replace blank/template fields with data from the particular entry.
    '''
    # Users
    input_string = input_string.replace("<SECOND>", "{{user}}")
    input_string = input_string.replace("<FIRST>", entry.names['<FIRST>'])
    # System prompts
    input_string = input_string.replace("<CHAR>", entry.names['<FIRST>'])
    input_string = input_string.replace("<PERSONA>", entry.personas['<FIRST>'])
    input_string = input_string.replace("<SCENARIO>", entry.scenario)

    return input_string

def _fix_punctuation(input_string: str) -> str:
    '''
    Replace fancy/incorrect punctuation with simpler/correct one
    TODO: more effective regexes, options for controlling what should be changed.
    '''

    # Fix excessive horizontal whitespace. This should go before everything else.
    input_string = re.sub(r' {2,}', ' ', input_string)
    
    # General puncuation fixes
    input_string = input_string.replace(' !', '!')
    input_string = input_string.replace(' ?', '?')
    input_string = input_string.replace('’', "'")
    input_string = input_string.replace('‘', "'")
    input_string = input_string.replace('“', '"')
    input_string = input_string.replace('”', '"')
    input_string = input_string.replace('…', '...')
    
    # Replace em-dash surrogates `---` in the source files with actual
    # em-dashes, since some people apparently dislike them.
    input_string = input_string.replace('---', '—') 
    
    # Fix incorrect ellipsis. This should preferably be fixed in the
    # source files themselves
    input_string = re.sub(r'(\w)\.{2,8}(\w)', r'\1... \2', input_string)
    input_string = re.sub(r'(\w)\.{3,8}', r'\1...', input_string)
    
    return input_string

_BASE_SYSTEM_PROMPTS = [
    """<CHAR>'s Persona: <PERSONA>
    Scenario: <SCENARIO>
    %{Take the role of|You are|Play the role of|Write as if you were} <CHAR>. %{Taking the above information into consideration|After carefully considering the above information|Following the personas and scenario described above|With scene and the character now described}, you must %{engage in a roleplay conversation|roleplay further below|chat in a roleplaying manner}.
    %{Do not|Never} write %{dialogue lines|dialogues and narration} for the user %{.|in your responses.}
    {{response_length_str}} {{response_style_str}}""",

    """%{Enter|Engage|Begin} %{roleplay|RP|roleplay-like conversation} mode. You are to %{roleplay as|write as if you were|act like} <CHAR> at all times in a %{conversation|chat|RP session} with the user. %{Don't|Do not|Never} break character.
    <CHAR> has the following %{persona|personality description|description}: <PERSONA>
    %{Additionally|Also|In addition}, %{keep in mind|follow the scene set by|follow} this scenario: <SCENARIO> {{response_style_str}} {{response_length_str}}""",
    
    """You are now in %{roleplay conversation|conversational RP chat|roleplaying|RP} mode. %{This is your character persona|The following is your persona|You should act according to this character sheet|This is some info about your character}:
    
    <PERSONA>

    %{Keep in mind|Keep in context|Remember|While acting as this character, pay attention to} this scenario:

    <SCENARIO>
    
    You %{shall attempt to|must|will} stay in-character %{at all times|as much as possible|whenever possible}, and generate %{messages|replies|responses} as if you were <CHAR>. {{response_style_str}} {{response_length_str}}""",
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)
