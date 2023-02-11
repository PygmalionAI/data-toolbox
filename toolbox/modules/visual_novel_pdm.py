import logging
import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.core.models import Episode, Turn
from toolbox.datasets.visual_novels import VisualNovelDataset
from toolbox.modules import BaseModule
from toolbox.utils.strings import is_first_and_last_char

LOG = logging.getLogger(__name__)
MIN_SCENE_LENGTH = 2

class VisualNovelPDM(BaseModule):
    '''Persona Dialogue Model based on Visual Novels.'''

    def generator(self) -> t.Generator[Episode, None, None]:
        for scene in VisualNovelDataset():
            turns: list[Turn] = []
            personas: dict = {}

            # If the scene is extremely short, don't make an Episode out of it
            if len(scene.utterances) < MIN_SCENE_LENGTH:
                continue

            for utterance in scene.utterances:
                # Gather speaker of dialogue.
                # If it's an action/narration scene, it likely comes from the user.
                if is_first_and_last_char(line=utterance, char_to_find="*") or len(utterance.split(":")) == 1:
                    utterance = f"{PromptConstants.USER_TOKEN}: {utterance}"
                
                line_speakers = _extract_speakers(utterance)
                # Find out whether the protagonist/user is speaking the current message
                is_human: bool = PromptConstants.USER_TOKEN in line_speakers
                # Grab the message only. By this point anything should be annotated with a colon
                extracted_utterance = utterance.split(":")[1:]
                extracted_utterance = ":".join(extracted_utterance).strip()
                # Reminder: " & ".join(['character']) -> 'character'
                all_speakers = " & ".join(line_speakers)
                turns.append(Turn(
                    utterance=extracted_utterance,
                    speaker=all_speakers,
                    human_speaker=is_human
                ))

            # Gather the speakers present within the scene and if they are
            # in the VNDB, grab their personas
            if scene.chars is not None:
                for character in scene.chars.keys():
                    # _get_persona_for takes in an int, but we have a string
                    # Fix this.
                    char_id = scene.chars[character]
                    # Every id is "c[N]", where n is the actual ID
                    char_id = int(char_id[1:])
                    personas[character] = _get_persona_for(char_id)
                yield Episode(
                    turns=turns,
                    participant_personas=personas
                )
            else:
                # Char personas are not findable, treat as VDM
                yield Episode(turns=turns)

def _get_persona_for(char_id: int) -> str:
    '''Given a character ID from the VNDB, return a persona.'''
    # TODO(TG): am not good at database, do not sql
    # pls andastando
    raise NotImplementedError

# Copied over from datasets/visual_novels.py
def _extract_speakers(line: str) -> t.Optional[list[str]]:
    '''Attempts to extract a list of speakers in a dialogue line (since 2 or more people can speak a line at once in the data).'''
    # Anything that has no colon at all is likely to be a narration/action line
    if len(line.split(":")) <= 1:
        return None
    # Anything that starts and end with an asterisk is likely to be a narration/action line as well.
    if is_first_and_last_char(line=line, char_to_find="*"):
        return None

    # Grab everything before the first colon
    potential_dialogue = line.split(":")[0].strip()
    # If there's more than 4 words before the colon, it's likely a colon in a narration line
    # and not a separator between characters and dialogue
    if len(potential_dialogue.split(" ")) > 4:
        return None
    else:
        # Parse the "&" to reveal how many speakers there really are
        speakers = potential_dialogue.split("&")
        speakers = [speaker.strip() for speaker in speakers]
        return speakers
