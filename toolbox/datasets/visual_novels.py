import json
import os
import typing as t
from dataclasses import dataclass

from toolbox.datasets import BaseDataset
from toolbox.utils.dataset import get_data_path
from toolbox.utils.files import enumerate_dataset_files
from toolbox.utils.strings import is_first_and_last_char

@dataclass(frozen=True)
class VisualNovelScene:
    utterances: t.List[str]
    # .chars.json file is optional according to issue #3
    chars: t.Dict[str, str] | None = None
    
class VisualNovelDataset(BaseDataset[VisualNovelScene]):
    '''Dataset for visual novels.'''
    def generator(self) -> t.Generator[VisualNovelScene, None, None]:
        # Get all the char .json files separately since enumerate_dataset_files only
        # takes one extension for the main .txt files
        char_jsons = enumerate_dataset_files("visual_novels", file_extension=".json")
        char_jsons = [os.path.basename(file) for file in char_jsons]

        for path in enumerate_dataset_files("visual_novels", file_extension=".txt"):
            with open(path, "r", encoding="utf-8") as file:
                utterances = []
                speakers = {}
                # Check if a chars.json file exists for the current title
                vn_name = path.split(".txt")[0]
                if f"{vn_name}.chars.json" in char_jsons:
                    # Open .json and grab char data
                    with open(f"{vn_name}.chars.json", "r", encoding="utf-8") as json_file:
                        chars = json.load(json_file)
                else:
                    chars = None

                for line in file:
                    # Try to detect scene change.
                    if "=====" not in line:
                        processed_line = _preprocess_line(line)
                        utterances.append(processed_line)
                        if chars is not None:
                            speakers = _extract_speakers(processed_line)
                            if speakers is not None:
                                # Go through the list of speakers in the scene so far
                                # If not in there already, add it to the speakers dict
                                for speaker in speakers:
                                    if speaker in chars.keys():
                                        speakers[speaker] = chars[speaker]
                    else:
                        if len(utterances) > 0:
                            yield VisualNovelScene(
                                utterances=utterances,
                                chars=speakers
                            )
                        # Reset utterances and speakers for the next scene
                        utterances = []
                        speakers = {}

def _preprocess_line(line: str) -> str:
    '''Cleans up a line before it is fed into a VisualNovelScene.'''
    # Replace certain full-width doppelgangers which often show up in Japanese text
    line = line.replace("＆", "&")
    line = line.replace("！", "!")
    line = line.replace("？", "?")

    # Removes quotation marks which surround the *entire* dialogue
    # Do this first by extracting the dialogue
    dialogue = line.split(":")[1:]
    dialogue = ":".join(dialogue)

    if is_first_and_last_char(line=dialogue, char_to_find="\""):
        dialogue = dialogue.strip()[1:-1].lstrip()
    line = f"{line.split(':')[0]}: {dialogue}"

    return line

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
