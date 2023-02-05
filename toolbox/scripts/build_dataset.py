#!/usr/bin/env python3
import argparse
import hashlib
import importlib
import json
import logging
import os
import random
import subprocess
import sys
import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.modules import BaseModule
from toolbox.utils.strings import contains_suspect_unicode

import langdetect

# TODO(11b): Needs manual maintenance to keep up-to-date. Consider doing some
# metaprogramming trickery to build this list out instead.
DEFAULT_MODULE_LIST = [
    "characterai_pdm:CharacterAiPDM",
    # "discord_vdm:DiscordVDM",
    # KajiwotoPDM has a bunch of garbage I need to filter, disabling in favor
    # of the vanilla dialogue module for now.
    # "kajiwoto_pdm:KajiwotoPDM",
    # "kajiwoto_vdm:KajiwotoVDM",
    # "light_dialogue_pdm:LightDialoguePDM",
]
DEFAULT_MODULES_STRING = ",".join(DEFAULT_MODULE_LIST)


def main() -> None:
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-name",
        help="Path to write to. Should not include a file extension.")

    parser.add_argument("-m",
                        "--modules",
                        default=DEFAULT_MODULES_STRING,
                        help="List of modules to use, comma-separated.")

    parser.add_argument(
        "-p",
        "--print",
        type=int,
        help="If given, print this many episodes instead of writing to a file.")

    parser.add_argument(
        "-s",
        "--skip",
        type=int,
        help="If given, skip over this many episodes before printing.")

    parser.add_argument(
        "-u",
        "--supervised",
        action="store_true",
        help="If passed, output data for supervised fine-tuning instead.")

    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Enable verbose logging.")

    args = parser.parse_args()

    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    # Sanity checks.
    if args.output_name and args.print:
        raise Exception("--output-name and --print are mutually exclusive.")
    if args.skip and not args.print:
        raise Exception("--skip can only be used in conjunction with --print.")

    modules = _import_modules_from_string(args.modules)

    #
    # If the print argument was specified, print and exit.
    #
    if args.print:
        idx = 0
        episodes_to_skip = args.skip if args.skip is not None else None
        for module in modules:
            for episode in module():
                if episodes_to_skip:
                    episodes_to_skip -= 1
                    continue

                idx += 1
                if idx > args.print:
                    sys.exit()

                # Print a newline to visually separate different episodes.
                if idx != 1:
                    print()

                print("---| New Episode |---")
                print("---------------------")
                for turn in episode.turns:
                    print(f"{turn.speaker}: {turn.utterance}")

                # for ep in _episode_augmentations(episode):
                #     print("---| New Episode |---")
                #     print("---------------------")
                #     print("\n---\n".join(ep + [PromptConstants.EOS_TOKEN]))
        sys.exit()

    #
    # Otherwise, proceed with the writing logic.
    #

    # If no output name is given, we build one from the current git revision
    # plus a hash of the given arguments. That way, the same dataset should
    # theoretically always have the same output name, which is helpful for
    # reproducibility and bailing out early (e.g. if the file already exists).
    if args.output_name is None:
        args_hash = hashlib.sha256(str(args).encode("utf-8")).hexdigest()[:7]
        output_name = f"rev-{_get_git_revision_short_hash()}-args-{args_hash}"
    else:
        output_name = args.output_name

    # Open the output file.
    output_filename = f"{output_name}.jsonl"
    if os.path.exists(output_filename):
        raise Exception(f"{output_filename} already exists, aborting.")

    with open(output_filename, "w", encoding="utf-8") as output_file:
        # Keep track of episode hashes so we avoid duplicates.
        seen_episode_hashes = set()
        dropped_episodes_due_to_hash_collision = 0
        total_episode_count = 0

        def _calculate_hash_for(text: str) -> str:
            return hashlib.sha512(str(text).encode("utf-8")).hexdigest()

        # Enforce determinism in the language detection code.
        langdetect.DetectorFactory.seed = 0

        # Iterate over each module sequentially, and write the data out into the
        # file.
        for module in modules:
            for episode in module():
                text = "\n".join(episode)
                if contains_suspect_unicode(text):
                    print(
                        f"Skipping. Found suspect unicode contents in `{text}`")
                    continue

                episode_lang = langdetect.detect(text)
                if episode_lang != "en":
                    print(f"Skipping non-English episode ({episode_lang})")
                    continue

                for augmented_episode in _episode_augmentations(episode):
                    try:
                        total_episode_count += 1
                        text = "\n".join(augmented_episode +
                                         [PromptConstants.EOS_TOKEN])

                        # Skip over this episode if there's a hash collision.
                        episode_hash = _calculate_hash_for(text)
                        if episode_hash in seen_episode_hashes:
                            dropped_episodes_due_to_hash_collision += 1
                            continue

                        # TODO(11b): This code sucks. Refactor so we can share
                        # the looping and augmentation logic with --print, and
                        # use a logger to keep track of things happening.
                        if args.supervised:
                            offset_idx = 1
                            while augmented_episode[-offset_idx].startswith(
                                    "You: "):
                                offset_idx += 1
                                if offset_idx >= len(augmented_episode):
                                    pass  # print("Skipping episode where user speaks last.")

                            prompt = "\n".join(augmented_episode[:-offset_idx])
                            response = augmented_episode[-offset_idx]

                            separator_idx = response.find(": ")
                            bot_name = response[:separator_idx]
                            response = response.replace(f"{bot_name}:", "")
                            prompt += f"\n{bot_name}:"

                            if "<START>" in response:
                                continue  # print("skipping start")

                            json_line = json.dumps({
                                "input": prompt,
                                "output": response,
                                "reward": 1.0
                            })
                        else:
                            json_line = json.dumps({"text": text})

                        output_file.write(f"{json_line}\n")
                        seen_episode_hashes.add(episode_hash)
                    except Exception as ex:
                        print(f"Skipping episode:", ex)

        print(
            f"Dropped {dropped_episodes_due_to_hash_collision} seemingly duplicate episodes out of {total_episode_count} generated episodes."
        )


#
# Helpers and CLI entrypoint.
#


def _episode_augmentations(
        episode: list[str]) -> t.Generator[list[str], None, None]:
    '''
    Generates augmented data for the given episode.

    The first 1.3B model had wildly unpredictable performance at the start of
    conversations, which I attributed to the fact that originally we always fed
    the model entire episodes to train on, so there were no examples of freshly
    started conversations, in a sense.

    This function takes a complete episode and yields different permutations of
    it in an attempt to provide that data (e.g. with/without persona, with only
    X messages in the history, X+2, X+4 and so on).
    '''
    permutated_episode = []
    offset_idx = 0

    # Don't discard the original episode.
    yield episode

    for turn in episode:
        if "'s Persona: " in turn or "Scenario: " in turn or PromptConstants.CHAT_START_TOKEN in turn:
            permutated_episode.append(turn.strip())
            offset_idx += 1
            continue

        while len(episode) > 1 + offset_idx:
            permutated_episode.append(episode.pop(offset_idx))
            permutated_episode.append(episode.pop(offset_idx))

            # Yielding every single instance results in too much data
            # repetition, so instead we take a random sample.
            should_yield = random.randint(0, 100) < 25
            if should_yield:
                yield permutated_episode

            # Also, yield a version with _just_ dialogue if we've been yielding
            # with persona/scenario data this entire time.
            if offset_idx == 0:
                continue

            should_yield = random.randint(0, 100) < 25
            if should_yield:
                yield permutated_episode[offset_idx:]


def _get_git_revision_short_hash() -> str:
    '''Returns the project's short git revision hash.'''
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",
                         "..")).decode("ascii").strip()


def _import_modules_from_string(string: str) -> t.List[t.Type[BaseModule]]:
    '''Imports all the module classes from the given, comma-separated string.'''
    modules: t.List[t.Type[BaseModule]] = []
    for module_and_class_name in string.split(","):
        qualified_module_name = "toolbox.modules"
        try:
            module_name, class_name = module_and_class_name.split(":")
            qualified_module_name = f"toolbox.modules.{module_name}"
        except ValueError:
            class_name = module_and_class_name

        module = importlib.import_module(qualified_module_name)
        modules.append(getattr(module, class_name))

    return modules


if __name__ == "__main__":
    main()
