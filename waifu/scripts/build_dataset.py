#!/usr/bin/env python3
import argparse
import hashlib
import importlib
import json
import logging
import os
import subprocess
import sys
import typing as t

from waifu.modules import BaseModule

# TODO(11b): Needs manual maintenance to keep up-to-date. Consider doing some
# metaprogramming trickery to build this list out instead.
DEFAULT_MODULE_LIST = [
    "characterai_pdm:CharacterAiPDM",
    "discord_vdm:DiscordVDM",
    # KajiwotoPDM has a bunch of garbage I need to filter, disabling in favor
    # of the vanilla dialogue module for now.
    # "kajiwoto_pdm:KajiwotoPDM",
    "kajiwoto_vdm:KajiwotoVDM",
    "light_dialogue_pdm:LightDialoguePDM",
]
DEFAULT_MODULES_STRING = ",".join(DEFAULT_MODULE_LIST)


def main() -> None:
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

    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Enable verbose logging.")

    args = parser.parse_args()

    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    # Sanity check.
    if args.output_name and args.print:
        raise Exception("--output-name and --print are mutually exclusive.")

    modules = _import_modules_from_string(args.modules)

    #
    # If the print argument was specified, print and exit.
    #
    if args.print:
        idx = 0
        for module in modules:
            for episode in module():
                idx += 1
                if idx > args.print:
                    sys.exit()

                # Print a newline to visually separate different episodes.
                if idx != 1:
                    print()
                print("---| New Episode |---")
                print("---------------------")
                print(episode)
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
        # Iterate over each module sequentially, and write the data out into the
        # file.
        for module in modules:
            for episode in module():
                json_line = json.dumps({"text": episode})
                output_file.write(f"{json_line}\n")


#
# Helpers and CLI entrypoint.
#


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
        qualified_module_name = "waifu.modules"
        try:
            module_name, class_name = module_and_class_name.split(":")
            qualified_module_name = f"waifu.modules.{module_name}"
        except ValueError:
            class_name = module_and_class_name

        module = importlib.import_module(qualified_module_name)
        modules.append(getattr(module, class_name))

    return modules


if __name__ == "__main__":
    main()
