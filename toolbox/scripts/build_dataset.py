#!/usr/bin/env python3
import argparse
import collections
import hashlib
import importlib
import json
import logging
import os
import random
import subprocess
import sys
import typing as t

from colors import color

from toolbox.core.episode import SupervisedExampleGenerator
from toolbox.core.filter_criteria import FilterCriteria
from toolbox.modules import BaseModule

# TODO(11b): Needs manual maintenance to keep up-to-date. Consider doing some
# metaprogramming trickery to build this list out instead.
DEFAULT_MODULE_LIST = [
    "characterai_pdm:CharacterAiPDM", "discord_vdm:DiscordVDM",
    "kajiwoto_pdm:KajiwotoPDM", "light_dialogue_pdm:LightPDM",
    "soda_pdm:SodaPDM"
]
DEFAULT_MODULES_STRING = ",".join(DEFAULT_MODULE_LIST)

DEFAULT_FILTER_LIST = [
    "length_filter:LengthFilter",
    "duplicate_filter:DuplicateFilter",
    "suspect_unicode_filter:SuspectUnicodeFilter",
    "tomato_filter:TomatoFilter",
    "similarity_filter:SimilarityFilter",
    # This is a whopping ~48% of total runtime when enabled, and it ends up
    # filtering less than 3% of the episodes. I don't think this is worth the
    # performance hit, so I'll disable it by default.
    # "language_filter:LanguageFilter",
]
DEAFULT_FILTERS_STRING = ",".join(DEFAULT_FILTER_LIST)

LOG = logging.getLogger(__name__)


def main() -> None:
    random.seed(42)

    args = _parse_args_from_argv()

    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    # Sanity checks.
    if args.output_name and args.print is not None:
        raise Exception("--output-name and --print are mutually exclusive.")
    if args.skip and args.print is None:
        raise Exception("--skip can only be used in conjunction with --print.")

    # If the print argument was specified, print and exit.
    if args.print is not None:
        _iterate_through_examples(args, do_print=True)
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
        _iterate_through_examples(args, write_to=output_file)


#
# Helpers and CLI entrypoint.
#


def _parse_args_from_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-name",
        help="Path to write to. Should not include a file extension.")

    parser.add_argument("-m",
                        "--modules",
                        default=DEFAULT_MODULES_STRING,
                        help="List of modules to use, comma-separated.")

    parser.add_argument("-f",
                        "--filters",
                        default=DEAFULT_FILTERS_STRING,
                        help="List of filters to use, comma-separated.")

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

    parser.add_argument(
        "-t",
        "--tokenizer-name",
        default="PygmalionAI/pygmalion-6b",
        help=
        "Which tokenizer to use when calculating target length of training examples."
    )

    parser.add_argument(
        "-l",
        "--target-example-length",
        default=2048,
        type=int,
        help="Target length (in tokens) for the training examples.")

    parser.add_argument(
        "-r",
        "--use-real-token-counts",
        action="store_true",
        help=
        "Use real token counts instead of an approximation based on word count (aruond 3x slower)"
    )

    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Enable verbose logging.")

    return parser.parse_args()


def _iterate_through_examples(args: argparse.Namespace,
                              do_print: bool = False,
                              write_to: t.Any = None) -> None:
    modules = _import_from_string(args.modules,
                                  qualifier_prefix="toolbox.modules")
    filters: list[FilterCriteria] = [
        x() for x in _import_from_string(args.filters,
                                         qualifier_prefix="toolbox.filters")
    ]
    filter_drop_count: dict[str, int] = collections.defaultdict(int)
    filter_keep_count: dict[str, int] = collections.defaultdict(int)
    processor = SupervisedExampleGenerator(args.tokenizer_name,
                                           args.target_example_length,
                                           args.use_real_token_counts)

    idx = 0
    episodes_to_skip = args.skip if args.skip is not None else None
    for module in modules:
        for episode in module():
            if episodes_to_skip:
                episodes_to_skip -= 1
                continue

            # Print a newline to visually separate different episodes.
            if idx != 1 and do_print:
                print()

            if do_print:
                print(color("---| New Episode", fg="yellow", style="bold"))

            for episode, example in processor.process(episode):
                idx += 1
                if args.print is not None and idx > args.print:
                    sys.exit()

                should_drop_episode = False
                for _filter in filters:
                    filter_name = str(_filter.__class__.__name__)
                    if _filter.keep(episode):
                        filter_keep_count[filter_name] += 1
                    else:
                        filter_drop_count[filter_name] += 1
                        LOG.debug("Dropping episode due to %s filter",
                                  filter_name)
                        should_drop_episode = True
                        break

                if should_drop_episode:
                    continue

                if do_print:
                    print(color("   | Training Example:", fg="orange"))
                    print(color(example.prompt, fg="gray"),
                          color(example.response, fg="green"))

                if write_to is not None:
                    data = {
                        "input": example.prompt,
                        "output": example.response,
                        "reward": 1.0
                    } if args.supervised else {
                        "text":
                            f"{example.prompt.strip()} {example.response.strip()}"
                    }
                    line = json.dumps(data)
                    write_to.write(f"{line}\n")

    LOG.info("About to log filter statistics")
    for filter_name, dropped in filter_drop_count.items():
        kept = filter_keep_count[filter_name]
        total = dropped + kept
        LOG.info("%s: %i out of %i examples dropped (%f%%)", filter_name,
                 dropped, total, round((dropped / total) * 100, 2))

    LOG.info("Finished!")


def _get_git_revision_short_hash() -> str:
    '''Returns the project's short git revision hash.'''
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",
                         "..")).decode("ascii").strip()


def _import_from_string(string: str,
                        qualifier_prefix: str) -> t.List[t.Type[t.Any]]:
    '''Imports all the classes from the given, comma-separated string.'''
    modules: t.List[t.Type[BaseModule | FilterCriteria]] = []
    for module_and_class_name in string.split(","):
        qualified_module_name = qualifier_prefix
        try:
            module_name, class_name = module_and_class_name.split(":")
            qualified_module_name = f"{qualifier_prefix}.{module_name}"
        except ValueError:
            class_name = module_and_class_name

        module = importlib.import_module(qualified_module_name)
        modules.append(getattr(module, class_name))

    return modules


if __name__ == "__main__":
    main()
