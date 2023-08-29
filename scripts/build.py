#!/usr/bin/env python3
import argparse
import logging
import random
import json

from colors import color

from toolbox.core.task import BaseTask
from toolbox.core.training_example import TrainingExampleGenerator, TurnTooLargeError
from toolbox.filters.training_example_filter import TrainingExampleFilter
from toolbox.tasks import NAME_TO_TASK_MAPPING
from toolbox.filters import NAME_TO_TRAINING_EXAMPLE_FILTER_MAPPING

LOG = logging.getLogger(__name__)


def main() -> None:
    args = _parse_args_from_argv()
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    random.seed(args.seed)

    if not args.print and args.output_file.strip() == "":
        raise ValueError("Invalid directory specified! Did you mean to enable the `print` flag?")

    idx = 0
    print_new_episode_header = True

    # Generate tasks and example filters
    tasks: list[BaseTask] = [NAME_TO_TASK_MAPPING[task]() for task in args.tasks.split(",")]
    example_filters: list[TrainingExampleFilter] = [
        NAME_TO_TRAINING_EXAMPLE_FILTER_MAPPING[filter_name]()
        for filter_name in args.example_filters.split(",")
    ] if args.filters else []

    if not args.print:
        f = open(args.output_file, "w", encoding="utf-8")

    for task in tasks:
        for episode in task:
            if args.print and print_new_episode_header:
                print(
                    color("     new episode      ",
                        fg="black",
                        bg="green",
                        style="bold")
                )
                print_new_episode_header = False
            
            try:
                for example in TrainingExampleGenerator(episode, target_token_count=args.max_length, format=args.format):
                    # Right off the bat, if this training example gets caught by one
                    # of the filters, skip over and don't even count it.
                    should_keep = True
                    for filter in example_filters:
                        if not filter.should_keep(example):
                            should_keep = False
                            break
                    if not should_keep:
                        continue

                    idx += 1
                    if idx < args.starting_index:
                        continue
                    if args.max_count and (idx >
                                        args.starting_index + args.max_count):
                        quit()

                    print_new_episode_header = True

                    if args.print:
                        print(
                            color("   training example   ",
                                fg="black",
                                bg="orange",
                                style="bold")
                        )
                        print(color(example.prompt, fg="gray"), end="")
                        print(color(example.generation, fg="green"))
                    else:
                        dict_to_write = {
                            "prompt": example.prompt,
                            "generation": example.generation,
                            "identifier": example.identifier,
                        }
                        f.write(json.dumps(dict_to_write) + "\n")
            except TurnTooLargeError:
                LOG.info("Skipping over episode (%s) due to a TurnTooLargeError",
                        episode.identifier)
                
    if not args.print:
        f.close()

#
# Helpers and CLI entrypoint.
#


def _parse_args_from_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--tasks",
        type=str,
        required=True,
        help="The tasks to build data for, comma-separated."
    )

    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="", # Not required if examples just need to be printed
        help="The tasks to build data for, comma-separated."
    )

    parser.add_argument(
        "-f",
        "--filters",
        type=str,
        help="List of comma-separated filters to apply to training examples."
    )

    parser.add_argument(
        "-l",
        "--max-length",
        type=int,
        default=2048,
        # TODO(TG): Explain this more clearly
        help="The (approximate) amount of tokens to limit episodes to."
    )

    parser.add_argument(
        "-m",
        "--format",
        type=str,
        default="metharme",
        help="The format for the training data to use (accepted inputs: 'pygmalion', 'metharme'). Defaults  'metharme'"
    )

    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="Print training examples instead of writing to STDOUT."
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for the random number generator."
    )

    parser.add_argument(
        "--starting-index",
        type=int,
        default=0,
        help="Used to skip over training examples."
    )

    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Limit how many training examples to generate."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()