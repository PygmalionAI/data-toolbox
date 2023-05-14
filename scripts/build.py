#!/usr/bin/env python3
import argparse
import logging
import random
import json

from colors import color

from toolbox.core.training_example import TrainingExampleGenerator, TurnTooLargeError
from toolbox.filters.training_example_filter import TrainingExampleFilter
from toolbox.tasks import NAME_TO_TASK_MAPPING
from toolbox.filters import NAME_TO_TRAINING_EXAMPLE_FILTER_MAPPING

LOG = logging.getLogger(__name__)


def main() -> None:
    random.seed(42)

    args = _parse_args_from_argv()
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    idx = 0
    print_new_episode_header = True

    example_filters: list[TrainingExampleFilter] = [
        NAME_TO_TRAINING_EXAMPLE_FILTER_MAPPING[filter_name]()
        for filter_name in args.example_filters.split(",")
    ] if args.example_filters else []

    Task = NAME_TO_TASK_MAPPING[args.task]
    for episode in Task():
        if args.print and print_new_episode_header:
            print(
                color("     new episode      ",
                      fg="black",
                      bg="green",
                      style="bold"))
            print_new_episode_header = False

        try:
            for example in TrainingExampleGenerator(episode):
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
                              style="bold"))

                    print(color(example.prompt, fg="gray"), end="")
                    print(color(example.generation, fg="green"))
                else:
                    print(
                        json.dumps({
                            "prompt": example.prompt,
                            "generation": example.generation,
                            "identifier": example.identifier,
                        }))
        except TurnTooLargeError:
            LOG.info("Skipping over episode (%s) due to a TurnTooLargeError",
                     episode.identifier)


#
# Helpers and CLI entrypoint.
#


def _parse_args_from_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-t",
                        "--task",
                        required=True,
                        help="Which task to build data for.")

    parser.add_argument(
        "--example-filters",
        type=str,
        help="List of comma-separated filters to apply to training examples.")

    parser.add_argument("--starting-index",
                        type=int,
                        default=0,
                        help="Used to skip over training examples.")

    parser.add_argument("--max-count",
                        type=int,
                        default=None,
                        help="Limit how many training examples to generate.")

    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="Print training examples instead of writing to STDOUT.")

    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Enable verbose logging.")

    return parser.parse_args()


if __name__ == "__main__":
    main()