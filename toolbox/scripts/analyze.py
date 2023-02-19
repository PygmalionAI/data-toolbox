import argparse
import glob
import json
import os

from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FileInfo:
    # Source name.
    source: str

    # Word count (prompt).
    prompt_wc: int

    # Word count (response).
    response_wc: int

    # Average response word count
    avg_response_wc: float

    # Training examples
    example_count: int

    # Filesize (MBs)
    filesize_mb: float


def main() -> None:
    args = _parse_args_from_argv()

    paths = glob.glob(os.path.join(args.directory, "*.jsonl"))

    infos: list[FileInfo] = []
    for path in paths:
        infos.append(_get_info_from_file(path))

    _create_plots_from_infos(infos)


def _parse_args_from_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--directory",
                        required=True,
                        help="Directory containing the training JSONL files.")

    return parser.parse_args()


def _get_info_from_file(path: str) -> FileInfo:
    prompt_wc = 0
    response_wc = 0
    avg_response_wc = 0
    example_count = 0

    with open(path, "r") as file:
        for line in file:
            example = json.loads(line)

            prompt_wc += len(example["input"].split())
            response_wc += len(example["output"].split())

            example_count += 1

            avg_response_wc += len(example["output"].split())
            avg_response_wc /= 2

    return FileInfo(
        source=os.path.basename(path),
        prompt_wc=prompt_wc,
        response_wc=response_wc,
        avg_response_wc=response_wc,
        example_count=example_count,
        filesize_mb=_get_size_in_megabytes(path),
    )


def _get_size_in_megabytes(path: str) -> float:
    return os.path.getsize(path) / 1024 / 1024


def _create_plots_from_infos(infos: list[FileInfo]) -> None:
    labels = [x.source.split(":")[1].replace(".jsonl", "") for x in infos]
    response_wc_avgs = [x.avg_response_wc for x in infos]
    filesizes_in_mb = [x.filesize_mb for x in infos]
    example_counts = [x.example_count for x in infos]

    _, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.pie(response_wc_avgs, labels=labels, autopct="%1.1f%%")
    ax1.set_title("By average response word count")

    ax2.pie(filesizes_in_mb, labels=labels, autopct="%1.1f%%")
    ax2.set_title("By filesize")

    ax3.pie(example_counts, labels=labels, autopct="%1.1f%%")
    ax3.set_title("By example count")
    plt.show()


if __name__ == "__main__":
    main()
