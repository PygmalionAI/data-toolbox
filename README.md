# PygmalionAI Data Toolbox

*(Note: this documentation is still a heavy work in progress.)*

This repository contains the implementation of our data munging code, used for creating datasets for our fine-tuned models.

## How does it work?

In short, it takes raw data from several different sources and parses it. From there, we can quickly experiment with different ways of formatting or augmenting the parsed data to generate a final representation, ready to be used as training data for our models.

The general data flow goes something like this:

- We start off with raw datasets (see [./toolbox/datasets/](./toolbox/datasets/))
  - These are basically classes reponsible for giving us raw data. They might, for example, download a `.zip` off the internet, unzip it, read a `.json` file from in there and then return its contents.
- Tasks then make use of these datasets ([./toolbox/tasks/](./toolbox/tasks/)) ti create episodes
  - In general, each task is responsible for using a dataset as an input and processing that data down into "episodes" consisting of turns between a user and what will be the output of a language model, usually alongside a system prompt. Our system prompts vary depending on the task, but one can supply their own system prompts for a specific task, as seen later.
- At the task level, filters ([./toolbox/filters](./toolbox/filters/)) are applied to processed episodes based on certain criteria (for example, deduplicating a dataset or discarding low-quality data).
- The episodes are then processed into a specified format ([./toolbox/formats](./toolbox/formats/)) which will represent the structure of the data in the form of a JSON object. These JSON objects will be written to an output file.

## Installation
The usual:

`pip3 install -r requirements.txt`

## Processing

Processing the data is all done through the `build_data.py` file. Here are the arguments that can be specified:

```
usage: build_data.py [-h] --tasks TASKS --config CONFIG --output-file OUTPUT_FILE --format FORMAT [--max-length MAX_LENGTH] [--seed SEED]

options:
  -h, --help            show this help message and exit
  --tasks TASKS         The tasks to build data for, comma-separated.
  --config CONFIG       The path to the task configuration file.
  --output-file OUTPUT_FILE
                        The JSONL file to write the data to.
  --format FORMAT       The format to represent the data in.
  --max-length MAX_LENGTH
                        The number of tokens (exact or approximate depending on settings) to cap conversations to. Set to -1 (default) to disable.
  --seed SEED           The seed to use when applying random chance to anything.
```

**TODO: Further explanation of these options**

## Configuration
**TODO**

## Wiki
**TODO**

## Notes
**TODO**