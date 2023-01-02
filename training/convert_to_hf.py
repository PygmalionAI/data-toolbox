#!/usr/bin/env python3

# Utility to convert ColossalAI checkpoints to a HuggingFace pre-trained model.

import argparse
import logging

import transformers
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main() -> None:
    args = _parse_args_from_argv()
    model = _build_model(args)

    output_dir = args.output_dir
    logger.info("Saving pre-trained HF model to `%s`...", output_dir)
    model.save_pretrained(output_dir)


def _parse_args_from_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name",
        default="EleutherAI/pythia-1.3b-deduped",
        help="HuggingFace Transformers base model name.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="Fine-tune checkpoint to load into the base model.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Name of the output folder to save the pre-trained HF model to.",
        required=True,
    )

    return parser.parse_args()


def _build_model(args: argparse.Namespace) -> transformers.AutoModelForCausalLM:
    logger.info(f"Loading checkpoint from `{args.checkpoint}`")
    state_dict = torch.load(args.checkpoint, map_location="cuda").pop("model")

    logger.info(f"Loading the `{args.model_name}` model")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, state_dict=state_dict)
    model.eval().half()  # .to("cuda")

    return model


if __name__ == "__main__":
    main()
