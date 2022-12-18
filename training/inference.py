#!/usr/bin/env python3
import argparse
import logging
import sys

import torch
import transformers

logger = logging.getLogger(__name__)
logging.basicConfig()


def main() -> None:
    #
    # CLI argument handling.
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",
                        "--model-name",
                        default="facebook/opt-350m",
                        help="HuggingFace Transformers model name.")
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="Fine-tune checkpoint to load into the base model.")

    args = parser.parse_args()

    #
    # Model setup.
    #
    logger.info(f"Loading tokenizer for {args.model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

    state_dict = None
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint,
                                map_location="cuda").pop("model")

    logger.info(f"Loading the {args.model_name} model")
    bad_words_ids = [
        tokenizer(bad_word, add_prefix_space=True,
                  add_special_tokens=False).input_ids
        for bad_word in ["_", "__", "___", "____", "_____", "Description:"]
    ]

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, state_dict=state_dict, bad_words_ids=bad_words_ids)
    model.eval().half().to("cuda")

    #
    # We're good to go! Main loop.
    #
    try:
        while True:
            prompt = input("Prompt: ").replace("<EOL>", "\n")

            # First, sampling-based generation.
            input_ids = tokenizer(prompt,
                                  return_tensors='pt').input_ids.to("cuda")
            logits = model.generate(
                input_ids,
                do_sample=True,
                max_new_tokens=8,
                top_k=50,
                top_p=0.90,
            )
            output = tokenizer.decode(logits[0], skip_special_tokens=True)

            # Then, contrastive search.
            input_ids = tokenizer(output,
                                  return_tensors="pt").input_ids.to("cuda")
            logits = model.generate(input_ids,
                                    max_new_tokens=128,
                                    penalty_alpha=0.6,
                                    top_k=6)

            print(f"Output:\n{64 * '-'}")
            output = tokenizer.decode(logits[0], skip_special_tokens=True)
            cleaned_output = output.replace(prompt, "")
            print(cleaned_output)  # print(output)
            print(64 * "-")
    except EOFError:
        sys.exit()


if __name__ == "__main__":
    main()
