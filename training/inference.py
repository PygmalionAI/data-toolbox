#!/usr/bin/env python3
import argparse
import logging
import typing as t
import re

import torch
import transformers
import gradio as gr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# TODO(11b): Type these functions up properly.


def main() -> None:
    '''Script entrypoint.'''
    args = _parse_args_from_argv()
    # TODO(11b): We don't have the bot name at this point, since it's dynamic
    # on the UI, so we can't build `bad_word_ids` as perfectly as I'd like. See
    # if we can improve this later.
    model, tokenizer = _build_model_and_tokenizer_for(args, bot_name="")
    ui = _build_gradio_ui_for(model, tokenizer)
    ui.launch(server_port=3000, share=False)


def _parse_args_from_argv() -> argparse.Namespace:
    '''Parses arguments coming in from the command line.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name",
        default="facebook/opt-350m",
        help="HuggingFace Transformers model name.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="Fine-tune checkpoint to load into the base model. Optional.",
    )

    return parser.parse_args()


def _build_blacklist_for(bot_name: str) -> list[str]:
    '''
    Builds a blacklist for the given bot name.

    This is used to stop the model from invoking modules when we haven't
    prompted it to.
    '''

    # NOTE(11b): This should _ideally_ be shared with the actual implementations
    # inside the package's .core.consts, but for simplicity's sake I'm
    # re-implementing here (so there's no need to install the package just to
    # run inference).
    pdm_prefix = f"{bot_name}'s Persona: "

    # Not sure why, but the pre-trained OPT likes to generate these and it leaks
    # out to the fine-tuned models as well.
    bad_opt_generations = ["___", "____", "_____"]

    # And Pythia likes to do this.
    bad_pythia_generations = ["...."]

    return [pdm_prefix, *bad_opt_generations, *bad_pythia_generations]


def _build_model_and_tokenizer_for(args: argparse.Namespace,
                                   bot_name: str) -> t.Tuple[t.Any, t.Any]:
    '''Sets up the model and accompanying tokenizer.'''
    logger.info(f"Loading tokenizer for {args.model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

    state_dict = None
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint from {args.checkpoint}")

        # NOTE(11b): `.pop("model")` is specific to checkpoints saved by
        # the ColossalAI helper. If using a regular HF Transformers checkpoint,
        # comment that out.
        state_dict = torch.load(args.checkpoint,
                                map_location="cuda").pop("model")

    tokenizer_kwargs = {"add_special_tokens": False}
    if "facebook/opt-" in args.model_name:
        tokenizer_kwargs["add_prefix_space"] = True

    bad_words_ids = [
        tokenizer(bad_word, **tokenizer_kwargs).input_ids
        for bad_word in _build_blacklist_for(bot_name)
    ]

    logger.info(f"Loading the {args.model_name} model")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, state_dict=state_dict, bad_words_ids=bad_words_ids)
    model.eval().half().to("cuda")

    logger.info("Model and tokenizer are ready")
    return model, tokenizer


def _run_raw_inference(model: t.Any, tokenizer: t.Any, prompt: str,
                       user_message: str) -> str:
    '''Runs raw inference on the model, and returns just the generated text.'''

    # First, sampling-based generation.
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")
    logits = model.generate(
        input_ids,
        do_sample=True,
        max_new_tokens=32,
        top_k=50,
        top_p=0.90,
    )
    output = tokenizer.decode(logits[0], skip_special_tokens=True)

    # Then, contrastive search.
    input_ids = tokenizer(output, return_tensors="pt").input_ids.to("cuda")
    logits = model.generate(input_ids,
                            max_new_tokens=96,
                            penalty_alpha=0.6,
                            top_k=6)

    # Then, we trim out the input prompt from the generated output.
    output = tokenizer.decode(logits[0], skip_special_tokens=True)
    if (idx := prompt.rfind(user_message)) != -1:
        trimmed_output = output[idx + len(user_message):].strip()
        return trimmed_output
    else:
        raise ValueError("Couldn't find user message in the prompt. What?")


BAD_CHARS_FOR_REGEX_REGEX = re.compile(r"[-\/\\^$*+?.()|[\]{}]")


def _sanitize_string_for_use_in_a_regex(string: str) -> str:
    '''Sanitizes `string` so it can be used inside of a regexp.'''
    return BAD_CHARS_FOR_REGEX_REGEX.sub(r"\\\g<0>", string)


def _parse_messages_from_str(string: str, names: list[str]) -> list[str]:
    '''
    Given a big string containing raw chat history, this function attempts to
    parse it out into a list where each item is an individual message.
    '''
    sanitized_names = [
        _sanitize_string_for_use_in_a_regex(name) for name in names
    ]

    speaker_regex = re.compile(rf"^({'|'.join(sanitized_names)}): ",
                               re.MULTILINE)

    message_start_indexes = []
    for match in speaker_regex.finditer(string):
        message_start_indexes.append(match.start())

    if len(message_start_indexes) < 2:
        # Single message in the string.
        return [string.strip()]

    prev_start_idx = message_start_indexes[0]
    messages = []

    for start_idx in message_start_indexes[1:]:
        message = string[prev_start_idx:start_idx].strip()
        messages.append(message)
        prev_start_idx = start_idx

    return messages


def _serialize_chat_history(history: list[str]) -> str:
    '''Given a structured chat history object, collapses it down to a string.'''
    return "\n".join(history)


def _gr_run_inference(model: t.Any, tokenizer: t.Any, context: str,
                      history: list[str], character_name: str,
                      user_message: str) -> t.Tuple[list[str], str]:
    '''
    With `context` and `history` prompt-engineered into the model's input, feed
    it `user_message` and return everything the Gradio UI expects.
    '''

    # TODO(11b): Lots of assumptions to fix here. We need to make sure
    # everything fits, we need to use "You" from the `.core.consts` module, etc.
    prompt = "\n".join(
        [context, "", *history, f"You: {user_message}", f"{character_name}: "])

    output = _run_raw_inference(model, tokenizer, prompt, user_message).strip()
    logger.debug("_run_raw_inference returned `%s` after .strip()", output)

    # If there's enough space, the model will likely generate more than just its
    # own message, so we need to trim that out and just remove the first
    # generated message.
    generated_messages = _parse_messages_from_str(output,
                                                  ["You", character_name])
    logger.debug("Generated messages is `%s`", generated_messages)
    bot_message = generated_messages[0]

    logger.info("Generated message: `%s`", bot_message)

    history.append(f"You: {user_message}")
    history.append(bot_message)
    serialized_history = _serialize_chat_history(history)
    return history, serialized_history


def _gr_regenerate_last_output(model: t.Any, tokenizer: t.Any, context: str,
                               history: list[str], character_name: str,
                               user_message: str) -> t.Tuple[list[str], str]:
    history_without_last_message = history[:-2]
    return _gr_run_inference(model, tokenizer, context,
                             history_without_last_message, character_name,
                             user_message)


def _gr_undo(history: list[str]) -> t.Tuple[list[str], str]:
    updated_history = history[:-2]
    return updated_history, _serialize_chat_history(updated_history)


def _build_gradio_ui_for(model: t.Any, tokenizer: t.Any) -> t.Any:
    '''
    Builds a Gradio UI to interact with the model. Big thanks to TearGosling for
    the initial version of this.
    '''
    with gr.Blocks() as interface:
        history = gr.State([])

        with gr.Row():
            with gr.Column():
                user_message = gr.Textbox(
                    label="Input",
                    placeholder="Say something here",
                    interactive=True,
                )
                character_name = gr.Textbox(
                    label="Name of character",
                    placeholder="Insert the name of your character here",
                )
                context = gr.Textbox(
                    label="Long context",
                    lines=4,
                    placeholder=
                    "Insert the context of your character here, such as personality and scenario. Think of this as akin to CAI's short and long description put together.",
                    interactive=True,
                )
            history_text = gr.Textbox(
                label="Output",
                lines=4,
                placeholder="Your conversation will show up here!",
                interactive=False,
            )

        with gr.Row():
            submit_btn = gr.Button("Submit input")
            submit_fn = lambda context, history, character_name, user_message: _gr_run_inference(
                model, tokenizer, context, history, character_name, user_message
            )
            submit_btn.click(
                fn=submit_fn,
                inputs=[context, history, character_name, user_message],
                outputs=[history, history_text])

            regenerate_btn = gr.Button("Regenerate last output")
            regenerate_fn = lambda context, history, character_name, user_message: _gr_regenerate_last_output(
                model, tokenizer, context, history, character_name, user_message
            )
            regenerate_btn.click(
                fn=regenerate_fn,
                inputs=[context, history, character_name, user_message],
                outputs=[history, history_text])

            undo_btn = gr.Button("Undo last exchange")
            undo_btn.click(fn=_gr_undo,
                           inputs=[history],
                           outputs=[history, history_text])

    return interface


if __name__ == "__main__":
    main()
