# data-toolbox

This repository contains the implementation of our data munging code.

Currently undergoing a massive refactor, I still need to document everything.

___

Outline:

Three parts for finished LLM data:
- The DATASET gathers the data and converts it to a HuggingFace Dataset if it is not one already.
- The TASK takes the Dataset and creates ShareGPT conversations out of it according to a specific role (RP, chat, instructions, whatever)
- The FILTERS eliminate data that match a certain criteria of low quality.
- Optional AUGMENTATIONS create new instruction data based off already-present examples in the compiled dataset, to buff up examples count if needed or if wanting to have instructions related to examples from your dataset domain.