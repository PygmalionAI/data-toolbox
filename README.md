# 11b's /wAIfu/ Toolbox

**Note**: This is a _very_ early work-in-progress. Expect the unexpected.

As of the moment I'm writing this, the roadmap for the project's prototype model is basically:

- Build a dataset
- Fine-tune a pre-trained language model on that dataset
- Play around, observe behavior and identify what's subpar
- Adjust dataset accordingly as to try and address the relevant shortcomings
- Repeat.

This repository is where I'm versioning all the code I've written to accomplish the above.

In short, here's how it works:

- We start off with raw datasets (see [/waifu/datasets/](/waifu/datasets/)).
  - These are basically classes reponsible for giving us raw data. They might, for example, download a `.zip` off the internet, unzip it, read a `.json` file from in there and then return its contents.
- Modules then make use of these datasets ([/waifu/modules/](/waifu/modules/)).
  - These are heavily inspired by the papers that introduced LaMDA and BlenderBot3 (and their relevant supporting papers as well).
  - In general, each module is responsible for using a dataset as an input, and processing that data down into text that will be used in the fine-tuning process.
- A final data file is produced by concatenating the outputs of all the modules. This file is used as an input for the fine-tuning process.
