# 11b's /wAIfu/ Toolbox

**This is a copy of the original repo, found at https://git.fuwafuwa.moe/waifu-collective/toolbox. It will be updated side-by-side with this mirror on GitHub.**

**Note**: This is a _very_ early work-in-progress. Expect the unexpected.

## Summary

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
  - These are heavily inspired by the papers that introduced [LaMDA](https://arxiv.org/pdf/2201.08239.pdf) and [BlenderBot3](https://arxiv.org/pdf/2208.03188.pdf) (and their relevant supporting papers as well).
  - In general, each module is responsible for using a dataset as an input, and processing that data down into text that will be used in the fine-tuning process.
- A final data file is produced by concatenating the outputs of all the modules. This file is used as an input for the fine-tuning process.

Here's how I do that:

## Building the data file(s)

The final data file is created with the [build_dataset.py](./waifu/scripts/build_dataset.py) script:

```
$ ./waifu/scripts/build_dataset.py --help
usage: build_dataset.py [-h] [-o OUTPUT_NAME] [-m MODULES] [-p PRINT] [-v]

options:
  -h, --help            show this help message and exit
  -o OUTPUT_NAME, --output-name OUTPUT_NAME
                        File to write the dataset to. Should not include a file extension.
  -m MODULES, --modules MODULES
                        List of modules to use, comma-separated.
  -p PRINT, --print PRINT
                        If given, print this many episodes instead of writing out to a file.
  -v, --verbose         Enable verbose logging.
```

The default behavior is to write a file called `rev-{GIT_REVISION_HASH}-args{HASH_OF_USED_ARGS}.jsonl` to the current directory, with all the modules enabled. Behavior can be customized via the flags shown above.

The script also has an option to print some examples instead of writing to a file, for debugging/dev purposes. Example usage:

```bash
$ ./waifu/scripts/build_dataset.py --print 1 --modules 'light_dialogue_vdm:LightDialogueVDM' # or -p 1 and -m ...
```

Example output:

```
--- new episode ---
Context: You are in the Watchtower.
The tower is the largest section of the castle. It contains an observatory for nighttime scouting, but is also used by the wise men to study the stars. Armed
guardsmen are always to be found keeping watch.
There's an alarm horn here.
A soldier is here. You are carrying nothing.

Court Wizard: A quiet night this evening...
Soldier: Yes it is
Court Wizard: *ponder* Have any else come up this eve? I had hoped for a quiet night to examine the stars
Soldier: *nod* Yes, a few came through, but it is a cold night for me, I am used to warmer weather
Court Wizard: *sigh* Well, you are but a common soldier. No doubt you are used to such a lot. Thankfully I have my spells to keep me warm.
Soldier: *grin* I am a soldier doing my job
Court Wizard: Yes... Well... Very well then. See that you do! No slacking off while your betters are about.
Soldier: No sir
Court Wizard: When, for example, was this horn last tested? It looks dented. How can we be sure it will work?
Soldier: A year ago, test it out or cause a need to use it
Court Wizard: *frown* Mayhap I will speak to the king about such lackness. Or perhaps I can sell him a spell that will serve just as well.
Soldier: Good idea, I agree, go do that *hug court wizard*
Court Wizard: Get off of me, you fool! Who gave you permission to touch me! *hit soldier*
Soldier: To the jail with you *hit court wizard*
```

## Fine-tuning a model

Due to hardware limitations (read: lack of GPUs with massive amounts of VRAM), I need to make use of ColossalAI's optimizations to be able to fine-tune models. However, their example code for fine-tuning OPT lacks some important stuff. Notably: metric logging (so we can know what is going on) and checkpoint saving/loading.

I've gone ahead and, using [their example scripts](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/opt) as a starting point, made a slightly adjusted version that's actually usable for real-world scenarios. All that stuff is inside the [training folder](./training/).

If you don't want to mess with anything, all you need to do is put the built data file at `/training/data/train.json` and invoke [finetune.bash](./training/finetune.bash). To see metrics, you can use Tensorboard by visiting http://localhost:6006 after starting the server like this:

```bash
tensorboard serve --port 6006 --logdir training/checkpoints/runs
```

## Running inference on the fine-tuned model

To-do: write this up.
