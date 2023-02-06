# data-toolbox

This repository contains the implementation of our data munging code.

**Note:** Not very well documented at the moment. Still need to implement automatic downloading of data files and document how to install the project with PDM.

## How does it work?

In short, it takes raw data from several different sources and parses it. From there, we can quickly experiment with different ways of formatting or augmenting the parsed data to generate a final representation, ready to be used as training data for our models.

The general data flow goes something like this:

- We start off with raw datasets (see [./toolbox/datasets/](./toolbox/datasets/))
  - These are basically classes reponsible for giving us raw data. They might, for example, download a `.zip` off the internet, unzip it, read a `.json` file from in there and then return its contents.
- Modules then make use of these datasets ([./toolbox/modules/](./toolbox/modules/))
  - These are heavily inspired by the papers that introduced LaMDA and BlenderBot3 (and their relevant supporting papers)
  - In general, each module is responsible for using a dataset as an input, and processing that data down into episodes, which will then be formatted into a proper dataset to be used in the fine-tuning process.

## Building a training dataset

The final data file is created with the [build_dataset.py](./toolbox/scripts/build_dataset.py) script:

```
$ ./toolbox/scripts/build_dataset.py --help
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
$ ./toolbox/scripts/build_dataset.py --print 1 --modules 'light_dialogue_pdm:LightPDM' # or -p 1 and -m ...
```

Example output:

```
--- new episode ---
Scenario: You are in the Watchtower.
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
