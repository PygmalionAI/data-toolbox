from .buzz import BuzzData, BuzzInstructTask
from .slimorca import SlimOrcaData, SlimOrcaInstructTask

# Format: (Task class name, task shorthand) -> (Task class, TrainingData class)
# Used in compile_dataset.py to dynamically instantiate tasks and their corresponding datasets.
TASK_MAPPINGS: dict[tuple[str, str], tuple["Task", "TrainingData"]] = {
    ("BuzzInstructTask", "buzz_instruct"): (BuzzInstructTask, BuzzData),
    ("SlimOrcaInstructTask", "slimorca_instruct"): (SlimOrcaInstructTask, SlimOrcaData),
}
