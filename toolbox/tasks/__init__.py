import typing as t

from toolbox.core.task import BaseTask
from toolbox.tasks.aidungeon_text_adventure import AiDungeonTextAdventureTask
from toolbox.tasks.characterai_roleplay import CharacterAiRoleplayTask
from toolbox.tasks.clubfloyd_text_adventure import ClubFloydTextAdventureTask
from toolbox.tasks.evol_instruct import EvolInstructTask
from toolbox.tasks.gpt4all_question_answering import \
    Gpt4AllQuestionAnsweringTask
from toolbox.tasks.mcstories_writing import McStoriesWritingTask
from toolbox.tasks.rp_forums_writing import RpForumsWritingTask
from toolbox.tasks.sharegpt_instruction_following import \
    ShareGptInstructionFollowingTask
from toolbox.tasks.single_turn_instruction_following import \
    SingleTurnInstructionFollowingTask
from toolbox.tasks.soda_reply_generation import SodaReplyGenerationTask
from toolbox.tasks.soda_summarization import SodaSummarizationTask
from toolbox.tasks.whocars_roleplay import WhocarsRoleplayTask
from toolbox.tasks.wizard_vicuna_question_answering import \
    WizardVicunaQuestionAnsweringTask

NAME_TO_TASK_MAPPING: dict[str, t.Type[BaseTask]] = {
    cls.__name__: cls for cls in [
        AiDungeonTextAdventureTask,
        CharacterAiRoleplayTask,
        ClubFloydTextAdventureTask,
        EvolInstructTask,
        Gpt4AllQuestionAnsweringTask,
        McStoriesWritingTask,
        RpForumsWritingTask,
        ShareGptInstructionFollowingTask,
        SingleTurnInstructionFollowingTask,
        SodaReplyGenerationTask,
        SodaSummarizationTask,
        WhocarsRoleplayTask,
        WizardVicunaQuestionAnsweringTask,
    ]
}
