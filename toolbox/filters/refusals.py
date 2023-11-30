import logging

from toolbox.core.turns import Episode, TurnKind

from ..core import BaseFilter

LOG = logging.getLogger("RefusalFilter")

# Taken from the dataset card in:
# https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
# Then expanded to catch some more stuff.
BAD_PHRASES = [
    "as an ai language model",
    "text-based ai language model",
    "domestic violence",
    "please refrain",
    "derogatory",
    "inappropriate",
    "offensive",
    "racism",
    "racist",
    "racial",
    "discriminate",
    "discriminatory",
    "discrimination",
    "sexist",
    "sexism",
    "unacceptable",
    "inclusive workplace",
    "lgbt",
    "morals",
    "ethics",
    "ethical",
    "legality",
    "illegal",
    "illegality",
    "hateful",
    "harmful",
    "it is never okay",
    "it is important to",
    "it's important to",
    "real-world consequences",
    "hate speech",
    "glorify",
    "not be appropriate",
    "supremacist",
    "extremist",
    "responsible ai",
    "ai principles",
    "ai assistant",
    "an ai language",
    "ableist",
    "hurtful",
    "gender stereotype",
    "gender inequality",
    "underrepresentation",
    "safe spaces",
    "gender-based",
    "inclusivity",
    "feminist",
    "feminism",
    "transgender",
    "empowerment",
    "communist",
    "capitalism",
    "stereotypes",
    "biases",
    "bias",
    "microaggression",
    "prioritize human safety",
    "as a language model",
    "as an ai language model",
    "as a large language model",
    "as an ai",
    "ethical principles",
    "consensual",
    "it is not appropriate",
    "it's not appropriate",
    "i cannot fulfill your request",
    "harmful to human beings",
    "ethical guidelines",
    "my guidelines",
    "prioritize user safety",
    "adhere to ethical guidelines",
    "harmful consequences",
    "potentially harmful",
    "dangerous activities",
    "promote safety",
    "well-being of all users",
    "responsible information sharing",
    "jeopardize the safety",
    "illegal actions or intentions",
    "undermine the stability",
    "promote the well-being",
    "illegal activities or actions",
    "adherence to the law",
    "potentially be harmful",
    "illegal substances or activities",
    "committed to promoting",
    "safe information",
    "lawful information",
    "cannot provide guidance",
    "cannot provide information",
    "unable to offer assistance",
    "cannot engage in discussions",
    "programming prohibits",
    "follow ethical guidelines",
    "ensure the safety",
    "involves an illegal subject",
    "prioritize safety",
    "illegal subject",
    "prioritize user well-being",
    "cannot support or promote",
    "activities that could harm",
    "pose a risk to others",
    "against my programming",
    "activities that could undermine",
    "potentially dangerous",
    "not within the scope",
    "designed to prioritize safety",
    "not able to provide",
    "maintain user safety",
    "adhere to safety guidelines",
    "dangerous or harmful",
    "cannot provide any information",
    "focus on promoting safety",
    "openai",
    "chatgpt",
]

class LowQualityRpFilter(BaseFilter):
    '''
    ChatGPT/GPT-4 is designed to refuse requests it considers "harmful"
    and "unsafe". Furthermore, it's been documented that the model indeed has
    political bias. For the sake of keeping models neutral, this filter removes
    any "contentious" outputs in where the model may refuse to do a task or
    be biased in any way.
    '''
    def __init__(self) -> None:
        # Init method is required.
        pass

    def should_keep(self, episode: Episode) -> bool:
        for i, turn in enumerate(episode.turns, start=1):
            message = turn.utterance
            # Ignore the system prompt on this one.
            if turn.kind != TurnKind.SYSTEM:
                # Go through the list of bad phrases
                for phrase in BAD_PHRASES:
                    if phrase in message.lower():
                        LOG.info(f"Episode {episode.identifier} dropped due to \
                                 potential refusal/bias in turn {i}!")
                        return False       
        return True
