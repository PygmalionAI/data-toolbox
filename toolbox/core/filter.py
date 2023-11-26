from .training_example import TrainingExample

class BaseFilter:
    '''
    Any filter for data should inherit from this base class.
    Filters work on the task level and discards any data that does not meet
    a certain criteria (must be English, must not be a duplicate, etc etc.)
    '''
    def should_keep(self, example: TrainingExample) -> bool:
        '''
        Whether or not the given training example should be kept and used for training.
        '''
        raise NotImplementedError
