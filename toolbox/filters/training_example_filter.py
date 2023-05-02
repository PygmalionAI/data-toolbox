from toolbox.core.training_example import TrainingExample


class TrainingExampleFilter:
    '''Filter implementations should inherit from this base class.'''

    def should_keep(self, _example: TrainingExample) -> bool:
        '''
        Whether or not the given training example should be kept and used for
        training.
        '''
        raise NotImplementedError