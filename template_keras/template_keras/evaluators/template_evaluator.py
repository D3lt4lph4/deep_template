from abc import ABC, abstractmethod

class TemplateEvaluator(ABC):

    @abstractmethod
    def __str__(self):
        """ Should print the results after the call to this class. """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, model, test_generator=None):
        """ Should calculate the evaluation for a given generator. """
        raise NotImplementedError

    @abstractmethod
    def make_runs(self, model, test_generator=None, number_of_runs=10):
        """ Should calculate the evaluation for a given generator. """
        raise NotImplementedError
    
    @abstractmethod
    def display_results(self):
        raise NotImplementedError
    @property
    @abstractmethod
    def test_generator(self):
        raise NotImplementedError
