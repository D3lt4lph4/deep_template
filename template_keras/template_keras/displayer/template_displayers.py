from abc import ABC, abstractmethod

class TemplateDisplayer(ABC):

    @abstractmethod
    def display(self, predictions, inputs):
        raise NotImplementedError

    @abstractmethod
    def display_with_gt(self, predictions, inputs, groundtruth):
        raise NotImplementedError
