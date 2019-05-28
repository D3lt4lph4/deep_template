import time

import numpy as np

from template.evaluators import TemplateEvaluator

class Evaluator(TemplateEvaluator):

    def __init__(self, generator=None):
        self.score = None
        self._generator = generator
        self.runs = False
        self.number_of_runs = None
        self.times = None

    def __call__(self, model, test_generator=None):
        if self._generator is None and test_generator is None:
            raise RuntimeError("A generator should be specified using the init or parameters.")
        self.runs = False
        if test_generator is not None:
            self._generator = test_generator

        self.score = model.evaluate_generator(self._generator)

    def make_runs(self, model, test_generator=None, number_of_runs=10):

        if self._generator is None and test_generator is None:
            raise RuntimeError("A generator should be specified using the init or parameters.")

        scores = []
        self.runs = True
        self.times = np.zeros((number_of_runs))

        if test_generator is not None:
            self._generator = test_generator

        for i in range(number_of_runs):
            start_time = time.time()
            scores.append(model.evaluate_generator(self._generator))
            self.times[i] = time.time() - start_time

        self.score = np.mean(np.array(scores), axis=0)
        self.number_of_runs = number_of_runs

    def __str__(self):
        if self.runs:
            return "Number of runs: {}\nAverage score: {}\nAverage time: {}".format(self.number_of_runs, self.score, np.mean(self.times))
        else:
            return "The evaluated score is {}.".format(self.score)

    @property
    def test_generator(self):
        return self._generator
    
    def display_results(self):
        pass
