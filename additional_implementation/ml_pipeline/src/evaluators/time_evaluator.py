from src.evaluators.evaluator import Evaluator
from time import time
import numpy as np


class TimeEvaluator(Evaluator):
    """Class to find the time the classifier needs for each image and in total
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        """
        Resets the overall report of this evaluation metric and everything else to do with this evaluator to its default values.
        """
        self.last_time = time()
        self.result_times = []

    def evaluate_step(self, msk: np.ndarray, gold_msk: np.ndarray) -> float:
        """calculate the time between this and the last evaluate step. Adds it to the overall report.

        Args:
            msk (np.ndarray): An Image Mask - is ignored
            gold_msk (np.ndarray): The correct Mask - is ignored

        Returns:
            float: the time between this and the last calling of this method
        """
        now = time()
        result_time = now-self.last_time
        self.result_times.append(result_time)
        self.last_time = now
        return result_time

    def report(self):
        print(
            f"The average time is {sum(self.result_times)/len(self.result_times)}")
        print(
            f"In total the classifier needed {sum(self.result_times)} for {len(self.result_times)} images")
