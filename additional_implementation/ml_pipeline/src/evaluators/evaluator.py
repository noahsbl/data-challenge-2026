
from abc import ABC, abstractmethod
import numpy as np


class Evaluator(ABC):
    """
    An abstract class for evaluating results of an image segmentation process
    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def evaluate_step(self, msk: np.ndarray, gold_msk: np.ndarray) -> float:
        """
        A Method to evaluate two images against each other and simultaniously collect an overall report.
        It returns the result of this specific comparison and adds it to 
        the overall report of the evaluation.

        Args:
            msk (np.ndarray): An Image Mask
            gold_msk (np.ndarray): The correct Mask.

        Returns:
            float: the result of this specific comparison
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the overall report of this evaluation metric and everything else to do with this evaluator to its default values.
        """
        pass

    @abstractmethod
    def report(self):
        """
        generate an overall report that describes an overall report for this metric
        """
        pass
