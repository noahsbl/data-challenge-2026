from abc import ABC, abstractmethod
import numpy as np
from pandas import Series


class Classifier(ABC):
    """
    The abstract class from which all other classifiers in this challenge should inherit from.
    """

    def __init__(self):
        super().__init__()
        self.__name__ = self.__class__.__name__
        pass

    @abstractmethod
    def classify(self, img: np.ndarray, row: Series) -> np.ndarray:
        """
        Takes an Image img and row of a df that contains information about that image

        it then returns a black and white mask, that differentiates the classes in the image.

        Args:
            img (np.ndarray): the original image
            row (Series): a series that contains extra information about the image

        Returns:
            np.ndarray: A black and white mask, that differentiates the classes in the image
        """
        pass
