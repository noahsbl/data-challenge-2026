from src.evaluators.evaluator import Evaluator
import cv2
import numpy as np


class IntersectionOverUnionEvaluator(Evaluator):
    def __init__(self, zero_division):
        """
        A Class to evaluate how many of the pixels in the msk overlap with the pixels in the gold_mask.
        The calculated score is called Intersection over Union or Jaccard-Score

        Args:
            zero_division (int): The value that is to be returned in case there is no union between the two masks
        """
        self.zero_division = zero_division
        super().__init__()
        self.reset()

    def reset(self):
        """
        Resets the overall report of this evaluation metric and everything else to do with this evaluator to its default values.
        """
        self.result_intersections = []
        self.result_unions = []
        self.union_sum = 0
        self.intersection_sum = 0

    def evaluate_step(self, msk: np.ndarray, gold_msk: np.ndarray) -> float:
        """
        A Method to evaluate how many of the pixels in the msk overlap with the pixels in the gold_mask.
        The calculated score is called Intersection over Union or Jaccard-Score


        Args:
            msk (np.ndarray): An Image Mask
            gold_msk (np.ndarray): The correct Mask.

        Returns:
            float: the result of this specific comparison
        """
        if msk.shape != gold_msk.shape:
            raise ValueError(f"The shape of the msk must be the same, not msk:{msk.shape}, gold:{gold_msk.shape}")
        msk_bin = (msk > 0).astype(np.uint8)
        gold_bin = (gold_msk > 0).astype(np.uint8)

        intersection = np.sum(cv2.bitwise_and(src1=msk_bin, src2=gold_bin))
        union = np.sum(cv2.bitwise_or(src1=msk_bin, src2=gold_bin))

        self.intersection_sum += intersection
        self.union_sum += union

        self.result_intersections.append(intersection)
        self.result_unions.append(union)

        if union != 0:
            return intersection/union
        else:
            return self.zero_division

    def report(self):
        """
        generate an overall report that calculates the IoU for all masks together
        """
        print(
            f"The IoU is {self.intersection_sum/self.union_sum}, based on {self.intersection_sum} pixels of intersections and {self.union_sum} pixels of unions")
