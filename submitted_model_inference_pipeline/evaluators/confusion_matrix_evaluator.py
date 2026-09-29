import numpy as np
import cv2
from src.evaluators.evaluator import Evaluator


class ConfusionMatrixEvaluator(Evaluator):
    """
    An Evaluator that calculates the confusion matrix
    https://medium.com/data-science/accuracy-precision-recall-or-f1-331fb37c5cb9
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        self.tp_counter = 0
        self.fp_counter = 0
        self.tn_counter = 0
        self.fn_counter = 0

    def evaluate_step(self, msk: np.ndarray, gold_msk: np.ndarray) -> str:
        """
        A Method to evaluate wether the classifier and the gold standard agree on there being pools in the picture or not
        The Overall report calculates the derived values, like accuracy, precision, 
        recall, f1-score and the counts for all true positives, false positives, 
        false negatives and true negatives.

        Args:
            msk (np.ndarray): An Image Mask
            gold_msk (np.ndarray): The correct Mask.

        Returns:
            str: the result of this specific comparison
        """
        msk_has_content = np.sum(msk) > 0
        gold_has_content = np.sum(gold_msk) > 0
        if msk_has_content and gold_has_content:
            self.tp_counter += 1
            return "TP"
        elif not(msk_has_content)  and not(gold_has_content):
            self.tn_counter += 1
            return "TN"
        elif not(msk_has_content) and gold_has_content:
            self.fn_counter += 1
            return "FN"
        elif msk_has_content and not(gold_has_content):
            self.fp_counter += 1
            return "FP"

    def report(self):
        """
        The Overall report calculates the derived values, like accuracy, precision, 
        recall, f1-score and the counts for all true positives, false positives, 
        false negatives and true negatives and then prints them
        """
        print("True Positives", self.tp_counter)
        print("False Positives", self.fp_counter)
        print("True Negatives", self.tn_counter)
        print("False Negatives", self.fn_counter)

        accuracy = (self.tp_counter+self.tn_counter)/(self.tp_counter +
                                                      self.tn_counter+self.fp_counter+self.fn_counter)
        print(f"accuracy: {accuracy}")

        # only calculate when denominator is not 0, since that would be undefined. Therefore return nan
        denominator_recall = (self.tp_counter+self.fn_counter)
        if denominator_recall != 0:
            recall = self.tp_counter/denominator_recall
        else:
            recall = np.nan
        print(f"recall: {recall}")

        # only calculate when denominator is not 0, since that would be undefined. Therefore return nan
        denominator_precision = (self.tp_counter+self.fp_counter)
        if denominator_precision != 0:
            precision = self.tp_counter/denominator_precision
        else:
            precision = np.nan
        print(f"precision: {precision}")

        if precision + recall != 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = np.nan
        print(f"f1-score: {f1_score}")
