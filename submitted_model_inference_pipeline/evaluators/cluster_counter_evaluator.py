import numpy as np
import cv2
from src.evaluators.evaluator import Evaluator


class ClusterCounterEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def count_clusters(self, mask: np.ndarray) -> int:
        """
        Takes a binary mask and counts the clusters based on cv2.connected Components

        Args:
            mask (np.ndarray): binary mask (0/1)

        Returns:
            int: amount of clusters
        """

        # In case mask is 0/1 → transform to 0/255
        if mask.max() == 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # Connected Components (4-Neighbourhood)
        num_labels, labels = cv2.connectedComponents(mask, connectivity=4)

        # num_labels contains:
        #   0 = Backround
        #   1..N = clustes
        return num_labels - 1

    def reset(self):
        """
        Resets the overall report of this evaluation metric and everything else to do with this evaluator to its default values
        """
        self.results = []
        return super().reset()

    def evaluate_step(self, msk: np.ndarray, gold_msk: np.ndarray) -> float:
        """
        Calculates by how many clusters the generated mask deviates from the gold mask.
        Saves the result for later reporting

        Args:
            msk (np.ndarray): the generated mask
            gold_msk (np.ndarray): the gold mask

        Returns:
            float: the deviation between the amount of clusters in the mask and the gold_mask. 
            Positive if the predicted mask contains more clusters then the gold mask,
            Negative if the predicted mask contains fewer clusters then the gold mask.
        """
        connected_components_predict = self.count_clusters(mask=msk)
        connected_components_gold = self.count_clusters(mask=gold_msk)
        result = connected_components_predict - connected_components_gold
        self.results.append(result)
        return result

    def report(self):
        print(
            f"The average diversion from the correct amount of clusters is {sum(self.results)/len(self.results)}")
