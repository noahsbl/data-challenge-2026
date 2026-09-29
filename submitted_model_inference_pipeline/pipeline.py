from src.classifiers.classifier import Classifier
from src.evaluators.evaluator import Evaluator
from src.evaluators.iou_evaluator import IntersectionOverUnionEvaluator
from src.evaluators.cluster_counter_evaluator import ClusterCounterEvaluator
from src.evaluators.time_evaluator import TimeEvaluator
from src.evaluators.confusion_matrix_evaluator import ConfusionMatrixEvaluator
from src.classifiers.segformer_segmentation_classifier import SegFormerSegmentationClassifier
import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm
import os

# set directorys where the image data is located
DATA_DIR = os.path.join("data", "labeled")
image_directory = os.path.join(DATA_DIR, "imgs")
mask_directory = os.path.join(DATA_DIR, "msks")

# load csv with image information
df = pd.read_csv(os.path.join(DATA_DIR, "image_info.csv"))

# set up classifiers to compete
classifier = [
    SegFormerSegmentationClassifier(
        model_path=os.path.join("classifiers", "segformer_segmentation_classifier", "segformer_iou_optimized.pt"),
        threshold=0.3,
        image_size=512,
        device="cpu",
        min_area=30,
    )
]

# For these evaluators it is only relevent, wether the image contains a pool or not
binary_evaluators = [
    TimeEvaluator(),
    ConfusionMatrixEvaluator(),
]

# These evaluators only judge how good the classifier found the shape of the pools on images that contain pools
shape_evaluators = [
    ClusterCounterEvaluator(),
    IntersectionOverUnionEvaluator(1.0)
]


for c in classifier:
    c: Classifier
    print("\n\n=================================================================")
    print(c.__name__)

    show_shape_evaluation = False

    # reset all evaluators
    for e in binary_evaluators + shape_evaluators:
        e: Evaluator
        e.reset()

    # segment images
    for index, row in tqdm(df.iterrows()):
        # load gold standard mask and image
        gold_path = os.path.join(mask_directory, row["mask_path"])
        gold_msk = tiff.imread(gold_path)

        image_path = os.path.join(image_directory, row["filename"])
        image = tiff.imread(image_path)
        result_msk = c.classify(img=image, row=row)

        # evaluate
        for e in binary_evaluators:
            e: Evaluator
            e.evaluate_step(msk=result_msk, gold_msk=gold_msk)
        # check wether the image contains a pool and if we therefore should evaluate the shape
        if np.sum(gold_msk) > 0:
            show_shape_evaluation = True
            for e in shape_evaluators:
                e: Evaluator
                e.evaluate_step(msk=result_msk, gold_msk=gold_msk)

    # report results
    print("------------binary evaluation----------------")
    for e in binary_evaluators:
        e: Evaluator
        e.report()

    print("------------shape evaluation-----------------")

    if show_shape_evaluation:
        for e in shape_evaluators:
            e: Evaluator
            e.report()
    else:
        print("No shapes have been identified.")
