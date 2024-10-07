import os
import sys
import warnings

import numpy as np
import torch
from tqdm import tqdm

sys.path.append('./models/backbone')
sys.path.append(os.getcwd())

from models.base import BaseEval
from utils.helpers import json_to_dict

warnings.filterwarnings("ignore")


class JsonScoreEvaluator(BaseEval):
    """
        Evaluates anomaly detection performance based on pre-computed scores stored in JSON files.

        This class extends the BaseEval class and specializes in reading scores from JSON files,
        computing evaluation metrics, and optionally saving results to Excel format.

        Notes:
            - Score files are expected to follow the MVTec AD dataset structure.
                    `{category}/{split}/{anomaly_type}/{image_name}_scores.json`
              e.g., `cable/test/good/039_scores.json`
            - Score files are expected be at `self.score_dir`.
            - Evaluation metrics are calculated using methods inherited from BaseEval.

        Example usage:
            >>> evaluator = JsonScoreEvaluator(cfg)
            >>> results = evaluator.main()
    """

    def __init__(self, cfg, seed=0):
        super().__init__(cfg, seed)
        self.preprocess = None
        print(f'Overriding should_save_scores=False, save_excel=True')
        self.should_save_scores = False
        self.save_excel = True

    def get_scores_for_image(self, image_path):
        image_scores_path = self.get_scores_path_for_image(image_path)
        image_scores = json_to_dict(image_scores_path)

        return image_scores

    def get_category_scores(self, category):
        # divide sub-datasets
        divide_num = self.divide_num

        anomaly_maps = []  # pr_px
        cls_scores_list = []  # pr_sp
        gt_list = []  # gt_sp
        img_masks = []  # gt_px

        image_path_list = []
        for divide_iter in range(divide_num):
            test_dataset = self.load_datasets(category, divide_num=divide_num, divide_iter=divide_iter)
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            for image_info in tqdm(test_dataloader):
                if not isinstance(image_info, dict):
                    raise ValueError('Encountered non-dict image in dataloader')

                del image_info["image"]

                image_path = image_info["image_path"][0] # 0 because of batch_size = 1
                image_path_list.extend(image_path)

                img_masks.append(image_info["mask"])
                gt_list.extend(list(image_info["is_anomaly"].numpy()))

                image_scores = self.get_scores_for_image(image_path)
                cls_scores = image_scores['img_level_score']
                anomaly_maps_iter = image_scores['pix_level_score']

                cls_scores_list.append(cls_scores)
                anomaly_maps.append(anomaly_maps_iter)

        pr_sp = np.array(cls_scores_list)
        gt_sp = np.array(gt_list)
        pr_px = np.array(anomaly_maps)
        gt_px = torch.cat(img_masks, dim=0).numpy().astype(np.int32)
        print(f'pr_sp: {pr_sp.shape} | gt_sp: {gt_sp.shape} | gt_px: {gt_px.shape} | pr_px: {pr_px.shape}')

        return gt_sp, pr_sp, gt_px, pr_px, image_path_list
