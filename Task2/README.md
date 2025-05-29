Model Configs:
- Config for masked rcnn: Task2/config/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py
- Config for sparse rcnn: Task2/config/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py

Train File:
- Task2/tools/train.py

Visualization Files
- Task2/visualization/visualize_log.py
- Task2/visualization/visualize_voc_compare.py
- Task2/visualization/visualize_external_compare.py

Other file: use bbox as mask for VOC2017 dataset
- Task2/config/bbox_to_mask_transform.py