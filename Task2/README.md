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

Output file:
- Mask RCNN
    - Training Log: Task2/output/mask_rcnn_r50_fpn_1x_voc/20250527_160710.json
    - Config File: Task2/output/mask_rcnn_r50_fpn_1x_voc/config.py
    - Scalars: Task2/output/mask_rcnn_r50_fpn_1x_voc/scalars.json
    - Train loss/ val map json: Task2/output/mask_rcnn_r50_fpn_1x_voc/20250527_160710.json

- Sparse RCNN:
    - Training Log: Task2/output/sparse_rcnn_r50_fpn_1x_voc/20250528_121748.log
    - Config File: Task2/output/sparse_rcnn_r50_fpn_1x_voc/config.py
    - Scalars: Task2/output/sparse_rcnn_r50_fpn_1x_voc/scalars.json
    - Train loss/ val map json: Task2/output/sparse_rcnn_r50_fpn_1x_voc/20250528_121748.json