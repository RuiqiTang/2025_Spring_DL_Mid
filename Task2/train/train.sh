# train for Mask R-CNN
mkdir Task2/output/mask_rcnn_r50_fpn_1x_voc
python \
    Task2/mmdetection/tools/train.py    \
    Task2/mmdetection/configs/voc/mask-rcnn_r50_fpn_1x_voc.py  \
    Task2/output/mask_rcnn_r50_fpn_1x_voc/latest.pth    \
    --out results_mask_rcnn.pkl \
    --show-dir vis_mask_rcnn

# train for Sparse R-CNN
mkdir Task2/output/sparse_rcnn_r50_fpn_1x_voc
python  \
    Task2/mmdetection/tools/train.py    \
    Task2/mmdetection/configs/voc/sparse-rcnn_r50_fpn_1x_voc.py   \
    Task2/output/sparse_rcnn_r50_fpn_1x_voc/latest.pth  \
    --out results_sparse_rcnn.pkl   \
    --show-dir vis_sparse_rcnn
