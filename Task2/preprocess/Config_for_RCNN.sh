# 拷贝配置文件（以 Mask R-CNN 为例）
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html  # 替换为你实际使用的 CUDA 和 torch 版本

cp configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py configs/voc/mask-rcnn_r50_fpn_1x_voc.py
cp configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py configs/voc/sparse-rcnn_r50_fpn_1x_voc.py