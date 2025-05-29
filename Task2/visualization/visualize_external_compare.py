import os
import cv2
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector

import numpy as np
import torch


def get_latest_checkpoint(work_dir):
    last_ckpt_path = os.path.join(work_dir, 'last_checkpoint')
    if os.path.exists(last_ckpt_path):
        with open(last_ckpt_path, 'r') as f:
            ckpt_name = f.read().strip()
        checkpoint_file = os.path.join(work_dir, ckpt_name)
        if os.path.exists(checkpoint_file):
            return checkpoint_file
        else:
            raise FileNotFoundError(f"权重文件 {checkpoint_file} 不存在")
    else:
        raise FileNotFoundError(f"last_checkpoint 文件不存在于 {last_ckpt_path}")


def draw_mask_rcnn_result(img, result, score_thr=0.3, alpha=0.5):
    img = img.copy()
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy() if pred_instances.bboxes is not None else np.array([])
    scores = pred_instances.scores.cpu().numpy() if pred_instances.scores is not None else np.array([])
    labels = pred_instances.labels.cpu().numpy() if pred_instances.labels is not None else np.array([])
    masks = pred_instances.masks.cpu().numpy() if hasattr(pred_instances, 'masks') and pred_instances.masks is not None else None


    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i in range(len(bboxes)):
        if scores[i] < score_thr:
            continue
        x1, y1, x2, y2 = bboxes[i].astype(int)
        cls_id = labels[i]
        color = colors[cls_id % len(colors)]
        color_bgr = (color[2], color[1], color[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
        label = f'cls{cls_id}: {scores[i]:.2f}'
        cv2.putText(img, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
        
        if masks is None:
            print("未检测到分割 mask")
        else:
            print(f"检测到 {len(masks)} 个 mask")


        if masks is not None:
            mask = masks[i]
            if mask.dtype != np.uint8:
               mask = (mask > 0.5).astype(np.uint8) * 255  # 推荐处理方式
            mask_img = np.zeros_like(img, dtype=np.uint8)
            mask_img[mask > 0] = color_bgr
            img = cv2.addWeighted(img, 1.0, mask_img, alpha, 0)

    return img


def draw_sparse_rcnn_result(img, result, score_thr=0.3):
    img = img.copy()
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy() if pred_instances.bboxes is not None else np.array([])
    scores = pred_instances.scores.cpu().numpy() if pred_instances.scores is not None else np.array([])
    labels = pred_instances.labels.cpu().numpy() if pred_instances.labels is not None else np.array([])

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i in range(len(bboxes)):
        if scores[i] < score_thr:
            continue
        x1, y1, x2, y2 = bboxes[i].astype(int)
        cls_id = labels[i]
        color = colors[cls_id % len(colors)]
        color_bgr = (color[2], color[1], color[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
        label = f'cls{cls_id}: {scores[i]:.2f}'
        cv2.putText(img, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

    return img



def visualize_external_image(img_path, mask_model, sparse_model, save_dir='./external_results/', score_thr=0.3):
    os.makedirs(save_dir, exist_ok=True)

    # 加载图像
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mask R-CNN 推理与可视化
    mask_result = inference_detector(mask_model, img_path)
    img_mask = draw_mask_rcnn_result(img_rgb, mask_result, score_thr=score_thr)
    save_path_mask = os.path.join(save_dir, f'mask_rcnn_{os.path.basename(img_path)}')
    plt.imsave(save_path_mask, img_mask)
    print(f'Mask R-CNN 可视化保存于: {save_path_mask}')

    # Sparse R-CNN 推理与可视化
    sparse_result = inference_detector(sparse_model, img_path)
    img_sparse = draw_sparse_rcnn_result(img_rgb, sparse_result, score_thr=score_thr)
    save_path_sparse = os.path.join(save_dir, f'sparse_rcnn_{os.path.basename(img_path)}')
    plt.imsave(save_path_sparse, img_sparse)
    print(f'Sparse R-CNN 可视化保存于: {save_path_sparse}')


def main():
    # 模型工作路径
    mask_work_dir = './work_dirs/mask_rcnn_voc_1x'
    sparse_work_dir = './work_dirs/sparse_rcnn_voc_1x'

    # 模型配置与权重
    mask_config = 'configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
    sparse_config = 'configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py'
    mask_checkpoint = get_latest_checkpoint(mask_work_dir)
    sparse_checkpoint = get_latest_checkpoint(sparse_work_dir)

    # 初始化模型
    mask_model = init_detector(mask_config, mask_checkpoint, device='cuda:0')
    sparse_model = init_detector(sparse_config, sparse_checkpoint, device='cuda:0')

    # 3张包含 VOC 类别但不属于 VOC 的测试图像
    external_images = [
        './data/VOCdevkit/VOC2007/JPEGImages/000001.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000002.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000003.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000004.jpg',
         './data/VOCdevkit/VOC2007/JPEGImages/000005.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000006.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000007.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000008.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000009.jpg',
        './test_images/bike_street.png',
        './test_images/cat_chair.png',
        './test_images/person_dog_car.png'
    ]

    for img_path in external_images:
        if not os.path.exists(img_path):
            print(f"图像不存在，跳过: {img_path}")
            continue
        visualize_external_image(img_path, mask_model, sparse_model)

if __name__ == '__main__':
    main()
