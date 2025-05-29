import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mmcv
import torch

from mmcv.transforms import Compose
from mmdet.apis import init_detector, inference_detector


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


def visualize_results(model, img_path, result, title='', save_dir='./results/', score_thr=0.3):
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if 'mask_rcnn' in title.lower():
        img_vis = draw_mask_rcnn_result(img_rgb, result, score_thr=score_thr)
    else:
        img_vis = draw_sparse_rcnn_result(img_rgb, result, score_thr=score_thr)

    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}_{os.path.basename(img_path)}")
    plt.imsave(save_path, img_vis)
    print(f"{title} 可视化结果已保存到: {save_path}")


def get_rpn_proposals(model, img_path):
    import mmcv
    from mmengine.dataset import Compose

    pipeline = Compose([
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        dict(type='PackDetInputs')
    ])

    data = dict(img_path=img_path)
    data = pipeline(data)

    if data['inputs'].dim() == 3:
        data['inputs'] = data['inputs'].unsqueeze(0)

    # 将单个 DetDataSample 包装成 list
    if not isinstance(data['data_samples'], (list, tuple)):
        data['data_samples'] = [data['data_samples']]

   
    processed = model.data_preprocessor(data, training=False)


    inputs = processed['inputs'].to(next(model.parameters()).device)
    data_samples = processed['data_samples']
    
    
   
    feats = model.backbone(inputs)
    feats = model.neck(feats)

    proposal_list = model.rpn_head.predict(feats, data_samples, model.test_cfg.rpn)
   
    proposals = proposal_list[0]  # 第一个图像的 proposals，类型是 InstanceData
    bboxes = proposals.bboxes  # Tensor 或 ndarray，形状是 (N, 4)
    return bboxes.cpu().numpy()  # 返回 numpy 数组

    





def visualize_maskrcnn_with_proposals(model, img_path, score_thr=0.3, save_dir='./results_with_proposals/'):
    os.makedirs(save_dir, exist_ok=True)

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_draw = img_rgb.copy()

    result = inference_detector(model, img_path)
    result.pred_instances = result.pred_instances[result.pred_instances.scores > 0.05]

    print(result.pred_instances)
    
    img_draw = draw_mask_rcnn_result(img_draw, result, score_thr=score_thr)

    proposals = get_rpn_proposals(model, img_path)
    for prop in proposals:
        x1, y1, x2, y2 = prop.astype(int)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 1)

    save_path = os.path.join(save_dir, f"mask_rcnn_with_proposals_{os.path.basename(img_path)}")
    plt.imsave(save_path, img_draw)
    print(f"Mask R-CNN + RPN proposals 可视化结果已保存到: {save_path}")


def main():
    mask_work_dir = './work_dirs/mask_rcnn_voc_1x'
    sparse_work_dir = './work_dirs/sparse_rcnn_voc_1x'

    mask_config = 'configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
    sparse_config = 'configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py'

    mask_checkpoint = get_latest_checkpoint(mask_work_dir)
    sparse_checkpoint = get_latest_checkpoint(sparse_work_dir)

    print(f"Mask R-CNN 权重路径: {mask_checkpoint}")
    print(f"Sparse R-CNN 权重路径: {sparse_checkpoint}")

    mask_model = init_detector(mask_config, mask_checkpoint, device='cuda:0')
    sparse_model = init_detector(sparse_config, sparse_checkpoint, device='cuda:0')

    test_images = [
        './data/VOCdevkit/VOC2007/JPEGImages/000001.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000002.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000003.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000004.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000005.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000006.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000007.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000008.jpg',
        './data/VOCdevkit/VOC2007/JPEGImages/000009.jpg',
    ]

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"图片路径不存在，跳过: {img_path}")
            continue

        mask_results = inference_detector(mask_model, img_path)
        visualize_results(mask_model, img_path, mask_results, title='Mask_R-CNN_Result')

        sparse_results = inference_detector(sparse_model, img_path)
        visualize_results(sparse_model, img_path, sparse_results, title='Sparse_R-CNN_Result')

        visualize_maskrcnn_with_proposals(mask_model, img_path, score_thr=0.3)


if __name__ == '__main__':
    main()
