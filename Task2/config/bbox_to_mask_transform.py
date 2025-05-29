import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures.mask import BitmapMasks


@TRANSFORMS.register_module()
class BboxToMask(BaseTransform):
    """Convert bounding boxes to masks.
    
    This transform generates rectangular masks from bounding boxes,
    which is useful for datasets that only have bbox annotations
    but need to train instance segmentation models.
    """
    
    def transform(self, results):
        """Transform function to convert bboxes to masks.
        
        Args:
            results (dict): Result dict containing the data to transform.
            
        Returns:
            dict: The result dict with masks added.
        """
        if 'gt_bboxes' not in results:
            return results
            
        gt_bboxes = results['gt_bboxes']
        img_shape = results['img_shape']
        
        if len(gt_bboxes) == 0:
            # No bboxes, create empty masks
            masks = np.zeros((0, img_shape[0], img_shape[1]), dtype=np.uint8)
        else:
            # Create rectangular masks from bboxes
            masks = []
            # Convert HorizontalBoxes to numpy array
            bbox_array = gt_bboxes.tensor.cpu().numpy()
            
            for bbox in bbox_array:
                mask = np.zeros(img_shape[:2], dtype=np.uint8)
                x1, y1, x2, y2 = bbox.astype(int)
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, img_shape[1] - 1))
                y1 = max(0, min(y1, img_shape[0] - 1))
                x2 = max(0, min(x2, img_shape[1] - 1))
                y2 = max(0, min(y2, img_shape[0] - 1))
                
                if x2 > x1 and y2 > y1:
                    mask[y1:y2+1, x1:x2+1] = 1
                masks.append(mask)
            
            masks = np.stack(masks, axis=0)
        
        # Convert to BitmapMasks format
        results['gt_masks'] = BitmapMasks(masks, img_shape[0], img_shape[1])
        
        return results
    
    def __repr__(self):
        return f'{self.__class__.__name__}()' 