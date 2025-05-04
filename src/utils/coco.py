import numpy as np
import os
import json
from skimage import measure
from sklearn.model_selection import train_test_split
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple

from src.utils.pretty_print import *

class COCODataset(Dataset):
    """Dataset for COCO formatted data with train/test splitting capability."""
    
    def __init__(
        self,
        annotation_file: Path | str,
        train: bool = True,
        train_ratio: float = 0.8,
        transform: Optional[Callable] = None,
        random_seed: int = 42,
        bbox_format: str = "coco",
        split_file: Optional[Path | str] = None
    ) -> None:
        """Initialize dataset.
        
        Args:
            annotation_file: Path to COCO annotation file
            train: If True, creates training dataset, else test dataset
            train_ratio: Ratio of images to use for training (default: 0.8)
            transform: Optional transform to be applied
            random_seed: Random seed for reproducible splits
            bbox_format: Format of bounding boxes (default: "coco"). Available formats:
                - "coco": [x, y, w, h]
                - "pascal_voc": [x1, y1, x2, y2]
            split_file: Optional path to a JSON file containing predefined train/test splits.
                        If provided, train_ratio is ignored.
        """

        self.annotation_file = Path(annotation_file)
        self.train = train
        self.train_ratio = train_ratio
        self.transform = transform
        self.random_seed = random_seed
        self.bbox_format = bbox_format
        self.split_file = split_file
        
        self.coco = COCO(annotation_file)
        
        all_image_ids = list(sorted(self.coco.imgs.keys()))
        
        # If split file is provided, use it for train/test split
        if split_file is not None:
            self._split_from_file(all_image_ids)
        else:
            # Otherwise random split
            train_ids, test_ids = train_test_split(
                all_image_ids,
                train_size=train_ratio,
                random_state=random_seed,
                shuffle=True
            )
            
            # Select appropriate images
            self.image_ids = train_ids if train else test_ids
        
        # Category mapping
        self.categories = {cat['id']: cat['name'] 
                         for cat in self.coco.loadCats(self.coco.getCatIds())}
        
        print_success(f"Loaded {'training' if train else 'test'} set with {len(self.image_ids)} images\n")

    def _split_from_file(self, all_image_ids: List[int]) -> None:
        """Load predefined train/test split from a JSON file.
        
        Args:
            all_image_ids: List of all image IDs in the dataset
        """
        try:
            with open(self.split_file, 'r') as f:
                split_data = json.load(f)
                
            if "train_set" not in split_data or "test_set" not in split_data:
                raise ValueError(f"Split file must contain 'train_set' and 'test_set' keys")
                
            # Get image names to use for filtering
            train_filenames = set(split_data["train_set"])
            test_filenames = set(split_data["test_set"])
            
            # Filter image IDs of the COCO dataset based on the source filename in the split file
            train_ids = []
            test_ids = []
            
            for image_id in all_image_ids:
                img_info = self.coco.loadImgs(image_id)[0]
                wsi_source = img_info.get("wsi_source", "")
                
                # Remove extension
                if wsi_source.endswith(".mrxs"):
                    wsi_source = wsi_source[:-5]
                
                # Check which set this image belongs to
                if wsi_source in train_filenames:
                    train_ids.append(image_id)
                elif wsi_source in test_filenames:
                    test_ids.append(image_id)
            
            self.image_ids = train_ids if self.train else test_ids
            
            # Checks
            if len(train_ids) == 0 and self.train:
                raise ValueError("No training images found with the given split file")
            if len(test_ids) == 0 and not self.train:
                raise ValueError("No test images found with the given split file")
                
            print_success(f"Loaded split from {self.split_file}")
            print_success(f"Found {len(train_ids)} training images and {len(test_ids)} test images")
            
        except Exception as e:
            print_error(f"Error loading split file: {e}")
            print_warning("Falling back to random split")
            
            # Fallback to random split
            train_ids, test_ids = train_test_split(
                all_image_ids,
                train_size=self.train_ratio,
                random_state=self.random_seed,
                shuffle=True
            )
            
            # Select appropriate images
            self.image_ids = train_ids if self.train else test_ids

    def __getitem__(self, idx: int) -> Dict:
        """Get one image from the dataset.
        
        Args:
            idx: Index of the image
            
        Returns:
            Dictionary containing:
                - image: Tensor of shape [C, H, W]
                - target: Dictionary with annotations
        """

        # Load image info
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        
        # Load image
        image_path = self.annotation_file.parent / image_info['file_name']
        image = Image.open(image_path).convert('RGB')
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Prepare target
        boxes = []
        labels = []
        masks = []

        # Expected target format by Mask R-CNN
        """
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
        """
        
        for ann in annotations:
            # Get bbox
            if "bbox" in ann.keys() and ann["bbox"]:
                x, y, w, h = ann['bbox']

                # Convert to apropriate format (is in COCO format by default)
                if self.bbox_format == "coco":
                    boxes.append([x, y, w, h])
                elif self.bbox_format == "pascal_voc":
                    # Pascal VOC format: [x1, y1, x2, y2]
                    boxes.append([x, y, x + w, y + h])
                else:
                    raise ValueError(f"Unknown bbox format: {self.bbox_format}. Use 'coco' or 'pascal_voc'.")

            # Get label
            labels.append(ann['category_id'])
            
            # Get mask
            if "segmentation" in ann.keys() and ann["segmentation"]:
                mask = self.coco.annToMask(ann)
                masks.append(mask)
        
        # Convert to tensors
        # "bboxes" and "boxes" are both present because of a KeyError in the model's code
        # When only using one, the model throws a KeyError trying to access the other 
        target = {
            "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "bboxes": torch.zeros((0, 4), dtype=torch.float32),
            "masks": torch.zeros((0, image.height, image.width), dtype=torch.uint8)
        }

        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["bboxes"] = torch.as_tensor(boxes, dtype=torch.float32)

        if len(masks) > 0:
            target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)

        
        # Apply transforms if any
        if self.transform is not None:
            transformed = self.transform(image=np.array(image),
                                      masks=np.array(masks),
                                      bboxes=np.array(boxes),
                                      class_labels=labels)
            
            image = transformed["image"].to(torch.float32)
            target["masks"] = torch.as_tensor(transformed["masks"], dtype=torch.uint8)
            target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            target["bboxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)

        else:
            image = torch.as_tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
        
        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)
    
    
def convert_predictions_to_coco_format(predictions, annotations):
    """
    Converts Mask R-CNN predictions to COCO format.
    
    Args:
        predictions (list): List of predictions from the model.
        annotations (list): List of annotations for the images.
        
    Returns:
        coco_predictions (list): List of dictionaries in COCO format.
    """
    coco_predictions = []

    for i, prediction in enumerate(predictions):
        image_id = int(annotations[i]["image_id"].cpu().numpy()[0])
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        masks = prediction["masks"].cpu().numpy()

        for j in range(len(boxes)):
            bbox = boxes[j].tolist()
            score = float(scores[j])
            label = int(labels[j])
            mask = masks[j]

            # Convert bbox to COCO format [x, y, width, height]
            x_min, y_min, x_max, y_max = bbox
            bbox_coco = [x_min, y_min, x_max - x_min, y_max - y_min]

            """# Thresholding mask to binary
            binary_mask = (mask.squeeze(0) >= 0.5).astype(np.uint8)
            # Converting binary mask to RLE
            binary_mask_fortran = np.asfortranarray(binary_mask)
            rle = maskUtils.encode(binary_mask_fortran)
            # Converting bytes to string for JSON serialization
            rle['counts'] = rle['counts'].decode('utf-8')
            """

            contours = measure.find_contours(mask.squeeze(), 0.5)

            contour = np.flip(contours[0], axis=1)
            segmentation = contour.ravel().tolist()

            prediction_dict = {
                "image_id": image_id,
                "category_id": label,
                "bbox": bbox_coco,
                "score": score,
                "segmentation": [segmentation]
            }

            coco_predictions.append(prediction_dict)

    return coco_predictions

def save_predictions_in_coco_format(predictions, annotations, save_path):
    coco_predictions = convert_predictions_to_coco_format(predictions, annotations)
    with open(os.path.join(save_path, 'predictions.json'), 'w') as f:
        json.dump(coco_predictions, f)