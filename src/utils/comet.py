import comet_ml
import torch
from torch.utils.data import DataLoader
from PIL import Image
from skimage import measure
import numpy as np
from torchvision.transforms.functional import to_pil_image

from src.utils.image_tools import get_image

class CometLogger(comet_ml.Experiment):
    """
    Child of the comet_ml.Experiment class with the main purpose to implement image logging with the
    same functionalities as the parent.

    """
    def __init__(self, experiment_name: str, *args, **kwargs) -> None:
        """
        Creates a new experiment on the Comet.ml frontend.

        Args:
            experiment_name (string): name of the experiment inside the project.

            Other options can be retrieved from the comet_ml.Experiment class.
        """
        super().__init__(*args, **kwargs)
        self.set_name(experiment_name)

    def convert_predictions_to_comet_annotations(self, pred: dict) -> list[dict]:
        """
        Convert model predictions to comet_ml annotation format.

        Args:
            predictions (dict): Model predictions containing boxes, binary masks, labels, and scores.

        Returns:
            List[dict]: A list of annotations in comet_ml format.
        """

        boxes = pred["boxes"]
        scores = pred["scores"]
        masks = pred["masks"]

        # If there are no predictions
        if boxes == []:
            annotation_data = [{
                "boxes": [],
                "label": None,
                "score": None
            }]
            return [{"name": "Predictions", "data": annotation_data}]
        
        else:
            annotation_data = []

            # Iterating over the bounding boxes, masks and corresponding scores
            for i, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
                
                # Converting box to COCO format
                box = box.cpu().detach().tolist()

                xmin = box[0]
                ymin = box[1]
                width = box[2] - xmin
                height = box[3] - ymin

                box = [xmin, ymin, width, height]

                # Converting binary mask to polygon
                contours = measure.find_contours(mask.cpu().numpy().squeeze(), 0.5)
                points = []

                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    segmentation = contour.ravel().tolist()
                    points.append(segmentation)

                # Filling annotation data array
                annotation_data.append({
                    "boxes": [box],
                    "points": points,
                    "label": pred["labels"][i],
                    "score": np.round(score.item(), 3)
                })

            return [{"name": "Predictions", "data": annotation_data}]

    def upload_images(self, 
                      model, 
                      model_type: str="maskrcnn",
                      device:str="cuda", 
                      dataloader: torch.utils.data.DataLoader=None, 
                      confidence_thr: float=0.5, 
                      entry_name: str="sample_image", 
                      overwrite: bool=True, 
                      indices: list=[],
                      ) -> None:
        """Uploads images to the comet.ml frontend

        Args:
            model (_type_): variable containing the trained model.
            model_type (str): type of the model. Supported: "unet", "maskrcnn".
            device (str): device name.
            dataset (torch.utils.data.DataLoader): DataLoader for images.
            confidence_thr (float): predictions having equal or higher confidence score will be accepted and visualized.
            entry_name (str): name of the image in comet.ml.
            overwrite (bool, optional): whether to overwrite the images in the cloud with existing names. Defaults to True.
            indices (list, optional): the indices of the images to be selected from the dataset. Defaults to [].

        Raises:
            ValueError: The length of the indices cannot be zero.
        """    

        model.eval()
        model.to(device)

        num_samples = len(indices)

        if num_samples == 0:
            raise ValueError("Number of samples cannot be zero")

        # Preparing images
        imgs=[]
        
        dataset = dataloader.dataset
        for i in indices:
            img = dataset[i][0]
            img = img.squeeze().to(device)
            imgs.append(img)
            
        imgs = [img.to(torch.float32) for img in imgs]
        
        # Prediction
        with torch.no_grad():
            preds = model(imgs)

        # Filtering predictions based on confidence threshold
        for p, pred in enumerate(preds):
            valid_preds={"masks": [], "boxes": [], "labels": [], "scores" : []}
            for i, score in enumerate(pred["scores"]): 
                if score >= confidence_thr:
                    for k in valid_preds.keys():
                        valid_preds[k].append(pred[k][i])
            preds[p] = valid_preds

        for i, pred in enumerate(preds):

            annotations = self.convert_predictions_to_comet_annotations(pred)
            
            img = imgs[i].cpu()
            if img.dtype != torch.uint8:
                img = img.to(torch.uint8)
            if img.min() < 0 or img.max() > 255:
                img = torch.clamp(img, 0, 255)

            img = to_pil_image(img, mode="RGB")
            
            # Upload the image with annotations to Comet
            self.log_image(image_data=img, overwrite=overwrite, name=f"{entry_name}_{i}", annotations=annotations)