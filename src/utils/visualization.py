import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import cv2
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from typing import List, Dict
from src.wsilib import WSITile, AnnotatedWSI, WSI

def visualize_masks(wsi_name:str, matched_results:Dict, wsi_folder:Path, wsi_name_index_map:Dict, slide_dataset:torch.utils.data.Dataset, alpha:float=0.4):
    """
    Visualizes the original image with overlayed masks from different methods.

    Args:
        wsi_name (str): The name of the whole slide image (WSI) to visualize.
        matched_results (dict): A dictionary containing the results from different methods.
        wsi_folder (Path): The folder containing the WSI files.
        slide_dataset (Dataset): The dataset containing the images and their annotations.
        alpha (float): The transparency level for the overlayed masks. Default is 0.4.

    """
    
    # Define colors for each method (RGB format)
    method_colors = {
        "Tile-based": (242, 137, 99),       # F28963
        "Semantic segmentation": (68, 202, 73),   # 44CA49
        "Instance segmentation": (120, 132, 196)  # 7884C4
    }
    
    # Find the image in the dataset
    original_image = None
    ground_truth_mask = None
    
    for idx in range(len(slide_dataset)):
        name = wsi_name_index_map.get(idx)
        
        folder_name = name.upper().replace('-', '_') 
        if folder_name.rfind("_40") != -1:
            folder_name = folder_name[:folder_name.rfind("_40")]
        if folder_name.rfind("HE_") != -1:
            folder_name = folder_name[:folder_name.rfind("HE_")+2]
            
        wsi_path = wsi_folder / folder_name
        # Check path validity
        if not wsi_path.exists():
            raise FileNotFoundError(f"WSI path '{wsi_path}' does not exist.")

        if name == wsi_name:
            # Get the image and its annotations
            wsi = WSI(wsi_path)
            level_size = wsi.level_dimensions
            print(level_size)

            image, target = slide_dataset[idx]
            
            # Convert tensor to numpy for plotting
            original_image = image.permute(1, 2, 0).cpu().numpy()
            
            # Normalize image if needed (if values > 1.0)
            if original_image.max() > 1.0:
                original_image = original_image / 255.0
                
            # Get ground truth mask (binary)
            if 'masks' in target and len(target['masks']) > 0:
                ground_truth_mask = target['masks'][0].cpu().numpy()
            break
    
    if original_image is None:
        print(f"WSI '{wsi_name}' not found in the dataset")
        return
    
    # Find predictions for this WSI
    method_masks = {}
    
    for i, name in enumerate(matched_results['wsi_name']):
        if name == wsi_name:
            method = matched_results['method'][i]
            prediction = matched_results['prediction'][i]
            if prediction is not None:
                method_masks[method] = prediction
            else:
                print(f"Did not found prediction for method '{method}'.")
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Show original image
    plt.imshow(original_image)
    plt.title(f"WSI: {wsi_name}", fontsize=16)
    
    # Create legend patches
    legend_elements = []
    
    # Add ground truth contour (black)
    if ground_truth_mask is not None:
        # Find contours
        contours, _ = cv2.findContours(
            (ground_truth_mask * 255).astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Plot contours
        for contour in contours:
            plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'k-', linewidth=2)
        
        legend_elements.append(Patch(facecolor='black', edgecolor='black', label='Ground Truth', alpha=0.7))
    
    print(image.shape)
    for method, mask in method_masks.items():
        print(mask.shape)

    # Add prediction contours and semi-transparent masks
    for method, mask in method_masks.items():
        if mask is None or not isinstance(mask, (np.ndarray, torch.Tensor)):
            raise ValueError(f"Invalid mask for method '{method}'. Expected numpy array or torch tensor.")
            
        # Convert to numpy if tensor
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

             # Handle different tensor shapes
            if mask.ndim == 4:  # Shape: [1, 1, H, W]
                mask = mask.squeeze(0).squeeze(0)
            elif mask.ndim == 3:  # Shape: [1, H, W]
                mask = mask.squeeze(0)
        
        # Ensure binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Get color
        color = method_colors.get(method, (255, 0, 0))  # Default to red if not found
        
        # Find contours
        contours, _ = cv2.findContours(
            (binary_mask * 255).astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Plot contours with method color
        for contour in contours:
            plt.plot(contour[:, 0, 0], contour[:, 0, 1], color=[c/255 for c in color], linewidth=2)
        
        # Create colored mask overlay with transparency
        colored_mask = np.zeros_like(original_image)
        colored_mask[:,:,0] = color[0]/255
        colored_mask[:,:,1] = color[1]/255
        colored_mask[:,:,2] = color[2]/255
        
        # Apply the mask with alpha blending
        mask_overlay = np.zeros_like(original_image)
        mask_overlay[binary_mask > 0] = colored_mask[binary_mask > 0]
        # plt.imshow(mask_overlay, alpha=alpha)
        
        # Add to legend
        legend_elements.append(Patch(facecolor=np.array(color)/255, 
                                     edgecolor=np.array(color)/255, 
                                     label=method, 
                                     alpha=alpha))
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    save_dir = Path('results/figures/mask_overlays')
    save_dir.mkdir(exist_ok=True, parents=True)
    # plt.savefig(save_dir / f"{wsi_name}_mask_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()