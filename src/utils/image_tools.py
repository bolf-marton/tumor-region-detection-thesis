from pathlib import Path
import numpy as np
from PIL import Image
from typing import Union, List, Tuple

def get_image(image_path: Union[Path, Tuple[Path, ...]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Load and return the image(s) corresponding to the given file path(s).
    
    Args:
        image_path (Path | tuple[Path]): Path or tuple of paths to image files
        
    Returns:
        np.ndarray | list[np.ndarray]: Loaded image(s) as float32 numpy array(s) normalized to [0,1]
    """
    if isinstance(image_path, tuple):
        images = []
        for path in image_path:
            img = np.array(Image.open(path)).astype(np.float32)
            img /= 255
            images.append(img)
        return images
    elif isinstance(image_path, (str, Path)):
        path = Path(image_path)
        img = np.array(Image.open(path)).astype(np.float32)
        img /= 255
        return img
    else:
        raise ValueError("Wrong input format provided. Please use Path or tuple of Paths")