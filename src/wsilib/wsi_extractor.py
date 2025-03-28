import openslide
import numpy as np
from pathlib import Path
from .classes import WSI, WSITile, WSITileContainer
import xml.etree.ElementTree as ET


def load_WSI(image:WSI, level:int) -> tuple[np.array, float, tuple]:
        """Returns the selected level of the Whole Slide Image as a numpy array.

        Args:
            image (WSI): Whole Slide Image instance.
            level (int): Selected layer.

        Returns:
            tuple[np.array, float, tuple] : desired image layer as a numpy array, scale factor compared to the highest resolution layer (level = 0), image dimensions (resolution).
        """

        with openslide.open_slide(image.mrxs_path) as slide:
            try:
                if level >= len(slide.level_dimensions):
                    raise IndexError(f"Level {level} does not exist. Available levels: 0-{len(slide.level_dimensions)-1}")
                    
                level_dimensions = slide.level_dimensions[level] # Get the dimensions of the selected level
                
                scale_factor = slide.level_downsamples[level] # Get the scale factor compared to the highest resolution layer (level = 0)
    
                region = slide.read_region((0, 0), level, level_dimensions) # Read the selected level
                
                image = np.asarray(region) # Convert to np array
                
                return image, scale_factor, level_dimensions
                
            except IndexError as e:
                raise IndexError(str(e))
            
# Custom exception for annotation-related errors
class AnnotationError(Exception):
    """Custom exception for annotation-related errors"""
    pass

def get_annotations(image:WSI, level:int, tags:list = ["header", 'object_privilege version="0100"']) -> tuple:
        """Get the annotations (segmentations, bounding boxes) for the given image.

        Args:
            image (WSI): Whole Slide Image instance.
            level (int): Selected layer. The annotations are rescaled to match the dimensions of this layer.
            tags (list[str]): List of XML tags (present only in the annotation file) to search for in the .dat files. Default: ["header", 'object_privilege version="0100"'].

        Returns:
            The corresponding bounding boxes and segmentation polygons.
        """

        # List of files in the data folder
        files = list(image.data_path.glob("*.dat"))

        # Get the image dimensions
        with openslide.open_slide(image.mrxs_path) as slide:
            image_sizes = slide.dimensions
            image_scale = slide.level_downsamples[level]

        # Iterate through all .dat files
        for file in files:

            # Check if all tags are found in the file
            found_tags = all(find_xml_start(file_path=file, tag=tag) != -1 for tag in tags)

            if not found_tags:
                continue
            
            #############################################################################
            # ↓ This part only runs for one of the .dat files which has the annotations #
            #############################################################################

            xml_content = extract_xml_from_dat(file)

            # Parsing xml
            try:
                xml_root = ET.fromstring(xml_content)
            except ET.ParseError as e:
                raise ET.ParseError(f"Failed to parse XML in file {file}: {e}")

            # Annotations are present inside the <data> tags
            annotations = xml_root.findall('.//data')

            bounding_boxes = []
            segmentations = []
            
            # Iterate through each <data> element to plot annotations
            for annotation in annotations:
                
                # Skipping annotation if it has been deleted (but still stored in the file)
                descriptor = annotation.find('object_privilege/descriptor')
                deletion_date = descriptor.get('DeletionDate', '')

                if deletion_date != "":
                    continue
                
                # Reading all the points of the annotation polygon
                points = extract_polygon_points(annotation)

                # flattening x, y pairs to one list
                points_flat = np.array(points).flatten().tolist()

                if points_flat:
                    scaled_points = []
                    # scale segmentation polygon coordinates
                    scaled_points = [p / image_scale for p in points_flat]

                    # Separate the points into X and Y coordinates
                    x_coords, y_coords = zip(*points)

                    # Create bounding boxes
                    bb = create_bounding_box(x_coords, y_coords, image_scale, image_sizes)

                    if bb is not None:
                        top_left_x, top_left_y, width, height = bb
                    else:
                        continue
                    
                    segmentations.append(scaled_points)
                    bounding_boxes.append([top_left_x, top_left_y, width, height])
            
            if bounding_boxes == [] or segmentations == []:
                raise AnnotationError("No annotations found on the image")

            return bounding_boxes, segmentations
        
        raise AnnotationError("No files found matching the annotation tags")

def find_xml_start(file_path:str, tag:str) -> int:
    """Searches for the tag in the file at file_path.

    Args:
        file_path: Path of the .dat file.
        tag: The name of the XML tag (without "<" and ">") indicating the beggining of the file, eg. in case of <header> -> tag = header.

    Returns:
        Starting index of the XML formatted part in the file.
    """

    with open(file_path, 'rb') as file:
        buffer_size = 2048
        buffer = b""
        offset = 0
        
        while True:
            # Reading buffer sized region from the beggining of the file
            chunk = file.read(buffer_size)
            if not chunk:
                return -1
            buffer += chunk
            
            # Search for the specified tag in the buffer
            xml_start = buffer.find(f'<{tag}>'.encode())
            if xml_start != -1:
                return offset + xml_start
            
            # Move the offset and buffer window
            offset += buffer_size
            buffer = buffer[-buffer_size:]  # Keep the last part of the buffer in case the tag is split between chunks

def extract_xml_from_dat(file_path:str) -> str:
    """Extracts the XML file from the given .dat file.

    Args:
        file_path: Path of the .dat file.

    Returns:
        XML file in string format.
    """

    xml_start = find_xml_start(file_path, "header")
    
    with open(file_path, 'rb') as file:
        # Skip to the start of the XML content
        file.seek(xml_start)
        
        # Read the remaining content as XML
        xml_content = file.read()
        # Wrap the extracted content with XML tags
        wrapped_xml_content = wrap_with_xml_tags(xml_content.strip())

        return wrapped_xml_content

def wrap_with_xml_tags(xml_content:str) -> str:
    """Wraps the incoming incomplete XML string with a <root> tag and a <xml> header.

    Args:
        xml_content: Incomplete XML.

    Returns:
        Complete XML.
    """

    # XML declaration and root tags
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>'
    root_start_tag = '<root>'
    root_end_tag = '</root>'
    
    # Combine the XML declaration, start tag, content, and end tag
    wrapped_content = f"{xml_declaration}\n{root_start_tag}\n{xml_content}\n{root_end_tag}"
    
    return wrapped_content

def extract_polygon_points(data_element:ET.Element) -> list[tuple[int,int]]:
    """Returns the x, y coordinates from an XML data element (which corresponds to a single annotated area).

    Args:
        data_element: XML data element.

    Returns:
        The extracted points.
    """

    points = []

    for point in data_element.findall('.//polygon_point'):
        x = int(point.get('polypointX'))
        y = int(point.get('polypointY'))
        points.append((x, y))

    return points

def create_bounding_box(x_coords:list[int], y_coords:list[int], image_scale:int, image_sizes:tuple[int|float, int|float]) -> tuple[int, int, int, int] | None:
    """Generates a bounding box around the given polygon.

    Args:
        x_coords: list of x coordinates belonging to an annotated segment.
        y_coords: list of y coordinates belonging to an annotated segment.
        image_scale: scale of the image compared to the full size (level = 0).
        image_sizes: width and height of the image.

    Returns:
        generated (and scaled) bounding boxes in COCO format. None if the bounding box is degenerate.
    """

    # Calculating values while cropping to image size
    min_x = max(min(x_coords), 0)
    max_x = min(max(x_coords), image_sizes[1]*image_scale)
    min_y = max(min(y_coords), 0)
    max_y = min(max(y_coords), image_sizes[0]*image_scale)

    
    width = max_x - min_x
    height = max_y - min_y

    # Calculating new dimensions based on scale factor and image dimensions while cropping to image size
    scaled_width = np.round(width / image_scale)
    scaled_height = np.round(height / image_scale)
    scaled_min_x = np.round(min_x / image_scale)
    scaled_min_y = np.round(min_y / image_scale)

    # Filtering faulty bounding boxe (coming from faulty/incomplete annotation) where the width or height is 0
    if width <= 0 or height <= 0:
        return None
    
    return scaled_min_x, scaled_min_y, scaled_width, scaled_height

def alpha_crop(image: np.ndarray, bboxes: list, segmentations: list) -> tuple:
    """
    Crop the image to the non-transparent area, remove the alpha channel,
    and adjust bounding boxes and segmentation polygons accordingly.
    
    Args:
        image (np.ndarray): Input image with alpha channel. Shape [H, W, 4].
        bboxes (list): List of bounding boxes in float32 pixel coordinates [(x_min, y_min, width, height), ...].
        segmentations (list): List of segmentation points in float32 pixel coordinates [[x1, y1 x2, y2, ...], [...], ...].

    Returns:
        tuple: A tuple containing the cropped RGB image and adjusted bounding boxes and segmentation polygons.
    """

    if image.dtype != np.uint8:
        raise ValueError("Input image must be of dtype uint8.")

    if image.shape[2] != 4:
        raise ValueError("Input image must have an alpha channel (4 channels).")
    
    alpha_channel = image[:, :, 3]
    y_indices, x_indices = np.where(alpha_channel != 0)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # No non-transparent pixels, return original image and bboxes
        return image[:, :, :3], bboxes

    # Calculate crop coordinates
    x_min_crop, y_min_crop = np.min(x_indices), np.min(y_indices)
    x_max_crop, y_max_crop = np.max(x_indices), np.max(y_indices)
    
    # Crop the image and remove the alpha channel
    cropped_image = image[y_min_crop:y_max_crop+1, x_min_crop:x_max_crop+1, :3]

    # Mask for remaining transparent regions
    cropped_alpha = alpha_channel[y_min_crop:y_max_crop+1, x_min_crop:x_max_crop+1]
    transparent_mask = cropped_alpha == 0

    # Replace transparent pixels with white in the cropped image
    cropped_image = cropped_image.copy()
    cropped_image[transparent_mask] = 255
    
    # Get new dimensions
    cropped_height, cropped_width = cropped_image.shape[0], cropped_image.shape[1]
    
    # Adjust bounding boxes
    adjusted_bboxes = []
    for bbox in bboxes:

        # Converting from COCO to Pascal Voc format
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = x_min + bbox[2]
        y_max = y_min + bbox[3]
        
        # Adjust by crop offsets
        x_min -= x_min_crop
        x_max -= x_min_crop
        y_min -= y_min_crop
        y_max -= y_min_crop
        
        # Clip the bounding box to the bounds of the cropped image
        x_min = np.clip(x_min, 0, cropped_width - 1)
        x_max = np.clip(x_max, 0, cropped_width - 1)
        y_min = np.clip(y_min, 0, cropped_height - 1)
        y_max = np.clip(y_max, 0, cropped_height - 1)
        
        # Only include bounding boxes that are within the cropped image
        if x_min < x_max and y_min < y_max:
            adjusted_bboxes.append([x_min, y_min, x_max-x_min, y_max-y_min])

    # Adjust segmentation polygons
    adjusted_segmentations = []
    for segmentation in segmentations:
        adjusted = []
        for i in range(0, len(segmentation), 2):
            # Adjust by crop offsets
            x_new = segmentation[i] - x_min_crop
            y_new = segmentation[i+1] - y_min_crop
            
            # Clip the points to the bounds of the cropped image
            x_new = np.clip(x_new, 0, cropped_width - 1)
            y_new = np.clip(y_new, 0, cropped_height - 1)
            
            adjusted.append(x_new)
            adjusted.append(y_new)

        adjusted_segmentations.append(adjusted)

    return cropped_image, adjusted_bboxes, adjusted_segmentations

def extract_tiles(image: WSI, level: int, tile_size: int, output_dir: Path = Path("tiles")) -> WSITileContainer:
    """Extract tiles from a WSI at given level with specified tile size.
    
    Args:
        image (WSI): Whole Slide Image instance
        level (int): Level to extract tiles from
        tile_size (int): Size of each tile (width and height)
        output_dir (Path, optional): Directory to save tiles. Defaults to Path("tiles")
    
    Returns:
        WSITileContainer: Container with extracted tiles and visualization capabilities
    """
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Create container
    container = WSITileContainer(wsi=image, level=level, tile_size=tile_size)
    
    with openslide.open_slide(image.mrxs_path) as slide:
        # Get dimensions and scale
        level_dimensions = slide.level_dimensions[level]
        scale_factor = slide.level_downsamples[level]
        
        # Calculate dimensions for center crop
        crop_width = (level_dimensions[0] // tile_size) * tile_size
        crop_height = (level_dimensions[1] // tile_size) * tile_size
        
        # Calculate offsets for center crop
        x_offset = (level_dimensions[0] - crop_width) // 2
        y_offset = (level_dimensions[1] - crop_height) // 2
        
        # Read cropped region
        region = slide.read_region(
            (int(x_offset * scale_factor), int(y_offset * scale_factor)),
            level,
            (crop_width, crop_height)
        )
        
        # Extract tiles

        tiles = []

        for y in range(0, crop_height, tile_size):
            for x in range(0, crop_width, tile_size):
                # Extract tile
                tile_img = region.crop((x, y, x + tile_size, y + tile_size))
                
                # Convert to array for transparency check
                tile_array = np.array(tile_img)
                
                # Skip fully transparent tiles
                if not tile_array[:,:,3].any():
                    continue
                
                # Calculate original coordinates (without crop)
                orig_x = x + x_offset
                orig_y = y + y_offset
                
                # Create tile path and save image
                tile_path = output_dir / f"{image.mrxs_path.stem}_l{level}_x{orig_x}_y{orig_y}.png"
                tile_img.save(tile_path)
                
                # Create tile metadata
                metadata = {
                    'source_wsi': image.mrxs_path.stem,
                    'level': level,
                    'scale_factor': scale_factor,
                    'original_coords': {
                        'x': orig_x,
                        'y': orig_y
                    },
                    'crop_offset': {
                        'x': x_offset,
                        'y': y_offset
                    }
                }
                
                # Create WSITile instance
                wsi_tile = WSITile(
                    level=level,
                    x=orig_x,
                    y=orig_y,
                    width=tile_size,
                    height=tile_size,
                    image_path=tile_path,
                    class_id=0,  # Default class, should be set based on annotations
                    metadata=metadata
                )
                
                tiles.append(wsi_tile)
        
        # Add tiles to the container
        container.add_tiles(tiles)
        
        return container