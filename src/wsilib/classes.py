import os
from pathlib import Path
import shutil
import openslide
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional, ClassVar, Dict
from datetime import datetime
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.draw import polygon2mask


@dataclass
class WSI():
    """Class for storing information about a Whole Slide Image."""

    mrxs_path: Path = None
    data_path: Path = None
    level_count: int = None
    level_dimensions: list[tuple[int, int]] = None # (width, height) for each level
    level_downsamples: list[float] = None
    properties: dict = None


    def __init__(self, source_dir: Path | str) -> None:
        """Initialize WSI with path and metadata.
        
        Args:
            source_dir (Path | str): Path to the directory containing the .mrxs file and the data folder.
        """ 
        if isinstance(source_dir, str):
            source_dir = Path(source_dir)
            
        if not source_dir.exists():
            raise FileNotFoundError(f"Directory {source_dir} not found")
        
        if not os.path.isdir(source_dir):
            raise NotADirectoryError(f"{source_dir} is not a directory")
        
        mrxs_files = list(Path(source_dir).glob("*.mrxs"))

        if not mrxs_files:
            raise FileNotFoundError("No .mrxs file found in source directory")
        
        if len(mrxs_files) > 1:
            raise ValueError("Multiple .mrxs files found in source directory")

        self.mrxs_path = mrxs_files[0]
        self.data_path = Path(source_dir) / self.mrxs_path.stem

        if not self.data_path.is_dir():
            raise FileNotFoundError("Data folder not found")
            
        # Load OpenSlide metadata
        with openslide.OpenSlide(str(self.mrxs_path)) as slide:
            self.level_count = slide.level_count
            self.level_dimensions = [slide.level_dimensions[i] for i in range(self.level_count)] 
            self.level_downsamples = [slide.level_downsamples[i] for i in range(self.level_count)]
            self.properties = dict(slide.properties)
    
    def print_info(self) -> None:
        """Print metadata of the WSI."""
        print(f"WSI info for: {self.mrxs_path.name}")
        print("-" * 50)
        print(f"Number of levels: {self.level_count}")
        print("\nLevel dimensions:")
        for level in range(self.level_count):
            width, height = self.level_dimensions[level]
            scale = self.level_downsamples[level]
            print(f"  Level {level:2d}: {width:7d} x {height:<7d} pixels (downscale: {scale:3.0f}x)")

    def get_level_size(self, level: int) -> tuple[int, int]:
        """Get dimensions of specified level.
        
        Args:
            level (int): Level to get dimensions for
            
        Returns:
            tuple[int, int]: Width and height of the level
        """
        if level >= self.level_count:
            raise IndexError(f"Level {level} does not exist. Available levels: 0-{self.level_count-1}")
        return self.level_dimensions[level]
    
    def get_level_scale(self, level: int) -> float:
        """Get downsample factor for specified level.
        
        Args:
            level (int): Level to get scale for
            
        Returns:
            float: Downsample factor relative to level 0
        """
        if level >= self.level_count:
            raise IndexError(f"Level {level} does not exist. Available levels: 0-{self.level_count-1}")
        return self.level_downsamples[level]
    
    def visualize(self, level: int, figsize: tuple = (20, 20)) -> None:
        """Visualize the WSI at specified level.
        
        Args:
            level (int): Level to visualize
            figsize (tuple): Figure size in inches
        """
        with openslide.OpenSlide(str(self.mrxs_path)) as slide:
            dimensions = slide.level_dimensions[level]
            image = slide.read_region((0, 0), level, dimensions)
            plt.figure(figsize=figsize)
            plt.imshow(image)
            plt.title(f"WSI at Level {level}")
            plt.axis('off')
            plt.show()

class AnnotationError(Exception):
        """Custom exception for annotation-related errors"""
        pass

@dataclass
class AnnotatedWSI:
    """Class for storing and managing annotations from a Whole Slide Image."""
    
    wsi: WSI
    _bounding_boxes: list[list[float]] = None  # [x, y, width, height] format at level 0
    _segmentations: list[list[float]] = None   # [x1, y1, x2, y2, ...] format at level 0

    def __post_init__(self):
        """Initialize annotations and validate attributes."""
        if not isinstance(self.wsi, WSI):
            raise ValueError("wsi must be an instance of WSI")
            
        # Loading annotations if they are not provided
        if self._bounding_boxes is None and self._segmentations is None:
            try:
                self._bounding_boxes, self._segmentations = self.get_annotations(level=0)
            except Exception as e:
                raise AnnotationError(f"Failed to load annotations: {str(e)}")

    
    def get_annotations_at_level(self, level: int) -> tuple[list, list]:
        """Get annotations scaled to specified level.
        
        Args:
            level (int): Target level for annotations
            
        Returns:
            tuple[list, list]: Scaled bounding boxes and segmentations
        """
        if level >= self.wsi.level_count:
            raise IndexError(f"Level {level} does not exist. Available levels: 0-{self.wsi.level_count-1}")
                
        scale_factor = self.wsi.get_level_scale(level)
        
        # Scale bounding boxes
        scaled_bboxes = [
            [coord / scale_factor for coord in bbox]
            for bbox in self._bounding_boxes
        ]
        
        # Scale segmentations
        scaled_segments = [
            [coord / scale_factor for coord in segment]
            for segment in self._segmentations
        ]
        
        return scaled_bboxes, scaled_segments
            
    def visualize_at_level(self, level: int, figsize: tuple = (20, 20)) -> None:
        """Visualize annotations at specified level.
        
        Args:
            level (int): Level to visualize
            figsize (tuple): Figure size in inches
        """
                
        # Get scaled annotations
        bboxes, segments = self.get_annotations_at_level(level)
        
        # Create figure
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
        # Load and display WSI
        with openslide.OpenSlide(str(self.wsi.mrxs_path)) as slide:
            dimensions = slide.level_dimensions[level]
            image = slide.read_region((0, 0), level, dimensions)
            ax.imshow(image)
        
        # Draw segmentations
        for segment in segments:
            # Convert flat list to x,y pairs
            points = np.array(segment).reshape(-1, 2)
            plt.plot(points[:, 0], points[:, 1], 'g-', linewidth=1)
        
        plt.title(f"WSI Annotations at Level {level}")
        plt.axis('off')
        plt.show()

    def get_annotations(self, level:int, tags:list[str] = ["header", 'object_privilege version="0100"']) -> tuple:
            """Get the annotations (segmentations, bounding boxes) for the given image.

            Args:
                level (int): Selected layer. The annotations are rescaled to match the dimensions of this layer.
                tags (list[str]): List of XML tags (present only in the annotation file) to search for in the .dat files. Default: ["header", 'object_privilege version="0100"'].

            Returns:
                The corresponding bounding boxes and segmentation polygons.
            """

            # List of files in the data folder
            files = list(self.wsi.data_path.glob("*.dat"))

            # Get the image dimensions
            image_sizes = self.wsi.level_dimensions[0]  # Base dimensions
            image_scale = self.wsi.get_level_scale(level)

            # Iterate through all .dat files
            for file in files:
                # Check file size
                if os.path.getsize(file) > 10 * 1024 * 1024:
                    # Skipping if too large 
                    continue

                # Check if all tags are found in the file
                found_tags = all(self.find_xml_start(file_path=file, tag=tag) != -1 for tag in tags)

                if not found_tags:
                    continue
                
                #############################################################################
                # ↓ This part only runs for one of the .dat files which has the annotations #
                #############################################################################

                xml_content = self.extract_xml_from_dat(file)

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
                    points = self.extract_polygon_points(annotation)

                    # flattening x, y pairs to one list
                    points_flat = np.array(points).flatten().tolist()

                    if points_flat:
                        scaled_points = []
                        # scale segmentation polygon coordinates
                        scaled_points = [p / image_scale for p in points_flat]

                        # Separate the points into X and Y coordinates
                        x_coords, y_coords = zip(*points)

                        # Create bounding boxes
                        bb = self.create_bounding_box(x_coords, y_coords, image_scale, image_sizes)

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

    def extract_xml_from_dat(self, file_path: Path) -> str:
        """Extracts the XML file from the given .dat file.

        Args:
            file_path (Path): Path of the .dat file.

        Returns:
            XML file in string format.
        """

        xml_start = self.find_xml_start(file_path, "header")
        
        with open(file_path, 'rb') as file:
            # Skip to the start of the XML content
            file.seek(xml_start)
            
            # Read the remaining content as XML
            xml_content = file.read()
            # Wrap the extracted content with XML tags
            wrapped_xml_content = self.wrap_with_xml_tags(xml_content.strip())

            return wrapped_xml_content

    @staticmethod
    def find_xml_start(file_path: Path, tag: str) -> int:
        """Searches for the tag in the file at file_path.

        Args:
            file_path (Path): Path of the .dat file.
            tag (str): The name of the XML tag (without "<" and ">") indicating the beggining of the file, eg. in case of <header> -> tag = header.

        Returns:
            Starting index of the XML formatted part in the file.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        with open(file_path, 'rb') as file:
            buffer_size = 4096
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

@dataclass
class WSITile:
    """Class representing a tile from a Whole Slide Image."""
    
    level: int
    x: int
    y: int
    width: int
    height: int
    image_path: Path
    class_id: int
    metadata: Optional[dict] = None

    def __post_init__(self):
        """Validate tile attributes after initialization."""

        # Check if image file exists
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image file not found: {self.image_path}")

    def save(self, base_path: Path) -> None:
        """Save tile and metadata in COCO format.

        Args:
            base_path (Path): Path to the directory where the tile and metadata should be saved.
        """

        os.makedirs(base_path, exist_ok=True)
        
        # Save tile image
        if self.image_path.parent != base_path:
            Image.open(self.image_path).save(base_path / self.image_path.name)
        
        # Create COCO format metadata
        coco_data = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": f"Tile from {self.metadata.get('source_wsi', 'unknown')}",
                "date_created": datetime.now().isoformat()
            },
            "images": [{
                "id": 0,
                "file_name": self.image_path.name,
                "width": self.width,
                "height": self.height,
                "tile_x": self.x,
                "tile_y": self.y,
                "level": self.level
            }],
            "annotations": [{
                "id": 0,
                "image_id": 0,
                "category_id": self.class_id,
                "bbox": [0, 0, self.width, self.height],
                "area": self.width * self.height,
                "iscrowd": 0
            }],
            "categories": [{
                "id": self.class_id,
                "name": f"class_{self.class_id}",
                "supercategory": "tissue"
            }]
        }
        
        # Save COCO metadata
        meta_path = base_path / f"{self.image_path.stem}_coco.json"
        with open(meta_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

@dataclass
class WSITileContainer:
    """Container for WSI tiles with extraction and visualization capabilities."""
    
    wsi: WSI
    level: int
    tile_size: int
    annotations: Optional[AnnotatedWSI] = None
    _tiles: list[WSITile] = field(default_factory=list)

    def __post_init__(self):
        """Validate attributes after initialization."""
        if self.annotations and self.annotations.wsi.mrxs_path != self.wsi.mrxs_path:
            raise ValueError("Annotations must be from the same WSI")

    def _annotate_tiles(self, threshold:float = 0.4) -> None:
        """Modifies the tiles' class id to the corresponding values from the annotation mask.

        Args:
            threshold (float): Threshold for the tumor area in the tile to be considered as a tumor class. Defaults to 0.4.
        """

        bb, segmentations = self.annotations.get_annotations_at_level(self.level) # only using the segmentation polygon

        segmentation_polygons = []
        for segmentation in segmentations:
            segmentation_polygons.append(np.array(segmentation).reshape(-1, 2))

        tile_shape = (self.tile_size, self.tile_size)

        for tile in self._tiles:
            x = tile.x
            y = tile.y

            # Initialize an empty mask
            combined_mask = np.zeros(tile_shape, dtype=bool)

            for segmentation_polygon in segmentation_polygons:
                # Adjust polygon coordinates relative to the tile's origin
                tile_polygon_coords = segmentation_polygon - [x, y]

                mask = polygon2mask(tile_shape, tile_polygon_coords)

                # Combine the masks
                combined_mask |= mask

            # Determining areas of the tumor and the tile
            tumor_area = combined_mask.sum()
            tile_area = combined_mask.size

            # If the tumor area is greater than 10% of the tile area, the tile is considered as tumor class
            if (tumor_area / tile_area) > threshold:
                tile.class_id = 1  # tumor
            else:
                tile.class_id = 0  # other

    def extract_tiles(self, output_dir: Path = None) -> None:
        """Extract tiles from the WSI at specified level. Not whole tiles (with sizes smaller than the tile size) are skipped at the ends of the image.
        
        Args:
            output_dir (Path, optional): Directory to save tiles. Defaults to Path("tiles").
        """
        if output_dir is None:
            output_dir = Path("tiles")
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)
            
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        with openslide.open_slide(self.wsi.mrxs_path) as slide:
            # Get dimensions and scale
            level_dimensions = self.wsi.level_dimensions[self.level]
            scale_factor = self.wsi.level_downsamples[self.level]
            
            # Read cropped region
            region = slide.read_region((0,0), self.level, level_dimensions)
            
            total_tiles = (level_dimensions[0] // self.tile_size) * (level_dimensions[1] // self.tile_size)

            # Extract tiles (y for hight, x for width)
            with tqdm(total=total_tiles, desc=f"Extracting tiles from {self.wsi.mrxs_path.stem}", unit=" tiles") as pbar:
                for x in range(0, level_dimensions[0], self.tile_size):
                    for y in range(0, level_dimensions[1], self.tile_size): 
                        # Extract tile
                        tile_img = region.crop((x, y, x + self.tile_size, y + self.tile_size))
                        
                        # Update progress bar
                        pbar.update(1)

                        # Convert to array for transparency check
                        tile_array = np.array(tile_img)
                        
                        # Skip fully transparent tiles
                        if not tile_array[:,:,3].any():
                            continue
                        
                        # Create tile path and save image
                        tile_path = output_dir / f"{self.wsi.mrxs_path.stem}_l{self.level}_x{x}_y{y}.png"
                        tile_img.save(tile_path)
                        
                        # Create tile metadata
                        metadata = {
                            'source_wsi': self.wsi.mrxs_path.stem,
                            'level': self.level,
                            'scale_factor': scale_factor,
                            'coords': {
                                'x': x,
                                'y': y
                            },
                        }
                        
                        # Create WSITile instance
                        wsi_tile = WSITile(
                            level=self.level,
                            x=x,
                            y=y,
                            width=self.tile_size,
                            height=self.tile_size,
                            image_path=tile_path,
                            class_id=0,  # Default class, should be set based on annotations
                            metadata=metadata
                        )
                        
                        self._tiles.append(wsi_tile)
                
            # Save COCO annotations if tiles were extracted
            if self._tiles:
                self._save_coco_annotations(output_dir)

    def _save_coco_annotations(self, output_dir: Path) -> None:
        """Save tile information in COCO format.
        
        Args:
            output_dir (Path): Directory where tiles are saved
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
            
        coco_data = {
            "info": {
                "description": f"Tiles extracted from {self.wsi.mrxs_path.name}",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "images": [],
            "annotations": []
        }
        
        for idx, tile in enumerate(self._tiles):
            # Add image info
            coco_data["images"].append({
                "id": idx,
                "file_name": tile.image_path.name,
                "width": tile.width,
                "height": tile.height,
                "metadata": tile.metadata
            })
            
            # Add annotation info if tile has class_id
            if tile.class_id > 0:
                coco_data["annotations"].append({
                    "image_id": idx,
                    "category_id": tile.class_id,
                    "bbox": [0, 0, tile.width, tile.height],  # Full tile annotation
                    "area": tile.width * tile.height
                })
        
        # Save COCO JSON
        coco_path = output_dir / f"{self.wsi.mrxs_path.stem}_tiles.json"
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

    def visualize(self, figsize: tuple = (20, 20), alpha: float = 0.3,
                 color: str = 'red', show_stats: bool = True) -> None:
        """Visualize WSI with tile overlay.
        
        Args:
            figsize (tuple): Figure size in inches
            alpha (float): Transparency of tile overlays
            color (str): Color of tile overlays
            show_stats (bool): Whether to print statistics
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Create figure and axis
        plt.figure(figsize=figsize)
        ax = plt.gca()

        dimensions = self.wsi.get_level_size(self.level)

        with openslide.OpenSlide(str(self.wsi.mrxs_path)) as slide:
            image = slide.read_region((0, 0), self.level, dimensions)
        
        # Display WSI
        ax.imshow(image)
        
        # Add tile overlays
        for tile in self._tiles:
            rect = patches.Rectangle(
                (tile.x, tile.y),
                tile.width,
                tile.height,
                linewidth=1,
                edgecolor=color,
                facecolor=color,
                alpha=alpha
            )
            ax.add_patch(rect)

            if self.annotations:
                # Add class ID text in center of tile
                ax.text(
                    tile.x + tile.width/2,
                    tile.y + tile.height/2,
                    str(tile.class_id),
                    color='white',
                    fontsize=8,
                    horizontalalignment='center',
                    verticalalignment='center'
                )

        # Get scaled annotations
        if self.annotations:
            bboxes, segments = self.annotations.get_annotations_at_level(self.level)
            
            # Draw segmentations
            for segment in segments:
                # Convert flat list to x,y pairs
                points = np.array(segment).reshape(-1, 2)
                plt.plot(points[:, 0], points[:, 1], 'g-', linewidth=1)
        
        # Set title and display
        plt.title(f"WSI Layer {self.level} with Tile Coverage")
        plt.axis('off')

        if show_stats:
            print(f"Total number of tiles: {len(self._tiles)}")
            print(f"Image dimensions: {self.wsi.get_level_size(self.level)}")
        
        plt.show()

    def save_tiles(self, output_dir: Path) -> None:
        """Save all tiles to specified directory.
        
        Args:
            output_dir (Path): Directory to save tiles
        """
        output_dir.mkdir(exist_ok=True)
        for tile in self._tiles:
            tile.save(output_dir)
            
    def get_tile_at(self, x: int, y: int) -> Optional[WSITile]:
        """Get tile at specific coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            Optional[WSITile]: Tile at coordinates if found
        """
        for tile in self._tiles:
            if (x in range(tile.x, tile.x + tile.width) and 
                y in range(tile.y, tile.y + tile.height)):
                return tile
        return None

@dataclass
class WSIDatabase:
    """Class for managing multiple AnnotatedWSI instances."""
    
    wsi_dir: Path
    annotated_wsis: list[AnnotatedWSI] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the database by loading all WSIs and their annotations from the directory."""
        
        if isinstance(self.wsi_dir, str):
            self.wsi_dir = Path(self.wsi_dir)

        if not self.wsi_dir.exists():
            raise FileNotFoundError(f"Directory {self.wsi_dir} not found")
        
        # Find all .mrxs files in directory and subdirectories
        mrxs_files = list(self.wsi_dir.glob("**/*.mrxs"))

        print(f"[INFO]: Found {len(mrxs_files)} .mrxs files in {self.wsi_dir}")
        
        if not mrxs_files:
            raise FileNotFoundError("No .mrxs files found in directory")
        
        # Load each WSI with annotations
        print(f"Loading annotations...")
        for mrxs_file in tqdm(mrxs_files):
            try:
                wsi = WSI(str(mrxs_file.parent))
                try:
                    annotated_wsi = AnnotatedWSI(wsi)
                    self.annotated_wsis.append(annotated_wsi)
                except AnnotationError:
                    raise AnnotationError(f"No annotations found for {mrxs_file.name}")
            except Exception as e:
                raise RuntimeError(f"Error loading {mrxs_file.name}: {str(e)}")
            
        print(f"\n[SUCCESS]: Loaded {len(self.annotated_wsis)} WSIs with annotations")

    def save_coco_dataset(self, output_dir: Path, level: int) -> None:
        """Save all WSIs as complete images in COCO dataset format at specified level.
        
        Args:
            output_dir (Path): Directory to save the dataset
            level (int): WSI level to extract
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        (output_dir / "images").mkdir(exist_ok=True)
        
        # COCO dataset structure
        dataset = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": "WSI Dataset",
                "date_created": datetime.now().isoformat()
            },
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "background", "supercategory": "tissue"},
                {"id": 1, "name": "tumor", "supercategory": "tissue"}
            ]
        }
        
        image_id = 0
        annotation_id = 0

        print(f"Processing images...")
        # Processing each AnnotatedWSI
        for annotated_wsi in tqdm(self.annotated_wsis):
            
            width, height = annotated_wsi.wsi.get_level_size(level)
            
            # Save the level as image
            image_filename = f"{annotated_wsi.wsi.mrxs_path.stem}_lvl{level}.png"
            image_path = output_dir / "images" / image_filename
            
            with openslide.OpenSlide(str(annotated_wsi.wsi.mrxs_path)) as slide:
                image = slide.read_region((0, 0), level, (width, height))
                image.save(image_path)
            
            # Image entry in COCO annotation file
            dataset["images"].append({
                "id": image_id,
                "file_name": f"images/{image_filename}",
                "width": width,
                "height": height,
                "level": level,
                "wsi_source": annotated_wsi.wsi.mrxs_path.name
            })
            
            # Get annotations at the specified level
            bboxes, segmentations = annotated_wsi.get_annotations_at_level(level)
            
            # Add annotations
            for bbox, segmentation in zip(bboxes, segmentations):
                x, y, w, h = bbox
                dataset["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # tumor class
                    "bbox": bbox,
                    "segmentation": [segmentation],  # COCO format expects list of lists
                    "area": w * h,
                    "iscrowd": 0
                })
                annotation_id += 1
            
            image_id += 1
        
        # Save COCO dataset
        with open(output_dir / "annotations.json", 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n[SUCCESS]: Dataset saved to {output_dir}. Number of images: {len(dataset['images'])}")

@dataclass
class WSITileDatabase:
    """Class for managing multiple WSITileContainer instances."""
    
    wsi_dir: Path
    tile_size: int
    level: int
    tile_containers: list[WSITileContainer] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the database by loading all WSIs and creating tile containers."""
        if isinstance(self.wsi_dir, str):
            self.wsi_dir = Path(self.wsi_dir)

        if not self.wsi_dir.exists():
            raise FileNotFoundError(f"Directory {self.wsi_dir} not found")
        
        # Find all .mrxs files
        mrxs_files = list(self.wsi_dir.glob("**/*.mrxs"))
        print(f"[INFO]: Found {len(mrxs_files)} .mrxs files in {self.wsi_dir}")
        
        if not mrxs_files:
            raise FileNotFoundError("No .mrxs files found in directory")
        
        # Create tile containers for each WSI
        for mrxs_file in tqdm(mrxs_files, desc="Creating tile containers for each WSI"):
            try:
                wsi = WSI(str(mrxs_file.parent))
                try:
                    annotated_wsi = AnnotatedWSI(wsi)
                    container = WSITileContainer(
                        wsi=wsi,
                        level=self.level,
                        tile_size=self.tile_size,
                        annotations=annotated_wsi
                    )
                    self.tile_containers.append(container)
                except AnnotationError:
                    raise AnnotationError(f"No annotations found for {mrxs_file.name}")
            except Exception as e:
                raise RuntimeError(f"Error processing {mrxs_file.name}: {str(e)}")
                
        print(f"\n[SUCCESS]: Created {len(self.tile_containers)} tile containers")

    def save_dataset(self, output_dir: Path) -> None:
        """Save all tiles and merge annotations into a single COCO dataset.
        
        Args:
            output_dir (Path): Directory to save the dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Extract tiles from all containers
        for container in tqdm(self.tile_containers, desc="Extracting tiles from containers"):
            container.extract_tiles(output_dir)
            if container.annotations:
                container._annotate_tiles()

        # Merge all annotation files
        print("\nMerging annotations...")
        merged_coco = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": "WSI Tile Dataset",
                "date_created": datetime.now().isoformat()
            },
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "background", "supercategory": "tissue"},
                {"id": 1, "name": "tumor", "supercategory": "tissue"}
            ]
        }
        
        image_id = 0
        annotation_id = 0
        
        # Collect all tiles and their annotations
        for container in self.tile_containers:
            for tile in container._tiles:
                # Add image info
                merged_coco["images"].append({
                    "id": image_id,
                    "file_name": "images/" + tile.image_path.name,
                    "width": tile.width,
                    "height": tile.height,
                    "tile_x": tile.x,
                    "tile_y": tile.y,
                    "level": tile.level,
                    "wsi_source": tile.metadata['source_wsi']
                })
                
                # Add annotation if tumor class
                if tile.class_id == 1:
                    merged_coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [0, 0, tile.width, tile.height],
                        "area": tile.width * tile.height,
                        "iscrowd": 0
                    })
                    annotation_id += 1
                
                image_id += 1
        
        # Save merged annotations
        with open(output_dir / "annotations.json", 'w') as f:
            json.dump(merged_coco, f, indent=2)

        # Move images to img folder
        img_dir = output_dir / "images"
        img_dir.mkdir(exist_ok=True)
        for tile in container._tiles:
            shutil.move(tile.image_path, img_dir / tile.image_path.name)

        # Delete individual COCO annotation files except the merged one
        for container in self.tile_containers:
            coco_path = container.wsi.mrxs_path.stem + "_tiles.json"
            coco_path = output_dir / coco_path
            if coco_path.exists():
                coco_path.unlink()

        print(f"\n[SUCCESS]: Dataset saved to {output_dir}. Total images: {len(merged_coco['images'])}")