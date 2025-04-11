import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import matplotlib.patches as Patch
import io
from PIL import Image, ImageDraw
import numpy as np
import csv
import pandas as pd

from torchvision import transforms
from transformers import AutoModelForObjectDetection
import torch
import easyocr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MaxResize(object):
    def __init__(self, max_size=800): #constructor inicializa el tamaño maximo de la imagen
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image

detection_transform = transforms.Compose([ #for the detection of the table
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])    

structure_transform = transforms.Compose([ #for the structure detection of the table
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
        
model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm").to(device)

structure_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all").to(device)

reader = easyocr.Reader(['en'])

def box_cxcywh_to_xyxy(x: torch) -> torch.Tensor:
    """
    Convert bounding boxes from (center_x, center_y, width, height) format to (x_min, y_min, x_max, y_max) format.

    Args:
        x (torch.Tensor): A tensor of shape (..., 4) containing bounding boxes in (cx, cy, w, h) format.

    Returns:
        torch.Tensor: A tensor of shape (..., 4) containing bounding boxes in (x_min, y_min, x_max, y_max) format.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Rescale bounding boxes from normalized (0-1) format to absolute pixel coordinates.

    Args:
        out_bbox (torch.Tensor): A tensor of shape (..., 4) containing bounding boxes 
                                 in (center_x, center_y, width, height) format, normalized in [0,1].
        size (tuple): A tuple (width, height) representing the actual image dimensions in pixels.

    Returns:
        torch.Tensor: A tensor of shape (..., 4) with bounding boxes in (x_min, y_min, x_max, y_max) format, 
                      scaled to the given image size.
    """
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)  # Convert to (x_min, y_min, x_max, y_max)
    boxes = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)  # Rescale
    return boxes

import torch

def outputs_to_objects(outputs, img_size: tuple, id2label: dict) -> list[dict]:
    """
    Convert raw model outputs to a structured list of detected objects with labels, scores, and bounding boxes.

    Args:
        outputs: Model output containing classification scores and bounding boxes.
        img_size (tuple): A tuple (width, height) representing the actual image dimensions in pixels.
        id2label (dict): A dictionary mapping class indices to human-readable labels.

    Returns:
        list[dict]: A list of detected objects, each represented as a dictionary with:
                    - 'label': Predicted class label
                    - 'score': Confidence score
                    - 'bbox': Bounding box (x_min, y_min, x_max, y_max) in pixel coordinates
    """
    # Apply softmax to logits and extract the maximum score for each object
    m = outputs.logits.softmax(-1).max(-1)

    # Get predicted labels and confidence scores
    pred_labels = list(m.indices.detach().cpu().numpy())[0]  # Class indices
    pred_scores = list(m.values.detach().cpu().numpy())[0]   # Confidence scores

    # Get bounding boxes and rescale to image size
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]  # Extract boxes
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]  # Rescale to pixel coordinates

    # Construct list of detected objects
    detected_objects = [
        {"label": id2label[pred_labels[i]], "score": float(pred_scores[i]), "bbox": pred_bboxes[i]}
        for i in range(len(pred_labels))
    ]

    return detected_objects

def fig2img(fig):
    """
    Convert a Matplotlib figure to a PIL Image.

    Args:
        fig (matplotlib.figure.Figure): A Matplotlib figure instance.

    Returns:
        PIL.Image.Image: The figure converted to a PIL Image.
    """
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    image = Image.open(buf)
    return image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch

def visualize_detected_tables(img, det_tables) -> matplotlib.figure.Figure:
    """
    Visualize detected tables on an image using Matplotlib.

    Args:
        img (numpy.ndarray or PIL.Image.Image): The image on which to overlay the table detections.
        det_tables (list of dict): A list of detected tables, where each table is represented as a dictionary with:
            - 'label' (str): Either 'table' or 'table rotated'.
            - 'bbox' (list of float): Bounding box coordinates in (x_min, y_min, x_max, y_max) format.

    Returns:
        matplotlib.figure.Figure: The generated Matplotlib figure with overlaid table detections.
    """
    plt.imshow(img, interpolation="lanczos")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)  # Red-Pink for normal tables
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)  # Orange for rotated tables
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        else:
            continue  # Ignore other labels

        # Draw filled rectangle with transparency
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor='none', facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)

        # Draw rectangle border
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', alpha=alpha)
        ax.add_patch(rect)

        # Draw hatched pattern inside rectangle
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=0,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    # Remove ticks
    plt.xticks([], [])
    plt.yticks([], [])

    # Define legend
    legend_elements = [
        Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45), label='Table', hatch='//////', alpha=0.3),
        Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1), label='Table (rotated)', hatch='//////', alpha=0.3)
    ]

    # Add legend below the figure
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
               fontsize=10, ncol=2)

    # Set figure size and hide axis
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')

    return fig


def detect_and_crop_table_list(image):
    """
    Detect tables in an image and return a list of cropped table images.
    """
    pixel_values = detection_transform(image).unsqueeze(0).to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(pixel_values)

    # Post-process to get detected tables
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)

    # Ensure output is always a list
    cropped_tables = []
    for table in detected_tables:
        cropped_tables.append(image.crop(table["bbox"]))

    return cropped_tables  # Always returns a list

def detect_and_crop_table(image: Image.Image) -> Image.Image:
    """
    Detect tables in an image using a deep learning model and crop the first detected table.

    Args:
        image (PIL.Image.Image): The input image to process.

    Returns:
        PIL.Image.Image: A cropped image containing the first detected table.
    """
    # Prepare image for the model
    # pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = detection_transform(image).unsqueeze(0).to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(pixel_values)

    # Post-process to get detected tables
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)

    # Visualize detection (commented out)
    # fig = visualize_detected_tables(image, detected_tables)
    # image = fig2img(fig)

    # Crop the first detected table from the image
    cropped_table = image.crop(detected_tables[0]["bbox"])
    
    return cropped_table

def recognize_table(image) -> tuple[Image.Image, list[dict]]:
    """
    Recognize and annotate table structure by detecting individual cells in an image.

    Args:
        image (PIL.Image.Image): The cropped table image to process.

    Returns:
        tuple:
            - PIL.Image.Image: The annotated table image with detected cells outlined in red.
            - list[dict]: A list of detected cells, where each cell is represented as a dictionary with:
                - 'label' (str): The detected cell type.
                - 'score' (float): Confidence score of the detection.
                - 'bbox' (list of float): Bounding box coordinates in (x_min, y_min, x_max, y_max) format.
    """
    # Prepare image for the model
    # pixel_values = structure_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = structure_transform(image).unsqueeze(0).to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = structure_model(pixel_values)

    # Post-process to get detected cells
    id2label = structure_model.config.id2label
    id2label[len(structure_model.config.id2label)] = "no object"
    cells = outputs_to_objects(outputs, image.size, id2label)

    # Draw bounding boxes around detected cells
    draw = ImageDraw.Draw(image)
    for cell in cells:
        draw.rectangle(cell["bbox"], outline="red")

    return image, cells

def get_cell_coordinates_by_row(table_data) -> list:
    
    '''
    Organize table structure data into cell coordinates grouped by row.

    This function processes a list of table structure elements, where each element is
    represented as a dictionary containing a 'label' and a 'bbox'. It extracts elements
    labeled as 'table row' and 'table column', sorts them based on their spatial coordinates,
    and then computes individual cell bounding boxes by combining the x-coordinates of columns
    with the y-coordinates of rows.

    Args:
        table_data (list of dict): A list of dictionaries representing detected table elements.
            Each dictionary should have:
                - 'label' (str): The type of element (expected to be 'table row' or 'table column').
                - 'bbox' (list of float): Bounding box coordinates in the format [x_min, y_min, x_max, y_max].

    Returns:
        list of dict: A list where each element corresponds to a table row and contains:
            - 'row' (list of float): The bounding box for the row.
            - 'cells' (list of dict): A list of dictionaries for each cell in the row, where each dictionary has:
                - 'column' (list of float): The bounding box for the corresponding column.
                - 'cell' (list of float): The computed cell bounding box, combining the column's x-coordinates
                  and the row's y-coordinates.
            - 'cell_count' (int): The number of cells in that row.                
    '''

    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information to cell_coordinates
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates


def apply_ocr(cell_coordinates, cropped_table):
    """
    Apply Optical Character Recognition (OCR) to detected table cells and save the extracted data to a CSV file.

    Args:
        cell_coordinates (list of dict): A list of detected rows, where each row contains:
            - 'cells' (list of dict): List of detected cells in the row, with bounding boxes.
        cropped_table (PIL.Image.Image): The cropped table image.

    Returns:
        tuple:
            - pd.DataFrame: A Pandas DataFrame containing the extracted table data.
            - dict: A dictionary where keys are row indices and values are lists of extracted text from each cell.
    """
    # Initialize storage for table data
    data = dict()
    max_num_columns = 0

    # Process rows one by one
    for idx, row in enumerate(cell_coordinates):
        row_text = []
        for cell in row["cells"]:
            # Crop cell from image
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            # Apply OCR to extract text
            result = reader.readtext(np.array(cell_image))
            if len(result) > 0:
                text = " ".join([x[1] for x in result])  # Concatenate OCR results
                row_text.append(text)

        # Update max column count
        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        # Store row data
        data[str(idx)] = row_text

    # Ensure all rows have the same number of columns by padding with empty strings
    for idx, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[str(idx)] = row_data
    
  
    #verify that there's text
    if not any(data.values()):
        return None, None
    
    # Write extracted text to a CSV file
    """with open('output.csv', 'w', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        for row, row_text in data.items():
            wr.writerow(row_text)"""

    # Load the extracted data into a Pandas DataFrame
    
    df = pd.DataFrame.from_dict(data, orient='index')

    return df, data

def process_pdf(image):
    cropped_table = detect_and_crop_table(image)

    image, cells = recognize_table(cropped_table)
    
    print('recognized table')
    image.show()
    
    cell_coordinates = get_cell_coordinates_by_row(cells)
    
    print('getting to ocr')
    df, data = apply_ocr(cell_coordinates, image)

    return image, df, data