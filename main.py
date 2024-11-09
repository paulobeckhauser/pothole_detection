from ground_truth_bounding_boxes import GroundTruthBoundingBoxes
from selective_search import SelectiveSearch
import os
import xml.etree.ElementTree as ET

def resize_bounding_boxes(bounding_boxes, original_size, new_size):
    x_scale = new_size[0] / original_size[0]
    y_scale = new_size[1] / original_size[1]
    resized_boxes = []
    for (xmin, ymin, xmax, ymax) in bounding_boxes:
        xmin_resized = int(xmin * x_scale)
        ymin_resized = int(ymin * y_scale)
        xmax_resized = int(xmax * x_scale)
        ymax_resized = int(ymax * y_scale)
        resized_boxes.append((xmin_resized, ymin_resized, xmax_resized, ymax_resized))
    return resized_boxes

def load_bounding_boxes_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bounding_boxes = []
    for object_elem in root.findall('object'):
        xmin = int(object_elem.find('xmin').text)
        ymin = int(object_elem.find('ymin').text)
        xmax = int(object_elem.find('xmax').text)
        ymax = int(object_elem.find('ymax').text)
        bounding_boxes.append((xmin, ymin, xmax, ymax))
    return bounding_boxes

# Function to parse XML and extract bounding boxes
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # Calculate the (x, y)-coordinates of the intersection rectangle
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    # Calculate the area of intersection rectangle
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Calculate the area of both the prediction and ground-truth rectangles
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    # Calculate the union area
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def main():

    # ------------ TASK 1 ------------
    # gt_bound_box = GroundTruthBoundingBoxes(name="Data Loader")
    # gt_bound_box.load_dataset(data_folder='images', output_folder = 'bounding_boxes_images')

    # ------------ TASK 2 ------------
    # Selective Search
    # resize_dim = (400, 400)
    # nb_boxes_list = [5, 10]
    # selective_search = SelectiveSearch(input_folder='images', output_folder='ss_images', resize_dim=resize_dim, mode='fast')
    # selective_search.process_all_images(nb_boxes_list=nb_boxes_list)

    mabo_results = []
    image_folder = 'ss_images/5_boxes/img-1.xml'

    # for filename in os.listdir(image_folder):
    #     if filename.lower().endswith(('.xml')):  # Filter for image files
    #         print(filename)
    bd_result_ss = load_bounding_boxes_from_xml(image_folder)

    b_box_path = 'images/img-1.xml'
    bd_g_box = parse_annotation(b_box_path)

    print(bd_result_ss)
    print(bd_g_box)

    # Edge Boxes

    # ------------ TASK 3 ------------
    


# image 1: [0.5, 0.6, 0.7]
# image 2: [0.5, 0.6, 0.7]

    # Apply selective search
    

 

if __name__ == "__main__":
    main()
