from ground_truth_bounding_boxes import GroundTruthBoundingBoxes
from selective_search import SelectiveSearch
import os
import cv2
import xml.etree.ElementTree as ET
import re
import torch

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

def resize_bounding_boxes(bounding_boxes, original_size, new_size):
    """
    Resize bounding boxes from the original image size to the new image size.

    Parameters:
    bounding_boxes (list of tuples): List of bounding boxes in the format (xmin, ymin, xmax, ymax).
    original_size (tuple): Original image size in the format (width, height).
    new_size (tuple): New image size in the format (width, height).

    Returns:
    list of tuples: Resized bounding boxes in the format (xmin, ymin, xmax, ymax).
    """
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



def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ------------ TASK 1 ------------
    gt_bound_box = GroundTruthBoundingBoxes(name="Data Loader")
    gt_bound_box.load_dataset(data_folder='images', output_folder = 'bounding_boxes_images')

        # Iterate over all images in the ground truth folder
    for filename in os.listdir(ground_truth_folder):
        if filename.lower().endswith(('.jpg', 'jpeg', '.png')):  # Filter for image files
            image_path = os.path.join(ground_truth_folder, filename)
            image = cv2.imread(image_path)
            if image is not None:
                height, width, channels = image.shape
                print(f"Image: {filename}, Size: {width}x{height}")

    # # ------------ TASK 2 ------------
    # Selective Search
    # resize_dim = (400, 400)
    # nb_boxes_list = [5, 10, 20, 50, 100, 200, 500, 100, 1500, 2000, 2500, 3000]
    # selective_search = SelectiveSearch(input_folder='images', output_folder='ss_images', resize_dim=resize_dim, mode='fast')
    # selective_search.process_all_images(nb_boxes_list=nb_boxes_list)

    mabo_results = []
    nb_boxes = []
    max_iou_results = []
    original_size = (720, 720)
    new_size = (400, 400)

    ground_truth_folder = 'images'
    # selective_search_folder_5_boxes = 'ss_images/5_boxes'
    ss_images_folder = 'ss_images'

    # Iterate over all subfolders in ss_images
    for subfolder in os.listdir(ss_images_folder):
        nb_boxes_str = extract_number_from_string(subfolder)
        nb_boxes.append(nb_boxes_str)
        selective_search_folder = os.path.join(ss_images_folder, subfolder)
        if not os.path.isdir(selective_search_folder):
            continue

        images_folder_len = count_xml_files(ground_truth_folder)
        print(f"Number of .xml files in '{ground_truth_folder}': {images_folder_len}")

        max_iou_results = []
        i = 1
        while i <= images_folder_len:
            image_folder = f'{selective_search_folder}/img-{i}.xml'
            b_box_path = f'images/img-{i}.xml'
            print(image_folder)

            if not os.path.exists(image_folder) or not os.path.exists(b_box_path):
                i += 1
                continue

            bd_result_ss = load_bounding_boxes_from_xml(image_folder)
            bd_g_box = parse_annotation(b_box_path)
            resized_boxes = resize_bounding_boxes(bd_g_box, original_size, new_size)

            iou = []
            for element in bd_result_ss:
                iou_list = calculate_iou(element, resized_boxes[0])
                iou.append(iou_list)

            iou_max = max(iou)
            max_iou_results.append(iou_max)

            i += 1

        print(max_iou_results)
        # Calculate the average of max_iou_results
        if max_iou_results:
            average_max_iou_results = sum(max_iou_results) / len(max_iou_results)
            mabo_results.append(average_max_iou_results)
            print(f"Average MABO for {subfolder}: {average_max_iou_results}")
        else:
            print(f"No MABO results to average for {subfolder}.")

    print(f"MABO results: {mabo_results}")
    print(f"Number boxes: {nb_boxes}")

    # Pair the elements from both lists
    paired_results = list(zip(nb_boxes, mabo_results))

    # Save results to a text file
    with open('mabo_results.txt', 'w') as file:
        file.write(f"MABO results: {mabo_results}\n")
        file.write(f"Number boxes: {nb_boxes}\n")

    
        


    # images_folder_len = count_xml_files(ground_truth_folder)
    # selective_search_folder_5_boxes_len = count_xml_files('ss_images/5_boxes/')

    
    # i = 1
    # while i <= images_folder_len:
    #     image_folder = f'{selective_search_folder_5_boxes}/img-{i}.xml'
    #     b_box_path = f'images/img-{i}.xml'
    #     print(image_folder)

    #     if not os.path.exists(image_folder) or not os.path.exists(b_box_path):
    #         i += 1
    #         continue

    #     bd_result_ss = load_bounding_boxes_from_xml(image_folder)
    #     bd_g_box = parse_annotation(b_box_path)
    #     resized_boxes = resize_bounding_boxes(bd_g_box, original_size, new_size)
    
    #     iou = []
    #     for element in bd_result_ss:
    #         iou_list = calculate_iou(element, resized_boxes[0])
    #         iou.append(iou_list)

    #     iou_max = max(iou)
    #     max_iou_results.append(iou_max)

    #     i += 1

    # print(max_iou_results)
    #     # Calculate the average of max_iou_results
    # if max_iou_results:
    #     average_max_iou_results = sum(max_iou_results) / len(max_iou_results)
    #     mabo_results.append(average_max_iou_results)
    #     print(f"Average MABO: {average_max_iou_results}")
    # else:
    #     print("No MABO results to average.")

    
    # print(f"MABO results: {mabo_results}")

 

if __name__ == "__main__":
    main()
