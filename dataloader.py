import xml.etree.ElementTree as ET

#parse xml file

# tree = ET.parse("annotated_images/img-1.xml")

# root = tree.getroot() # get root object

# height = int(root.find("size")[0].text)
# width = int(root.find("size")[1].text)
# channels = int(root.find("size")[2].text)

# bbox_coordinates = []

# for member in root.findall('object'):
#     class_name = member[0].text # class name

#     # bbox coordinates
#     xmin = int(member[4][0].text)
#     ymin = int(member[4][1].text)
#     xmax = int(member[4][2].text)
#     ymax = int(member[4][3].text)

#     # store data in list

#     bbox_coordinates.append([class_name, xmin, ymin, xmax, ymax])

# print(bbox_coordinates)


# def read_content(xml_file: str):

#     tree = ET.parse(xml_file)
#     root = tree.getroot()

#     list_with_all_boxes = []

#     for boxes in root.iter('object'):

#         filename = root.find('filename').text

#         ymin = int(boxes.find("bndbox/ymin").text)
#         xmin = int(boxes.find("bndbox/xmin").text)
#         ymax = int(boxes.find("bndbox/ymax").text)
#         xmax = int(boxes.find("bndbox/xmax").text)

#         list_with_single_boxes = [xmin, ymin, xmax, ymax]
#         list_with_all_boxes.append(list_with_single_boxes)
    
#     return (filename, list_with_all_boxes)

# name, boxes = read_content("annotated_images/img-1.xml")


# print(name)
# print(boxes)


import os
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt

# Path to the folder containing both images and XML annotations
data_folder = 'annotated_images'
output_folder = 'output'

os.makedirs(output_folder, exist_ok=True)

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

# Function to draw bounding boxes on an image
def draw_and_save_bounding_boxes(image_path, bboxes, output_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for (xmin, ymin, xmax, ymax) in bboxes:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=2)
    

    # Save the image with bounding boxes to the output folder
    cv2.imwrite(output_path, image)

# Iterate through the folder and find pairs of image and XML files
for filename in os.listdir(data_folder):
    if filename.endswith('.xml'):
        image_name = filename.replace('.xml', '.jpg') # Change the extension to match image file
        image_path = os.path.join(data_folder, image_name)
        annotation_path = os.path.join(data_folder, filename)

        if os.path.exists(image_path):
            bboxes = parse_annotation(annotation_path)
            output_path = os.path.join(output_folder, image_name)
            draw_and_save_bounding_boxes(image_path, bboxes, output_path)
