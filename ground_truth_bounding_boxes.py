import xml.etree.ElementTree as ET
import os
import cv2
import matplotlib.pyplot as plt

class GroundTruthBoundingBoxes:

    def __init__(self, name):
        self.name = name

    # Function to parse XML and extract bounding boxes
    def parse_annotation(self, annotation_path):
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
    def draw_and_save_bounding_boxes(self, image_path, bboxes, output_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for (xmin, ymin, xmax, ymax) in bboxes:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR before saving
        # Save the image with bounding boxes to the output folder
        cv2.imwrite(output_path, image)

    def load_dataset(self, data_folder='images', output_folder = 'annotated_images'):
        # Create output folder if not exist yet
        os.makedirs(output_folder, exist_ok=True)
        # Iterate through the folder and find pairs of image and XML files
        for filename in os.listdir(data_folder):
            if filename.endswith('.xml'):
                image_name = filename.replace('.xml', '.jpg') # Change the extension to match image file
                image_path = os.path.join(data_folder, image_name)
                annotation_path = os.path.join(data_folder, filename)
                if os.path.exists(image_path):
                    bboxes = self.parse_annotation(annotation_path)
                    output_path = os.path.join(output_folder, image_name)
                    self.draw_and_save_bounding_boxes(image_path, bboxes, output_path)
