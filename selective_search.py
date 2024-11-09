import cv2
import os
import xml.etree.ElementTree as ET

class SelectiveSearch:
    def __init__(self, input_folder, output_folder, resize_dim=(600, 600), mode='fast'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.resize_dim = resize_dim
        self.mode = mode.lower()
        os.makedirs(self.output_folder, exist_ok=True)

    def run_selective_search(self, image_path):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return None, None

        # Resize the image for efficiency
        image_resized = cv2.resize(image, self.resize_dim)

        # Initialize Selective Search
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image_resized)

        # Set mode
        if self.mode == 'fast':
            ss.switchToSelectiveSearchFast()
        elif self.mode == 'quality':
            ss.switchToSelectiveSearchQuality()
        else:
            raise ValueError("Invalid mode: choose 'fast' or 'quality'")
        
        # Run Selective Search
        rects = ss.process()
        return image_resized, rects

    def save_proposals_to_xml(self, proposals, output_xml_path):
        root = ET.Element("annotations")
        for (x, y, w, h) in proposals:
            object_elem = ET.SubElement(root, "object")
            ET.SubElement(object_elem, "xmin").text = str(x)
            ET.SubElement(object_elem, "ymin").text = str(y)
            ET.SubElement(object_elem, "xmax").text = str(x + w)
            ET.SubElement(object_elem, "ymax").text = str(y + h)

        tree = ET.ElementTree(root)
        tree.write(output_xml_path)

    def process_all_images(self, nb_boxes_list):
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(('.jpg', 'jpeg', '.png')): # Filter for image files
                image_path = os.path.join(self.input_folder, filename)
                image_resized, proposals = self.run_selective_search(image_path)

                if image_resized is not None:
                    for nb_boxes in nb_boxes_list:
                        image_copy = image_resized.copy()
                        for (x, y, w, h) in proposals[:nb_boxes]:
                            cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Create subdirectory for each nb_boxes value
                        output_subfolder = os.path.join(self.output_folder, f"{nb_boxes}_boxes")
                        os.makedirs(output_subfolder, exist_ok=True)

                        output_path = os.path.join(output_subfolder, f"{nb_boxes}_boxes_{filename}")
                        cv2.imwrite(output_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                        print(f"Saved processed image with proposal to {output_path}")

                        # Save the coordinates in an XML file
                        output_xml_path = os.path.join(output_subfolder, f"{os.path.splitext(filename)[0]}.xml")
                        self.save_proposals_to_xml(proposals[:nb_boxes], output_xml_path)
                        print(f"Saved proposals coordinates to {output_xml_path}")
