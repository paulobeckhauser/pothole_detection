import cv2
import os
import torch


class SelectiveSearch:
    def __init__(self, input_folder, output_folder, resize_dim=(600, 600), max_num_boxes = None, mode='fast'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.resize_dim = resize_dim
        self.max_num_boxes = max_num_boxes
        self.mode = mode.lower()
        os.makedirs(self.output_folder, exist_ok=True)
    
    def create_proposals(self, image_path, device):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return None, None

        # Initialize Selective Search
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)

        # Set mode
        if self.mode == 'fast':
            ss.switchToSelectiveSearchFast()
        elif self.mode == 'quality':
            ss.switchToSelectiveSearchQuality()
        else:
            raise ValueError("Invalid mode: choose 'fast' or 'quality'")
        
        # Run Selective Search
        rects = ss.process()
        rects = torch.tensor(rects, dtype=torch.float32).to(device)
        if self.max_num_boxes is not None:
            rects = rects[:self.max_num_boxes]
        return image, rects

    def process_all_images(self, device):
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(('.jpg', 'jpeg', '.png')): # Filter for image files
                image_path = os.path.join(self.input_folder, filename)
                image_resized, proposals = self.create_proposals(image_path, device)

                if image_resized is not None:
                    # Draw the first 100 proposals for visualization
                    for (x, y, w, h) in proposals[:100]:
                        cv2.rectangle(image_resized, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

                    output_path = os.path.join(self.output_folder, f"proposals_{filename}")
                    cv2.imwrite(output_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                    print(f"Saved processed image with proposal to {output_path}")
    
