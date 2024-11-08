from ground_truth_bounding_boxes import GroundTruthBoundingBoxes
from selective_search import SelectiveSearch

import cv2
import numpy as np

def calculate_overlap(box1, box2):
    # Calculate the intersection over union (IoU) of two boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union = box1_area + box2_area - intersection
    return intersection / union

def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects

def calculate_mabo(images, ground_truth_boxes, num_boxes_list):
    mabo_results = {}
    
    for num_boxes in num_boxes_list:
        total_overlap = 0
        count = 0
        
        for image, gt_boxes in zip(images, ground_truth_boxes):
            proposed_boxes = selective_search(image)[:num_boxes]
            best_overlaps = []
            
            for gt_box in gt_boxes:
                best_overlap = 0
                for prop_box in proposed_boxes:
                    overlap = calculate_overlap(gt_box, prop_box)
                    if overlap > best_overlap:
                        best_overlap = overlap
                best_overlaps.append(best_overlap)
            
            total_overlap += np.mean(best_overlaps)
            count += 1
        
        mabo_results[num_boxes] = total_overlap / count
    
    return mabo_results

# Example usage

def main():

    # ------------ TASK 1 ------------
    # gt_bound_box = GroundTruthBoundingBoxes(name="Data Loader")
    # gt_bound_box.load_dataset(data_folder='images', output_folder = 'bounding_boxes_images')

    # ------------ TASK 2 ------------
    # Selective Search
    # resize_dim = (400, 400)
    # selective_search = SelectiveSearch(input_folder='images', output_folder='ss_images',
    #     resize_dim=resize_dim, mode='fast')
    # selective_search.process_all_images()

    # Edge Boxes

    # ------------ TASK 3 ------------
    images = [...]  # List of images
    ground_truth_boxes = [...]  # List of ground truth boxes for each image
    num_boxes_list = [5, 10, 20]

    mabo_results = calculate_mabo(images, ground_truth_boxes, num_boxes_list)
    print(mabo_results)


# image 1: [0.5, 0.6, 0.7]
# image 2: [0.5, 0.6, 0.7]

if __name__ == "__main__":
    main()
