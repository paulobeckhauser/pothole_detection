import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

class test_object_detection:

    def __init__(self, test_files_path, network, search_method, ground_truth_boxes, convert_box, iou, NMS, iou_threshold=0.5, device='cpu'):
        
        self.name = test_files_path
        self.network = network.to(device)
        self.search_method = search_method
        self.ground_truth_boxes = ground_truth_boxes
        self.convert_box = convert_box
        self.iou = iou
        self.NMS = NMS
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5263633, 0.5145254, 0.49127004], std=[0.16789266, 0.16256736, 0.1608751])
        ])
        self.network.eval()

    def create_proposals_and_ground_truth(self, image_path):
        image, proposals = self.search_method.create_proposals(image_path, self.device)
        proposals = self.convert_box(proposals)
        annotation_path = image_path.replace('.jpg', '.xml')
        ground_truth_boxes = self.ground_truth_boxes.parse_annotation(annotation_path)
        return image, proposals, ground_truth_boxes
    
    def precision_recall(self, predicted_boxes, ground_truth_boxes, confidense, index_predicted, index_groundtruth, iou_threshold=0.5):
        number_of_ground_truth_boxes = len(ground_truth_boxes)
        sorted_indices = np.argsort(confidense)[::-1]
        
        predicted_boxes = predicted_boxes[sorted_indices]
        index_predicted = index_predicted[sorted_indices]

        true_positives = torch.zeros(len(predicted_boxes)).to(self.device)
        false_positives = torch.zeros(len(predicted_boxes)).to(self.device)

        for i, predicted_box in enumerate(predicted_boxes):

            # get image
            image_index = index_predicted[i]
            mask = (index_groundtruth == image_index)
            # get ground truth boxes for the image
            ground_truth_boxes_image = ground_truth_boxes[mask]
            # i want to know the indexes where mask is true
            index_list = np.where(mask)

            for j, ground_truth_box in enumerate(ground_truth_boxes_image):
                iou = self.iou(predicted_box, ground_truth_box)
                if iou > iou_threshold:
                    # remove the ground truth box using numpy  
                    index = index_list[0][j] 
                    ground_truth_boxes = np.delete(ground_truth_boxes, index, axis=0)
                    index_groundtruth = np.delete(index_groundtruth, index, axis=0)
                    true_positives[i] = 1
                    break
            false_positives[i] = 1 - true_positives[i]


        true_pos = torch.cumsum(true_positives, dim=0)
        false_pos= torch.cumsum(false_positives, dim=0)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / number_of_ground_truth_boxes

        return precision, recall
    
    def avarage_precision(self, precision, recall):
        # get area under the curve
        precision = torch.cat((torch.tensor([0]), precision, torch.tensor([0])))
        recall = torch.cat((torch.tensor([0]), recall, torch.tensor([1])))

        for i in range(len(precision) - 2, -1, -1):
            precision[i] = torch.max(precision[i], precision[i+1])

        indices = torch.where(recall[1:] != recall[:-1])[0] + 1
        average_precision = torch.sum((recall[indices] - recall[indices-1]) * precision[indices])
        return average_precision
                
    
    def test_all_images(self, test_files_path):
        with torch.no_grad():
            # Initialize tensors
            image_box_tensor = []
            proposals_list = []
            confidense_tensor = []
            ground_truth_boxes_list = []
            ground_truth_boxes_image = []

            for i, image_path in enumerate(test_files_path):
                image, proposals, ground_truth_boxes = self.create_proposals_and_ground_truth(image_path)
                
                # Crop images
                cropped_images = torch.zeros(len(proposals), 3, 128, 128).to(self.device)
                for j, proposal in enumerate(proposals):
                    xmin, ymin, xmax, ymax = map(int, proposal)
                    crop = image[ymin:ymax, xmin:xmax]
                    crop_pil = Image.fromarray(crop)
                    cropped_image = self.transform(crop_pil)
                    cropped_images[j] = cropped_image.to(self.device)

                
                confidense = torch.zeros(len(proposals)).to(self.device)
                # Predict in batches of 64
                
                outputs = self.network(cropped_images)
                confidense = outputs[:, 0]
                
                # Run NMS
                print(image_path)
                predicted_boxes, confidense = self.NMS(proposals, confidense, iou_threshold=0.2, image_path=image_path)
            
                # Update tensors
                for j in range(len(confidense)):
                    image_box_tensor.append(i)
                    proposals_list.append(predicted_boxes[j])
                    confidense_tensor.append(confidense[j])
                
                for j in range(len(ground_truth_boxes)):
                    ground_truth_boxes_image.append(i)
                    ground_truth_boxes_list.append(ground_truth_boxes[j])
                
                # Clear unused variables
                del cropped_images, outputs, confidense, predicted_boxes
            
            # Calculate precision and recall
            precision, recall = self.precision_recall(np.array(proposals_list), np.array(ground_truth_boxes_list), np.array(confidense_tensor), np.array(image_box_tensor), np.array(ground_truth_boxes_image))
            self.plot_precision_recall(precision, recall)
            AP = self.avarage_precision(precision, recall)

            print("precision", precision)
            print("recall", recall)

        return AP
    
    def plot_precision_recall(self, precision, recall):
        plt.clf() # clear plot
        plt.plot(recall, precision, color='b')
        plt.scatter(recall, precision, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0, 1.001])
        plt.ylim([0, 1.001])
        plt.grid()
        plt.title('Precision-Recall Curve')
        plt.savefig('results/pr_curve.png')
        plt.show()
    

if __name__ == '__main__':

    test_object_detection = test_object_detection(test_files_path=None, network=None, search_method=None, ground_truth_boxes=None, convert_box=None, iou=None, NMS=None) 

    precision = torch.tensor([1.0, 1.0, 2/3, 0.5, 0.6])
    recall = torch.tensor([1/3, 2/3, 2/3, 2/3, 1.0])

    average_precision = test_object_detection.avarage_precision(precision, recall)
    print(average_precision)

    sorted_indices = np.argsort(np.array([1,12,2]))[::-1]
    print(sorted_indices)