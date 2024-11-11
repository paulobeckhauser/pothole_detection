import torch
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def convert_boxes(rects):
    return [(x, y, x + w, y + h) for (x, y, w, h) in rects]

def iou(box1, box2):
    """
    Compute the IoU between two boxes using PyTorch tensors.
    Each box is represented as [x_min, y_min, x_max, y_max].
    """
    x1_inter = torch.max(box1[0], box2[0])
    y1_inter = torch.max(box1[1], box2[1])
    x2_inter = torch.min(box1[2], box2[2])
    y2_inter = torch.min(box1[3], box2[3])
    
    # Calculate the area of intersection
    inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
    
    # Calculate the area of each box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate the IoU
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else torch.tensor(0.0)

def mean_average_best_overlap(ground_truth_boxes, generated_boxes):
    """
    Compute the Mean Average Best Overlap (MABO) using PyTorch tensors.
    
    Parameters:
    - ground_truth_boxes: Tensor of shape (N, 4) where N is the number of ground truth boxes.
    - generated_boxes: Tensor of shape (M, 4) where M is the number of generated boxes.
    
    Returns:
    - mabo: Mean Average Best Overlap score as a torch scalar.
    """
    generated_boxes = convert_boxes(generated_boxes)
    best_ious = []

    # Convert input lists to PyTorch tensors if they aren't already
    if not isinstance(ground_truth_boxes, torch.Tensor):
        ground_truth_boxes = torch.tensor(ground_truth_boxes, dtype=torch.float32)
    if not isinstance(generated_boxes, torch.Tensor):
        generated_boxes = torch.tensor(generated_boxes, dtype=torch.float32)

    # For each ground truth box, find the best IoU with any of the generated boxes
    for gt_box in ground_truth_boxes:
        max_iou = torch.tensor(0.0)
        for gen_box in generated_boxes:
            iou_score = iou(gt_box, gen_box)
            max_iou = torch.max(max_iou, iou_score)
        best_ious.append(max_iou)
    
    # Calculate the mean of the best IoUs
    best_ious = torch.stack(best_ious)
    mabo = torch.mean(best_ious)
    return mabo

# Main function to calculate and plot MABO vs Number of Boxes
def plot_mabo_vs_boxes(ground_truth_boxes, rects, save_path='/work3/s214598/mabo_vs_boxes.png'):
    step = 100
    # Convert ground truth boxes to tensors
    ground_truth_tensor = torch.tensor(ground_truth_boxes, dtype=torch.float32).cpu()
    
    num_boxes_list = np.arange(100, min(3000+step, len(rects)+step), step)
    mabo_scores = []

    # Calculate MABO for different numbers of proposals
    for num_boxes in num_boxes_list:
        selected_boxes = rects[:num_boxes]
        mabo = mean_average_best_overlap(ground_truth_tensor, selected_boxes)
        mabo_scores.append(mabo)
    
    # Convert lists to NumPy arrays for plotting
    num_boxes_list = np.array(num_boxes_list)
    mabo_scores = np.array([mabo.cpu().numpy() if isinstance(mabo, torch.Tensor) else mabo for mabo in mabo_scores])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(num_boxes_list, mabo_scores, label='Selective Search', color='red')
    plt.xlabel("Number of Object Boxes")
    plt.ylabel("Mean Average Best Overlap (MABO)")
    plt.title("MABO vs. Number of Object Boxes")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

