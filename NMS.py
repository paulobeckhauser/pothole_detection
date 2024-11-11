import torch
from torchvision.ops import box_iou

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    # Convert boxes and scores to PyTorch tensors
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)


    # Sort boxes by scores in descending order
    _, indices = torch.sort(scores, descending=True)
    selected_boxes = []

    while len(indices) > 0:
        # Select the box with the highest score
        current = indices[0]
        selected_boxes.append(boxes[current].tolist())

        if len(indices) == 1:
            break  # No more boxes to compare

        # Calculate IoU of the current box with the remaining boxes
        remaining_boxes = boxes[indices[1:]]
        ious = box_iou(boxes[current].unsqueeze(0), remaining_boxes).squeeze(0)

        # Keep only boxes with IoU below the threshold
        indices = indices[1:][ious < iou_threshold]

    return selected_boxes

# Example usage
boxes = [
    [100, 100, 210, 210],
    [105, 105, 215, 215],
    [150, 150, 260, 260],
    [200, 200, 310, 310]
]
scores = [0.9, 0.75, 0.8, 0.7]
iou_threshold = 0.5

selected_boxes = non_max_suppression(boxes, scores, iou_threshold)
print("Selected boxes after NMS:", selected_boxes)