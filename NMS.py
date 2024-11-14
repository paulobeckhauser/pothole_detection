import torch
# from torchvision.ops import box_iou
from support import iou
import cv2
import os

def non_max_suppression(boxes, scores, iou_threshold=0.5, image_path=None):
    """
    Perform Non-Max Suppression (NMS) on bounding boxes with confidence scores.

    Parameters:
    - boxes (list or torch.Tensor): A list of bounding boxes, each represented as [xmin, ymin, xmax, ymax].
      If it's a list, it will be converted to a torch.Tensor.
    - scores (list or torch.Tensor): A list of confidence scores corresponding to each bounding box.
      If it's a list, it will be converted to a torch.Tensor.
    - iou_threshold (float): IoU threshold for NMS. Boxes with IoU above this threshold will be suppressed.
      Default is 0.5.

    Returns:
    - tuple: A tuple containing two lists:
      - selected_boxes (list): A list of selected bounding boxes in the format [xmin, ymin, xmax, ymax].
      - selected_scores (list): A list of confidence scores corresponding to each selected bounding box.
    """

    # create a folder to save the results
    os.makedirs('results/nms', exist_ok=True)

    # display before nms
    if image_path is not None:
        img_name = image_path.split('/')[-1][4:-4] # images/img-454.jpg
        # plot before NMS
        draw_and_save_bounding_boxes(image_path=image_path, bboxes=boxes, output_path=f'results/nms/before_nms{img_name}.jpg')
        len_before = len(boxes)


    # Ensure boxes and scores are tensors
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)

    # Sort boxes by scores in descending order
    _, indices = torch.sort(scores, descending=True)
    selected_boxes = []
    selected_scores = []

    # Start with the box with the highest score, remove boxes with high IoU
    # Stop when there are no more overlapping boxes
    while len(indices) > 0:
        # Select the box with the highest score
        current = indices[0]
        selected_boxes.append(boxes[current].tolist())
        selected_scores.append(scores[current].item())

        # Calculate IoU of the current box with the remaining boxes
        remaining_boxes = boxes[indices[1:]]
        ious = torch.tensor([iou(boxes[current], b) for b in remaining_boxes])

        # Keep only boxes with IoU below the threshold
        indices = indices[1:][ious < iou_threshold]

    # display after nms
    if image_path is not None:
        # plot after NMS
        draw_and_save_bounding_boxes(image_path=image_path, bboxes=selected_boxes, output_path=f'results/nms/after_nms{img_name}.jpg')
        len_after = len(selected_boxes)
        print(f'image: {img_name}, before NMS: {len_before}, after NMS: {len_after}')

    return selected_boxes, selected_scores


def draw_and_save_bounding_boxes(image_path, bboxes, output_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for (xmin, ymin, xmax, ymax) in bboxes:
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        # print(xmin, ymin, xmax, ymax)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=1)

    # Save the image with bounding boxes to the output folder
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if output_path is not None:
        cv2.imwrite(output_path, image)
    else:
        # convert to a numpy and return'
        return image

if __name__ == '__main__':
    # Example usage
    boxes = [
        [100, 100, 210, 210],
        [105, 105, 215, 215],
        [150, 150, 260, 260],
        [200, 200, 310, 310]
    ]
    scores = [0.9, 0.75, 0.8, 0.7]
    iou_threshold = 0.5

    selected_boxes, selected_scores = non_max_suppression(boxes, scores, iou_threshold)
    selected_scores = [round(score, 2) for score in selected_scores]  # Round scores for display
    print("Selected boxes after NMS:", (selected_boxes, selected_scores))