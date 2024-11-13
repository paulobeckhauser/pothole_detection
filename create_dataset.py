import torch
import random
import matplotlib.pyplot as plt
import os
import cv2
from ground_truth_bounding_boxes import GroundTruthBoundingBoxes
from selective_search import SelectiveSearch
from support import iou, convert_boxes

def split_images_train_validation_test(image_paths, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, save_path='/work3/s214598/train_validation_test.png', GTBB = GroundTruthBoundingBoxes(name="Data Loader")):
    """
    Split the dataset into train, validation, and test sets.

    Parameters:
    - image_paths: List of image paths.
    - annotation_path: Path to the annotation file.
    - train_ratio: Ratio of the dataset to use for training.
    - validation_ratio: Ratio of the dataset to use for validation.
    - test_ratio: Ratio of the dataset to use for testing.

    Returns:
    - train_images: List of image paths for the training set.
    - validation_images: List of image paths for the validation set.
    - test_images: List of image paths for the test set.
    """
    # set random seed
    random.seed(42)
    
    # Calculate the number of images for each set
    num_images = len(image_paths)
    num_train = int(train_ratio * num_images)
    num_validation = int(validation_ratio * num_images)
    num_test = num_images - num_train - num_validation

    # Find a new sequence of image paths
    index_list = list(range(num_images))
    index_shuffle = random.shuffle(index_list)
    new_image_paths = image_paths
    
    annotation_path = []
    for i in range(len(new_image_paths)):
        annotation_path.append(new_image_paths[i].replace('.jpg', '.xml'))
        

    # Split the image paths
    train_images = new_image_paths[:num_train]
    validation_images = new_image_paths[num_train:num_train + num_validation]
    test_images = new_image_paths[num_train + num_validation:]

    # annotate the images
    train_annotations = annotation_path[:num_train]
    validation_annotations = annotation_path[num_train:num_train + num_validation]
    test_annotations = annotation_path[num_train + num_validation:]
    
    # Draw and save bounding boxes for the first image in the training set
    image_path = train_images[0]
    annotation_path = train_annotations[0]
    bboxes_train = GTBB.parse_annotation(annotation_path)
    image_train = GTBB.draw_and_save_bounding_boxes(image_path, bboxes_train, output_path=None)

    # Draw and save bounding boxes for the first image in the validation set
    image_path = validation_images[0]
    annotation_path = validation_annotations[0]
    bboxes_validation = GTBB.parse_annotation(annotation_path)
    image_validation = GTBB.draw_and_save_bounding_boxes(image_path, bboxes_validation, output_path=None)

    # Draw and save bounding boxes for the first image in the test set
    image_path = test_images[0]
    annotation_path = test_annotations[0]
    bboxes_test = GTBB.parse_annotation(annotation_path)
    image_test = GTBB.draw_and_save_bounding_boxes(image_path, bboxes_test, output_path=None)

    # plot one of each in a figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 5)) 
    ax[0].imshow(image_train)
    ax[0].set_title('Training Image')
    ax[0].axis('off')
    ax[1].imshow(image_validation)
    ax[1].set_title('Validation Image')
    ax[1].axis('off')
    ax[2].imshow(image_test)
    ax[2].set_title('Test Image')
    ax[2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return train_images, validation_images, test_images, train_annotations, validation_annotations, test_annotations

def put_images_and_labels_in_folders(train_images, validation_images, train_anno, validation_anno, base_folder='/work3/s214598/pothole_detection/data', GTBB = GroundTruthBoundingBoxes(name="Data Loader"), bb = SelectiveSearch(input_folder='/work3/s214598/pothole_detection/images', output_folder='ss_images', resize_dim=(600, 600), mode='quality'), limit=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create the output folder if it doesn't exist
    os.makedirs(base_folder, exist_ok=True)

    # create the train
    for i, file in enumerate(train_images):
        image = cv2.imread(file)

        # get real bounding boxes
        bboxes = GTBB.parse_annotation(train_anno[i])

        # do the search
        image, proposals = bb.create_proposals(file, device)
        # add proposals limit if not None
        if limit is not None:
            proposals = proposals[:limit]

        proposals = convert_boxes(proposals)
        if not isinstance(bboxes, torch.Tensor):
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
        if not isinstance(proposals, torch.Tensor):
            proposals = torch.tensor(proposals, dtype=torch.float32)
        # for each proposal, check if the IoU is greater than 0.5 with any of the real bounding boxes 
        # and label it as a pothole or not
        labels = []
        for proposal in proposals:
            
            # check if the IoU is greater than 0.5 with any of the real bounding boxes
            label = 0
            for bbox in bboxes:
                if iou(proposal, bbox) > 0.5:
                    label = 1
                    break
            labels.append(label)
        pothole_count = 0
        background_count = 0
        # save 
        for true_bb in bboxes:
            x_min, y_min, x_max, y_max = true_bb
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            # crop the image
            crop = image[y_min:y_max, x_min:x_max]
            crop = cv2.resize(crop, (128, 128))
            pothole_count += 1
            os.makedirs(os.path.join(base_folder, 'train', f'{i}', 'pothole'), exist_ok=True)
            cv2.imwrite(os.path.join(base_folder, 'train', f'{i}', 'pothole', f'pothole_{pothole_count}.jpg'), crop)

        # shuffe all the bounding boxes with its labels
        zipped = list(zip(proposals, labels))
        random.shuffle(zipped)
        for proposal, label in zipped:
            x_min, y_min, x_max, y_max = proposal
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            # crop the image
            if label==1 and pothole_count<16:
                crop = image[y_min:y_max, x_min:x_max]
                crop = cv2.resize(crop, (128, 128))
                pothole_count += 1
                cv2.imwrite(os.path.join(base_folder, 'train', f'{i}','pothole', f'pothole_{pothole_count}.jpg'), crop)
            elif label==0 and background_count<48:
                crop = image[y_min:y_max, x_min:x_max]
                # resize the image to 128x128
                crop = cv2.resize(crop, (128, 128))

                background_count += 1
                os.makedirs(os.path.join(base_folder, 'train', f'{i}', 'background'), exist_ok=True)
                cv2.imwrite(os.path.join(base_folder, 'train', f'{i}', 'background', f'background_{background_count}.jpg'), crop)
            elif pothole_count==16 and background_count==48:
                break
    # create the validation
    for i, file in enumerate(validation_images):
        image = cv2.imread(file)

        # get real bounding boxes
        bboxes = GTBB.parse_annotation(validation_anno[i])

        # do the search
        image, proposals = bb.create_proposals(file, device)
        # add proposals limit if not None
        if limit is not None:
            proposals = proposals[:limit]

        proposals = convert_boxes(proposals)
        if not isinstance(bboxes, torch.Tensor):
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
        if not isinstance(proposals, torch.Tensor):
            proposals = torch.tensor(proposals, dtype=torch.float32)
        # for each proposal, check if the IoU is greater than 0.5 with any of the real bounding boxes 
        # and label it as a pothole or not
        labels = []
        for proposal in proposals:
            
            # check if the IoU is greater than 0.5 with any of the real bounding boxes
            label = 0
            for bbox in bboxes:
                if iou(proposal, bbox) > 0.5:
                    label = 1
                    break
            labels.append(label)
        pothole_count = 0
        background_count = 0
        # save 
        for true_bb in bboxes:
            x_min, y_min, x_max, y_max = true_bb
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            # crop the image
            crop = image[y_min:y_max, x_min:x_max]
            crop = cv2.resize(crop, (128, 128))
            pothole_count += 1
            os.makedirs(os.path.join(base_folder, 'validation', f'{i}', 'pothole'), exist_ok=True)
            cv2.imwrite(os.path.join(base_folder, 'validation', f'{i}', 'pothole', f'pothole_{pothole_count}.jpg'), crop)

        # shuffe all the bounding boxes with its labels
        zipped = list(zip(proposals, labels))
        random.shuffle(zipped)
        for proposal, label in zipped:
            x_min, y_min, x_max, y_max = proposal
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            # crop the image
            if label==1 and pothole_count<16:
                crop = image[y_min:y_max, x_min:x_max]
                crop = cv2.resize(crop, (128, 128))
                pothole_count += 1
                cv2.imwrite(os.path.join(base_folder, 'validation', f'{i}','pothole', f'pothole_{pothole_count}.jpg'), crop)
            elif label==0 and background_count<48:
                crop = image[y_min:y_max, x_min:x_max]
                # resize the image to 128x128
                crop = cv2.resize(crop, (128, 128))

                background_count += 1
                os.makedirs(os.path.join(base_folder, 'validation', f'{i}', 'background'), exist_ok=True)
                cv2.imwrite(os.path.join(base_folder, 'validation', f'{i}', 'background', f'background_{background_count}.jpg'), crop)
            elif pothole_count==16 and background_count==48:
                break


        

        


        