import torch
import glob
import os
from ground_truth_bounding_boxes import GroundTruthBoundingBoxes
from selective_search import SelectiveSearch
from support import *
# from support import iou, mean_average_best_overlap, plot_mabo_vs_boxes
# from support import convert_boxes, calculate_mabo
from create_dataset import split_images_train_validation_test, put_images_and_labels_in_folders



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # ------------ TASK 1 ------------
    # os.makedirs('results', exist_ok=True)
    # gt_bound_box = GroundTruthBoundingBoxes(name="Data Loader")
    # gt_bound_box.load_dataset(data_folder='images', output_folder = 'bounding_boxes_images') # edit the path
    # print("Loaded dataset")

    
    # # ------------ TASK 2 ------------
    # # Selective Search
    # resize_dim = (400, 400)
    # selective_search = SelectiveSearch(input_folder='images', output_folder='ss_images',
    #                                    resize_dim=resize_dim, mode='fast') # edit the path
    # selective_search.process_all_images(device=device)

    # Edge Boxes
    
    # ------------ TASK 3 ------------
    # MABO for one image ss
    groundtruth_bound_box = GroundTruthBoundingBoxes(name="Data Loader")
    SS = SelectiveSearch(input_folder='images', output_folder='ss_images',
                         resize_dim=(600, 600), mode='quality') # edit the path
    annotation_paths = glob.glob('images/*.xml') # list all xml files
    
    # MABO
    n_boxes_list, mabo = calculate_mabo(annotation_paths=annotation_paths, 
                                        ground_truth_bound_box=groundtruth_bound_box, 
                                        proposal_method=SS, 
                                        device=device,
                                        max_runs=None,
                                        do_plotting=True)
    
    print("Final n_boxes_list", n_boxes_list)
    print("Final mabo", mabo)



    # bb = gt_bound_box.parse_annotation('images/img-2.xml') # edit the path
    # print(bb)
    # image, proposals = SS.create_proposals('images/img-2.jpg', device) # edit the path
    # print(len(proposals))

    # # compute mabo
    # mabo_score = mean_average_best_overlap(bb, proposals)
    # print(mabo_score)
    # plot_mabo_vs_boxes(bb, proposals, save_path='results/mabo_vs_boxes.png') # edit the path

    # # ------------ TASK 4 ------------
    # # get all image paths
    # image_paths = glob.glob('images/*.jpg') # edit the path

    # # split the dataset
    # train_images, validation_images, test_images, train_anno, validation_anno, test_anno = split_images_train_validation_test(image_paths, 
    #                                                                                                                           train_ratio=0.8, validation_ratio=0.1, 
    #                                                                                                                           test_ratio=0.1, 
    #                                                                                                                           save_path='results/train_validation_test.png') # edit the path 

    # put_images_and_labels_in_folders(train_images, validation_images, train_anno, validation_anno, base_folder='data', GTBB = gt_bound_box, bb = SS, limit=None)


    

if __name__ == "__main__":
    main()