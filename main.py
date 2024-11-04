from ground_truth_bounding_boxes import GroundTruthBoundingBoxes
from selective_search import SelectiveSearch

def main():

    # ------------ TASK 1 ------------
    gt_bound_box = GroundTruthBoundingBoxes(name="Data Loader")
    gt_bound_box.load_dataset(data_folder='images', output_folder = 'bounding_boxes_images')


    # ------------ TASK 2 ------------
    # Selective Search
    resize_dim = (400, 400)
    selective_search = SelectiveSearch(input_folder='images', output_folder='ss_images',
        resize_dim=resize_dim, mode='fast')
    selective_search.process_all_images()

    # Edge Boxes
    

if __name__ == "__main__":
    main()