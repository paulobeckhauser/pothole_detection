from ground_truth_bounding_boxes import GroundTruthBoundingBoxes
from selective_search import SelectiveSearch

def main():

    # ------------ TASK 1 ------------
    gt_bound_box = GroundTruthBoundingBoxes(name="Data Loader")
    gt_bound_box.load_dataset(data_folder='images', output_folder = 'bounding_boxes_images')


    # ------------ TASK 2 ------------
    # Selective Search
    # resize_dim = (400, 400)
    # selective_search = SelectiveSearch(input_folder='images', output_folder='ss_images',
    #     resize_dim=resize_dim, mode='fast')
    # selective_search.process_all_images()

    # Edge Boxes



    # ------------ TASK 3 ------------
    n_region_proposals = [5, 10, 50, 100]

    os.makedirs(self.output_folder, exist_ok=True)
    output_folder='mabo_results'

    mabos = mabo_ss_function(n_region_proposals, input_folder='images', output_folder=output_folder)








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

def mabo_function(n_region_proposals, input_folder='images', output_folder=output_folder):
    
    # For every image
    for filename in os.listdir(self.input_folder):
        if filename.lower().endswith(('.jpg', 'jpeg', '.png')): # Filter for image files
        image_path = os.path.join(self.input_folder, filename)

        # Compute region proposals
        image_resized, proposals = self.run_selective_search(image_path)

        image_resized, proposals = self.run_edge_boxes(image_path)

        # Compute IoU for all bounding boxes
        consider_proposals = proposals[max(n_region_proposals)]

        ious = calc_ious(consider_proposals)

        abos = []
        for n_regions in n_region_proposals:
            max_iou = max(ious[:n_regions])
            abos.append(max_iou)

    mabos = 
    
    
    return mabos



# image 1: [0.5, 0.6, 0.7]
# image 2: [0.5, 0.6, 0.7]

    

if __name__ == "__main__":
    main()