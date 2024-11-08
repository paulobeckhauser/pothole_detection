## function that will plot mabo -> Will help to determine Number of proposals

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