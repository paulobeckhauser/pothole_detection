import torch


class test_object_detection:

    def __init__(self, test_files_path, network, search_method, ground_truth_boxes, convert_box, iou, NMS, iou_threshold=0.5, device='cpu'):
        self.name = test_files_path
        self.network = network
        self.search_method = search_method
        self.ground_truth_boxes = ground_truth_boxes
        self.convert_box = convert_box
        self.iou = iou
        self.NMS = NMS
        self.device = device

    def create_proposals_and_ground_truth(self, image_path):
        image, proposals = self.search_method.create_proposals(image_path, self.device)
        proposals = self.convert_box(proposals)
        annotation_path = image_path.replace('images', 'bounding_boxes_images').replace('.jpg', '.xml')
        ground_truth_boxes = self.ground_truth_boxes.parse_annotation(annotation_path)
        return image, proposals, ground_truth_boxes
    
    def precision_recall(self, predicted_boxes, ground_truth_boxes, confidense, iou_threshold=0.5):

        sorted_indices = torch.argsort(confidense, descending=True)
        predicted_boxes = predicted_boxes[sorted_indices]
        true_positives = torch.zeros(len(predicted_boxes)).to(self.device)
        false_positives = torch.zeros(len(predicted_boxes)).to(self.device)

        for i, predicted_box in enumerate(predicted_boxes):
            
            for ground_truth_box in ground_truth_boxes:
                iou = self.iou(predicted_box, ground_truth_box)
                if iou > iou_threshold:
                    # remove the ground truth box
                    ground_truth_boxes.remove(ground_truth_box)
                    true_positives[i] = 1
                    break
            false_positives[i] = 1 - true_positives[i]


        true_pos = torch.cumsum(true_positives, dim=0)
        false_pos= torch.cumsum(false_positives, dim=0)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / len(ground_truth_boxes)

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
        avarage_precision = []
        for image_path in test_files_path:
            image, proposals, ground_truth_boxes = self.create_proposals_and_ground_truth(image_path)
            image = image.to(self.device)
            proposals = proposals.to(self.device)

            # crop image
            cropepd_images = torch.zeros(len(proposals), 3, 128, 128)
            for i, proposal in enumerate(proposals):
                xmin, ymin, xmax, ymax = proposal
                cropped_image = image[:, :, ymin:ymax, xmin:xmax]
                cropepd_images[i] = cropped_image

            # predict
            outputs = self.network(cropepd_images)
            confidense = outputs.squeeze()
            confidense = confidense.cpu().detach().numpy()
            predicted_boxes = proposals[confidense > 0.5]

            # run NMS
            predicted_boxes = self.NMS(predicted_boxes, confidense)

            # calculate avarage precision
            precision, recall = self.precision_recall(predicted_boxes, ground_truth_boxes, confidense)
            AP =  self.avarage_precision(precision, recall)
            avarage_precision.append(AP)
        AP = torch.mean(torch.tensor(avarage_precision))
        return AP
    
if __name__ == '__main__':

    test_object_detection = test_object_detection(test_files_path=None, network=None, search_method=None, ground_truth_boxes=None, convert_box=None, iou=None, NMS=None) 

    precision = torch.tensor([1.0, 1.0, 2/3, 0.5, 0.6])
    recall = torch.tensor([1/3, 2/3, 2/3, 2/3, 1.0])

    average_precision = test_object_detection.avarage_precision(precision, recall)
    print(average_precision)