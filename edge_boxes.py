import cv2
import os

class EdgeBoxes:
    def __init__(self, max_boxes=1000, min_score=0.01):
        # Initialize EdgeBoxes
        self.edge_detector = cv2.ximgproc.createStructuredEdgeDetection('pre_trained_edge_detect_model.yml.gz') # 
