import cv2
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.morphology import binary_dilation, disk
from typing import Tuple

EXTREMELY_LARGE_NUMBER = 1000.0
PNI_LABEL = 2

bbox_enlargement_scale = 1.5          
min_bbox_overlap_iou = 0.00
line_overlap_penalty_alpha = 8

max_dist_score = 10

def bbox_map(
    line_map: np.ndarray,
    ) -> Tuple[np.ndarray, int, int, int, int]:
    # Compute coordinates
    (img_h, img_w) = line_map.shape[:2]
    x, y, w, h = cv2.boundingRect(line_map)

    # Enlarge bounding box
    h_margin_half = round((h * bbox_enlargement_scale - h) / 2)
    w_margin_half = round((w * bbox_enlargement_scale - w) / 2)
    
    start_y = max(y - h_margin_half, 0)
    start_x = max(x - w_margin_half, 0)
    end_y = min(y + h + h_margin_half, img_h - 1)
    end_x = min(x + w + w_margin_half, img_w - 1)
    
    bbox_map = np.zeros_like(line_map)
    bbox_map[start_y: end_y,
             start_x: end_x] = 1
       
    return bbox_map, start_x, end_x, start_y, end_y

def compute_avg_min_dist(
    line_region_a: np.ndarray,
    line_region_b: np.ndarray,
    length_norm_gamma: int = 200,
    ) -> float:
    
    min_dists = list()
    for i in range(line_region_a.shape[0]):
        for j in range(line_region_a.shape[1]):
            # Skip for empty pixel
            if line_region_a[i][j] == 0:
                continue
            
            min_dist = math.inf
            for k in range(line_region_b.shape[0]):
                for l in range(line_region_b.shape[1]):
                    # Skip for empty pixel
                    if line_region_b[k][l] == 0:
                        continue
                    
                    dist = compute_dist_bw_pts((i, j), (k, l))
                    if dist < min_dist:
                        min_dist = dist
    
            min_dists.append(min_dist)
    avg_min_dist = sum(min_dists) / len(min_dists)
    
    # Normalize based on line length
    line_a_length = np.sum(line_region_a) 
    line_a_length_norm = math.exp(line_a_length / length_norm_gamma)
    avg_min_dist = avg_min_dist * line_a_length_norm
    
    return avg_min_dist

def compute_bidirect_avg_min_dist(
    line_region_a: np.ndarray,
    line_region_b: np.ndarray,
    ) -> float:
    
    avg_min_dist_ab = compute_avg_min_dist(line_region_a, line_region_b)
    avg_min_dist_ba = compute_avg_min_dist(line_region_b, line_region_a)
    bidirect_avg_min_dist = (avg_min_dist_ab + avg_min_dist_ba) / 2
    
    return bidirect_avg_min_dist

def compute_dist_bw_pts(
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    ) -> float:
    
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]
    pt_dist = math.sqrt(math.pow((x2 - x1), 2) + math.pow(y2 - y1, 2))
    
    return pt_dist

def compute_dist_score(
    line_region_a: np.ndarray,
    line_region_b: np.ndarray,
    ) -> float:
    
    # Bidirectional average of minimum distances
    bidirect_avg_min_dist = compute_bidirect_avg_min_dist(line_region_a, 
                                                          line_region_b)
    
    # Line overlap penalty
    line_iou = compute_line_iou(line_region_a, line_region_b)
    line_overlap_penalty = (1.00 - line_iou) ** line_overlap_penalty_alpha
    
    # Final distance score
    dist_score = bidirect_avg_min_dist * line_overlap_penalty
    
    return dist_score

def compute_f1_score(
    score_table: np.ndarray,
    gt_labels: int,
    pred_labels: int,
    ) -> float:
    
    gt_count = gt_labels - 1
    pred_count = pred_labels - 1
    
    score_table[score_table > EXTREMELY_LARGE_NUMBER] = EXTREMELY_LARGE_NUMBER

    matched_gt_labels, matched_pred_labels = linear_sum_assignment(score_table)
            
    reasonable_match = np.full_like(matched_gt_labels, True, dtype=bool)
    for i, matched_gt_label in enumerate(matched_gt_labels):
        matched_pred_label = matched_pred_labels[i]
        dist_score = score_table[matched_gt_label][matched_pred_label]
        
        if dist_score > max_dist_score:
            reasonable_match[i] = False
    
    matched_gt_labels = matched_gt_labels[reasonable_match]
    matched_pred_labels = matched_pred_labels[reasonable_match]
            
    unmatched_pred_labels = set(range(pred_count)) - set(matched_pred_labels)
                
    for unmatched_pred_label in unmatched_pred_labels.copy():
        min_unmatched_dist_score = np.min(score_table[..., unmatched_pred_label])
        if min_unmatched_dist_score <= max_dist_score:
            unmatched_pred_labels.remove(unmatched_pred_label)
            
    tp = len(matched_gt_labels)
    fp = len(unmatched_pred_labels)
    fn = gt_count - len(matched_gt_labels)
        
    precision = tp / (tp + fp) if (tp + fp) != 0.0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) != 0.0 else 1.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0.0 else 0.0
    
    return f1_score

def compute_iou(
    outputs: np.array, 
    labels: np.array
    ) -> float:
    
    outputs = outputs > 0
    labels = labels > 0
    
    intersection = np.sum(np.logical_and(outputs, labels))
    union = np.sum(np.logical_or(outputs, labels))

    return intersection / union

def compute_line_iou(
    line_region_a: np.ndarray,
    line_region_b: np.ndarray,
    dilation_size: int = 1,
    ) -> float:          
    
    line_region_a = binary_dilation(line_region_a > 0, disk(dilation_size))
    line_region_b = binary_dilation(line_region_b > 0, disk(dilation_size))
    
    line_iou = compute_iou(line_region_a, line_region_b)
    
    return line_iou

def compute_dist_f1_score(
    pred: np.ndarray,
    gt: np.ndarray,
    ) -> float:
                
    # Extract only PNI line labels
    gt = (gt == PNI_LABEL).astype(np.uint8)
        
    # Connected components
    gt_labels, gt_label_map = cv2.connectedComponents(gt, connectivity=8)
    pred_labels, pred_label_map = cv2.connectedComponents(pred, connectivity=8)
        
    # Score Table
    score_table = np.full((gt_labels - 1, pred_labels - 1), np.inf, dtype=np.float32)
                        
    for i in range(1, gt_labels): # Exclude background label
        gt_i = (gt_label_map == i).astype(np.uint8)
        gt_i_bbox, gt_start_x, gt_end_x, gt_start_y, gt_end_y = \
            bbox_map(gt_i)
        
        for j in range(1, pred_labels):
            pred_j = (pred_label_map == j).astype(np.uint8)
            pred_j_bbox, pred_start_x, pred_end_x, pred_start_y, pred_end_y = \
                bbox_map(pred_j)
    
            overlap_iou = compute_iou(gt_i_bbox, pred_j_bbox)
    
            if overlap_iou <= min_bbox_overlap_iou:
                continue
    
            # Calculate minimum region to compute distance
            start_y = min(gt_start_y, pred_start_y)
            start_x = min(gt_start_x, pred_start_x)
            end_y = max(gt_end_y, pred_end_y)
            end_x = max(gt_end_x, pred_end_x)
    
            gt_line_region = gt_i[start_y:end_y, start_x:end_x]
            pred_line_region = pred_j[start_y:end_y, start_x:end_x]
            
            dist_score_ij = compute_dist_score(gt_line_region, pred_line_region)

            # Update distance score
            score_table[i - 1, j - 1] = dist_score_ij
    
    f1_score = compute_f1_score(score_table, gt_labels, pred_labels)
    
    return f1_score

if __name__ == "__main__":
    pred = None
    gt = None
    
    dist_f1_score = compute_dist_f1_score(pred=pred, 
                                          gt=gt)