import numpy as np
from medpy.metric.binary import dc, hd95
import torch


def convert_mask_argmax(img_t):
    """
    Convert 10x256x256 multi-channel mask to single-channel semantic segmentation mask
    Background is 255, classes from 0-9
    """
    # Get predicted class for each position (0-9)
    pred_mask = torch.argmax(img_t, dim=0)  # Output shape: 256x256

    # Check if any class has prediction (max value > 0.5)
    has_prediction = torch.max(img_t, dim=0)[0] > 0.5

    # Set positions with no prediction to 255 (background)
    final_mask = torch.where(has_prediction, pred_mask, 255)
    return final_mask


def calculate_metrics(mask1, mask2, num_classes=10):
    """
    Calculate normalized 95% Hausdorff distance, Dice coefficient and IoU between two masks

    Parameters:
        mask1: numpy array, shape (1, H, W) or (H, W)
        mask2: numpy array, shape (1, H, W) or (H, W)
        num_classes: number of classes (excluding background)

    Returns:
        dict: Dictionary containing metrics for each class and average metrics
    """

    # Ensure masks are numpy arrays and remove single dimensions
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()

    mask1 = np.squeeze(mask1)
    mask2 = np.squeeze(mask2)

    assert mask1.shape == mask2.shape, "Masks must have the same shape"

    results = {}

    # Calculate metrics for each class
    for class_id in range(num_classes):
        # Extract binary mask for current class
        binary_mask1 = (mask1 == class_id).astype(np.uint8)
        binary_mask2 = (mask2 == class_id).astype(np.uint8)

        # Calculate Dice coefficient
        dice = calculate_dice(binary_mask1, binary_mask2)

        # Calculate IoU
        iou = calculate_iou(binary_mask1, binary_mask2)

        # Calculate normalized 95% Hausdorff distance
        nhd95 = calculate_normalized_hausdorff95(binary_mask1, binary_mask2)

        results[f'class_{class_id}'] = {
            'dice': dice,
            'iou': iou,
            'normalized_hausdorff95': nhd95
        }

    # ---------- Calculate average metrics ----------
    def valid_mean(values):
        """Calculate mean excluding invalid values (-1, NaN, negative values)"""
        valid = [v for v in values if (v is not None and not np.isnan(v) and v >= 0)]
        if len(valid) == 0:
            return -1.0  # All classes are invalid
        return float(np.mean(valid))  # Ensure float return

    # Calculate average for all 10 classes (0~9)
    results['mean_0_9'] = {
        'dice': valid_mean([results[f'class_{i}']['dice'] for i in range(num_classes)]),
        'iou': valid_mean([results[f'class_{i}']['iou'] for i in range(num_classes)]),
        'normalized_hausdorff95': valid_mean(
            [results[f'class_{i}']['normalized_hausdorff95'] for i in range(num_classes)]
        )
    }

    # Calculate average for classes 1~8
    selected_classes = range(1, 9)
    results['mean_1_8'] = {
        'dice': valid_mean([results[f'class_{i}']['dice'] for i in selected_classes]),
        'iou': valid_mean([results[f'class_{i}']['iou'] for i in selected_classes]),
        'normalized_hausdorff95': valid_mean(
            [results[f'class_{i}']['normalized_hausdorff95'] for i in selected_classes]
        )
    }

    return results


def calculate_dice(mask1, mask2):
    """
    Calculate Dice coefficient (using medpy, fast)
    Returns -1 if both masks have no foreground (all background)
    """
    # Exclude background 255
    mask1 = np.where(mask1 == 255, 0, mask1)
    mask2 = np.where(mask2 == 255, 0, mask2)

    mask1_bool = mask1 > 0
    mask2_bool = mask2 > 0

    if not mask1_bool.any() and not mask2_bool.any():
        return -1.0
    if not mask1_bool.any() or not mask2_bool.any():
        return 0.0

    try:
        val = dc(mask1_bool, mask2_bool)
        if val < 0:
            raise ValueError("Dice coefficient is negative")
            exit()
        if np.isnan(val) or val < 0:
            return 0.0
        ans = min(1.0, max(0.0, val))
        # print(ans)
        return ans
    except Exception:
        return 0.0


def calculate_iou(mask1, mask2):
    """
    Calculate IoU (Jaccard index)
    Returns -1 if both masks have no foreground (all background)
    """
    dice = calculate_dice(mask1, mask2)
    if dice == -1:
        return -1.0
    iou = dice / (2 - dice) if 0 <= dice < 1.0 else 1.0
    if np.isnan(iou) or iou < 0:
        iou = 0.0
    return min(1.0, max(0.0, iou))


def calculate_normalized_hausdorff95(mask1, mask2):
    """
    Calculate normalized 95% Hausdorff distance (using medpy, fast)
    Returns -1 if both masks have no foreground (all background)
    """
    mask1 = np.where(mask1 == 255, 0, mask1)
    mask2 = np.where(mask2 == 255, 0, mask2)

    mask1_bool = mask1 > 0
    mask2_bool = mask2 > 0

    if not mask1_bool.any() and not mask2_bool.any():
        return -1.0
    if not mask1_bool.any() or not mask2_bool.any():
        return 1.0

    try:
        hd95_value = hd95(mask1_bool, mask2_bool)
        image_diagonal = np.sqrt(mask1.shape[0] ** 2 + mask1.shape[1] ** 2)
        normalized_hd95 = hd95_value / image_diagonal
        if np.isnan(normalized_hd95) or normalized_hd95 < 0:
            normalized_hd95 = 0.0
        return min(1.0, max(0.0, normalized_hd95))
    except Exception:
        return 1.0


# Example usage
if __name__ == "__main__":
    # Create example masks (1, 256, 256)
    H, W = 1080, 1920
    mask1 = np.random.randint(0, 11, (1, H, W))  # 0-9 represent classes
    mask2 = np.random.randint(0, 11, (1, H, W))

    # Set some pixels as background (255)
    mask1[:, 100:150, 100:150] = 255
    mask2[:, 120:170, 120:170] = 255

    # Calculate metrics
    results = calculate_metrics(mask1, mask2, num_classes=10)

    # Print results
    print("Evaluation Results:")
    print("=" * 60)
    for class_id in range(10):
        class_result = results[f'class_{class_id}']
        print(f"Class {class_id}:")
        print(f"  Dice: {class_result['dice']:.4f}")
        print(f"  IoU: {class_result['iou']:.4f}")
        print(f"  Normalized 95% Hausdorff Distance: {class_result['normalized_hausdorff95']:.4f}")
        print()

    print("Average Metrics:")
    print(f"  Dice: {results['mean_0_9']['dice']:.4f}")
    print(f"  IoU: {results['mean_0_9']['iou']:.4f}")
    print(f"  Normalized 95% Hausdorff Distance: {results['mean_0_9']['normalized_hausdorff95']:.4f}")