import numpy as np
from medpy.metric.binary import dc, hd95
import torch


def convert_mask_argmax(img_t):
    """<br/>    将10x256x256的多通道mask转换为单通道语义分割mask<br/>    背景为255，类别从0-9<br/>    """
    # 获取每个位置预测的类别（0-9）<br/>    
    pred_mask = torch.argmax(img_t, dim=0)  # 输出形状: 256x256<br/>    <br/>   
    #  # 检查每个位置是否有任何类别预测（max值>0.5）<br/>   
    has_prediction = torch.max(img_t, dim=0)[0] > 0.5
    #  <br/>    
    # # 将没有预测的位置设为255（背景）<br/>    
    final_mask = torch.where(has_prediction, pred_mask, 255)
    return final_mask


def calculate_metrics(mask1, mask2, num_classes=10):
    """
    计算两个mask之间的标准化95% Hausdorff距离、Dice系数和IoU

    参数:
        mask1: numpy数组, 形状为 (1, H, W) 或 (H, W)
        mask2: numpy数组, 形状为 (1, H, W) 或 (H, W)
        num_classes: 类别数量(不包括背景)

    返回:
        dict: 包含各类别及平均指标结果的字典
    """

    # 确保mask是numpy数组并去除单维度
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()

    mask1 = np.squeeze(mask1)
    mask2 = np.squeeze(mask2)

    assert mask1.shape == mask2.shape, "Masks must have the same shape"

    results = {}

    # 计算每个类别的指标
    for class_id in range(num_classes):
        # 提取当前类别的二值mask
        binary_mask1 = (mask1 == class_id).astype(np.uint8)
        binary_mask2 = (mask2 == class_id).astype(np.uint8)

        # 计算Dice系数
        dice = calculate_dice(binary_mask1, binary_mask2)

        # 计算IoU
        iou = calculate_iou(binary_mask1, binary_mask2)

        # 计算标准化95% Hausdorff距离
        nhd95 = calculate_normalized_hausdorff95(binary_mask1, binary_mask2)

        results[f'class_{class_id}'] = {
            'dice': dice,
            'iou': iou,
            'normalized_hausdorff95': nhd95
        }

    # ---------- 计算平均值部分 ----------
    def valid_mean(values):
        """计算排除无效值(-1, NaN, 负数)后的平均值"""
        valid = [v for v in values if (v is not None and not np.isnan(v) and v >= 0)]
        if len(valid) == 0:
            return -1.0  # 所有类别都无效
        return float(np.mean(valid))  # 保证返回float

    # 计算所有10类 (0~9) 的平均
    results['mean_0_9'] = {
        'dice': valid_mean([results[f'class_{i}']['dice'] for i in range(num_classes)]),
        'iou': valid_mean([results[f'class_{i}']['iou'] for i in range(num_classes)]),
        'normalized_hausdorff95': valid_mean(
            [results[f'class_{i}']['normalized_hausdorff95'] for i in range(num_classes)]
        )
    }

    # 计算类别 1~8 的平均
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
    计算Dice系数（使用medpy，高速）
    若两个mask都无前景（全为背景），返回-1
    """
    # 排除背景255
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
    计算IoU（Jaccard指数）
    若两个mask都无前景（全为背景），返回-1
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
    计算标准化95% Hausdorff距离（使用medpy，快速）
    若两个mask都无前景（全为背景），返回-1
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


# 使用示例
if __name__ == "__main__":
    # 创建示例mask (1, 256, 256)
    H, W = 1080, 1920
    mask1 = np.random.randint(0, 11, (1, H, W))  # 0-9表示类别
    mask2 = np.random.randint(0, 11, (1, H, W))

    # 将部分像素设置为背景(255)
    mask1[:, 100:150, 100:150] = 255
    mask2[:, 120:170, 120:170] = 255

    # 计算指标
    results = calculate_metrics(mask1, mask2, num_classes=10)

    # 打印结果
    print("评估结果:")
    print("=" * 60)
    for class_id in range(10):
        class_result = results[f'class_{class_id}']
        print(f"类别 {class_id}:")
        print(f"  Dice: {class_result['dice']:.4f}")
        print(f"  IoU: {class_result['iou']:.4f}")
        print(f"  标准化95% Hausdorff距离: {class_result['normalized_hausdorff95']:.4f}")
        print()

    print("平均值:")
    print(f"  Dice: {results['mean_0_9']['dice']:.4f}")
    print(f"  IoU: {results['mean_0_9']['iou']:.4f}")
    print(f"  标准化95% Hausdorff距离: {results['mean_0_9']['normalized_hausdorff95']:.4f}")