from Metrics import calculate_metrics, convert_mask_argmax
from pathlib import Path
from itertools import combinations
import pandas as pd
from collections import defaultdict
import torch
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import re

colors = {
    "IMV": (112, 48, 160),
    "AA": (255, 102, 224),
    "LCIA": (255, 102, 224),
    "RCIA": (255, 102, 224),
    "IMA": (229, 76, 94),
    "LCA": (238, 130, 47),
    "SA": (242, 186, 2),
    "SRA": (255, 255, 102),
    "Stem of LCA + SA": (1, 176, 80),
    "vessels": (72, 116, 203)
}

class_names = list(colors.keys())
label2id = {name: idx for idx, name in enumerate(class_names)}
num_classes = len(class_names)


def normalize_label(label: str) -> str:
    label_low = label.lower()
    for cname in class_names:
        if label_low == cname.lower():
            return cname
    return None

def convertPolygonToSemanticMask(jsonfilePath):
    with open(jsonfilePath, "r", encoding='utf-8') as jsonf:
        jsonData = json.load(jsonf)
        img_h = jsonData["imageHeight"]
        img_w = jsonData["imageWidth"]
        mask = np.zeros((10, img_h, img_w), np.uint8)

        for obj in jsonData["shapes"]:
            polygonPoints = np.array(obj["points"], np.int32)
            raw_label = obj.get("label", "").strip()
            norm_label = normalize_label(raw_label)

            if norm_label is None:
                continue

            cls_id = label2id[norm_label]
            cv2.drawContours(mask[cls_id], [polygonPoints], -1, (1), -1)

    return mask

def extract_id_from_filename(filename: str) -> int:

    filename = filename.split('/')[-1].split('/')[-1]
    match = re.match(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def extract_mask_from_name(filename: str) -> int:
    json_files = [str(p) for p in Path(filename).rglob('*.json')]
    json_files = sorted(json_files, key=lambda x: os.path.basename(x))
    return json_files



def aggregate_video_metrics_per_class(mask_pairs, video_ids, num_classes=10, save_path="video_metrics.xlsx"):

    video_sums = defaultdict(lambda: defaultdict(lambda: {'dice': 0, 'iou': 0, 'nhd95': 0, 'count': 0}))

    for (mask1, mask2), vid in tqdm(zip(mask_pairs, video_ids), total=len(mask_pairs)):
        results = calculate_metrics(mask1, mask2, num_classes=num_classes)

        for class_id in range(num_classes):
            class_metrics = results[f'class_{class_id}']
            dice = class_metrics['dice']
            iou = class_metrics['iou']
            nhd = class_metrics['normalized_hausdorff95']

            if dice == -1 or iou == -1 or nhd == -1:
                continue

            video_sums[vid][class_id]['dice'] += dice
            video_sums[vid][class_id]['iou'] += iou
            video_sums[vid][class_id]['nhd95'] += nhd
            video_sums[vid][class_id]['count'] += 1

    final_results = []

    for vid, class_dict in video_sums.items():
        row = {'video_id': vid}
        dice_list_all, iou_list_all, nhd_list_all = [], [], []
        dice_list_1_8, iou_list_1_8, nhd_list_1_8 = [], [], []

        for class_id in range(num_classes):
            stats = class_dict[class_id]
            count = stats['count']

            if count == 0:
                row[f'class_{class_id}_dice'] = -1
                row[f'class_{class_id}_iou'] = -1
                row[f'class_{class_id}_nhd95'] = -1
                continue

            dice_avg = stats['dice'] / count
            iou_avg = stats['iou'] / count
            nhd_avg = stats['nhd95'] / count

            row[f'class_{class_id}_dice'] = dice_avg
            row[f'class_{class_id}_iou'] = iou_avg
            row[f'class_{class_id}_nhd95'] = nhd_avg

            dice_list_all.append(dice_avg)
            iou_list_all.append(iou_avg)
            nhd_list_all.append(nhd_avg)

            if 1 <= class_id <= 8:
                dice_list_1_8.append(dice_avg)
                iou_list_1_8.append(iou_avg)
                nhd_list_1_8.append(nhd_avg)

        row['mean_dice'] = np.mean([v for v in dice_list_all if v != -1]) if dice_list_all else -1
        row['mean_iou'] = np.mean([v for v in iou_list_all if v != -1]) if iou_list_all else -1
        row['mean_nhd95'] = np.mean([v for v in nhd_list_all if v != -1]) if nhd_list_all else -1

        row['mean_dice_1_8'] = np.mean([v for v in dice_list_1_8 if v != -1]) if dice_list_1_8 else -1
        row['mean_iou_1_8'] = np.mean([v for v in iou_list_1_8 if v != -1]) if iou_list_1_8 else -1
        row['mean_nhd95_1_8'] = np.mean([v for v in nhd_list_1_8 if v != -1]) if nhd_list_1_8 else -1

        final_results.append(row)

    df = pd.DataFrame(final_results).sort_values(by='video_id')
    df.to_excel(save_path, index=False)
    print(f"✅ Results saved in: {save_path}")

    return df


def find_file_in_folder(folder_path, target_filename):

    matched_files = []
    
    for root, dirs, files in os.walk(folder_path):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None

def run1():
    # Inter

    file_name = ""
    name_list = [name for name in os.listdir(file_name) if os.path.isdir(os.path.join(file_name, name))]
    save_dir = file_name + "_metrics"
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    name_dict = {}

    for name in name_list:
        name_path = os.path.join(file_name, name)
        name_dict[name] = extract_mask_from_name(name_path)
    
    for k1, k2 in combinations(name_dict.keys(), 2):
        mask_pairs = []
        video_ids = []
        for i in range(len(name_dict[k1])):
            assert os.path.basename(name_dict[k1][i])==os.path.basename(name_dict[k2][i])
            mask1 = convertPolygonToSemanticMask(name_dict[k1][i])
            mask2 = convertPolygonToSemanticMask(name_dict[k2][i])

            mask1 = convert_mask_argmax(torch.from_numpy(mask1))
            mask2 = convert_mask_argmax(torch.from_numpy(mask2))
            mask_pairs.append((mask1, mask2))

            id = extract_id_from_filename(os.path.basename(name_dict[k1][i]))
            video_ids.append(id)

        aggregate_video_metrics_per_class(mask_pairs=mask_pairs, video_ids=video_ids, num_classes=10, 
                                save_path=os.path.join(save_dir, f"{k1}_vs_{k2}_metrics.xlsx"))

def run2():
    # Intra
    file_name1 = ""
    file_name2 = ""
    name_list = [name for name in os.listdir(file_name1) if os.path.isdir(os.path.join(file_name1, name))]
    save_dir = file_name1 + "_metrics"
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    name_dict1 = {}



    for name in name_list:
        name_path1 = os.path.join(file_name1, name)
        name_dict1[name] = extract_mask_from_name(name_path1)



    for name in name_dict1.keys():
        mask_pairs = []
        video_ids = []  
        print(name)

        for i in range(len(name_dict1[name])):
            mask2_path = find_file_in_folder(file_name2, os.path.basename(name_dict1[name][i]))
            mask1 = convertPolygonToSemanticMask(name_dict1[name][i])
            mask2 = convertPolygonToSemanticMask(mask2_path)

            mask1 = convert_mask_argmax(torch.from_numpy(mask1))
            mask2 = convert_mask_argmax(torch.from_numpy(mask2))
            mask_pairs.append((mask1, mask2))

            id = extract_id_from_filename(os.path.basename(name_dict1[name][i]))
            video_ids.append(id)

        aggregate_video_metrics_per_class(mask_pairs=mask_pairs, video_ids=video_ids, num_classes=10, 
                                save_path=os.path.join(save_dir, f"{name}_intra_consistency_metrics.xlsx"))

    
if __name__ == "__main__":
    run2()