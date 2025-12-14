import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import re

# 类别配置
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

# 构建 label -> id 映射
class_names = list(colors.keys())
label2id = {name: idx for idx, name in enumerate(class_names)}
num_classes = len(class_names)


def normalize_label(label: str) -> str:
    """规范化标签名，所有以 'stem of' 开头的归为 'Stem of LCA + SA' """
    label_low = label.lower()

    if label_low == "ureter":
        return None
    if label_low.startswith("stem of"):
        return "Stem of LCA + SA"
    # 大小写不敏感匹配
    for cname in class_names:
        if label_low == cname.lower():
            return cname
    return None  # 未知类别


def check_dataset_integrity(dataset_root):
    """检查Train/Val/Test 三个子集的完整性"""
    for split in ['Train', 'Validation', 'Test']:
        image_dir = os.path.join(dataset_root, split, 'images')
        mask_dir = os.path.join(dataset_root, split, 'masks')

        if not os.path.exists(image_dir):
            print(f"[错误] {split} 图像文件夹不存在: {image_dir}")
            continue
        if not os.path.exists(mask_dir):
            print(f"[错误] {split} 掩码文件夹不存在: {mask_dir}")
            continue

        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.npy')])

        image_basenames = set([os.path.splitext(f)[0] for f in image_files])
        mask_basenames = set([os.path.splitext(f)[0] for f in mask_files])

        total_images = len(image_files)
        total_masks = len(mask_files)
        matching_files = image_basenames & mask_basenames
        missing_images = mask_basenames - image_basenames
        missing_masks = image_basenames - mask_basenames

        print(f"\n【{split}】")
        print(f"图像数量: {total_images}")
        print(f"掩码数量: {total_masks}")
        print(f"完全匹配的图像-掩码对数: {len(matching_files)}")

        if missing_images:
            print(f"[缺失图像] {len(missing_images)} 个掩码缺少对应图像:")
            print(sorted(missing_images))

        if missing_masks:
            print(f"[缺失掩码] {len(missing_masks)} 个图像缺少对应掩码:")
            print(sorted(missing_masks))

        if not missing_images and not missing_masks:
            print("✔ 图像与掩码一一对应，完整无缺。")


def imread_unicode(path):
    if not os.path.exists(path):
        print(f"[错误] 文件不存在: {path}")
        return None
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        print(f"[错误] 图像解码失败: {path}")
    return image


def convertPolygonToSemanticMask(jsonfilePath):
    """
    把多边形标注转换为语义分割 one-hot mask
    返回形状: (num_classes, H, W)
    """
    with open(jsonfilePath, "r", encoding='utf-8') as jsonf:
        jsonData = json.load(jsonf)
        img_h = jsonData["imageHeight"]
        img_w = jsonData["imageWidth"]
        mask = np.zeros((num_classes, img_h, img_w), np.uint8)

        for obj in jsonData["shapes"]:
            polygonPoints = np.array(obj["points"], np.int32)
            raw_label = obj.get("label", "").strip()
            norm_label = normalize_label(raw_label)

            if norm_label is None:
                continue

            cls_id = label2id[norm_label]
            cv2.drawContours(mask[cls_id], [polygonPoints], -1, (1), -1)

    return mask


def build_dataset(clips_root, output_root, Train_list, Val_list, Test_list, id_dict):
    """支持 Train / Validation / Test 三划分"""
    for split in ["Train", "Validation", "Test"]:
        os.makedirs(os.path.join(output_root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_root, split, "masks"), exist_ok=True)

    stats = {"Train": 0, "Validation": 0, "Test": 0}
    video_map = {}

    all_dirs = os.listdir(clips_root)

    for parent_dir in tqdm(all_dirs, desc="Processing parent directories"):
        parent_path = os.path.join(clips_root, parent_dir)
        if not os.path.isdir(parent_path):
            continue

        match = re.search(r'(\d+)', parent_dir)
        if not match:
            continue

        video_id = int(match.group(1))
        video_id_str = f"{id_dict[video_id]:03d}"

        if video_id in Train_list:
            split = "Train"
        elif video_id in Val_list:
            split = "Validation"
        elif video_id in Test_list:
            split = "Test"
        else:
            print(f"[警告] 视频ID {video_id} 不在Train/Validation/Test列表中，跳过。")
            continue

        subdirs = os.listdir(parent_path)
        video_mask_num = 0

        for subdir in subdirs:
            sub_path = os.path.join(parent_path, subdir)
            if not os.path.isdir(sub_path):
                continue

            files = os.listdir(sub_path)
            images = sorted([f for f in files if f.endswith(".jpg")])
            jsons = set([f.replace(".json", "") for f in files if f.endswith(".json")])

            video_mask_num += len(jsons)


            for img_name in images:
                frame_id = img_name.split(".")[0].split('_')[-1]
                basename = img_name.replace(".jpg", "")
                json_path = os.path.join(sub_path, basename + ".json")
                img_path = os.path.join(sub_path, img_name)

                if basename not in jsons:
                    continue

                mask = convertPolygonToSemanticMask(json_path)
                image = imread_unicode(img_path)

                save_name = f"{video_id_str}_{frame_id}"
                save_img_path = os.path.join(output_root, split, "images", save_name + ".png")
                save_mask_path = os.path.join(output_root, split, "masks", save_name + ".npy")

                try:
                    cv2.imwrite(save_img_path, image)
                    np.save(save_mask_path, mask)
                except Exception as e:
                    print(img_path)
                    print(json_path)
                    print(e)
                    exit()

                stats[split] += 1

        video_map[video_id] = video_mask_num

    return video_map, stats 


if __name__ == "__main__":
    clips_root = "F:\\BaiduNetdiskDownload\\renji\\打标"
    output_root = "F:\\BaiduNetdiskDownload\\renji\\dataset_semantic_1018_"

    total_videos = 33  # 总视频数量
    error_list = [6, 10, 31]  # 错误视频ID列表
    Test_list = [3,4,12,13,21,25,30]
    Val_list = [1,8,9,11,17,22,28,29]  # ✅ 你可以自定义验证集视频ID
    Train_list = [2,5,7,14,15,16,18,19,20,23,24,26,27,32,33]
    # Train_list = [i for i in range(1, total_videos + 1)
    #               if i not in Test_list and i not in Val_list and i not in error_list]

    id_dict= { # 视频编号--->实际id
        1: 1, 2:2, 3:3, 4:4, 5:5,
        7:6, 8:7, 9:8,
        11:9, 12:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21, 24:22, 25:23, 26:24, 27:25, 28:26, 29:27, 30:28,
        32:29, 33:30

    }

    video_map, stats = build_dataset(clips_root, output_root, Train_list, Val_list, Test_list, id_dict)

    print("\n===== 数据集统计 =====")
    print(f"训练视频数: {len(Train_list)}")
    print(f"验证视频数: {len(Val_list)}")
    print(f"测试视频数: {len(Test_list)}")
    print(f"训练样本数: {stats['Train']}")
    print(f"验证样本数: {stats['Validation']}")
    print(f"测试样本数: {stats['Test']}")

    for video_id, mask_count in sorted(video_map.items(), key=lambda x: int(x[0])):
        print(f"Video ID {video_id} has {mask_count} masks.")

    check_dataset_integrity(output_root)
