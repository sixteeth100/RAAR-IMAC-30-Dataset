import json
import os
import random
import shutil
from labelme.utils import shape_to_mask
import numpy as np
import csv
import cv2
import re

data = {
    1: {
        '显露': [[0, 22100]],
        '解剖': [[22600, 26600]],
        '结扎': [[26600, 28100]],
        '离断': [[28100, 40296]]
    },
    2: {
        '显露': [[0, 3300]],
        '解剖': [[4000, 4280], [20000, 20100]],
        '结扎': [[4280, 6300]],
        '离断': [[6500, 19700]]
    },
    3: {
        '显露': [[500, 11900]],
        '解剖': [[11900, 12100]],
        '结扎': [[12400, 14400]],
        '离断': [[14400, 17600]]
    },
    4: {
        '显露': [[3400, 5700]],
        '解剖': [[5900, 6200], [10000, 10100]],
        '结扎': [[6900, 8100]],
        '离断': [[8100, 8500]]
    },
    5: {
        '显露': [[15000, 26100]],
        '解剖': [[27500, 27672]],
        '结扎': [[27674, 29100]],
        '离断': [[29100, 32800]]
    },
    7: {
        '显露': [[0, 100]],
        '解剖': [[500, 600], [1000, 2200], [8500, 11300]],
        '结扎': [[3300, 4400]],
        '离断': [[4400, 4600]]
    },
    8: {
        '显露': [[3800, 4000]],
        '解剖': [[4000, 4300], [6800, 6900]],
        '结扎': [[4300, 5500]],
        '离断': [[5500, 5600]]
    },
    9: {
        '显露': [[0, 6100]],
        '解剖': [[45000, 47100]],
        '结扎': [[49500, 51900]],
        '离断': [[52100, 53300]]
    },
    11: {
        '显露': [[0, 2000]],
        '解剖': [[9300, 10000], [12700, 22600]],
        '结扎': [[10000, 11300]],
        '离断': [[11400, 11800]]
    },
    12: {
        '显露': [[2500, 3100]],
        '解剖': [[3900, 4200]],
        '结扎': [[4500, 4800]],
        '离断': [[6600, 7300]]
    },
    13: {
        '显露': [[0, 100]],
        '解剖': [[800, 4400], [9200, 9300]],
        '结扎': [[5800, 6500]],
        '离断': [[6900, 7000]]
    },
    14: {
        '显露': [[1500, 1600]],
        '解剖': [[2000, 2200], [10000, 14100]],
        '结扎': [[2600, 3400]],
        '离断': [[3900, 4100]]
    },
    15: {
        '显露': [[11500, 28600]],
        '解剖': [[28800, 29700], [32800, 33300]],
        '结扎': [[30000, 31600]],
        '离断': [[31600, 31900]]
    },
    16: {
        '显露': [[21700, 27900]],
        '解剖': [[28200, 28400]],
        '结扎': [[33000, 34400]],
        '离断': [[34400, 36400]]
    },
    17: {
        '显露': [[4800, 5000]],
        '解剖': [[5000, 5100], [8000, 8100]],
        '结扎': [[5500, 6900]],
        '离断': [[7000, 7200]]
    },
    18: {
        '显露': [[30000, 43900]],
        '解剖': [[44000, 44900]],
        '结扎': [[44900, 46100]],
        '离断': [[46100, 46500]]
    },
    19: {
        '显露': [[1700, 1800]],
        '解剖': [[1800, 1900]],
        '结扎': [[2000, 3000]],
        '离断': [[3200, 3300]]
    },
    20: {
        '显露': [[23900, 24100]],
        '解剖': [[24100, 24400]],
        '结扎': [[24400, 26000]],
        '离断': [[26000, 26200]]
    },
    21: {
        '显露': [[32100, 33200]],
        '解剖': [[33200, 33800], [35500, 36000]],
        '结扎': [[33900, 34500], [36100, 36400]],
        '离断': [[34700, 34800]]
    },
    22: {
        '显露': [[15800, 16100]],
        '解剖': [[16100, 16200], [18200, 18300]],
        '结扎': [[16500, 16800]],
        '离断': [[17500, 17600]]
    },
    23: {
        '显露': [[100, 23700]],
        '解剖': [[25000, 25500], [29000, 29100]],
        '结扎': [[26100, 27218]],
        '离断': [[27282, 27500]]
    },
    24: {
        '显露': [[700, 4900]],
        '解剖': [[4900, 5300], [10300, 10600]],
        '结扎': [[5600, 6700], [10600, 10800]],
        '离断': [[6800, 7900]]
    },
    25: {
        '显露': [[1000, 17300]],
        '解剖': [[17900, 18300]],
        '结扎': [[18800, 20300]],
        '离断': [[20900, 21200]]
    },
    26: {
        '显露': [[20100, 21932]],
        '解剖': [[21934, 22300]],
        '结扎': [[31100, 32200]],
        '离断': [[32400, 32500]]
    },
    27: {
        '显露': [[4400, 11900]],
        '解剖': [[11900, 12300], [13832, 14500]],
        '结扎': [[13200, 13500]],
        '离断': [[13700, 13822]]
    },
    28: {
        '显露': [[9900, 10078]],
        '解剖': [[10080, 10300]],
        '结扎': [[10300, 10900]],
        '离断': [[11000, 11200]]
    },
    29: {
        '显露': [[3000, 3100]],
        '解剖': [[4000, 4200]],
        '结扎': [[4500, 4600]],
        '离断': [[6800, 6900]]
    },
    30: {
        '显露': [[10900, 11000]],
        '解剖': [[11000, 11600]],
        '结扎': [[11700, 13100]],
        '离断': [[13100, 13300]]
    },
    32: {
        '显露': [[13500, 13600]],
        '解剖': [[14700, 15000]],
        '结扎': [[15100, 15300]],
        '离断': [[16700, 16900]]
    },
    33: {
        '显露': [[15000, 15100]],
        '解剖': [[29100, 29300]],
        '结扎': [[30000, 30100]],
        '离断': [[38400, 38500]]
    }
}

p_dict = {
    '显露': 'P1',
    '解剖': 'P2',
    '结扎': 'P3',
    '离断': 'P4'
}

video_stage_dict = {
    2: "T0",
    5: "T2",
    7: "T0",
    14: "T0",
    15: "T1",
    16: "T3",
    18: "T3",
    19: "T0",
    20: "T1",
    23: "T0",
    24: "T0",
    26: "T2",
    27: "T2",
    32: "T3",
    33: "T1",
    1: "T2",
    8: "T0",
    9: "T0",
    11: "T3",
    17: "T0",
    22: "T0",
    28: "T1",
    29: "T0",
    3: "T0",
    4: "T0",
    12: "T0",
    13: "T0",
    21: "T2",
    25: "T1",
    30: "T3"
}

real_id_dicr = {}

for real_id, id in enumerate(data.keys()):
    real_id_dicr[id] = str(real_id + 1)


def sample_paths(paths, sample_size):
    if sample_size > len(paths):
        raise ValueError(f"Sample size {sample_size} is greater than total paths {len(paths)}")
    return random.sample(paths, sample_size)


video_root = ''
out_root = ''
path_dict = {}

for path in os.listdir(video_root):
    vid = int(path.split('_')[1].split(' ')[0])
    path_dict[vid] = os.path.join(video_root, path)

gts = {
    "IMV": (1, 1, 1),
    "AA": (2, 2, 2),
    "LCIA": (3, 3, 3),
    "RCIA": (4, 4, 4),
    "IMA": (5, 5, 5),
    "LCA": (6, 6, 6),
    "SA": (7, 7, 7),
    "SRA": (8, 8, 8),
    "Stem of LCA + SA": (9, 9, 9),
    "vessels": (10, 10, 10)
}

colors = {
    "IMV": (27, 158, 119),
    "AA": (217, 95, 2),
    "LCIA": (217, 95, 2),
    "RCIA": (217, 95, 2),
    "IMA": (231, 41, 138),
    "LCA": (255, 166, 0),
    "SA": (102, 166, 30),
    "SRA": (230, 171, 2),
    "Stem of LCA + SA": (0, 114, 178),
    "vessels": (126, 0, 128)
}

person_dict = {
    "Person1": [1, 8, 9, 11, 17, 18, 25],
    "Person2": [2, 12, 13, 14, 19, 20, 26],
    "Person3": [3, 4, 5, 7, 15, 16, 23, 24],
    "Person4": [21, 22, 27, 28, 30],
    "Person5": [29, 32, 33]
}

for name, vids in person_dict.items():
    for vid in vids:
        if vid not in data.keys():
            continue

        out_dict = os.path.join(out_root, real_id_dicr[vid])
        os.makedirs(out_dict, exist_ok=True)

        with open(os.path.join(out_dict, real_id_dicr[vid] + '.csv'), 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['No.', 'ID', 'Type', 'FileName', 'MaskName', 'Vessel'])

            No_i = 1
            district = data[vid]
            label_list = list(colors.keys())
            video_path = path_dict[vid]
            frames_group_tmp = os.listdir(video_path)


            def get_underscore_numbers(filename):
                match = re.search(r'_(\d+)_(\d+)$', filename)
                if match:
                    return int(match.group(1)), int(match.group(2))
                return (0, 0)


            frames_group = sorted(frames_group_tmp, key=get_underscore_numbers)

            for frames in frames_group:
                def get_underscore_number(filename):
                    match = int(filename.split('.')[0].split('_')[-1])
                    return match


                frame_a_group_tmp = os.listdir(os.path.join(video_path, frames))
                frame_a_group = sorted(frame_a_group_tmp, key=get_underscore_number)

                for frame in frame_a_group:
                    if '.jpg' not in frame:
                        continue

                    exsists = {
                        "IMV": 0,
                        "AA": 0,
                        "LCIA": 0,
                        "RCIA": 0,
                        "IMA": 0,
                        "LCA": 0,
                        "SA": 0,
                        "SRA": 0,
                        "Stem of LCA + SA": 0,
                        "vessels": 0
                    }

                    frame_id = int(frame.split('.')[0].split('_')[-1])

                    for procedure, distr in district.items():
                        for len_peroid in range(len(distr)):
                            start = distr[len_peroid][0]
                            end = distr[len_peroid][1]

                            if start <= frame_id <= end:
                                print(vid, frame_id, distr[len_peroid], procedure)

                                out_dict_img = os.path.join(out_root, real_id_dicr[vid], video_stage_dict[vid],
                                                            p_dict[procedure], 'Frames')
                                out_dict_anno = os.path.join(out_root, real_id_dicr[vid], video_stage_dict[vid],
                                                             p_dict[procedure], 'Annotations')
                                out_dict_vis = os.path.join(out_root, real_id_dicr[vid], video_stage_dict[vid],
                                                            p_dict[procedure], 'vis')

                                img_name = real_id_dicr[vid] + '_' + video_stage_dict[vid] + '_' + p_dict[
                                    procedure] + '_' + str(No_i) + '.jpg'
                                mask_name = real_id_dicr[vid] + '_' + video_stage_dict[vid] + '_' + p_dict[
                                    procedure] + '_' + str(No_i) + '_mask.npy'

                                class_masks = {}

                                if os.path.exists(os.path.join(video_path, frames, frame)[:-3] + 'json'):
                                    with open(os.path.join(video_path, frames, frame)[:-3] + 'json', 'r') as f:
                                        label_data = json.load(f)

                                height, width = label_data['imageHeight'], label_data['imageWidth']

                                for shape in label_data['shapes']:
                                    if shape['shape_type'] == 'polygon':
                                        label = shape['label']

                                        if label not in label_list:
                                            print(label, os.path.join(video_path, frames, frame), name)
                                            continue

                                        exsists[label] = 1

                                        if 'all' not in class_masks:
                                            class_masks['all'] = [np.zeros((height, width), dtype=np.uint8) for _ in
                                                                  range(3)]

                                        points = shape['points']
                                        shape_mask = shape_to_mask((height, width), points, shape_type='polygon')
                                        for i in range(3):
                                            class_masks['all'][i][shape_mask] = gts[label][i]

                                if list(class_masks.keys()) == []:
                                    print(111, os.path.join(video_path, frames, frame), name)
                                    continue

                                class_masks['all'] = np.array(class_masks['all'])[0].astype(np.uint8)

                                os.makedirs(out_dict_img, exist_ok=True)
                                os.makedirs(out_dict_anno, exist_ok=True)
                                os.makedirs(out_dict_vis, exist_ok=True)

                                shutil.copy(os.path.join(video_path, frames, frame),
                                            os.path.join(out_dict_img, img_name))
                                np.save(os.path.join(out_dict_anno, mask_name), class_masks['all'])

                                all_labels = []
                                for label in exsists.keys():
                                    if exsists[label] == 1:
                                        all_labels.append(label)

                                writer.writerow([No_i, real_id_dicr[vid], video_stage_dict[vid], img_name, mask_name,
                                                 str(all_labels)])

                                for gt_label in range(1, 11):
                                    mask = (class_masks['all'] == gt_label).astype(np.uint8)
                                    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

                                    for i in range(3):
                                        mask[:, :, i] = mask[:, :, i] * colors[label_list[gt_label - 1]][i]

                                    vis_name = real_id_dicr[vid] + '_' + video_stage_dict[vid] + '_' + p_dict[
                                        procedure] + '_' + str(No_i) + '_' + label_list[gt_label - 1] + '_mask.jpg'
                                    cv2.imwrite(os.path.join(out_dict_vis, vis_name), mask)

                                No_i += 1