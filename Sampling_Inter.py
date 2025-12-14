import json
import os
import random
import shutil

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


def sample_paths(paths, sample_size):
    if sample_size > len(paths):
        raise ValueError(f"Sample size {sample_size} is greater than total paths {len(paths)}")
    return random.sample(paths, sample_size)


video_root = ''
path_dict = {}
for path in os.listdir(video_root):
    vid = int(path.split('_')[1].split(' ')[0])
    path_dict[vid] = os.path.join(video_root, path)

out_root = 'Inter_Test'
os.makedirs(out_root, exist_ok=True)

global_district_path = {
    '显露': [],
    '解剖': [],
    '结扎': [],
    '离断': []
}

for vid in data.keys():
    district = data[vid]
    if vid not in path_dict:
        continue

    for procedure, distr in district.items():
        video_path = path_dict[vid]
        if not os.path.exists(video_path):
            continue

        frames_group = os.listdir(video_path)
        for frames in frames_group:
            if not os.path.isdir(os.path.join(video_path, frames)):
                continue

            for frame in os.listdir(os.path.join(video_path, frames)):
                if '.jpg' not in frame:
                    continue

                frame_id = int(frame.split('.')[0].split('_')[-1])
                for time_range in distr:
                    start, end = time_range
                    if start <= frame_id <= end:
                        json_path = os.path.join(video_path, frames, frame)[:-3] + 'json'
                        if os.path.exists(json_path):
                            global_district_path[procedure].append(os.path.join(video_path, frames, frame))
                        break

total_samples = 100
sample_ratios = {
    '显露': 0.20,
    '解剖': 0.60,
    '结扎': 0.10,
    '离断': 0.10
}

for procedure in global_district_path.keys():
    procedure_dir = os.path.join(out_root, procedure)
    os.makedirs(procedure_dir, exist_ok=True)

for procedure, distr in global_district_path.items():
    procedure_dir = os.path.join(out_root, procedure)

    sample_size = int(total_samples * sample_ratios[procedure])

    if len(distr) < sample_size:
        actual_sample_size = len(distr)
    else:
        actual_sample_size = sample_size

    sampled_paths = sample_paths(distr, actual_sample_size)

    for sampled_path in sampled_paths:
        shutil.copy(sampled_path, procedure_dir)

        json_path = sampled_path[:-3] + 'json'
        if os.path.exists(json_path):
            shutil.copy(json_path, procedure_dir)