import os
import cv2

file_names = os.listdir('stages')
if not os.path.exists('clips'):
    os.mkdir('clips')

for filename in file_names:
    clip_file_dir = os.path.join('clips', filename[:-4])
    if not os.path.exists(clip_file_dir):
        os.mkdir(clip_file_dir)

    video_path = os.path.join('stages', filename)

    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

    current_start = -100
    next_start = 0
    clip_path = None
    while True:
        print(int(pos))
        ret, frame = cap.read()
        if not ret:
            break
        if int(pos) % 100 == 0:
            current_start = next_start
            next_start = next_start + 100
            filename_clip = filename[:-4] + '_' + str(current_start) + '_' + str(next_start)
            clip_path = os.path.join(clip_file_dir, filename_clip)
            if not os.path.exists(clip_path):
                os.mkdir(clip_path)

        frame_name = filename[:-4] + '_' + str(int(pos)) + '.jpg'
        frame_path = os.path.join(clip_path, frame_name)
        if pos % 2 == 0:
            cv2.imwrite(frame_path, frame)
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

    if int(pos) != next_start:
        filename_clip = filename[:-4] + '_' + str(current_start) + '_' + str(next_start)
        clip_path = os.path.join(clip_file_dir, filename_clip)

        filename_clip_new = filename[:-4] + '_' + str(current_start) + '_' + str(int(pos))
        clip_path_new = os.path.join(clip_file_dir, filename_clip_new)
        os.rename(clip_path, clip_path_new)

    cap.release()




