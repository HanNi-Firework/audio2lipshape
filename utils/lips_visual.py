import os
import cv2
from utils.video import VideoWriter
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip


def read_lip_class():
    root_path = "E:\\DH\\lips_class_40"
    imgs = list()
    for i in range(27):
        img_path = os.path.join(root_path, f"class_{str(i)}.png")
        img = cv2.imread(img_path)
        imgs.append(img)
    h, w, _ = imgs[0].shape
    return imgs, h, w


def write_video(audio_file, class_idx, output_file, class_frames, h, w):
    frames = list()
    for idx in class_idx:
        frames.append(class_frames[int(idx)-1])

    with VideoWriter(filename=output_file + ".mp4", fps=25, height=h, width=w) as vw:
        for frame in frames:
            vw.write_frame(np.asarray(frame))

    video_clip = VideoFileClip(output_file + ".mp4")
    audio_clip = AudioFileClip(audio_file)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_file + "_with_audio.mp4", codec='libx264', audio_codec='aac')

#
# imgs, h, w = read_lip_class()
#
# write_video(audio_file, class_file, "./c", imgs, h, w)
