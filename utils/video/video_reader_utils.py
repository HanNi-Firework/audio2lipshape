from pathlib import Path
from typing import Dict, Optional, Iterator, Union

import av
import numpy as np


def get_video_frames(
    video_path: Union[str, Path],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """
    Read video frames using av
        this is a generator
    """
    with av.open(str(video_path), mode="r") as container:
        fps = container.streams.video[0].base_rate
        for frame in container.decode(container.streams.video[0]):
            frame_array = frame.to_ndarray(format="rgb24", width=width, height=height)
            yield frame_array


def get_video_frames_by_fps(
    video_path: Union[str, Path],
    target_fps: float,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """
    Read video frames at a specific FPS using av. This is a generator.

    Args:
    video_path (str): Path to the video file.
    target_fps (float): Desired frame rate to read the video.

    Yields:
    Iterator[np.ndarray]: Iterator of video frames as numpy arrays.
    """
    with av.open(str(video_path), mode="r") as container:
        stream = container.streams.video[0]

        # time interval for each frame (in seconds)
        target_interval = 1.0 / target_fps
        # Current time in the video and the next target time
        current_time = 0.0
        next_target_time = 0.0

        for frame in container.decode(stream):
            # Convert the frame's pts (presentation timestamp) to seconds
            frame_time = frame.pts * frame.time_base

            # Check if the current frame is close to the next target time
            if frame_time >= next_target_time:
                frame_array = frame.to_ndarray(format="rgb24", width=width, height=height)
                yield frame_array

                next_target_time += target_interval
            current_time = frame_time


def get_video_meta(
    path: Union[str, Path],
    flatten_fps: bool = False,
) -> Dict[str, int]:
    """
    Get video meta data
    path: str or Path
    """
    with av.open(str(path), mode="r") as container:
        fps_frac = container.streams.video[0].base_rate
        total_frames = container.streams.video[0].frames
        # if flatten fps will be shown as int
        if flatten_fps:
            fps = fps_frac.numerator // fps_frac.denominator
        else:
            fps = fps_frac
        data = dict(
            fps=fps,
            total_frames=total_frames,
            height=container.streams.video[0].height,
            width=container.streams.video[0].width,
        )
    return data


def get_frames_from_container(container) -> Iterator[np.ndarray]:
    """
    Read video frames using av
        this version of function is based on container input
        this is a generator
    """
    for frame in container.decode(container.streams.video[0]):
        yield np.array(frame.to_image())
