from pathlib import Path
from typing import Any, Dict, Optional, Iterator, Union

import av
from av import VideoFrame
import pandas as pd
import numpy as np
import torch

# from utils.audio import file_to_mel, wav_from_file
from .video_reader_utils import get_video_meta


class ReadVideoFramesFlexible:
    """
    Improved flexible version of the ReadVideoFrames

    If you are using the any of the iter function, you need to use the `with` statement
    to handle the closing of the container

    Example:
    ```python
    with ReadVideoFramesFlexible(video_path) as reader:
        for frame in reader.iter_array():
            print(frame.shape)
    ```


    Args:
    - video_path: the path to the video file
    - height: the **NEW** height of the frame, if you don't want to change the height, set it to None
    - width: the **NEW** width of the frame, if you don't want to change the width, set it to None
    - fps: the **NEW** fps of the frame, if you don't want to change the fps, set it to None

    If you don't want to use the iter but the `__call__` method, you don't need to use the `with` statement
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: Optional[float] = None,
    ):
        self.video_path = Path(video_path)
        self.fps: Optional[float] = fps
        self.height: Optional[int] = height
        self.width: Optional[int] = width

        # the version of meta data from the video it self
        video_meta = get_video_meta(self.video_path, flatten_fps=False)
        self.video_height = video_meta["height"]
        self.video_width = video_meta["width"]
        self.total_frames = video_meta["total_frames"]
        fps_frac = video_meta["fps"]
        self.video_fps = fps_frac
        self.video_fps_flatten: float = fps_frac.numerator / fps_frac.denominator

    def __enter__(self):
        self.container = av.open(str(self.video_path), mode="r")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.container.close()

    def __len__(self):
        return self.total_frames

    def __repr__(self):
        if self.fps is None:
            fps_statement = f"fps={self.video_fps}"
        else:
            fps_statement = f"fps=[{self.video_fps}=>{self.fps}]"

        if self.height is None:
            height_statement = f"height={self.video_height}"
        else:
            height_statement = f"height=[{self.video_height}=>{self.height}]"

        if self.width is None:
            width_statement = f"width={self.video_width}"
        else:
            width_statement = f"width=[{self.video_width}=>{self.width}]"
        return (
            f"ReadVideoFramesFlexible(\n\tvideo_path={self.video_path}, "
            f"\n\t{fps_statement}, \n\t{height_statement}, \n\t{width_statement})"
        )

    def _iter_frame(self) -> Iterator[Dict[str, Any]]:
        target_fps = self.fps
        if target_fps is None:
            # frame in default fps
            for frame in self.container.decode(video=0):
                yield {
                    "frame": frame,
                }
        else:
            # frame in target fps
            target_interval = 1 / target_fps
            next_target_frame_time = 0.0
            next_target_frame_idx = 0
            for frame in self.container.decode(video=0):
                frame_time = frame.pts / frame.time_base.denominator

                # scenario1: we want to lower the fps
                scenario1 = (self.video_fps_flatten > target_fps) and (frame_time >= next_target_frame_time)
                # scenario2: we want to increase the fps
                scenario2 = self.video_fps_flatten <= target_fps

                if scenario1 or scenario2:
                    yield {
                        "frame": frame,
                        "frame_time": frame_time,
                        "frame_idx": frame.index,
                        "target_frame_time": next_target_frame_time,
                        "target_frame_idx": next_target_frame_idx,
                    }
                    next_target_frame_time += target_interval
                    next_target_frame_idx += 1
                if scenario2:
                    while frame_time >= next_target_frame_time:
                        yield {
                            "frame": frame,
                            "frame_time": frame_time,
                            "frame_idx": frame.index,
                            "target_frame_time": next_target_frame_time,
                            "target_frame_idx": next_target_frame_idx,
                        }
                        next_target_frame_time += target_interval
                        next_target_frame_idx += 1

    def iter_frame(self) -> Iterator[VideoFrame]:
        for frame_dict in self._iter_frame():
            yield frame_dict["frame"]

    def _iter_array(self) -> Iterator[Dict[str, Any]]:
        """
        The underlying function to read the video frames as array

        This will output extra information about the frame and the target frame
        """
        kwargs = dict()
        if self.height is not None:
            kwargs["height"] = self.height
        if self.width is not None:
            kwargs["width"] = self.width
        for frame_data in self._iter_frame():
            frame = frame_data["frame"]
            frame_array = frame.to_ndarray(format="rgb24", **kwargs)

            frame_data["frame_array"] = frame_array
            yield frame_data

    def iter_array(self) -> Iterator[np.ndarray]:
        for frame_dict in self._iter_array():
            yield frame_dict["frame_array"]

    def _iter_pil(self) -> Iterator[Dict[str, Any]]:
        for frame_data in self._iter_frame():
            frame = frame_data["frame"]
            frame_pil = frame.to_image()
            frame_data["frame_pil"] = frame_pil
            yield frame_data

    def iter_pil(self) -> Iterator:
        for frame_dict in self._iter_pil():
            yield frame_dict["frame_pil"]

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        We choose use self.iter_array as the default iterator
        you can do
        ```python
        with ReadVideoFramesFlexible(video_path) as reader:
            for frame in reader:
                print(frame.shape)
        ```

        Please remember to use the `with` statement, it will close the video file handler
        """
        return self.iter_array()

    def timeline_debug(self) -> pd.DataFrame:
        """
        Return a dataframe of the timeline detail
        on debug mode
        """
        frame_timeline = []
        if self.fps is None:
            raise ValueError("fps is not set, hence we not changing the fps, no need to debug")
        with self:
            for frame_data in self._iter_frame():
                row_data = dict()
                row_data["frame_time"] = frame_data["frame_time"]
                row_data["target_frame_time"] = frame_data.get("target_frame_time")
                row_data["frame_idx"] = frame_data["frame_idx"]
                row_data["target_frame_idx"] = frame_data.get("target_frame_idx")
                frame_timeline.append(row_data)
        timeline_df = pd.DataFrame(frame_timeline)
        timeline_df["frame_time_diff"] = timeline_df["target_frame_time"] - timeline_df["frame_time"]
        return timeline_df

    def __call__(self, limit: Optional[int] = None) -> np.ndarray:
        """
        Read the video frames in 1 go

        You can place a limit to the number of frames to read

        If limit is None, it will read all the frames

        This function will handle the closing of the container with the `with` statement
        """
        with self:
            if limit is None:
                return np.array(list(self.iter_array()))
            else:
                array_list = []
                for _ in range(limit):
                    array_list.append(next(self.iter_array()))
                return np.array(array_list)


class VideoBatchReader:
    def __init__(
        self,
        video_path: Union[str, Path],
        batch_size: Optional[int] = None,
    ) -> None:
        self.batch_size = batch_size
        self.video_path = str(video_path)

    def iter_frame(self, container) -> Iterator[np.ndarray]:
        for frame in container.decode(container.streams.video[0]):
            yield np.array(frame.to_image())

    def __iter__(self) -> Iterator[np.ndarray]:
        with av.open(str(self.video_path), mode="r") as container:
            self.fps = container.streams.video[0].base_rate.numerator
            self.total_frames = container.streams.video[0].frames
            batch_size: int = self.fps if self.batch_size is None else self.batch_size

            a_batch = []

            for frame in self.iter_frame(container):
                a_batch.append(frame)
                if len(a_batch) >= batch_size:
                    return_array = np.stack(a_batch, axis=0)
                    a_batch = []
                    yield return_array
            if len(a_batch) > 0:
                yield np.stack(a_batch, axis=0)


class VideoReader:
    """
    A universal video reader that reads video frames

    For example:
    for frame in VideoReader("video.mp4"):
        print(frame.shape)

    Or being more specific:
    for frame in VideoReader("video.mp4", width=160, height=160, fps=12):
        print(frame.shape)
    """

    def __init__(
        self,
        path: Union[str, Path],
        width: Optional[int] = None,
        height: Optional[int] = None,
        separate_audio: bool = False,
        fps: int = 25,
        load_meta: bool = False,
    ):
        """
        audio: str, path to the audio file, default to None, which is using the audio from the video
        width: int, width of the output frame, default to None, which is keep the original width
        height: int, height of the output frame, default to None, which is keep the original height
        fps: int, frame rate of the output video, default to 25, this is a set value instead of the original fps
        """
        self.path = Path(path)
        self.separate_audio = separate_audio
        self.fps: int = fps
        if load_meta:

            # extract meta data from video
            video_meta = get_video_meta(self.path, flatten_fps=False)

            self.video_height = video_meta["height"]
            self.video_width = video_meta["width"]
            self.total_frames = video_meta["total_frames"]
            fps_frac = video_meta["fps"]
            self.video_fps = fps_frac
            self.video_fps_flatten: float = fps_frac.numerator / fps_frac.denominator

        self.width = self.video_width if width is None else width
        self.height = self.video_height if height is None else height

        # If we have to change any of the dimension, we set the do_resize flag to True
        if load_meta and (self.width != self.video_width or self.height != self.video_height):
            # We have to set load_meta to use resize
            self.do_resize: bool = True
        else:
            self.do_resize: bool = False

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Spit out frame by frame in designated fps and size
        """
        kwargs = dict()
        if self.do_resize:
            kwargs["width"] = self.width
            kwargs["height"] = self.height
        with ReadVideoFramesFlexible(self.path, fps=self.fps, **kwargs) as reader:
            for frame in reader.iter_array():
                yield frame

    def __repr__(self) -> str:
        return f"VideoReader({self.path.name}, wh: {self.width}x{self.height}, fps: {self.fps})"

    def __len__(self) -> int:
        """
        Return the number of frames in the video
        """
        return self.total_frames

    # def get_wav(self, sample_rate: Optional[int] = None) -> np.ndarray:
    #     """
    #     Get wav from the video
    #     """
    #     if self.separate_audio:
    #         wav, _ = wav_from_file(file_path=self.audio_path, sample_rate=sample_rate)
    #     else:
    #         wav, _ = wav_from_file(file_path=self.path, sample_rate=sample_rate)
    #     return wav

    # def get_mel(
    #     self,
    #     sample_rate: Optional[int] = None,
    #     seconds: Optional[float] = None,
    # ) -> np.ndarray:
    #     """
    #     Get mel spectrogram from the video
    #     """
    #     if self.separate_audio:
    #         path = self.audio_path
    #     else:
    #         path = self.path
    #     return file_to_mel(path, sample_rate=sample_rate, duration=seconds)

    @property
    def audio_path(self) -> Path:
        raise NotImplementedError("This is a base class, please use a subclass where this is specified")

    @property
    def frames(self) -> np.ndarray:
        """
        Return all frames in the video
        """
        return np.stack(list(self))

    def limited_frames(self, n_frames: int) -> np.ndarray:
        """
        Don't read all frames, just read n_frames
        """
        return ReadVideoFramesFlexible(
            self.path,
            height=self.height,
            width=self.width,
            fps=self.fps,
        )(limit=n_frames)

    def frames_tensor(self, num_frames: int) -> torch.Tensor:
        return torch.Tensor((self.limited_frames(num_frames).transpose(3, 0, 1, 2) / 255 - 0.5) / 0.5)[None, ...]


ReadVideoFrames = ReadVideoFramesFlexible
