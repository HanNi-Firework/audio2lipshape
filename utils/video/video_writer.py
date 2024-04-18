from pathlib import Path
from typing import Optional, Union

import cv2
from PIL import Image
import numpy as np
from subprocess import check_output


class VideoWriter:
    """
    A handy video writer
    """

    def __init__(
        self,
        filename: Union[str, Path],
        fps: int,
        height: int,
        width: int,
        fourcc: str = "mp4v",
    ) -> None:
        self.filename = str(filename)
        self.fps = int(fps)
        self.height = int(height)
        self.width = int(width)
        self.fourcc = fourcc
        self.writer = None

    def __enter__(self):
        self.writer = cv2.VideoWriter(
            str(self.filename),
            cv2.VideoWriter_fourcc(*self.fourcc),
            self.fps,
            (self.width, self.height),
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.release()

    def write_frame(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def write_pil_image(self, image: Image.Image) -> None:
        frame = np.array(image)
        self.write_frame(frame)

    def add_audio(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]],
        audio_rate: Optional[int] = None,
    ) -> None:
        """
        Add audio to the video
        """
        if output_path is None:
            output_path = self.filename

        if audio_rate is None:
            audio_rate_settings = ""
        else:
            audio_rate_settings = f"-ar {audio_rate}"

        command = (
            f"ffmpeg -y -i {self.filename} -i {audio_path} -c:a aac {audio_rate_settings} {output_path} -hide_banner"
        )

        check_output(command, shell=True)
