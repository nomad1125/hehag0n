import subprocess
import shlex
import time

import skimage.io


class FrameGrabber:

    def __init__(self, width=640, height=480, x_offset=0, y_offset=0, frame_rate=10, max_frames=99999999, max_retries=10, retry_delay=0.05):
        self.width = width
        self.height = height

        self.x_offset = x_offset
        self.y_offset = y_offset

        self.frame_rate = frame_rate

        self.max_frames = max_frames
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.process = None

    def start(self, on_frame_callback):
        ffmpeg_command = shlex.split(
            f"ffmpeg -loglevel quiet -draw_mouse 0 -video_size {self.width}x{self.height} -framerate {self.frame_rate} -f x11grab "
            f"-i :0.0+{self.x_offset},{self.y_offset} -r {self.frame_rate} frame_%015d.png"
        )

        self.process = subprocess.Popen(ffmpeg_command)

        try:
            current_frame = 1
            current_retry = 0

            while current_frame <= self.max_frames:
                try:
                    file_name = f"frame_{str(current_frame).zfill(15)}.png"

                    frame = skimage.io.imread(file_name)
                    on_frame_callback(frame)
                except FileNotFoundError as e:
                    if current_retry < self.max_retries:
                        time.sleep(self.retry_delay)
                        current_retry += 1
                        continue

                current_frame += 1
                current_retry = 0
        finally:
            subprocess.call("rm -f frame_*", shell=True)
            self.process.kill()