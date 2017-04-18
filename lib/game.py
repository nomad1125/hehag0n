import warnings
warnings.simplefilter("ignore")

import offshoot

import subprocess
import shlex
import time
import uuid
import re

import lib.ocr

import skimage.io

from lib.game_launchers.steam_game_launcher import SteamGameLauncher

from lib.frame_grabber import FrameGrabber


class Game(offshoot.Pluggable):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.platform = kwargs.get("platform")

        self.window_id = None
        self.window_name = kwargs.get("window_name")
        self.frame_grabbing_rate = kwargs.get("frame_rate")

        self.ocr_classifier_file_path = kwargs.get("ocr_classifier_file_path")
        self.ocr_classifier = lib.ocr.load_ocr_classifier(self.ocr_classifier_file_path) if self.ocr_classifier_file_path is not None else None

        self.kwargs = kwargs

    @property
    @offshoot.forbidden
    def game_launcher(self):
        return self.game_launchers.get(self.platform)

    @property
    @offshoot.forbidden
    def game_launchers(self):
        return {
            "steam": SteamGameLauncher
        }

    @property
    @offshoot.expected
    def screen_regions(self):
        raise NotImplementedError()

    @offshoot.forbidden
    def launch(self):
        self.before_launch()
        self.game_launcher().launch(**self.kwargs)
        self.after_launch()

    def before_launch(self):
        pass

    def after_launch(self):
        time.sleep(3)
        self.window_id = subprocess.check_output(shlex.split(f"xdotool search --name \"{self.window_name}\"")).decode("utf-8").strip()

        subprocess.call(shlex.split(f"xdotool windowmove {self.window_id} 0 0"))
        subprocess.call(shlex.split(f"xdotool windowactivate {self.window_id}"))

    @offshoot.forbidden
    def grab_frames(self, max_frames=500, on_frame=None):
        window_geometry = subprocess.check_output(shlex.split(f"xdotool getwindowgeometry {self.window_id}")).decode("utf-8").strip()
        size = re.match(r"\s+Geometry: ([0-9]+x[0-9]+)", window_geometry.split("\n")[2]).group(1).split("x")

        window_information = subprocess.check_output(shlex.split(f"xwininfo -id {self.window_id}")).decode("utf-8").strip()

        offset_x = re.match(r"\s+Absolute upper-left X:\s+([0-9]+)", window_information.split("\n")[2]).group(1)
        offset_y = re.match(r"\s+Absolute upper-left Y:\s+([0-9]+)", window_information.split("\n")[3]).group(1)

        frame_grabber = FrameGrabber(
            width=int(size[0]),
            height=int(size[1]),
            x_offset=int(offset_x),
            y_offset=int(offset_y),
            frame_rate=self.frame_grabbing_rate,
            max_frames=max_frames
        )

        frame_grabber.start(on_frame)

    @offshoot.expected
    def on_frame(self, frame):
        raise NotImplementedError()

    def on_frame_for_ocr(self, frame):
        frame_uuid = str(uuid.uuid4())

        skimage.io.imsave(f"datasets/ocr/frames/frame_{frame_uuid}.png", frame)
        lib.ocr.prepare_dataset_tokens(frame, frame_uuid)
