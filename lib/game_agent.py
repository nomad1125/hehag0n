import offshoot

from lib.config import config

import time
import uuid
import pickle

import skimage.io


class GameAgentError(BaseException):
    pass


class GameAgent(offshoot.Pluggable):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.game = kwargs["game"]
        self.config = config.get(f"{self.__class__.__name__}Plugin")

        self.input_controller = kwargs["input_controller"]
        self.machine_learning_models = dict()

        self.frame_handlers = dict(
            NOOP=self.handle_noop,
            COLLECT_FRAMES=self.handle_collect_frames
        )

    @offshoot.forbidden
    def on_frame(self, frame):
        frame_handler = self.frame_handlers.get(self.config.get("frame_handler", "NOOP"))

        frame_handler(frame)

    @offshoot.forbidden
    def load_machine_learning_model(self, file_path):
        with open(file_path, "rb") as f:
            serialized_classifier = f.read()

        return pickle.loads(serialized_classifier)

    def handle_noop(self, frame):
        time.sleep(1)

    def handle_collect_frames(self, frame):
        skimage.io.imsave(f"datasets/collect_frames/frame_{str(uuid.uuid4())}.png", frame)
        time.sleep(self.config.get("collect_frames_interval") or 1)

