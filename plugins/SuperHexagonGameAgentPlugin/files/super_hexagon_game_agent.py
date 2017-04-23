from lib.game_agent import GameAgent

import lib.ocr

import offshoot

import time
import random

from .helpers.frame_processing import *


class SuperHexagonGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        plugin_path = offshoot.config["file_paths"]["plugins"]

        ocr_classifier_path = f"{plugin_path}/SuperHexagonGameAgentPlugin/files/ml_models/super_hexagon_ocr.model"
        self.machine_learning_models["ocr_classifier"] = self.load_machine_learning_model(ocr_classifier_path)

        context_classifier_path = f"{plugin_path}/SuperHexagonGameAgentPlugin/files/ml_models/super_hexagon_context.model"
        self.machine_learning_models["context_classifier"] = self.load_machine_learning_model(context_classifier_path)

        self.frame_handlers["PLAY"] = self.handle_play

    @property
    def game_contexts(self):
        return dict(
            s="Splash Screen",
            l="Level Select Screen",
            g="Game Screen",
            d="Death Screen"
        )

    def handle_play(self, frame):
        processed_context_frame = process_frame_for_context(frame)

        context_prediction = self.machine_learning_models["context_classifier"].predict([processed_context_frame])[0]
        context = self.game_contexts.get(context_prediction, "Unknown")

        print(context)

        if context == "Splash Screen":
            splash_action = " ".join(lib.ocr.words_in_image_region(
                frame,
                self.game.screen_regions["SPLASH_ACTIONS"],
                self.machine_learning_models["ocr_classifier"],
                word_window_size=(1, 8)
            ))

            if "start game" in splash_action:
                self.input_controller.tap_key(self.input_controller.keyboard.enter_key)
            else:
                self.input_controller.tap_key(self.input_controller.keyboard.right_key)

            time.sleep(5 / 60)
        elif context == "Level Select Screen":
            self.input_controller.tap_key(self.input_controller.keyboard.right_key, duration=random.uniform(0.0, 1.0))
            self.input_controller.tap_key(self.input_controller.keyboard.enter_key)
            time.sleep(10 / 60)
        elif context == "Game Screen":
            time.sleep(5 / 60)
        elif context == "Death Screen":
            self.input_controller.tap_key(self.input_controller.keyboard.escape_key)
            time.sleep(10 / 60)



