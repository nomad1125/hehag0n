from lib.game_agent import GameAgent

import lib.ocr
import lib.trigonometry

import offshoot

import time
import json
import subprocess

from pprint import pprint

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

        self.frame_shape = (self.game.window_geometry["height"], self.game.window_geometry["width"])

        self.frame_angles_to_center = lib.trigonometry.angles_to_center(self.frame_shape)
        self.frame_distances_to_center = lib.trigonometry.distances_to_center(self.frame_shape)

        self.key_direction_mapping = {
            "+": self.input_controller.keyboard.left_key,
            "-": self.input_controller.keyboard.right_key
        }

        self.game_state = {
            "scores": {},
            "max_score": 0,
            "keypress_duration": 0.001,
            "max_keypress_duration": 0.04,
            "collision_threshold": 100,
            "max_run": 10,
            "total_runs": 1
        }

    @property
    def game_contexts(self):
        return dict(
            s="Splash Screen",
            l="Level Select Screen",
            g="Game Screen",
            d="Death Screen"
        )

    def handle_play(self, frame):
        gray_frame = grayscale_frame(frame)

        processed_context_frame = process_frame_for_context(gray_frame)

        context_prediction = self.machine_learning_models["context_classifier"].predict([processed_context_frame])[0]
        context = self.game_contexts.get(context_prediction, "Unknown")

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
            #self.input_controller.tap_key(self.input_controller.keyboard.right_key, duration=random.uniform(0.0, 1.0))
            self.input_controller.tap_key(self.input_controller.keyboard.enter_key)
            time.sleep(10 / 60)
        elif context == "Game Screen":
            # Preprocess
            processed_frame_for_game_play = process_frame_for_game_play(gray_frame)

            if processed_frame_for_game_play is None:
                return None

            # Detect Player Character
            player_bounding_box = get_player_character_bounding_box(processed_frame_for_game_play, self.game.screen_regions["GAME_PLAYER_AREA"])

            if player_bounding_box:
                player_bounding_box_center = (
                    (player_bounding_box[0] + player_bounding_box[2]) // 2,
                    (player_bounding_box[1] + player_bounding_box[3]) // 2,
                )

                player_to_center_angle = self.frame_angles_to_center[player_bounding_box_center]
                player_to_center_distance = self.frame_distances_to_center[player_bounding_box_center]

                # Mask out center & player
                processed_frame_for_game_play[self.frame_distances_to_center < (player_to_center_distance + (player_bounding_box[3] - player_bounding_box[1]))] = 0

                rays = {
                    "Ray Player + 180": (player_to_center_angle + 180 + 179) % 360 - 179,
                    "Ray Player + 150": (player_to_center_angle + 150 + 179) % 360 - 179,
                    "Ray Player + 120": (player_to_center_angle + 120 + 179) % 360 - 179,
                    "Ray Player + 90": (player_to_center_angle + 90 + 179) % 360 - 179,
                    "Ray Player + 60": (player_to_center_angle + 60 + 179) % 360 - 179,
                    "Ray Player + 30": (player_to_center_angle + 30 + 179) % 360 - 179,
                    "Ray Player": player_to_center_angle,
                    "Ray Player - 30": (player_to_center_angle - 30 + 179) % 360 - 179,
                    "Ray Player - 60": (player_to_center_angle - 60 + 179) % 360 - 179,
                    "Ray Player - 90": (player_to_center_angle - 90 + 179) % 360 - 179,
                    "Ray Player - 120": (player_to_center_angle - 120 + 179) % 360 - 179,
                    "Ray Player - 150": (player_to_center_angle - 150 + 179) % 360 - 179
                }

                ray_collision_distances = {
                    "Ray Player + 180": None,
                    "Ray Player + 150": None,
                    "Ray Player + 120": None,
                    "Ray Player + 90": None,
                    "Ray Player + 60": None,
                    "Ray Player + 30": None,
                    "Ray Player": None,
                    "Ray Player - 30": None,
                    "Ray Player - 60": None,
                    "Ray Player - 90": None,
                    "Ray Player - 120": None,
                    "Ray Player - 150": None
                }

                for label, angle in rays.items():
                    ray_collision_mask = ((self.frame_angles_to_center == angle) & (processed_frame_for_game_play == 1))
                    collision_distances = self.frame_distances_to_center[ray_collision_mask == True]

                    ray_collision_distances[label] = np.min(collision_distances) if collision_distances.size else 9999

                if ray_collision_distances["Ray Player"] <= 250:
                    best_ray = max(ray_collision_distances.items(), key=lambda i: i[1])[0]

                    if best_ray == "Ray Player":
                        return None

                    direction, magnitude = best_ray.split(" ")[2:]

                    self.input_controller.tap_key(
                        self.key_direction_mapping[direction],
                        duration=(int(magnitude) / 15) * self.game_state["keypress_duration"]
                    )
        elif context == "Death Screen":
            death_time_last = " ".join(lib.ocr.words_in_image_region(
                frame,
                self.game.screen_regions["DEATH_TIME_LAST"],
                self.machine_learning_models["ocr_classifier"],
                word_window_size=(1, 8)
            ))

            try:
                score = float(death_time_last.replace(":", ".").replace("o", "0").replace("b", "8"))
            except ValueError:
                score = None

            if "%.4f" % self.game_state["keypress_duration"] not in self.game_state["scores"]:
                self.game_state["scores"]["%.4f" % self.game_state["keypress_duration"]] = list()

            if score is not None:
                self.game_state["scores"]["%.4f" % self.game_state["keypress_duration"]].append(score)

            score_averages = {duration: (np.max(scores or [0]), np.mean(scores or [0])) for duration, scores in self.game_state["scores"].items()}

            subprocess.call(["clear"])

            pprint(score_averages)

            print("")
            print("Total Runs: " + str(self.game_state["total_runs"]))
            print("Current Keypress Duration: " + str(self.game_state["keypress_duration"]))
            print("Current Collision Threshold: " + str(self.game_state["collision_threshold"]))

            with open(f"scores_{self.game_state['collision_threshold']}_hexagon.json", "w") as f:
                f.write(json.dumps(score_averages))

            if len(self.game_state["scores"]["%.4f" % self.game_state["keypress_duration"]]) >= self.game_state["max_run"]:
                self.game_state["keypress_duration"] += 0.0005

                if self.game_state["keypress_duration"] > self.game_state["max_keypress_duration"]:
                    self.game_state["keypress_duration"] = 0.001
                    self.game_state["collision_threshold"] += 20
                    self.game_state["scores"] = {}

            self.input_controller.tap_key(self.input_controller.keyboard.escape_key)
            self.game_state["total_runs"] += 1
            time.sleep(10 / 60)
