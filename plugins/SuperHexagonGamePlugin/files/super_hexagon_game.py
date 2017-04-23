from lib.game import Game
from lib.config import config

import lib.ocr

import offshoot
from pprint import pprint


class SuperHexagonGame(Game):

    def __init__(self, **kwargs):
        kwargs["platform"] = "steam"
        kwargs["app_id"] = "221640"

        kwargs["window_name"] = "Super Hexagon"
        kwargs["frame_rate"] = config["SuperHexagonGamePlugin"].get("frame_rate") or 10

        plugin_path = offshoot.config["file_paths"]["plugins"]
        kwargs["ocr_classifier_file_path"] = f"{plugin_path}/SuperHexagonGamePlugin/files/super_hexagon_ocr.model"

        super().__init__(**kwargs)

    @property
    def screen_regions(self):
        return dict(
            SPLASH_ACTIONS=(349, 260, 391, 507),
            GAME_HUD_TIME=(0, 562, 52, 768)
        )