from lib.game import Game


class YouMustBuildABoatGame(Game):

    def __init__(self, **kwargs):
        kwargs["platform"] = "steam"
        kwargs["app_id"] = "290890"

        super().__init__(**kwargs)


