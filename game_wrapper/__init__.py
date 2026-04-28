"""Single interface to the running Clash Royale match: perception + actions.

Usage:
    from game_wrapper import GameWrapper
    game = GameWrapper()
    state = game.get_state()           # dict from perception.extract_state
    game.act(slot=0, col=8, row=4)     # play a card at a tile
    game.close()
"""
from .wrapper import GameWrapper, DEFAULT_WEIGHTS

__all__ = ["GameWrapper", "DEFAULT_WEIGHTS"]
