"""Single interface to the running Clash Royale match: perception + actions.

Usage:
    from game_wrapper import GameWrapper
    game = GameWrapper()
    state = game.get_state()           # dict from perception.extract_state
    game.act(slot=0, col=8, row=4)     # play a card at a tile
    game.close()
"""
__all__ = ["GameWrapper", "DEFAULT_WEIGHTS"]


# Lazy-load wrapper so `python -m game_wrapper.perception` (and other submodule
# entrypoints) don't trigger a circular import via wrapper -> perception.
def __getattr__(name):
    if name in __all__:
        from . import wrapper
        return getattr(wrapper, name)
    raise AttributeError(f"module 'game_wrapper' has no attribute {name!r}")
