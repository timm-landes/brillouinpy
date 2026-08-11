from . import fitmodel

__all__ = ["cluster", "decompose", "unmix", "fitmodel"]


def __getattr__(name):
    # Lazily import these (pull in sklearn/pysptools) so that multiprocessing
    # fit workers, which only need 'fitmodel', don't pay for them.
    if name in ("cluster", "decompose", "unmix"):
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

