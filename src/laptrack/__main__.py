"""Command-line interface."""
from tap import to_tap_class

from ._tracking import LapTrack


class LapTrackCommandLine(to_tap_class(LapTrack)):
    """Command-line interface for LapTrack."""

    def __init__(self):
        super().__init__()

    def run(self):
        """Run the command-line interface."""
        print(self)


def main():  # noqa: D103
    laptrack = LapTrackCommandLine()
    print(laptrack)


if __name__ == "__main__":
    main()  # pragma: no cover
