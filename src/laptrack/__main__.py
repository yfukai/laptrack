"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """LapTrack."""


if __name__ == "__main__":
    main(prog_name="laptrack")  # pragma: no cover
