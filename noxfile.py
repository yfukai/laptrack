"""Nox sessions."""
import shutil
import sys
from pathlib import Path
from textwrap import dedent

import nox

try:
    from nox_poetry import Session
    from nox_poetry import session
except ImportError:
    message = f"""\
    Nox failed to import the 'nox-poetry' package.

    Please install it using the following command:

    {sys.executable} -m pip install nox-poetry"""
    raise SystemExit(dedent(message))


package = "laptrack"
python_versions = ["3.13", "3.12", "3.11", "3.10"]
safety_ignore = [
    44717,
    44715,
    44716,
    51457,
    70612,
    71628,
]  # ignore numpy 1.21 CVEs and py 1.11.0
nox.needs_version = ">= 2021.6.6"
nox.options.sessions = (
    "pre-commit",
    "safety",
    "mypy",
    "tests",
    "typeguard",
    "docs-build",
    "docs",
)
# nox.options.reuse_existing_virtualenvs = True

doc_build_packages = [
    "sphinx",
    "sphinx-autobuild",
    "sphinx-click",
    "sphinx-rtd-theme",
    "autodoc_pydantic",
    "sphinx-gallery",
    "nbsphinx",
    "matplotlib",
    "ipykernel",
]


@session(name="pre-commit", python=python_versions[-1])
def precommit(session: Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    session.install("pre-commit")
    session.run("pre-commit", *args)


@session(python=python_versions[-1])
def safety(session: Session) -> None:
    """Scan dependencies for insecure packages."""
    session.install("safety==2.3.5", "poetry-plugin-export")
    requirements = session.poetry.export_requirements()
    session.run(
        "safety",
        "check",
        "--full-report",
        f"--file={requirements}",
        *[f"--ignore={ignore}" for ignore in safety_ignore],
    )


@session(python=python_versions)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or ["src", "tests", "docs/conf.py"]
    session.install(".")
    session.install("mypy", "pytest")
    session.run("mypy", *args)
    if not session.posargs:
        session.run("mypy", f"--python-executable={sys.executable}", "noxfile.py")


@session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    session.install(".")
    session.install("coverage[toml]", "pytest", "pytest-datadir", "pygments")
    try:
        session.install("ray")
    except nox.command.CommandFailed:
        session.warn("Ray not installed, skipping ray tests")

    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@session
def coverage(session: Session) -> None:
    """Produce the coverage report."""
    args = session.posargs or ["report"]

    session.install("coverage[toml]")

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)


@session(python=python_versions)
def typeguard(session: Session) -> None:
    """Runtime type checking using Typeguard."""
    session.install(".")
    session.install("pytest", "pytest-datadir", "typeguard", "pygments", "ray")
    session.run("pytest", f"--typeguard-packages={package}", *session.posargs)


@session(name="docs-build", python=python_versions[0])
def docs_build(session: Session) -> None:
    """Build the documentation."""
    args = session.posargs or ["docs", "docs/_build"]
    session.install(".", *doc_build_packages)

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)


@session(python=python_versions[0])
def docs(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session.install(".", *doc_build_packages)

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
