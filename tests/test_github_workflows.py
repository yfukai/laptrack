"""Tests for GitHub Actions workflow configurations."""

import pytest
import yaml
from pathlib import Path


@pytest.fixture
def github_workflows_dir():
    """Return the path to the GitHub workflows directory."""
    return Path(".github/workflows")


@pytest.fixture
def tests_workflow_file(github_workflows_dir):
    """Return the path to the tests workflow file."""
    return github_workflows_dir / "tests.yml"


def test_tests_workflow_exists(tests_workflow_file):
    """Test that the tests workflow file exists."""
    assert tests_workflow_file.exists(), "tests.yml workflow file should exist"


def test_tests_workflow_valid_yaml(tests_workflow_file):
    """Test that the tests workflow file contains valid YAML."""
    with open(tests_workflow_file, 'r') as f:
        try:
            yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"tests.yml contains invalid YAML: {e}")


def test_tests_workflow_has_concurrency_config(tests_workflow_file):
    """Test that the tests workflow has concurrency configuration."""
    with open(tests_workflow_file, 'r') as f:
        workflow = yaml.safe_load(f)
    
    assert "concurrency" in workflow, "tests.yml should have concurrency configuration"
    
    concurrency = workflow["concurrency"]
    assert "group" in concurrency, "concurrency section should have group field"
    assert "cancel-in-progress" in concurrency, "concurrency section should have cancel-in-progress field"


def test_tests_workflow_concurrency_values(tests_workflow_file):
    """Test that the concurrency configuration has correct values."""
    with open(tests_workflow_file, 'r') as f:
        workflow = yaml.safe_load(f)
    
    concurrency = workflow["concurrency"]
    assert concurrency["group"] == "${{ github.workflow }}-${{ github.head_ref || github.ref }}", "concurrency group should use correct template"
    assert concurrency["cancel-in-progress"] is True, "cancel-in-progress should be true"