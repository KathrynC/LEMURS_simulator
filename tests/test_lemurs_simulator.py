"""Tests for LEMURSSimulator including the shared output schema adapter.

Tests the to_standard_output() method that produces the zimmerman-toolkit
SimulatorOutput format for cross-simulator analysis and comparison.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lemurs_simulator import LEMURSSimulator


class TestStandardOutput:
    """Test shared output schema adapter."""

    @pytest.fixture(scope="class")
    def std_output(self):
        """Compute standard output once for all tests."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "zimmerman-toolkit"))
        sim = LEMURSSimulator()
        return sim.to_standard_output({})

    def test_validates_against_schema(self, std_output):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "zimmerman-toolkit"))
        from zimmerman.output_schema import validate_output
        errors = validate_output(std_output)
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_schema_version(self, std_output):
        assert std_output["schema_version"] == "1.0"

    def test_simulator_metadata(self, std_output):
        assert std_output["simulator"]["name"] == "lemurs"
        assert std_output["simulator"]["state_dim"] == 14
        assert std_output["simulator"]["param_dim"] == 12
        assert std_output["simulator"]["time_unit"] == "weeks"
        assert std_output["simulator"]["time_horizon"] == 15.0
        assert len(std_output["simulator"]["state_names"]) == 14

    def test_trajectory_shape(self, std_output):
        assert std_output["trajectory"]["n_steps"] == 106
        assert len(std_output["trajectory"]["times"]) == 106
        assert len(std_output["trajectory"]["states"]) == 106
        assert len(std_output["trajectory"]["states"][0]) == 14

    def test_analytics_present(self, std_output):
        assert "sleep_quality" in std_output["analytics"]["pillars"]
        assert "stress_anxiety" in std_output["analytics"]["pillars"]
        assert "physiological" in std_output["analytics"]["pillars"]
        assert "intervention_response" in std_output["analytics"]["pillars"]
        assert len(std_output["analytics"]["flat"]) > 0

    def test_parameters_present(self, std_output):
        assert len(std_output["parameters"]["bounds"]) == 12
