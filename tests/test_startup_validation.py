"""Unit tests for startup gating; no Pixhawk or model dependencies are needed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class StartupValidationTests(unittest.TestCase):
    def test_safe_default_does_not_start_a_mission(self) -> None:
        """A disabled startup must return before importing hardware dependencies."""
        with patch.object(main, "AUTO_START_MISSION", False):
            self.assertEqual(main.main(), 0)

    def test_validator_rejects_out_of_range_coordinates(self) -> None:
        """Latitude/longitude errors must stop opt-in execution before connecting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.pt"
            model.touch()
            with patch.multiple(
                main,
                LAP_WAYPOINTS=[(91.0, 20.0), (40.0, 30.0)],
                MAPPING_POLYGON=[(40.0, 30.0), (40.1, 30.0), (40.1, 30.1)],
                MAPPING_ALTITUDE_M=10.0,
                TAKEOFF_ALTITUDE_M=10.0,
                MAX_FLIGHT_ALTITUDE_M=30.0,
                TARGET_CLASS_NAMES=("target",),
                YOLO_MODEL_PATH=str(model),
            ):
                with self.assertRaisesRegex(ValueError, "valid coordinate range"):
                    main._validate_startup_configuration()

    def test_validator_accepts_well_formed_test_configuration(self) -> None:
        """A complete coordinate/model configuration should pass static checks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.pt"
            model.touch()
            with patch.multiple(
                main,
                LAP_WAYPOINTS=[(40.0, 30.0), (40.01, 30.01)],
                MAPPING_POLYGON=[(40.0, 30.0), (40.1, 30.0), (40.1, 30.1)],
                MAPPING_ALTITUDE_M=10.0,
                TAKEOFF_ALTITUDE_M=10.0,
                MAX_FLIGHT_ALTITUDE_M=30.0,
                TARGET_CLASS_NAMES=("target",),
                YOLO_MODEL_PATH=str(model),
            ):
                main._validate_startup_configuration()


if __name__ == "__main__":
    unittest.main()
