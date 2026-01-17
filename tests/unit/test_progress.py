"""Unit tests for progress tracking utilities."""

import time
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from src.depth_surge_3d.utils.progress import (
    ProgressReporter,
    ConsoleProgressReporter,
    ProgressTracker,
    ProgressCallback,
    create_progress_tracker,
    calculate_eta,
    format_time_duration,
    PROCESSING_PHASES,
)


class TestProcessingPhases:
    """Test PROCESSING_PHASES constant."""

    def test_phases_structure(self):
        """Test that PROCESSING_PHASES has correct structure."""
        assert isinstance(PROCESSING_PHASES, dict)
        assert len(PROCESSING_PHASES) > 0

        for phase_name, phase_info in PROCESSING_PHASES.items():
            assert "weight" in phase_info
            assert "description" in phase_info
            assert isinstance(phase_info["weight"], int)
            assert isinstance(phase_info["description"], str)

    def test_total_weight_is_100(self):
        """Test that phase weights sum to 100."""
        total_weight = sum(phase["weight"] for phase in PROCESSING_PHASES.values())
        assert total_weight == 100


class TestConsoleProgressReporter:
    """Test ConsoleProgressReporter class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        reporter = ConsoleProgressReporter()

        assert reporter.show_eta is True
        assert reporter.start_time > 0
        assert reporter.last_update_time == 0
        assert reporter.current_pbar is None
        assert reporter.current_phase is None

    def test_init_without_eta(self):
        """Test initialization without ETA."""
        reporter = ConsoleProgressReporter(show_eta=False)
        assert reporter.show_eta is False

    def test_init_without_tqdm(self):
        """Test initialization without tqdm."""
        reporter = ConsoleProgressReporter(use_tqdm=False)
        assert reporter.use_tqdm is False

    def test_get_phase_description_known_phase(self):
        """Test getting description for known phase."""
        reporter = ConsoleProgressReporter()
        desc = reporter._get_phase_description("depth_estimation")
        assert desc == "Estimating depth"

    def test_get_phase_description_unknown_phase(self):
        """Test getting description for unknown phase."""
        reporter = ConsoleProgressReporter()
        desc = reporter._get_phase_description("custom_phase")
        # Should title-case and replace underscores
        assert desc == "Custom Phase"

    @patch("sys.stdout", new_callable=StringIO)
    def test_report_progress_without_tqdm(self, mock_stdout):
        """Test reporting progress without tqdm."""
        reporter = ConsoleProgressReporter(use_tqdm=False)

        # First update (should print)
        reporter.report_progress(50, 100, "Processing")

        # Immediate second update (should be throttled)
        reporter.report_progress(51, 100, "Processing")

        output = mock_stdout.getvalue()
        assert "Processing" in output
        assert "50.0%" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_report_progress_with_eta(self, mock_stdout):
        """Test reporting progress with ETA calculation."""
        reporter = ConsoleProgressReporter(use_tqdm=False, show_eta=True)

        # Need some time to elapse for ETA calculation
        time.sleep(0.01)
        reporter.report_progress(1, 100, "Processing")

        output = mock_stdout.getvalue()
        assert "Processing" in output
        assert "ETA" in output or "1.0%" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_report_progress_zero_total(self, mock_stdout):
        """Test reporting progress with zero total."""
        reporter = ConsoleProgressReporter(use_tqdm=False)

        # Should not crash with zero total
        reporter.report_progress(0, 0, "Processing")

        # No output expected for zero total
        output = mock_stdout.getvalue()
        assert output == "" or "Processing" not in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_report_completion(self, mock_stdout):
        """Test reporting completion."""
        reporter = ConsoleProgressReporter(use_tqdm=False)

        reporter.report_completion("Task complete")

        output = mock_stdout.getvalue()
        assert "Task complete" in output
        assert "Completed in" in output

    def test_report_completion_closes_tqdm(self):
        """Test that completion closes tqdm progress bar."""
        reporter = ConsoleProgressReporter(use_tqdm=True)

        # Mock a progress bar
        mock_pbar = MagicMock()
        reporter.current_pbar = mock_pbar

        with patch("sys.stdout", new_callable=StringIO):
            reporter.report_completion("Done")

        # Should have closed the progress bar
        mock_pbar.close.assert_called_once()
        assert reporter.current_pbar is None


class TestProgressTracker:
    """Test ProgressTracker class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        tracker = ProgressTracker(total_frames=100)

        assert tracker.total_frames == 100
        assert tracker.processing_mode == "serial"
        assert tracker.reporter is not None
        assert tracker.start_time > 0
        assert tracker.current_step_index == 0

    def test_init_with_custom_reporter(self):
        """Test initialization with custom reporter."""
        mock_reporter = Mock(spec=ProgressReporter)
        tracker = ProgressTracker(total_frames=100, reporter=mock_reporter)

        assert tracker.reporter == mock_reporter

    def test_init_with_batch_mode(self):
        """Test initialization with batch processing mode."""
        tracker = ProgressTracker(total_frames=100, processing_mode="batch")

        assert tracker.processing_mode == "batch"

    def test_update_serial(self):
        """Test updating progress in serial mode."""
        mock_reporter = Mock(spec=ProgressReporter)
        tracker = ProgressTracker(total_frames=100, reporter=mock_reporter)

        tracker.update_serial(50, "Processing frame")

        # Should update internal state
        assert tracker.current_frame == 50
        assert tracker.current_step == "Processing frame"


class TestCreateProgressTracker:
    """Test create_progress_tracker factory function."""

    def test_create_with_defaults(self):
        """Test creating tracker with defaults."""
        tracker = create_progress_tracker(total_frames=100)

        assert isinstance(tracker, ProgressTracker)
        assert tracker.total_frames == 100
        assert tracker.processing_mode == "serial"

    def test_create_with_custom_mode(self):
        """Test creating tracker with custom mode."""
        tracker = create_progress_tracker(total_frames=100, processing_mode="batch")

        assert tracker.processing_mode == "batch"

    def test_create_with_console_reporter_type(self):
        """Test creating tracker with console reporter type."""
        tracker = create_progress_tracker(total_frames=100, reporter_type="console")

        assert isinstance(tracker.reporter, ConsoleProgressReporter)

    def test_create_with_unknown_reporter_type(self):
        """Test creating tracker with unknown reporter type defaults to console."""
        tracker = create_progress_tracker(total_frames=100, reporter_type="unknown")

        # Should default to console reporter
        assert isinstance(tracker.reporter, ConsoleProgressReporter)


class TestCalculateEta:
    """Test calculate_eta function."""

    def test_calculate_eta_half_complete(self):
        """Test ETA calculation when half complete."""
        start_time = time.time() - 10.0  # Started 10 seconds ago
        eta = calculate_eta(start_time, current=50, total=100)

        # Should be approximately 10 seconds (same time remaining as elapsed)
        assert 8.0 <= eta <= 12.0

    def test_calculate_eta_quarter_complete(self):
        """Test ETA calculation when quarter complete."""
        start_time = time.time() - 5.0  # Started 5 seconds ago
        eta = calculate_eta(start_time, current=25, total=100)

        # Should be approximately 15 seconds (3x the elapsed time remaining)
        assert 13.0 <= eta <= 17.0

    def test_calculate_eta_almost_complete(self):
        """Test ETA calculation when almost complete."""
        start_time = time.time() - 10.0
        eta = calculate_eta(start_time, current=99, total=100)

        # Should be very small
        assert 0.0 <= eta <= 1.0

    def test_calculate_eta_zero_current(self):
        """Test ETA calculation with zero current progress."""
        start_time = time.time()
        eta = calculate_eta(start_time, current=0, total=100)

        # Should handle gracefully (might return 0 or infinity)
        assert isinstance(eta, (int, float))

    def test_calculate_eta_zero_total(self):
        """Test ETA calculation with zero total."""
        start_time = time.time()
        eta = calculate_eta(start_time, current=0, total=0)

        # Should handle edge case without crashing
        assert isinstance(eta, (int, float))


class TestFormatTimeDuration:
    """Test format_time_duration function."""

    def test_format_seconds_only(self):
        """Test formatting duration with seconds only."""
        formatted = format_time_duration(45.0)

        assert isinstance(formatted, str)
        assert "45" in formatted or "0:45" in formatted

    def test_format_minutes_and_seconds(self):
        """Test formatting duration with minutes and seconds."""
        formatted = format_time_duration(125.0)  # 2 minutes 5 seconds

        assert isinstance(formatted, str)
        # Should contain minute and second info
        assert "2" in formatted

    def test_format_hours(self):
        """Test formatting duration with hours."""
        formatted = format_time_duration(3665.0)  # 1 hour, 1 minute, 5 seconds

        assert isinstance(formatted, str)
        assert "1" in formatted  # Should have hour component

    def test_format_zero_duration(self):
        """Test formatting zero duration."""
        formatted = format_time_duration(0.0)

        assert isinstance(formatted, str)
        assert "0" in formatted

    def test_format_fractional_seconds(self):
        """Test formatting fractional seconds."""
        formatted = format_time_duration(45.7)

        # Should handle fractional seconds
        assert isinstance(formatted, str)
        assert len(formatted) > 0


class MockProgressReporter(ProgressReporter):
    """Mock implementation of ProgressReporter for testing."""

    def __init__(self):
        self.progress_calls = []
        self.completion_calls = []

    def report_progress(self, current: int, total: int, message: str = "") -> None:
        """Record progress report."""
        self.progress_calls.append({"current": current, "total": total, "message": message})

    def report_completion(self, message: str = "") -> None:
        """Record completion report."""
        self.completion_calls.append({"message": message})


class TestProgressReporterInterface:
    """Test ProgressReporter abstract interface."""

    def test_mock_reporter_implementation(self):
        """Test that mock reporter implements interface correctly."""
        reporter = MockProgressReporter()

        # Should be able to call methods
        reporter.report_progress(50, 100, "Test")
        reporter.report_completion("Done")

        # Should have recorded calls
        assert len(reporter.progress_calls) == 1
        assert len(reporter.completion_calls) == 1

    def test_mock_reporter_records_progress(self):
        """Test that mock reporter records progress correctly."""
        reporter = MockProgressReporter()

        reporter.report_progress(25, 100, "Quarter done")

        assert reporter.progress_calls[0]["current"] == 25
        assert reporter.progress_calls[0]["total"] == 100
        assert reporter.progress_calls[0]["message"] == "Quarter done"

    def test_mock_reporter_records_completion(self):
        """Test that mock reporter records completion correctly."""
        reporter = MockProgressReporter()

        reporter.report_completion("All finished")

        assert reporter.completion_calls[0]["message"] == "All finished"
