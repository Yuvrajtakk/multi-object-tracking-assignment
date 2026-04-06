import unittest
from pathlib import Path

from mot_pipeline import DEFAULT_STABLE_TRACK_THRESHOLD, PipelineConfig, TrackingAnalytics, tracker_config_text


class TrackingAnalyticsTests(unittest.TestCase):
    def test_record_frame_tracks_unique_ids_and_lengths(self) -> None:
        analytics = TrackingAnalytics(min_track_len=3)

        analytics.record_frame(frame_idx=0, track_ids=[1, 2], detection_count=2)
        analytics.record_frame(frame_idx=1, track_ids=[1], detection_count=1)
        analytics.record_frame(frame_idx=2, track_ids=[1, 3], detection_count=2)

        self.assertEqual(analytics.total_unique_ids, 3)
        self.assertEqual(sorted(analytics.track_lengths()), [1, 1, 3])
        self.assertEqual(analytics.cumulative_unique_ids, [(0, 2), (1, 2), (2, 3)])
        self.assertEqual(analytics.stable_track_count(), 1)
        self.assertEqual(analytics.noise_track_count(), 2)
        self.assertEqual(analytics.first_active_count(), 2)
        self.assertEqual(analytics.last_active_count(), 2)

    def test_stable_track_count_supports_custom_threshold(self) -> None:
        analytics = TrackingAnalytics(min_track_len=2)
        analytics.record_frame(frame_idx=0, track_ids=[11], detection_count=1)
        analytics.record_frame(frame_idx=1, track_ids=[11, 12], detection_count=2)
        analytics.record_frame(frame_idx=2, track_ids=[11], detection_count=1)

        self.assertEqual(analytics.stable_track_count(), 1)
        self.assertEqual(analytics.stable_track_count(3), 1)
        self.assertEqual(analytics.stable_track_count(4), 0)


class TrackerConfigTextTests(unittest.TestCase):
    def test_botsort_config_contains_motion_compensation_fields(self) -> None:
        config = PipelineConfig(
            video_path=Path("input.mp4"),
            model_path=Path("yolov8m.pt"),
            output_path=Path("output.mp4"),
            analysis_path=Path("track_analysis.png"),
            tracker="botsort",
            conf_thresh=0.25,
            classes=(0,),
            track_thresh=0.7,
            match_thresh=0.8,
            buffer_size=90,
            min_track_len=3,
            stable_track_threshold=DEFAULT_STABLE_TRACK_THRESHOLD,
            max_frames=None,
            log_every=100,
            gmc_method="sparseOptFlow",
            with_reid=False,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
        )

        text = tracker_config_text(config)
        self.assertIn("tracker_type: botsort", text)
        self.assertIn("track_buffer: 90", text)
        self.assertIn("gmc_method: sparseOptFlow", text)
        self.assertIn("with_reid: false", text)

    def test_bytetrack_config_omits_botsort_only_fields(self) -> None:
        config = PipelineConfig(
            video_path=Path("input.mp4"),
            model_path=Path("yolov8m.pt"),
            output_path=Path("output.mp4"),
            analysis_path=Path("track_analysis.png"),
            tracker="bytetrack",
            conf_thresh=0.25,
            classes=(0,),
            track_thresh=0.5,
            match_thresh=0.8,
            buffer_size=30,
            min_track_len=3,
            stable_track_threshold=DEFAULT_STABLE_TRACK_THRESHOLD,
            max_frames=None,
            log_every=100,
            gmc_method="sparseOptFlow",
            with_reid=False,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
        )

        text = tracker_config_text(config)
        self.assertIn("tracker_type: bytetrack", text)
        self.assertNotIn("gmc_method", text)
        self.assertNotIn("with_reid", text)


if __name__ == "__main__":
    unittest.main()
