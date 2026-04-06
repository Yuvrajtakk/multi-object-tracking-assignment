from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_CLASSES = (0,)
DEFAULT_STABLE_TRACK_THRESHOLD = 200


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    total_frames: int


@dataclass(frozen=True)
class PipelineConfig:
    video_path: Path
    model_path: Path
    output_path: Path
    analysis_path: Path | None
    tracker: str
    conf_thresh: float
    classes: tuple[int, ...]
    track_thresh: float
    match_thresh: float
    buffer_size: int
    min_track_len: int
    stable_track_threshold: int
    max_frames: int | None
    log_every: int
    gmc_method: str
    with_reid: bool
    proximity_thresh: float
    appearance_thresh: float

    def frame_limit(self, total_frames: int) -> int:
        if self.max_frames is None:
            return total_frames
        return max(0, min(self.max_frames, total_frames))


@dataclass
class TrackStats:
    first_seen_frame: int
    last_seen_frame: int
    visible_frames: int = 1


@dataclass
class TrackingAnalytics:
    min_track_len: int
    track_stats: dict[int, TrackStats] = field(default_factory=dict)
    cumulative_unique_ids: list[tuple[int, int]] = field(default_factory=list)
    active_counts: list[tuple[int, int]] = field(default_factory=list)
    detection_counts: list[tuple[int, int]] = field(default_factory=list)

    def record_frame(self, frame_idx: int, track_ids: Iterable[int], detection_count: int | None = None) -> None:
        unique_ids = []
        for raw_track_id in track_ids:
            track_id = int(raw_track_id)
            unique_ids.append(track_id)
            if track_id not in self.track_stats:
                self.track_stats[track_id] = TrackStats(
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                )
                continue

            stats = self.track_stats[track_id]
            stats.last_seen_frame = frame_idx
            stats.visible_frames += 1

        self.cumulative_unique_ids.append((frame_idx, len(self.track_stats)))
        self.active_counts.append((frame_idx, len(unique_ids)))
        self.detection_counts.append(
            (frame_idx, detection_count if detection_count is not None else len(unique_ids))
        )

    @property
    def total_unique_ids(self) -> int:
        return len(self.track_stats)

    def track_lengths(self) -> list[int]:
        return [stats.visible_frames for stats in self.track_stats.values()]

    def stable_track_count(self, min_frames: int | None = None) -> int:
        threshold = self.min_track_len if min_frames is None else min_frames
        return sum(stats.visible_frames >= threshold for stats in self.track_stats.values())

    def noise_track_count(self) -> int:
        return sum(stats.visible_frames < self.min_track_len for stats in self.track_stats.values())

    def first_active_count(self) -> int:
        if not self.active_counts:
            return 0
        return self.active_counts[0][1]

    def last_active_count(self) -> int:
        if not self.active_counts:
            return 0
        return self.active_counts[-1][1]


@dataclass(frozen=True)
class RunSummary:
    frame_count: int
    elapsed_seconds: float
    total_unique_ids: int
    stable_ids: int
    long_lived_ids: int
    last_active_count: int

    @property
    def fps(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frame_count / self.elapsed_seconds


def build_parser() -> argparse.ArgumentParser:
    cwd = Path.cwd()
    parser = argparse.ArgumentParser(
        description="Production-ready multi-object tracking pipeline for the MOI project.",
    )
    parser.add_argument("--video", type=Path, default=cwd / "input.mp4", help="Input video path.")
    parser.add_argument("--model", type=Path, default=cwd / "yolov8m.pt", help="YOLO model path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=cwd / "output_compensated.mp4",
        help="Annotated output video path.",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=cwd / "track_analysis.png",
        help="Path for the generated track analysis plot.",
    )
    parser.add_argument(
        "--disable-analysis",
        action="store_true",
        help="Skip writing the analysis plot.",
    )
    parser.add_argument(
        "--tracker",
        choices=("botsort", "bytetrack"),
        default="botsort",
        help="Tracking backend to use.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument(
        "--track-thresh",
        type=float,
        default=0.7,
        help="Tracker activation threshold for new tracks.",
    )
    parser.add_argument("--match-thresh", type=float, default=0.8, help="Association matching threshold.")
    parser.add_argument("--buffer-size", type=int, default=90, help="Lost-track buffer in frames.")
    parser.add_argument(
        "--min-track-len",
        type=int,
        default=3,
        help="Minimum visible frames before a track stops counting as noise.",
    )
    parser.add_argument(
        "--stable-track-threshold",
        type=int,
        default=DEFAULT_STABLE_TRACK_THRESHOLD,
        help="Frames required for a track to count as long-lived in the analysis plot.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for smoke runs.")
    parser.add_argument("--log-every", type=int, default=100, help="Progress log frequency in frames.")
    parser.add_argument(
        "--gmc-method",
        type=str,
        default="sparseOptFlow",
        help="BoT-SORT global motion compensation method.",
    )
    parser.add_argument(
        "--with-reid",
        action="store_true",
        help="Enable BoT-SORT ReID if the local Ultralytics build supports it.",
    )
    parser.add_argument(
        "--proximity-thresh",
        type=float,
        default=0.5,
        help="BoT-SORT proximity threshold.",
    )
    parser.add_argument(
        "--appearance-thresh",
        type=float,
        default=0.25,
        help="BoT-SORT appearance threshold.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> PipelineConfig:
    args = build_parser().parse_args(argv)
    analysis_path = None if args.disable_analysis else args.analysis
    config = PipelineConfig(
        video_path=args.video.resolve(),
        model_path=args.model.resolve(),
        output_path=args.output.resolve(),
        analysis_path=analysis_path.resolve() if analysis_path else None,
        tracker=args.tracker,
        conf_thresh=args.conf,
        classes=DEFAULT_CLASSES,
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        buffer_size=args.buffer_size,
        min_track_len=args.min_track_len,
        stable_track_threshold=args.stable_track_threshold,
        max_frames=args.max_frames,
        log_every=max(1, args.log_every),
        gmc_method=args.gmc_method,
        with_reid=args.with_reid,
        proximity_thresh=args.proximity_thresh,
        appearance_thresh=args.appearance_thresh,
    )
    validate_config(config)
    return config


def validate_config(config: PipelineConfig) -> None:
    if not config.video_path.is_file():
        raise FileNotFoundError(f"Video not found: {config.video_path}")
    if not config.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {config.model_path}")
    for name, value in (
        ("conf_thresh", config.conf_thresh),
        ("track_thresh", config.track_thresh),
        ("match_thresh", config.match_thresh),
        ("proximity_thresh", config.proximity_thresh),
        ("appearance_thresh", config.appearance_thresh),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, received {value}")
    if config.buffer_size < 1:
        raise ValueError("buffer_size must be at least 1")
    if config.min_track_len < 1:
        raise ValueError("min_track_len must be at least 1")
    if config.stable_track_threshold < 1:
        raise ValueError("stable_track_threshold must be at least 1")
    if config.max_frames is not None and config.max_frames < 1:
        raise ValueError("max_frames must be at least 1 when supplied")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.analysis_path is not None:
        config.analysis_path.parent.mkdir(parents=True, exist_ok=True)


def read_video_info(video_path: Path) -> VideoInfo:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if width <= 0 or height <= 0 or total_frames <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}")
    return VideoInfo(fps=fps, width=width, height=height, total_frames=total_frames)


def print_config(config: PipelineConfig, video_info: VideoInfo) -> None:
    frame_limit = config.frame_limit(video_info.total_frames)
    print("MOT Pipeline Configuration")
    print("=" * 72)
    print(f"{'Video':<24}: {config.video_path}")
    print(f"{'Model':<24}: {config.model_path}")
    print(f"{'Tracker':<24}: {config.tracker}")
    print(f"{'Output':<24}: {config.output_path}")
    print(f"{'Analysis plot':<24}: {config.analysis_path if config.analysis_path else 'disabled'}")
    print(f"{'Video info':<24}: {video_info.width}x{video_info.height} @ {video_info.fps:.2f} FPS")
    print(f"{'Frames to process':<24}: {frame_limit}/{video_info.total_frames}")
    print(f"{'Confidence threshold':<24}: {config.conf_thresh}")
    print(f"{'Track threshold':<24}: {config.track_thresh}")
    print(f"{'Match threshold':<24}: {config.match_thresh}")
    print(f"{'Buffer size':<24}: {config.buffer_size}")
    print(f"{'Noise cutoff':<24}: {config.min_track_len} frames")
    if config.tracker == "botsort":
        print(f"{'GMC method':<24}: {config.gmc_method}")
        print(f"{'With ReID':<24}: {config.with_reid}")
    print("=" * 72)


def render_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def tracker_config_text(config: PipelineConfig) -> str:
    track_low_thresh = min(0.1, max(0.05, config.track_thresh * 0.5))
    values: list[tuple[str, object]] = [
        ("tracker_type", config.tracker),
        ("track_high_thresh", config.track_thresh),
        ("track_low_thresh", track_low_thresh),
        ("new_track_thresh", config.track_thresh),
        ("track_buffer", config.buffer_size),
        ("match_thresh", config.match_thresh),
        ("fuse_score", True),
    ]

    if config.tracker == "botsort":
        values.extend(
            [
                ("gmc_method", config.gmc_method),
                ("proximity_thresh", config.proximity_thresh),
                ("appearance_thresh", config.appearance_thresh),
                ("with_reid", config.with_reid),
                ("model", "auto"),
            ]
        )

    return "".join(f"{key}: {render_scalar(value)}\n" for key, value in values)


def write_tracker_config(config: PipelineConfig, target_dir: Path) -> Path:
    tracker_path = target_dir / f"{config.tracker}_generated.yaml"
    tracker_path.write_text(tracker_config_text(config), encoding="utf-8")
    return tracker_path


def color_for_track(track_id: int) -> tuple[int, int, int]:
    seed = (track_id * 2654435761) & 0xFFFFFFFF
    return (
        80 + (seed & 0x7F),
        80 + ((seed >> 8) & 0x7F),
        80 + ((seed >> 16) & 0x7F),
    )


def annotate_frame(frame, boxes: Sequence[Sequence[int]], track_ids: Sequence[int]) -> int:
    import cv2

    active_count = 0
    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = (int(value) for value in box)
        color = color_for_track(int(track_id))
        label = f"ID {int(track_id)}"
        active_count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1,
        )
        top = max(0, y1 - text_height - 8)
        cv2.rectangle(frame, (x1, top), (x1 + text_width + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
    return active_count


def open_video_writer(output_path: Path, video_info: VideoInfo):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        video_info.fps,
        (video_info.width, video_info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    return writer


def log_progress(
    frame_idx: int,
    frame_limit: int,
    active_count: int,
    unique_ids: int,
    start_time: float,
) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    fps = frame_idx / elapsed
    eta_seconds = max(frame_limit - frame_idx, 0) / fps if fps else 0.0
    print(
        f"Frame {frame_idx:>5}/{frame_limit:<5} | "
        f"active={active_count:<3} | "
        f"unique_ids={unique_ids:<3} | "
        f"fps={fps:>5.2f} | "
        f"eta={eta_seconds / 60:>5.2f} min"
    )


def run_botsort_pipeline(config: PipelineConfig, video_info: VideoInfo) -> tuple[TrackingAnalytics, RunSummary]:
    from ultralytics import YOLO

    analytics = TrackingAnalytics(min_track_len=config.min_track_len)
    frame_limit = config.frame_limit(video_info.total_frames)
    model = YOLO(str(config.model_path))

    with tempfile.TemporaryDirectory(prefix="moi_tracker_") as temp_dir_name:
        tracker_path = write_tracker_config(config, Path(temp_dir_name))
        writer = open_video_writer(config.output_path, video_info)
        start_time = time.perf_counter()
        frame_idx = 0
        active_count = 0

        try:
            results_stream = model.track(
                source=str(config.video_path),
                conf=config.conf_thresh,
                classes=list(config.classes),
                tracker=str(tracker_path),
                stream=True,
                verbose=False,
                persist=True,
            )

            for result in results_stream:
                if frame_idx >= frame_limit:
                    break

                frame = result.orig_img.copy()
                boxes: list[Sequence[int]] = []
                track_ids: list[int] = []
                detection_count = 0

                if result.boxes is not None:
                    detection_count = len(result.boxes)

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy().astype(int).tolist()
                    track_ids = [int(track_id) for track_id in result.boxes.id.cpu().tolist()]

                active_count = annotate_frame(frame, boxes, track_ids)
                analytics.record_frame(
                    frame_idx=frame_idx,
                    track_ids=track_ids,
                    detection_count=detection_count,
                )
                writer.write(frame)
                frame_idx += 1

                if frame_idx % config.log_every == 0 or frame_idx == frame_limit:
                    log_progress(frame_idx, frame_limit, active_count, analytics.total_unique_ids, start_time)
        finally:
            writer.release()

    elapsed = time.perf_counter() - start_time
    summary = RunSummary(
        frame_count=frame_idx,
        elapsed_seconds=elapsed,
        total_unique_ids=analytics.total_unique_ids,
        stable_ids=analytics.stable_track_count(),
        long_lived_ids=analytics.stable_track_count(config.stable_track_threshold),
        last_active_count=analytics.last_active_count(),
    )
    return analytics, summary


def run_bytetrack_pipeline(config: PipelineConfig, video_info: VideoInfo) -> tuple[TrackingAnalytics, RunSummary]:
    import cv2
    import supervision as sv
    from ultralytics import YOLO

    analytics = TrackingAnalytics(min_track_len=config.min_track_len)
    frame_limit = config.frame_limit(video_info.total_frames)
    model = YOLO(str(config.model_path))
    tracker = sv.ByteTrack(
        track_activation_threshold=config.track_thresh,
        lost_track_buffer=config.buffer_size,
        minimum_matching_threshold=config.match_thresh,
        frame_rate=max(1, int(round(video_info.fps))),
    )

    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {config.video_path}")

    writer = open_video_writer(config.output_path, video_info)
    start_time = time.perf_counter()
    frame_idx = 0
    active_count = 0

    try:
        while capture.isOpened() and frame_idx < frame_limit:
            success, frame = capture.read()
            if not success:
                break

            results = model(frame, conf=config.conf_thresh, classes=list(config.classes), verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            tracked = tracker.update_with_detections(detections)

            boxes: list[Sequence[int]] = []
            track_ids: list[int] = []
            if tracked.tracker_id is not None:
                boxes = tracked.xyxy.astype(int).tolist()
                track_ids = [int(track_id) for track_id in tracked.tracker_id.tolist()]

            active_count = annotate_frame(frame, boxes, track_ids)
            analytics.record_frame(
                frame_idx=frame_idx,
                track_ids=track_ids,
                detection_count=len(detections),
            )
            writer.write(frame)
            frame_idx += 1

            if frame_idx % config.log_every == 0 or frame_idx == frame_limit:
                log_progress(frame_idx, frame_limit, active_count, analytics.total_unique_ids, start_time)
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - start_time
    summary = RunSummary(
        frame_count=frame_idx,
        elapsed_seconds=elapsed,
        total_unique_ids=analytics.total_unique_ids,
        stable_ids=analytics.stable_track_count(),
        long_lived_ids=analytics.stable_track_count(config.stable_track_threshold),
        last_active_count=analytics.last_active_count(),
    )
    return analytics, summary


def write_analysis_plot(
    config: PipelineConfig,
    analytics: TrackingAnalytics,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    track_lengths = analytics.track_lengths()
    frame_numbers = [frame for frame, _ in analytics.cumulative_unique_ids]
    cumulative_ids = [count for _, count in analytics.cumulative_unique_ids]

    tracker_label = "BoT-SORT" if config.tracker == "botsort" else "ByteTrack"

    figure, axes = plt.subplots(1, 2, figsize=(15.5, 4.8))
    figure.suptitle(
        f"MOT Analysis — {tracker_label} run (buffer={config.buffer_size}, thresh={config.track_thresh})",
        fontsize=16,
    )

    axes[0].hist(track_lengths, bins=min(24, max(10, len(track_lengths) or 10)), color="#4f86e8")
    axes[0].axvline(config.min_track_len, linestyle="--", color="#ff9800", label=f"{config.min_track_len}f threshold")
    axes[0].axvline(
        config.stable_track_threshold,
        linestyle="--",
        color="#2eaf49",
        label=f"{config.stable_track_threshold}f (stable)",
    )
    axes[0].set_title("Track Length Distribution")
    axes[0].set_xlabel("Track length (frames)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].plot(frame_numbers, cumulative_ids, color="#4f86e8", linewidth=1.8)
    baseline = analytics.first_active_count()
    if baseline:
        axes[1].axhline(baseline, linestyle="--", color="#2eaf49", label=f"~{baseline} persons at start")
        axes[1].legend()
    axes[1].set_title("ID Growth Over Time")
    axes[1].set_xlabel("Frame number")
    axes[1].set_ylabel("Cumulative unique IDs")

    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def print_summary(config: PipelineConfig, video_info: VideoInfo, summary: RunSummary) -> None:
    processed_seconds = summary.frame_count / video_info.fps
    print("\nRun Summary")
    print("=" * 72)
    print(f"{'Frames processed':<28}: {summary.frame_count}")
    print(f"{'Video seconds processed':<28}: {processed_seconds:.1f}")
    print(f"{'Processing speed':<28}: {summary.fps:.2f} FPS")
    print(f"{'Unique IDs assigned':<28}: {summary.total_unique_ids}")
    print(f"{'Stable IDs (noise filtered)':<28}: {summary.stable_ids}")
    print(f"{'Long-lived IDs':<28}: {summary.long_lived_ids}")
    print(f"{'Active IDs on last frame':<28}: {summary.last_active_count}")
    print(f"{'Output video':<28}: {config.output_path}")
    if config.analysis_path is not None:
        print(f"{'Analysis plot':<28}: {config.analysis_path}")
    print("=" * 72)


def run_pipeline(config: PipelineConfig) -> tuple[TrackingAnalytics, RunSummary]:
    video_info = read_video_info(config.video_path)
    print_config(config, video_info)

    if config.tracker == "botsort":
        analytics, summary = run_botsort_pipeline(config, video_info)
    else:
        analytics, summary = run_bytetrack_pipeline(config, video_info)

    if config.analysis_path is not None:
        write_analysis_plot(config, analytics, config.analysis_path)

    print_summary(config, video_info, summary)
    return analytics, summary


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    run_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
