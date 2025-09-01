"""Video processing module for downloading videos and extracting transcripts."""

import asyncio
import os
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import structlog
import yt_dlp
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

# Import functions from YT-Navigator
import sys

sys.path.append("./yt-navigator")
try:
    from yt_navigator.app.services.scraping.transcript import (
        TranscriptScraper as YTTranscriptScraper,
    )
    from yt_navigator.app.services.scraping.video import (
        VideoScraper as YTVideoScraper,
    )
    from yt_navigator.app.helpers import (
        convert_seconds_to_timestamp as yt_convert_timestamp,
    )

    YT_NAVIGATOR_AVAILABLE = True
except ImportError:
    YT_NAVIGATOR_AVAILABLE = False
    print("Warning: YT-Navigator not available, using fallback implementations")

from .models import TranscriptSegment, VideoMetadata

logger = structlog.get_logger(__name__)


def convert_seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to human-readable timestamp format."""
    if YT_NAVIGATOR_AVAILABLE:
        return yt_convert_timestamp(seconds)
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class VideoProcessor:
    """Handles video downloading and transcript extraction."""

    def __init__(self, output_dir: str, max_concurrent: int = 4):
        """Initialize the video processor.

        Args:
            output_dir: Directory to save downloaded videos
            max_concurrent: Maximum number of concurrent downloads
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_video_folder(
        self, input_folder: str
    ) -> List[VideoMetadata]:
        """Process all videos in the input folder.

        Args:
            input_folder: Path to folder containing video files

        Returns:
            List of video metadata for processed videos
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            raise ValueError(f"Input folder does not exist: {input_folder}")

        video_files = self._find_video_files(input_path)
        logger.info(f"Found {len(video_files)} video files to process")

        tasks = []
        for video_file in video_files:
            task = self._process_single_video(video_file)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and collect successful results
        video_metadata = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Failed to process video: {result}")
            elif result is not None:
                video_metadata.append(result)

        logger.info(f"Successfully processed {len(video_metadata)} videos")
        return video_metadata

    def _find_video_files(self, folder_path: Path) -> List[Path]:
        """Find all video files in the given folder."""
        video_extensions = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
            ".flv",
            ".wmv",
        }
        video_files = []

        for file_path in folder_path.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in video_extensions
            ):
                video_files.append(file_path)

        return video_files

    async def _process_single_video(
        self, video_path: Path
    ) -> Optional[VideoMetadata]:
        """Process a single video file.

        Args:
            video_path: Path to the video file

        Returns:
            Video metadata if successful, None otherwise
        """
        async with self.semaphore:
            try:
                # Generate unique video ID
                video_id = str(uuid.uuid4())

                # Get video metadata
                metadata = await self._extract_video_metadata(video_path)
                if not metadata:
                    return None

                # Copy video to output directory with standardized name
                output_path = self.output_dir / f"{video_id}.mp4"
                await self._copy_video_file(video_path, output_path)

                return VideoMetadata(
                    video_id=video_id,
                    title=metadata.get("title", video_path.stem),
                    duration=metadata.get("duration", 0.0),
                    file_path=str(output_path),
                    source_url=None,
                )

            except Exception as e:
                logger.error(f"Failed to process video {video_path}: {e}")
                return None

    async def _extract_video_metadata(self, video_path: Path) -> Optional[dict]:
        """Extract metadata from a video file."""
        try:
            import ffmpeg

            probe = ffmpeg.probe(str(video_path))
            video_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "video"
                ),
                None,
            )

            if not video_stream:
                return None

            duration = float(probe["format"]["duration"])
            title = video_path.stem

            return {
                "title": title,
                "duration": duration,
                "width": int(video_stream["width"]),
                "height": int(video_stream["height"]),
            }

        except Exception as e:
            logger.error(f"Failed to extract metadata from {video_path}: {e}")
            return None

    async def _copy_video_file(self, source_path: Path, dest_path: Path):
        """Copy video file to destination."""
        import shutil

        shutil.copy2(source_path, dest_path)


class TranscriptExtractor:
    """Handles transcript extraction from videos."""

    def __init__(self, segment_duration: int = 30):
        """Initialize the transcript extractor.

        Args:
            segment_duration: Duration of transcript segments in seconds
        """
        self.segment_duration = segment_duration

    def extract_transcripts(
        self, video_metadata: VideoMetadata
    ) -> List[TranscriptSegment]:
        """Extract transcripts from a video.

        Args:
            video_metadata: Video metadata containing file path and ID

        Returns:
            List of transcript segments
        """
        try:
            # For local video files, we'll use a different approach
            # since YouTubeTranscriptApi is for YouTube videos
            # For now, we'll create placeholder segments
            # In a real implementation, you'd use speech-to-text services

            segments = self._create_consecutive_segments(video_metadata)
            logger.info(
                f"Extracted {len(segments)} transcript segments from {video_metadata.video_id}"
            )
            return segments

        except Exception as e:
            logger.error(
                f"Failed to extract transcripts from {video_metadata.video_id}: {e}"
            )
            return []

    def _create_consecutive_segments(
        self, video_metadata: VideoMetadata
    ) -> List[TranscriptSegment]:
        """Create consecutive transcript segments covering the entire video.

        Args:
            video_metadata: Video metadata containing duration and ID

        Returns:
            List of consecutive transcript segments
        """
        segments = []
        duration = video_metadata.duration

        # Create consecutive segments covering the entire video
        current_time = 0.0
        segment_index = 0

        while current_time < duration:
            end_time = min(current_time + self.segment_duration, duration)
            segment_duration = end_time - current_time

            # Create placeholder text (in real implementation, this would be actual transcript)
            placeholder_text = f"Transcript segment {segment_index + 1} from {convert_seconds_to_timestamp(current_time)} to {convert_seconds_to_timestamp(end_time)} of video {video_metadata.title}"

            segment = TranscriptSegment(
                video_id=video_metadata.video_id,
                start_time=current_time,
                end_time=end_time,
                duration=segment_duration,
                text=placeholder_text,
                timestamp=convert_seconds_to_timestamp(current_time),
            )
            segments.append(segment)

            current_time = end_time
            segment_index += 1

        return segments

    def extract_youtube_transcript(
        self, video_url: str, video_id: str
    ) -> List[TranscriptSegment]:
        """Extract transcript from a YouTube video.

        Args:
            video_url: YouTube video URL
            video_id: Unique identifier for the video

        Returns:
            List of transcript segments
        """
        try:
            # Extract YouTube video ID from URL
            youtube_id = self._extract_youtube_id(video_url)
            if not youtube_id:
                logger.error(
                    f"Could not extract YouTube ID from URL: {video_url}"
                )
                return []

            transcript = YouTubeTranscriptApi.get_transcript(youtube_id)
            return self._format_youtube_transcript(transcript, video_id)

        except NoTranscriptFound:
            logger.warning(
                f"No transcript available for YouTube video: {video_url}"
            )
            return []
        except TranscriptsDisabled:
            logger.warning(
                f"Transcripts are disabled for YouTube video: {video_url}"
            )
            return []
        except Exception as e:
            logger.error(
                f"Failed to fetch transcript for YouTube video {video_url}: {e}"
            )
            return []

    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL."""
        patterns = [
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)",
            r"youtube\.com\/watch\?.*v=([^&\n?#]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def _format_youtube_transcript(
        self, transcript: List[dict], video_id: str
    ) -> List[TranscriptSegment]:
        """Format YouTube transcript into consecutive segments."""
        if not transcript:
            return []

        segments = []
        current_time = 0.0
        segment_text = ""
        segment_start = 0.0
        segment_index = 0

        # Process each transcript item
        for item in transcript:
            item_start = item["start"]
            item_text = item["text"]

            # If this item would exceed the segment duration, create a new segment
            if item_start - segment_start >= self.segment_duration:
                # Save current segment if it has content
                if segment_text.strip():
                    segments.append(
                        TranscriptSegment(
                            video_id=video_id,
                            start_time=segment_start,
                            end_time=item_start,
                            duration=item_start - segment_start,
                            text=segment_text.strip(),
                            timestamp=convert_seconds_to_timestamp(
                                segment_start
                            ),
                        )
                    )

                # Start new segment
                segment_start = item_start
                segment_text = item_text
                segment_index += 1
            else:
                # Add to current segment
                if segment_text:
                    segment_text += " " + item_text
                else:
                    segment_text = item_text

        # Add the final segment
        if segment_text.strip():
            last_item = transcript[-1]
            end_time = last_item["start"] + last_item.get("duration", 0)
            segments.append(
                TranscriptSegment(
                    video_id=video_id,
                    start_time=segment_start,
                    end_time=end_time,
                    duration=end_time - segment_start,
                    text=segment_text.strip(),
                    timestamp=convert_seconds_to_timestamp(segment_start),
                )
            )

        return segments

    def extract_transcript_with_yt_navigator(
        self, video_metadata: VideoMetadata
    ) -> List[TranscriptSegment]:
        """Extract transcript using YT-Navigator functions if available."""
        if not YT_NAVIGATOR_AVAILABLE:
            logger.warning("YT-Navigator not available, using fallback method")
            return self.extract_transcripts(video_metadata)

        try:
            # Use YT-Navigator's transcript scraper
            yt_scraper = YTTranscriptScraper(
                max_transcript_segment_duration=self.segment_duration
            )

            # Create video metadata format expected by YT-Navigator
            video_metadata_dict = {
                "videoId": video_metadata.video_id,
                "title": video_metadata.title,
                "duration": video_metadata.duration,
            }

            # Extract transcript using YT-Navigator
            yt_segments = yt_scraper.get_video_transcript(video_metadata_dict)

            # Convert to our format
            segments = []
            for yt_segment in yt_segments:
                segment = TranscriptSegment(
                    video_id=video_metadata.video_id,
                    start_time=yt_segment["start_time"],
                    end_time=yt_segment["start_time"] + yt_segment["duration"],
                    duration=yt_segment["duration"],
                    text=yt_segment["text"],
                    timestamp=yt_segment["timestamp"],
                )
                segments.append(segment)

            return segments

        except Exception as e:
            logger.error(f"Failed to extract transcript with YT-Navigator: {e}")
            # Fallback to our method
            return self.extract_transcripts(video_metadata)
