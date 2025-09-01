"""Tests for the Video Transcript QA Dataset Generation Framework."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.models import (
    VideoMetadata,
    TranscriptSegment,
    SourceDocument,
    QAPair,
    DatasetConfig,
    LLMConfig,
)
from src.video_processor import VideoProcessor, TranscriptExtractor
from src.llm_processor import LLMProcessor


class TestVideoProcessor:
    """Test the VideoProcessor class."""

    @pytest.fixture
    def video_processor(self):
        """Create a VideoProcessor instance for testing."""
        return VideoProcessor(output_dir="./test_output", max_concurrent=2)

    @pytest.fixture
    def mock_video_metadata(self):
        """Create mock video metadata."""
        return VideoMetadata(
            video_id="test_video_123",
            title="Test Video",
            duration=120.0,
            file_path="./test_video.mp4",
            source_url=None,
        )

    def test_convert_seconds_to_timestamp(self):
        """Test timestamp conversion."""
        from src.video_processor import convert_seconds_to_timestamp

        assert convert_seconds_to_timestamp(0) == "00:00:00"
        assert convert_seconds_to_timestamp(61) == "00:01:01"
        assert convert_seconds_to_timestamp(3661) == "01:01:01"

    def test_find_video_files(self, video_processor, tmp_path):
        """Test finding video files in a directory."""
        # Create test video files
        video_files = [
            tmp_path / "video1.mp4",
            tmp_path / "video2.avi",
            tmp_path / "document.txt",  # Should be ignored
            tmp_path / "subdir" / "video3.mov",
        ]

        for file_path in video_files:
            file_path.parent.mkdir(exist_ok=True)
            file_path.touch()

        found_files = video_processor._find_video_files(tmp_path)
        assert len(found_files) == 3
        assert any("video1.mp4" in str(f) for f in found_files)
        assert any("video2.avi" in str(f) for f in found_files)
        assert any("video3.mov" in str(f) for f in found_files)


class TestTranscriptExtractor:
    """Test the TranscriptExtractor class."""

    @pytest.fixture
    def transcript_extractor(self):
        """Create a TranscriptExtractor instance for testing."""
        return TranscriptExtractor(segment_duration=30)

    @pytest.fixture
    def mock_video_metadata(self):
        """Create mock video metadata."""
        return VideoMetadata(
            video_id="test_video_123",
            title="Test Video",
            duration=90.0,
            file_path="./test_video.mp4",
            source_url=None,
        )

    def test_create_consecutive_segments(
        self, transcript_extractor, mock_video_metadata
    ):
        """Test creating consecutive transcript segments."""
        segments = transcript_extractor._create_consecutive_segments(
            mock_video_metadata
        )

        assert len(segments) == 3  # 90 seconds / 30 seconds = 3 segments
        assert all(
            isinstance(segment, TranscriptSegment) for segment in segments
        )
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 30.0
        assert segments[1].start_time == 30.0
        assert segments[1].end_time == 60.0
        assert segments[2].start_time == 60.0
        assert segments[2].end_time == 90.0


class TestLLMProcessor:
    """Test the LLMProcessor class."""

    @pytest.fixture
    def llm_processor(self):
        """Create a LLMProcessor instance for testing."""
        return LLMProcessor(
            llm_s_model="placeholder-llm-s",
            llm_a_model="placeholder-llm-a",
            api_key="test_key",
        )

    @pytest.fixture
    def mock_transcript_segment(self):
        """Create a mock transcript segment."""
        return TranscriptSegment(
            video_id="test_video_123",
            start_time=0.0,
            end_time=30.0,
            duration=30.0,
            text="This is a test transcript segment about machine learning.",
            timestamp="00:00:00",
        )

    def test_is_placeholder_model(self, llm_processor):
        """Test placeholder model detection."""
        assert llm_processor._is_placeholder_model("placeholder-llm-s") == True
        assert llm_processor._is_placeholder_model("placeholder-llm-a") == True
        assert llm_processor._is_placeholder_model("gpt-4") == False
        assert llm_processor._is_placeholder_model("gpt-3.5-turbo") == False

    def test_calculate_answer_similarity(self, llm_processor):
        """Test answer similarity calculation."""
        answer1 = "Machine learning is a subset of artificial intelligence"
        answer2 = "Machine learning is a subset of AI"
        answer3 = "The weather is sunny today"

        similarity1 = llm_processor._calculate_answer_similarity(
            answer1, answer2
        )
        similarity2 = llm_processor._calculate_answer_similarity(
            answer1, answer3
        )

        assert (
            similarity1 > similarity2
        )  # Similar answers should have higher similarity
        assert 0 <= similarity1 <= 1
        assert 0 <= similarity2 <= 1

    def test_parse_qa_response(self, llm_processor):
        """Test parsing LLM QA response."""
        response = """
        Question: What is machine learning?
        Answer: Machine learning is a subset of artificial intelligence.
        """

        question, answer = llm_processor._parse_qa_response(response)

        assert question == "What is machine learning?"
        assert (
            answer == "Machine learning is a subset of artificial intelligence."
        )

    def test_parse_qa_response_missing_fields(self, llm_processor):
        """Test parsing LLM QA response with missing fields."""
        response = """
        Question: What is machine learning?
        """

        question, answer = llm_processor._parse_qa_response(response)

        assert question == "What is machine learning?"
        assert answer is None


class TestModels:
    """Test the data models."""

    def test_video_metadata_creation(self):
        """Test VideoMetadata model creation."""
        metadata = VideoMetadata(
            video_id="test_123",
            title="Test Video",
            duration=120.0,
            file_path="./test.mp4",
        )

        assert metadata.video_id == "test_123"
        assert metadata.title == "Test Video"
        assert metadata.duration == 120.0
        assert metadata.file_path == "./test.mp4"

    def test_transcript_segment_creation(self):
        """Test TranscriptSegment model creation."""
        segment = TranscriptSegment(
            video_id="test_123",
            start_time=0.0,
            end_time=30.0,
            duration=30.0,
            text="Test transcript text",
            timestamp="00:00:00",
        )

        assert segment.video_id == "test_123"
        assert segment.start_time == 0.0
        assert segment.end_time == 30.0
        assert segment.text == "Test transcript text"

    def test_qa_pair_creation(self):
        """Test QAPair model creation."""
        source_doc = SourceDocument(
            transcript="Test transcript",
            video_ref="Test Video",
            timecode="00:00:00",
            video_id="test_123",
            start_time=0.0,
            end_time=30.0,
        )

        qa_pair = QAPair(
            question="What is this about?",
            answer="It's about testing.",
            source_document=source_doc,
            llm_s_model="placeholder-llm-s",
            llm_a_model="placeholder-llm-a",
            filtered_out=False,
        )

        assert qa_pair.question == "What is this about?"
        assert qa_pair.answer == "It's about testing."
        assert not qa_pair.filtered_out

    def test_llm_config_creation(self):
        """Test LLMConfig model creation."""
        llm_config = LLMConfig(
            llm_s_model="placeholder-llm-s", llm_a_model="placeholder-llm-a"
        )

        assert llm_config.llm_s_model == "placeholder-llm-s"
        assert llm_config.llm_a_model == "placeholder-llm-a"


class TestDatasetConfig:
    """Test the DatasetConfig model."""

    def test_dataset_config_creation(self):
        """Test DatasetConfig model creation."""
        training_llm_config = LLMConfig(
            llm_s_model="placeholder-llm-s", llm_a_model="placeholder-llm-a"
        )
        validation_llm_config = LLMConfig(
            llm_s_model="placeholder-llm-s", llm_a_model="placeholder-llm-a"
        )

        config = DatasetConfig(
            input_folder="./input",
            output_folder="./output",
            segment_duration=30,
            max_concurrent_videos=4,
            training_split_ratio=0.8,
            training_llm_config=training_llm_config,
            validation_llm_config=validation_llm_config,
        )

        assert config.input_folder == "./input"
        assert config.output_folder == "./output"
        assert config.segment_duration == 30
        assert config.max_concurrent_videos == 4
        assert config.training_split_ratio == 0.8
        assert config.training_llm_config.llm_s_model == "placeholder-llm-s"
        assert config.validation_llm_config.llm_a_model == "placeholder-llm-a"

    def test_dataset_config_defaults(self):
        """Test DatasetConfig default values."""
        training_llm_config = LLMConfig(
            llm_s_model="placeholder-llm-s", llm_a_model="placeholder-llm-a"
        )
        validation_llm_config = LLMConfig(
            llm_s_model="placeholder-llm-s", llm_a_model="placeholder-llm-a"
        )

        config = DatasetConfig(
            input_folder="./input",
            output_folder="./output",
            training_llm_config=training_llm_config,
            validation_llm_config=validation_llm_config,
        )

        assert config.segment_duration == 30
        assert config.max_concurrent_videos == 4
        assert config.training_split_ratio == 0.8


if __name__ == "__main__":
    pytest.main([__file__])
