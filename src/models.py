"""Data models for the Video Transcript QA Dataset Generation Framework."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Metadata for a video file."""

    video_id: str = Field(..., description="Unique identifier for the video")
    title: str = Field(..., description="Video title")
    duration: float = Field(..., description="Video duration in seconds")
    file_path: str = Field(..., description="Path to the video file")
    source_url: Optional[str] = Field(
        None, description="Original source URL if applicable"
    )


class TranscriptSegment(BaseModel):
    """A segment of video transcript with timing information."""

    video_id: str = Field(..., description="ID of the source video")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    duration: float = Field(
        ..., description="Duration of the segment in seconds"
    )
    text: str = Field(..., description="Transcribed text content")
    timestamp: str = Field(
        ..., description="Human-readable timestamp (e.g., '00:01:30')"
    )


class SourceDocument(BaseModel):
    """A source document containing transcript information."""

    transcript: str = Field(
        ..., description="Transcribed text of the segment fragment"
    )
    video_ref: str = Field(..., description="Reference to the source video")
    timecode: str = Field(..., description="Exact timecode from the video")
    video_id: str = Field(..., description="ID of the source video")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")


class QAPair(BaseModel):
    """A question-answer pair with source document."""

    question: str = Field(..., description="The question")
    answer: str = Field(..., description="The answer to the question")
    source_document: SourceDocument = Field(
        ..., description="Source document containing the answer"
    )
    llm_s_model: str = Field(
        ..., description="LLM-S model used to generate this pair"
    )
    llm_a_model: str = Field(..., description="LLM-A model used for validation")
    is_answerable_without_context: bool = Field(
        False,
        description="Whether this QA pair can be answered without context",
    )
    is_answerable_with_context: bool = Field(
        True,
        description="Whether LLM-A can answer the question with the provided context",
    )


class LLMConfig(BaseModel):
    """Configuration for LLM models."""

    llm_s_model: str = Field(
        ..., description="LLM-S model for generating QA pairs"
    )
    llm_a_model: str = Field(
        ..., description="LLM-A model for answering questions"
    )


class DatasetConfig(BaseModel):
    """Configuration for dataset generation."""

    input_folder: str = Field(..., description="Folder containing input videos")
    output_folder: str = Field(
        ..., description="Folder to save the generated dataset"
    )
    segment_duration: int = Field(
        30, description="Duration of transcript segments in seconds"
    )
    max_concurrent_videos: int = Field(
        4, description="Maximum number of videos to process concurrently"
    )
    training_split_ratio: float = Field(
        0.8, description="Ratio for training/validation split"
    )
    openai_api_base: Optional[str] = Field(
        None, description="OpenAI API base URL"
    )
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    training_llm_config: LLMConfig = Field(
        ..., description="LLM configuration for training split"
    )
    validation_llm_config: LLMConfig = Field(
        ..., description="LLM configuration for validation split"
    )


class DatasetSplit(BaseModel):
    """A dataset split (training or validation)."""

    name: str = Field(
        ..., description="Name of the split (training/validation)"
    )
    qa_pairs: List[QAPair] = Field(
        ..., description="List of QA pairs in this split"
    )
    llm_s_model: str = Field(..., description="LLM-S model used for this split")
    llm_a_model: str = Field(..., description="LLM-A model used for this split")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )


class GeneratedDataset(BaseModel):
    """Complete generated dataset with training and validation splits."""

    name: str = Field(..., description="Name of the dataset")
    description: str = Field(..., description="Description of the dataset")
    training_split: DatasetSplit = Field(..., description="Training split")
    validation_split: DatasetSplit = Field(..., description="Validation split")
    config: DatasetConfig = Field(
        ..., description="Configuration used for generation"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    total_videos: int = Field(
        ..., description="Total number of videos processed"
    )
    total_segments: int = Field(
        ..., description="Total number of transcript segments"
    )
    total_qa_pairs: int = Field(
        ..., description="Total number of QA pairs generated"
    )
