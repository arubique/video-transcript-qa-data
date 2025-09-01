"""Main dataset generator that orchestrates the entire pipeline."""

import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import structlog
from datasets import Dataset
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from .models import (
    DatasetConfig,
    DatasetSplit,
    GeneratedDataset,
    QAPair,
    VideoMetadata,
)
from .llm_processor import LLMProcessor
from .video_processor import VideoProcessor, TranscriptExtractor

logger = structlog.get_logger(__name__)
console = Console()


class DatasetGenerator:
    """Main class for generating synthetic QA datasets from videos."""

    def __init__(self, config: DatasetConfig):
        """Initialize the dataset generator.

        Args:
            config: Configuration for dataset generation
        """
        self.config = config
        self.video_processor = VideoProcessor(
            output_dir=f"{config.output_folder}/videos",
            max_concurrent=config.max_concurrent_videos,
        )
        self.transcript_extractor = TranscriptExtractor(
            segment_duration=config.segment_duration
        )

        # Initialize LLM processors for training and validation splits
        self.training_llm_processor = LLMProcessor(
            llm_s_model=config.training_llm_config.llm_s_model,
            llm_a_model=config.training_llm_config.llm_a_model,
            api_base=config.openai_api_base,
            api_key=config.openai_api_key,
            max_concurrent=4,
        )

        # For validation, use the validation LLM configuration
        self.validation_llm_processor = LLMProcessor(
            llm_s_model=config.validation_llm_config.llm_s_model,
            llm_a_model=config.validation_llm_config.llm_a_model,
            api_base=config.openai_api_base,
            api_key=config.openai_api_key,
            max_concurrent=4,
        )

    async def generate_dataset(
        self, dataset_name: str, dataset_description: str
    ) -> GeneratedDataset:
        """Generate the complete dataset.

        Args:
            dataset_name: Name of the dataset
            dataset_description: Description of the dataset

        Returns:
            Generated dataset with training and validation splits
        """
        console.print(
            f"[bold blue]🚀 Starting dataset generation: {dataset_name}[/bold blue]"
        )

        # Step 1: Process videos
        console.print("[bold green]📹 Step 1: Processing videos...[/bold green]")
        video_metadata_list = await self._process_videos()

        if not video_metadata_list:
            raise ValueError("No videos were successfully processed")

        # Step 2: Extract transcripts
        console.print(
            "[bold green]📝 Step 2: Extracting transcripts...[/bold green]"
        )
        all_transcript_segments = await self._extract_transcripts(
            video_metadata_list
        )

        if not all_transcript_segments:
            raise ValueError("No transcript segments were extracted")

        # Step 3: Generate QA pairs for training split
        console.print(
            "[bold green]🤖 Step 3: Generating QA pairs for training split...[/bold green]"
        )
        training_qa_pairs = await self._generate_qa_pairs(
            all_transcript_segments,
            video_metadata_list,
            self.training_llm_processor,
        )

        # Step 4: Generate QA pairs for validation split
        console.print(
            "[bold green]🤖 Step 4: Generating QA pairs for validation split...[/bold green]"
        )
        validation_qa_pairs = await self._generate_qa_pairs(
            all_transcript_segments,
            video_metadata_list,
            self.validation_llm_processor,
        )

        # Step 5: Create dataset splits
        console.print(
            "[bold green]📊 Step 5: Creating dataset splits...[/bold green]"
        )
        training_split, validation_split = self._create_dataset_splits(
            training_qa_pairs, validation_qa_pairs
        )

        # Step 6: Save dataset
        console.print("[bold green]💾 Step 6: Saving dataset...[/bold green]")
        dataset = GeneratedDataset(
            name=dataset_name,
            description=dataset_description,
            training_split=training_split,
            validation_split=validation_split,
            config=self.config,
            total_videos=len(video_metadata_list),
            total_segments=len(all_transcript_segments),
            total_qa_pairs=len(training_qa_pairs) + len(validation_qa_pairs),
        )

        await self._save_dataset(dataset)

        console.print(
            f"[bold green]✅ Dataset generation completed successfully![/bold green]"
        )
        console.print(f"[bold]📊 Dataset Statistics:[/bold]")
        console.print(f"  • Total videos: {dataset.total_videos}")
        console.print(f"  • Total segments: {dataset.total_segments}")
        console.print(f"  • Training QA pairs: {len(training_split.qa_pairs)}")
        console.print(
            f"  • Validation QA pairs: {len(validation_split.qa_pairs)}"
        )
        console.print(f"  • Total QA pairs: {dataset.total_qa_pairs}")

        return dataset

    async def _process_videos(self) -> List[VideoMetadata]:
        """Process all videos in the input folder."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing videos...", total=None)

            try:
                video_metadata = (
                    await self.video_processor.process_video_folder(
                        self.config.input_folder
                    )
                )
                progress.update(task, completed=True)
                return video_metadata
            except Exception as e:
                progress.update(task, description=f"Error: {e}")
                raise

    async def _extract_transcripts(
        self, video_metadata_list: List[VideoMetadata]
    ) -> List:
        """Extract transcripts from all videos."""
        all_segments = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Extracting transcripts...", total=len(video_metadata_list)
            )

            for video_metadata in video_metadata_list:
                try:
                    segments = self.transcript_extractor.extract_transcripts(
                        video_metadata
                    )
                    all_segments.extend(segments)
                    progress.advance(task)
                except Exception as e:
                    logger.error(
                        f"Failed to extract transcripts from {video_metadata.video_id}: {e}"
                    )
                    progress.advance(task)

        return all_segments

    async def _generate_qa_pairs(
        self,
        transcript_segments: List,
        video_metadata_list: List[VideoMetadata],
        llm_processor: LLMProcessor,
    ) -> List[QAPair]:
        """Generate QA pairs from transcript segments."""
        # Create a mapping from video_id to video metadata
        video_metadata_map = {vm.video_id: vm for vm in video_metadata_list}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Generating QA pairs...", total=len(transcript_segments)
            )

            qa_pairs = await llm_processor.generate_qa_pairs(
                transcript_segments, video_metadata_map
            )

            progress.update(task, completed=True)

        return qa_pairs

    def _create_dataset_splits(
        self, training_qa_pairs: List[QAPair], validation_qa_pairs: List[QAPair]
    ) -> Tuple[DatasetSplit, DatasetSplit]:
        """Create training and validation dataset splits."""

        # Create training split
        training_split = DatasetSplit(
            name="training",
            qa_pairs=training_qa_pairs,
            llm_s_model=self.training_llm_processor.llm_s_model,
            llm_a_model=self.training_llm_processor.llm_a_model,
        )

        # Create validation split
        validation_split = DatasetSplit(
            name="validation",
            qa_pairs=validation_qa_pairs,
            llm_s_model=self.validation_llm_processor.llm_s_model,
            llm_a_model=self.validation_llm_processor.llm_a_model,
        )

        return training_split, validation_split

    async def _save_dataset(self, dataset: GeneratedDataset):
        """Save the dataset to disk in multiple formats."""
        output_dir = Path(self.config.output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = output_dir / f"{dataset.name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dataset.model_dump(), f, indent=2, default=str)

        # Save as HuggingFace datasets format
        await self._save_huggingface_format(dataset, output_dir)

        # Save individual splits as JSON
        training_path = output_dir / f"{dataset.name}_training.json"
        validation_path = output_dir / f"{dataset.name}_validation.json"

        with open(training_path, "w", encoding="utf-8") as f:
            json.dump(
                dataset.training_split.model_dump(), f, indent=2, default=str
            )

        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(
                dataset.validation_split.model_dump(), f, indent=2, default=str
            )

        console.print(
            f"[bold green]💾 Dataset saved to: {output_dir}[/bold green]"
        )

    async def _save_huggingface_format(
        self, dataset: GeneratedDataset, output_dir: Path
    ):
        """Save dataset in HuggingFace datasets format."""
        try:
            # Convert training split to HuggingFace format
            training_data = []
            for qa_pair in dataset.training_split.qa_pairs:
                training_data.append(
                    {
                        "question": qa_pair.question,
                        "answer": qa_pair.answer,
                        "transcript": qa_pair.source_document.transcript,
                        "video_ref": qa_pair.source_document.video_ref,
                        "timecode": qa_pair.source_document.timecode,
                        "video_id": qa_pair.source_document.video_id,
                        "start_time": qa_pair.source_document.start_time,
                        "end_time": qa_pair.source_document.end_time,
                        "filtered_out": qa_pair.filtered_out,
                        "llm_s_model": qa_pair.llm_s_model,
                        "llm_a_model": qa_pair.llm_a_model,
                    }
                )

            # Convert validation split to HuggingFace format
            validation_data = []
            for qa_pair in dataset.validation_split.qa_pairs:
                validation_data.append(
                    {
                        "question": qa_pair.question,
                        "answer": qa_pair.answer,
                        "transcript": qa_pair.source_document.transcript,
                        "video_ref": qa_pair.source_document.video_ref,
                        "timecode": qa_pair.source_document.timecode,
                        "video_id": qa_pair.source_document.video_id,
                        "start_time": qa_pair.source_document.start_time,
                        "end_time": qa_pair.source_document.end_time,
                        "filtered_out": qa_pair.filtered_out,
                        "llm_s_model": qa_pair.llm_s_model,
                        "llm_a_model": qa_pair.llm_a_model,
                    }
                )

            # Create HuggingFace datasets
            training_dataset = Dataset.from_list(training_data)
            validation_dataset = Dataset.from_list(validation_data)

            # Save to disk
            hf_output_dir = output_dir / "huggingface"
            hf_output_dir.mkdir(exist_ok=True)

            training_dataset.save_to_disk(hf_output_dir / "train")
            validation_dataset.save_to_disk(hf_output_dir / "validation")

            # Save dataset info
            dataset_info = {
                "name": dataset.name,
                "description": dataset.description,
                "total_videos": dataset.total_videos,
                "total_segments": dataset.total_segments,
                "total_qa_pairs": dataset.total_qa_pairs,
                "training_qa_pairs": len(training_data),
                "validation_qa_pairs": len(validation_data),
                "created_at": dataset.created_at.isoformat(),
                "config": dataset.config.model_dump(),
            }

            with open(
                hf_output_dir / "dataset_info.json", "w", encoding="utf-8"
            ) as f:
                json.dump(dataset_info, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save HuggingFace format: {e}")
            console.print(
                f"[bold red]⚠️ Warning: Failed to save HuggingFace format: {e}[/bold red]"
            )

    def get_dataset_statistics(self, dataset: GeneratedDataset) -> dict:
        """Get comprehensive statistics about the generated dataset."""
        training_pairs = dataset.training_split.qa_pairs
        validation_pairs = dataset.validation_split.qa_pairs

        # Calculate filtering statistics
        training_filtered = sum(1 for p in training_pairs if p.filtered_out)
        validation_filtered = sum(1 for p in validation_pairs if p.filtered_out)

        stats = {
            "dataset_name": dataset.name,
            "total_videos": dataset.total_videos,
            "total_segments": dataset.total_segments,
            "total_qa_pairs": dataset.total_qa_pairs,
            "training_split": {
                "qa_pairs": len(training_pairs),
                "filtered_out": training_filtered,
                "filtered_ratio": training_filtered / len(training_pairs)
                if training_pairs
                else 0,
                "llm_s_model": dataset.training_split.llm_s_model,
                "llm_a_model": dataset.training_split.llm_a_model,
            },
            "validation_split": {
                "qa_pairs": len(validation_pairs),
                "filtered_out": validation_filtered,
                "filtered_ratio": validation_filtered / len(validation_pairs)
                if validation_pairs
                else 0,
                "llm_s_model": dataset.validation_split.llm_s_model,
                "llm_a_model": dataset.validation_split.llm_a_model,
            },
            "generation_config": dataset.config.model_dump(),
        }

        return stats
