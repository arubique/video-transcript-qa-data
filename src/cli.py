"""Command-line interface for the Video Transcript QA Dataset Generation Framework."""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import click
import structlog
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .models import DatasetConfig, LLMConfig
from .dataset_generator import DatasetGenerator

# Load environment variables
load_dotenv()

console = Console()
logger = structlog.get_logger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Video Transcript QA Dataset Generation Framework.

    A powerful tool for generating synthetic question-answer datasets from video content.
    """
    pass


@cli.command()
@click.option(
    "--input-folder", "-i", required=True, help="Folder containing input videos"
)
@click.option(
    "--output-folder",
    "-o",
    required=True,
    help="Folder to save the generated dataset",
)
@click.option("--dataset-name", "-n", required=True, help="Name of the dataset")
@click.option(
    "--dataset-description",
    "-d",
    default="Synthetic QA dataset generated from video transcripts",
    help="Description of the dataset",
)
@click.option(
    "--segment-duration",
    default=30,
    help="Duration of transcript segments in seconds",
)
@click.option(
    "--max-concurrent-videos",
    default=4,
    help="Maximum number of videos to process concurrently",
)
@click.option(
    "--training-split-ratio",
    default=0.8,
    help="Ratio for training/validation split",
)
@click.option(
    "--training-llm-s-model",
    default="placeholder-llm-s",
    help="LLM-S model for training split",
)
@click.option(
    "--training-llm-a-model",
    default="placeholder-llm-a",
    help="LLM-A model for training split",
)
@click.option(
    "--validation-llm-s-model",
    default="placeholder-llm-s",
    help="LLM-S model for validation split",
)
@click.option(
    "--validation-llm-a-model",
    default="placeholder-llm-a",
    help="LLM-A model for validation split",
)
@click.option(
    "--openai-api-base", envvar="OPENAI_API_BASE", help="OpenAI API base URL"
)
@click.option(
    "--openai-api-key", envvar="OPENAI_API_KEY", help="OpenAI API key"
)
@click.option(
    "--transcript-path",
    "-t",
    help="Path to pre-generated transcript file (JSON format). If provided, transcripts will be loaded from this file instead of being generated.",
)
@click.option(
    "--config-file", "-c", help="Path to configuration file (JSON format)"
)
def generate(
    input_folder: str,
    output_folder: str,
    dataset_name: str,
    dataset_description: str,
    segment_duration: int,
    max_concurrent_videos: int,
    training_split_ratio: float,
    training_llm_s_model: str,
    training_llm_a_model: str,
    validation_llm_s_model: str,
    validation_llm_a_model: str,
    openai_api_base: Optional[str],
    openai_api_key: Optional[str],
    transcript_path: Optional[str],
    config_file: Optional[str],
):
    """Generate a synthetic QA dataset from videos."""

    # Load configuration from file if provided
    if config_file:
        config = _load_config_from_file(config_file)
    else:
        # Create LLM configurations
        training_llm_config = LLMConfig(
            llm_s_model=training_llm_s_model, llm_a_model=training_llm_a_model
        )
        validation_llm_config = LLMConfig(
            llm_s_model=validation_llm_s_model,
            llm_a_model=validation_llm_a_model,
        )

        config = DatasetConfig(
            input_folder=input_folder,
            output_folder=output_folder,
            segment_duration=segment_duration,
            max_concurrent_videos=max_concurrent_videos,
            training_split_ratio=training_split_ratio,
            transcript_path=transcript_path,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            training_llm_config=training_llm_config,
            validation_llm_config=validation_llm_config,
        )

    # Validate configuration
    _validate_config(config)

    # Display configuration
    _display_config(config)

    # Run the dataset generation
    asyncio.run(_run_generation(config, dataset_name, dataset_description))


@cli.command()
@click.option(
    "--input-folder", "-i", required=True, help="Folder containing input videos"
)
@click.option(
    "--output-file",
    "-o",
    required=True,
    help="Path to save the generated transcript file (JSON format)",
)
@click.option(
    "--segment-duration",
    default=30,
    help="Duration of transcript segments in seconds",
)
@click.option(
    "--max-concurrent-videos",
    default=4,
    help="Maximum number of videos to process concurrently",
)
def extract_transcripts(
    input_folder: str,
    output_file: str,
    segment_duration: int,
    max_concurrent_videos: int,
):
    """Extract transcripts from videos and save to a JSON file."""

    # Validate input folder
    if not os.path.exists(input_folder):
        console.print(
            f"[bold red]Error: Input folder does not exist: {input_folder}[/bold red]"
        )
        return

    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run transcript extraction
    asyncio.run(
        _run_transcript_extraction(
            input_folder, output_file, segment_duration, max_concurrent_videos
        )
    )


@cli.command()
@click.option(
    "--dataset-path",
    "-p",
    required=True,
    help="Path to the generated dataset JSON file",
)
def analyze(dataset_path: str):
    """Analyze a generated dataset and display statistics."""

    if not os.path.exists(dataset_path):
        console.print(
            f"[bold red]Error: Dataset file not found: {dataset_path}[/bold red]"
        )
        return

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset_data = json.load(f)

        _display_dataset_analysis(dataset_data)

    except Exception as e:
        console.print(f"[bold red]Error analyzing dataset: {e}[/bold red]")


@cli.command()
@click.option(
    "--template-path",
    "-t",
    default="config_template.json",
    help="Path to save the configuration template",
)
def create_config_template(template_path: str):
    """Create a configuration template file."""

    # Create LLM configurations
    training_llm_config = LLMConfig(
        llm_s_model="placeholder-llm-s", llm_a_model="placeholder-llm-a"
    )
    validation_llm_config = LLMConfig(
        llm_s_model="placeholder-llm-s", llm_a_model="placeholder-llm-a"
    )

    template_config = DatasetConfig(
        input_folder="./input_videos",
        output_folder="./output_dataset",
        segment_duration=30,
        max_concurrent_videos=4,
        training_split_ratio=0.8,
        transcript_path=None,
        openai_api_base="https://api.openai.com/v1",
        openai_api_key="your-api-key-here",
        training_llm_config=training_llm_config,
        validation_llm_config=validation_llm_config,
    )

    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(template_config.model_dump(), f, indent=2, default=str)

    console.print(
        f"[bold green]✅ Configuration template created: {template_path}[/bold green]"
    )
    console.print(
        "[bold yellow]⚠️ Remember to update the API key and other settings before using![/bold yellow]"
    )


def _load_config_from_file(config_file: str) -> DatasetConfig:
    """Load configuration from a JSON file."""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        return DatasetConfig(**config_data)

    except Exception as e:
        raise click.ClickException(f"Failed to load configuration file: {e}")


def _validate_config(config: DatasetConfig):
    """Validate the configuration."""
    # Validate that only one of transcript_path or input_folder is provided
    if config.transcript_path and config.input_folder:
        raise click.ClickException(
            "Cannot provide both transcript_path and input_folder. "
            "If transcript_path is provided, transcripts will be loaded from file. "
            "If input_folder is provided, transcripts will be generated from videos."
        )

    # Check if transcript file exists (if provided)
    if config.transcript_path and not os.path.exists(config.transcript_path):
        raise click.ClickException(
            f"Transcript file does not exist: {config.transcript_path}"
        )

    # Check if input folder exists (if provided)
    if config.input_folder and not os.path.exists(config.input_folder):
        raise click.ClickException(
            f"Input folder does not exist: {config.input_folder}"
        )

    # Check if output folder can be created
    try:
        os.makedirs(config.output_folder, exist_ok=True)
    except Exception as e:
        raise click.ClickException(f"Cannot create output folder: {e}")

    # Validate API key (only if not using placeholder models)
    if not config.openai_api_key and (
        not config.training_llm_config.llm_s_model.startswith("placeholder-")
        or not config.training_llm_config.llm_a_model.startswith("placeholder-")
        or not config.validation_llm_config.llm_s_model.startswith(
            "placeholder-"
        )
        or not config.validation_llm_config.llm_a_model.startswith(
            "placeholder-"
        )
    ):
        raise click.ClickException(
            "OpenAI API key is required when not using placeholder models. Set OPENAI_API_KEY environment variable or use --openai-api-key option."
        )

    # Validate ratios
    if not 0 < config.training_split_ratio < 1:
        raise click.ClickException(
            "Training split ratio must be between 0 and 1"
        )


def _display_config(config: DatasetConfig):
    """Display the configuration in a nice format."""
    table = Table(title="Dataset Generation Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    if config.transcript_path:
        table.add_row("Mode", "External Transcripts")
        table.add_row("Transcript Path", config.transcript_path)
    else:
        table.add_row("Mode", "Generate Transcripts")
        table.add_row("Input Folder", config.input_folder)
        table.add_row("Segment Duration", f"{config.segment_duration} seconds")
        table.add_row(
            "Max Concurrent Videos", str(config.max_concurrent_videos)
        )

    table.add_row("Output Folder", config.output_folder)
    table.add_row("Training Split Ratio", f"{config.training_split_ratio:.2f}")

    # Training LLM config
    table.add_row(
        "Training LLM-S Model", config.training_llm_config.llm_s_model
    )
    table.add_row(
        "Training LLM-A Model", config.training_llm_config.llm_a_model
    )

    # Validation LLM config
    table.add_row(
        "Validation LLM-S Model", config.validation_llm_config.llm_s_model
    )
    table.add_row(
        "Validation LLM-A Model", config.validation_llm_config.llm_a_model
    )

    table.add_row("OpenAI API Base", config.openai_api_base or "Default")
    table.add_row(
        "OpenAI API Key", "***" if config.openai_api_key else "Not set"
    )

    console.print(table)


async def _run_generation(
    config: DatasetConfig, dataset_name: str, dataset_description: str
):
    """Run the dataset generation process."""
    try:
        generator = DatasetGenerator(config)
        dataset = await generator.generate_dataset(
            dataset_name, dataset_description
        )

        # Display final statistics
        stats = generator.get_dataset_statistics(dataset)
        _display_final_statistics(stats)

    except Exception as e:
        console.print(f"[bold red]❌ Dataset generation failed: {e}[/bold red]")
        raise click.ClickException(str(e))


async def _run_transcript_extraction(
    input_folder: str,
    output_file: str,
    segment_duration: int,
    max_concurrent_videos: int,
):
    """Run the transcript extraction process."""
    try:
        from .video_processor import VideoProcessor, TranscriptExtractor

        console.print(
            f"[bold blue]🚀 Starting transcript extraction...[/bold blue]"
        )

        # Initialize video processor and transcript extractor
        video_processor = VideoProcessor(
            output_dir=f"{Path(output_file).parent}/videos",
            max_concurrent=max_concurrent_videos,
        )
        transcript_extractor = TranscriptExtractor(
            segment_duration=segment_duration
        )

        # Process videos
        console.print("[bold green]📹 Processing videos...[/bold green]")
        video_metadata_list = await video_processor.process_video_folder(
            input_folder
        )

        if not video_metadata_list:
            raise ValueError("No videos were successfully processed")

        # Extract transcripts
        console.print("[bold green]📝 Extracting transcripts...[/bold green]")
        all_transcript_segments = []

        for video_metadata in video_metadata_list:
            try:
                segments = transcript_extractor.extract_transcripts(
                    video_metadata
                )
                all_transcript_segments.extend(segments)
            except Exception as e:
                logger.error(
                    f"Failed to extract transcripts from {video_metadata.video_id}: {e}"
                )

        if not all_transcript_segments:
            raise ValueError("No transcript segments were extracted")

        # Save transcripts to file
        console.print(
            f"[bold green]💾 Saving transcripts to {output_file}...[/bold green]"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                [segment.model_dump() for segment in all_transcript_segments],
                f,
                indent=2,
                default=str,
            )

        console.print(
            f"[bold green]✅ Transcript extraction completed successfully![/bold green]"
        )
        console.print(f"[bold]📊 Extraction Statistics:[/bold]")
        console.print(f"  • Total videos: {len(video_metadata_list)}")
        console.print(f"  • Total segments: {len(all_transcript_segments)}")
        console.print(f"  • Output file: {output_file}")

    except Exception as e:
        console.print(
            f"[bold red]❌ Transcript extraction failed: {e}[/bold red]"
        )
        raise click.ClickException(str(e))


def _display_dataset_analysis(dataset_data: dict):
    """Display dataset analysis."""
    console.print(
        f"[bold blue]📊 Dataset Analysis: {dataset_data.get('name', 'Unknown')}[/bold blue]"
    )

    # Create statistics table
    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Videos", str(dataset_data.get("total_videos", 0)))
    table.add_row("Total Segments", str(dataset_data.get("total_segments", 0)))
    table.add_row("Total QA Pairs", str(dataset_data.get("total_qa_pairs", 0)))

    # Training split stats
    training_split = dataset_data.get("training_split", {})
    table.add_row("Training QA Pairs", str(training_split.get("qa_pairs", 0)))
    table.add_row(
        "Training LLM-S Model", training_split.get("llm_s_model", "Unknown")
    )
    table.add_row(
        "Training LLM-A Model", training_split.get("llm_a_model", "Unknown")
    )

    # Validation split stats
    validation_split = dataset_data.get("validation_split", {})
    table.add_row(
        "Validation QA Pairs", str(validation_split.get("qa_pairs", 0))
    )
    table.add_row(
        "Validation LLM-S Model", validation_split.get("llm_s_model", "Unknown")
    )
    table.add_row(
        "Validation LLM-A Model", validation_split.get("llm_a_model", "Unknown")
    )

    console.print(table)

    # Display creation info
    created_at = dataset_data.get("created_at", "Unknown")
    console.print(f"[bold]Created:[/bold] {created_at}")

    # Display description
    description = dataset_data.get("description", "No description")
    console.print(f"[bold]Description:[/bold] {description}")


def _display_final_statistics(stats: dict):
    """Display final generation statistics."""
    console.print("\n[bold blue]📈 Final Generation Statistics[/bold blue]")

    table = Table(title="Generation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Training", style="green")
    table.add_column("Validation", style="yellow")

    training_stats = stats["training_split"]
    validation_stats = stats["validation_split"]

    table.add_row(
        "QA Pairs",
        str(training_stats["qa_pairs"]),
        str(validation_stats["qa_pairs"]),
    )
    table.add_row(
        "Answerable without Context",
        str(training_stats["is_answerable_without_context"]),
        str(validation_stats["is_answerable_without_context"]),
    )
    table.add_row(
        "Answerable without Context Ratio",
        f"{training_stats['is_answerable_without_context_ratio']:.3f}",
        f"{validation_stats['is_answerable_without_context_ratio']:.3f}",
    )
    table.add_row(
        "Answerable with Context",
        str(training_stats["answerable_with_context"]),
        str(validation_stats["answerable_with_context"]),
    )
    table.add_row(
        "Answerable with Context Ratio",
        f"{training_stats['answerable_with_context_ratio']:.3f}",
        f"{validation_stats['answerable_with_context_ratio']:.3f}",
    )
    table.add_row(
        "LLM-S Model",
        training_stats["llm_s_model"],
        validation_stats["llm_s_model"],
    )
    table.add_row(
        "LLM-A Model",
        training_stats["llm_a_model"],
        validation_stats["llm_a_model"],
    )

    console.print(table)

    console.print(f"\n[bold green]✅ Dataset saved successfully![/bold green]")
    console.print(
        f"[bold]Total QA Pairs Generated:[/bold] {stats['total_qa_pairs']}"
    )


if __name__ == "__main__":
    cli()
