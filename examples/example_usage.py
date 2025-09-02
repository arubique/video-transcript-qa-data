"""Example usage of the Video Transcript QA Dataset Generation Framework."""

import asyncio
import os
from pathlib import Path

from src.models import DatasetConfig, LLMConfig
from src.dataset_generator import DatasetGenerator


async def main():
    """Example of how to use the framework programmatically."""

    # Create LLM configurations
    training_llm_config = LLMConfig(
        llm_s_model="placeholder-llm-s", llm_a_model="placeholder-llm-a"
    )
    validation_llm_config = LLMConfig(
        llm_s_model="placeholder-llm-s", llm_a_model="placeholder-llm-a"
    )

    # Configuration
    config = DatasetConfig(
        input_folder="./input_videos",
        output_folder="./output_dataset",
        segment_duration=30,
        max_concurrent_videos=4,
        training_split_ratio=0.8,
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        training_llm_config=training_llm_config,
        validation_llm_config=validation_llm_config,
    )

    # Create dataset generator
    generator = DatasetGenerator(config)

    # Generate dataset
    dataset = await generator.generate_dataset(
        dataset_name="example_video_qa_dataset",
        dataset_description="Example dataset generated from educational videos",
    )

    # Get and print statistics
    stats = generator.get_dataset_statistics(dataset)
    print("Dataset Statistics:")
    print(f"Total videos: {stats['total_videos']}")
    print(f"Total segments: {stats['total_segments']}")
    print(f"Total QA pairs: {stats['total_qa_pairs']}")
    print(f"Training QA pairs: {stats['training_split']['qa_pairs']}")
    print(f"Validation QA pairs: {stats['validation_split']['qa_pairs']}")
    print(
        f"Training answerable without context: {stats['training_split']['is_answerable_without_context']}"
    )
    print(
        f"Validation answerable without context: {stats['validation_split']['is_answerable_without_context']}"
    )
    print(
        f"Training answerable with context: {stats['training_split']['is_answerable_with_context']}"
    )
    print(
        f"Validation answerable with context: {stats['validation_split']['is_answerable_with_context']}"
    )


if __name__ == "__main__":
    # Make sure you have set your OpenAI API key (only needed if not using placeholder models)
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Note: No OpenAI API key set. Using placeholder models for demonstration."
        )
        print("Set OPENAI_API_KEY environment variable to use real LLM models.")

    # Create input directory if it doesn't exist
    Path("./input_videos").mkdir(exist_ok=True)

    # Run the example
    asyncio.run(main())
