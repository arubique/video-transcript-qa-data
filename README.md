# 🎬 Video Transcript QA Dataset Generation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Transform video content into high-quality question-answer datasets using advanced LLM technology**

A framework for automatically generating synthetic question-answer (QA) datasets from video transcripts. This tool leverages language models to create contextually relevant QA pairs that are perfect for training and evaluating Retrieval Augmented Generation (RAG) systems for video content QA.

## ✨ Features

- **🎥 Multi-format Video Support**: Process MP4, AVI, MOV, MKV, WebM, FLV, and WMV files
- **🤖 Dual LLM Architecture**: Uses separate models for question generation (LLM-S) and answer validation (LLM-A)
- **📊 Intelligent Quality Control**: Filters out questions that can be answered without context
- **⚡ Concurrent Processing**: Parallel video processing for optimal performance
- **📈 Rich Analytics**: Comprehensive statistics and quality metrics
- **💾 Multiple Output Formats**: JSON, HuggingFace datasets, and individual splits
- **🎯 Configurable Parameters**: Fine-tune segment duration, confidence thresholds, and model selection

## 🏗️ Architecture

The framework follows a modular design with three core components:

### 1. Video Processing Pipeline
- **Video Processor**: Handles video file discovery, metadata extraction, and format standardization
- **Transcript Extractor**: Segments videos into 30-second chunks and extracts transcript text
- **Integration with YT-Navigator**: Leverages proven video processing tools from the [YT-Navigator](https://github.com/wassim249/YT-Navigator/tree/master) project

### 2. LLM Processing Engine
- **LLM-S (Synthesis)**: Generates contextually relevant question-answer pairs from transcript segments
- **LLM-A (Answer)**: Validates answers and tests context dependency: tells whether the question can be correctly answered without using the context
- **Quality Filtering**: Ensures questions require the provided context to answer correctly

### 3. Dataset Generation System
- **Training/Validation Splits**: Creates separate datasets with different LLM configurations: use different LLM-S and LLM-A pairs for training and validation splits to avoid overfitting
- **Multi-format Export**: Saves datasets in JSON and HuggingFace formats
- **Comprehensive Metadata**: Includes confidence scores, model information, and timing data

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/video-transcript-qa-data.git
   cd video-transcript-qa-data
   ```

2. **Set up the environment**
   ```bash
   # Create conda environment
   conda create --prefix ./envs/trqa python=3.10.0 --no-deps --channel conda-forge --channel defaults --override-channels

   # Activate environment
   conda activate ./envs/trqa

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure API credentials**
   ```bash
   # Set your OpenAI API key
   export OPENAI_API_KEY="your-api-key-here"

   # Optional: Set custom API endpoint
   export OPENAI_API_BASE="https://your-custom-endpoint.com/v1"
   ```

### Basic Usage

#### Command Line Interface

The framework provides a powerful command-line interface with three main commands that handle the entire dataset generation workflow:

**`create-config-template`** - Generates a configuration file template that you can customize with your specific settings for LLM models, processing parameters, and file paths.

**`generate`** - The main command that processes videos and creates QA datasets. It handles video processing, transcript extraction, LLM-powered QA generation, and exports the final dataset in multiple formats.

**`analyze`** - Allows you to examine existing datasets and view comprehensive statistics including video counts, segment information, QA pair counts, and filtering metrics.

**How to use the config template:**

1. **Generate the template**: Run `create-config-template` to create a JSON configuration file
2. **Customize settings**: Edit the generated file to set your input/output folders, LLM models, and processing parameters
3. **Use with generate**: Pass the config file to the `generate` command using `--config-file my_config.json`

This approach is ideal when you want to:
- Use the same configuration across multiple runs
- Set different LLM models for training vs validation splits
- Configure complex processing parameters
- Share configurations with team members

```bash
# Create a configuration template
python -m src.cli create-config-template --template-path my_config.json

# Customize the config as you wish
vim my_config.json

# Generate a dataset from videos using config
python -m src.cli generate \
    --config-file my_config.json

# Generate a dataset from videos without using config
python -m src.cli generate \
    --input-folder ./input_videos \
    --output-folder ./output_dataset \
    --dataset-name "educational_videos_qa" \
    --dataset-description "QA dataset from educational content"

# Analyze a generated dataset
python -m src.cli analyze --dataset-path ./output_dataset/educational_videos_qa.json
```

#### Programmatic Usage

This Python code demonstrates how to use the framework programmatically instead of through the command line. It shows the complete workflow of creating a configuration, initializing the dataset generator, and running the generation process.

**Why asyncio is needed:**
The framework uses asynchronous programming (`async`/`await`) to handle multiple operations concurrently:
- **Concurrent video processing** - Multiple videos can be processed simultaneously
- **Parallel LLM API calls** - Multiple API requests to LLM services can run in parallel
- **Efficient I/O operations** - File reading/writing and network requests don't block each other
- **Better performance** - Significantly faster than sequential processing, especially for large video collections

The `asyncio.run(main())` call starts the asynchronous event loop and executes the main function, allowing all the concurrent operations to run efficiently.

```python
import asyncio
from src.models import DatasetConfig, LLMConfig
from src.dataset_generator import DatasetGenerator

async def main():
    # Create LLM configurations for training and validation splits
    training_llm_config = LLMConfig(
        llm_s_model="placeholder-llm-s",
        llm_a_model="placeholder-llm-a"
    )
    validation_llm_config = LLMConfig(
        llm_s_model="placeholder-llm-s",
        llm_a_model="placeholder-llm-a"
    )

    # Configure the dataset generation
    config = DatasetConfig(
        input_folder="./input_videos",
        output_folder="./output_dataset",
        segment_duration=30,
        max_concurrent_videos=4,
        training_split_ratio=0.8,
        training_llm_config=training_llm_config,
        validation_llm_config=validation_llm_config
    )

    # Create and run the generator
    generator = DatasetGenerator(config)
    dataset = await generator.generate_dataset(
        dataset_name="my_video_qa_dataset",
        dataset_description="Custom QA dataset from my videos"
    )

    # Get statistics
    stats = generator.get_dataset_statistics(dataset)
    print(f"Generated {stats['total_qa_pairs']} QA pairs")

# Run the generation
asyncio.run(main())
```

## 📁 Project Structure

```
video-transcript-qa-data/
├── src/                          # Main source code
│   ├── __init__.py
│   ├── models.py                 # Pydantic data models
│   ├── video_processor.py        # Video processing and transcript extraction
│   ├── llm_processor.py          # LLM interaction and QA generation
│   ├── dataset_generator.py      # Main orchestration logic
│   └── cli.py                    # Command-line interface
├── examples/                     # Usage examples
│   └── example_usage.py
├── tests/                        # Test suite
│   └── test_framework.py
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

## 🔧 Configuration

The framework is highly configurable through the `DatasetConfig` class:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segment_duration` | 30 | Duration of transcript segments in seconds |
| `max_concurrent_videos` | 4 | Maximum concurrent video processing |
| `training_split_ratio` | 0.8 | Ratio for training/validation split |
| `min_qa_confidence` | 0.7 | Minimum confidence score for QA pairs |
| `llm_s_model` | "gpt-4" | Model for generating QA pairs |
| `llm_a_model` | "gpt-3.5-turbo" | Model for answer validation |

### Configuration File Example

```json
{
  "input_folder": "./input_videos",
  "output_folder": "./output_dataset",
  "segment_duration": 30,
  "max_concurrent_videos": 4,
  "training_split_ratio": 0.8,
  "min_qa_confidence": 0.7,
  "llm_s_model": "gpt-4",
  "llm_a_model": "gpt-3.5-turbo",
  "openai_api_base": "https://api.openai.com/v1",
  "openai_api_key": "your-api-key-here"
}
```

## 📊 Output Format

The framework generates datasets in multiple formats:

### JSON Format
```json
{
  "name": "educational_videos_qa",
  "description": "QA dataset from educational content",
  "training_split": {
    "name": "training",
    "qa_pairs": [
      {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of artificial intelligence...",
        "source_document": {
          "transcript": "In this video, we discuss machine learning...",
          "video_ref": "Introduction to ML",
          "timecode": "00:01:30",
          "video_id": "video_123",
          "start_time": 90.0,
          "end_time": 120.0
        },
        "confidence_score": 0.85,
        "is_answerable_without_context": false,
        "llm_s_model": "gpt-4",
        "llm_a_model": "gpt-3.5-turbo"
      }
    ]
  },
  "validation_split": { ... },
  "config": { ... },
  "total_videos": 10,
  "total_segments": 150,
  "total_qa_pairs": 120
}
```

### HuggingFace Datasets Format
The framework also exports datasets in HuggingFace format for easy integration with popular ML libraries.

## 🧪 Testing

Run the test suite to ensure everything works correctly:

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Performance

The framework is optimized for performance:

- **Concurrent Processing**: Videos are processed in parallel to maximize throughput
- **Efficient LLM Usage**: Batched API calls and intelligent caching reduce costs
- **Memory Management**: Streaming processing for large video collections
- **Quality Filtering**: Early filtering reduces unnecessary LLM calls

## 🔍 Quality Assurance

The framework includes several quality control mechanisms:

- **Confidence Scoring**: Each QA pair receives a confidence score based on LLM validation
- **Context Dependency Testing**: Filters out questions that can be answered without the provided context
- **Answer Validation**: LLM-A validates that answers are accurate and complete
- **Statistical Analysis**: Comprehensive metrics help assess dataset quality

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YT-Navigator**: Leverages video processing tools from the [YT-Navigator project](https://github.com/wassim249/YT-Navigator/tree/master)
- **OpenAI**: Provides the LLM APIs that power the QA generation
- **HuggingFace**: Dataset format compatibility and ML ecosystem integration
- **Pydantic**: Type safety and data validation

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/video-transcript-qa-data/issues) page
2. Create a new issue with detailed information
3. Include your configuration and error logs

---

**Made with ❤️ for the AI/ML community**

*Transform your video content into powerful training data for the next generation of AI systems.*

## 📚 Citation

If you use this framework in your research or projects, please cite:

```bibtex
@software{rubinstein2025videotranscriptqa,
  title={Video Transcript QA Dataset Generation Framework},
  author={Rubinstein, Alexander},
  year={2025},
  url={https://github.com/arubique/video-transcript-qa-data},
  note={A framework for generating synthetic question-answer datasets from video content using Large Language Models}
}
```
