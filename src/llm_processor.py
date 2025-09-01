"""LLM processing module for generating QA pairs and validating answers."""

import asyncio
from typing import List, Optional, Tuple

import openai
import structlog
from openai import AsyncOpenAI

from .models import QAPair, SourceDocument, TranscriptSegment

logger = structlog.get_logger(__name__)


class LLMProcessor:
    """Handles LLM interactions for QA pair generation and validation."""

    def __init__(
        self,
        llm_s_model: str = "placeholder-llm-s",
        llm_a_model: str = "placeholder-llm-a",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_concurrent: int = 4,
    ):
        """Initialize the LLM processor.

        Args:
            llm_s_model: Model for generating QA pairs (LLM-S)
            llm_a_model: Model for answering questions (LLM-A)
            api_base: OpenAI API base URL
            api_key: OpenAI API key
            max_concurrent: Maximum number of concurrent LLM calls
        """
        self.llm_s_model = llm_s_model
        self.llm_a_model = llm_a_model
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Initialize OpenAI client (only if not using placeholder models)
        if not self._is_placeholder_model(
            llm_s_model
        ) and not self._is_placeholder_model(llm_a_model):
            self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        else:
            self.client = None

    def _is_placeholder_model(self, model_name: str) -> bool:
        """Check if a model name is a placeholder."""
        return model_name.startswith("placeholder-")

    async def generate_qa_pairs(
        self, transcript_segments: List[TranscriptSegment], video_metadata: dict
    ) -> List[QAPair]:
        """Generate QA pairs from transcript segments.

        Args:
            transcript_segments: List of transcript segments
            video_metadata: Video metadata

        Returns:
            List of generated QA pairs
        """
        logger.info(
            f"Generating QA pairs for {len(transcript_segments)} transcript segments"
        )

        tasks = []
        for segment in transcript_segments:
            task = self._generate_qa_pair_for_segment(segment, video_metadata)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        qa_pairs = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Failed to generate QA pair: {result}")
            elif result is not None:
                qa_pairs.append(result)

        logger.info(f"Generated {len(qa_pairs)} QA pairs")
        return qa_pairs

    async def _generate_qa_pair_for_segment(
        self, segment: TranscriptSegment, video_metadata: dict
    ) -> Optional[QAPair]:
        """Generate a QA pair for a single transcript segment."""
        async with self.semaphore:
            try:
                # Create source document
                source_doc = SourceDocument(
                    transcript=segment.text,
                    video_ref=video_metadata.get("title", "Unknown Video"),
                    timecode=segment.timestamp,
                    video_id=segment.video_id,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                )

                # Generate question and answer using LLM-S
                if self._is_placeholder_model(self.llm_s_model):
                    question, answer = await self._generate_placeholder_qa(
                        segment.text
                    )
                else:
                    question, answer = await self._generate_question_answer(
                        segment.text
                    )

                if not question or not answer:
                    return None

                # Test if LLM-A can answer without context
                if self._is_placeholder_model(self.llm_a_model):
                    is_answerable_without_context = (
                        await self._test_placeholder_answer_without_context(
                            question, answer
                        )
                    )
                else:
                    is_answerable_without_context = (
                        await self._test_answer_without_context(
                            question, answer
                        )
                    )

                return QAPair(
                    question=question,
                    answer=answer,
                    source_document=source_doc,
                    llm_s_model=self.llm_s_model,
                    llm_a_model=self.llm_a_model,
                    filtered_out=is_answerable_without_context,
                )

            except Exception as e:
                logger.error(
                    f"Failed to generate QA pair for segment {segment.video_id}: {e}"
                )
                return None

    async def _generate_placeholder_qa(
        self, transcript_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate placeholder QA pair for demonstration."""
        # Simulate async delay
        await asyncio.sleep(0.1)

        # Create a simple QA pair based on the transcript
        question = f"What is discussed in this transcript segment?"
        answer = f"This segment discusses: {transcript_text[:100]}..."

        return question, answer

    async def _generate_question_answer(
        self, transcript_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate a question and answer from transcript text using LLM-S."""
        if not self.client:
            return await self._generate_placeholder_qa(transcript_text)

        try:
            prompt = f"""
            Based on the following transcript segment, generate a natural question and its corresponding answer.
            The question should be specific and answerable based on the content.
            The answer should be concise and directly address the question.

            Transcript: {transcript_text}

            Please respond in the following format:
            Question: [your question here]
            Answer: [your answer here]
            """

            response = await self.client.chat.completions.create(
                model=self.llm_s_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at generating high-quality question-answer pairs from video transcripts.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=300,
            )

            content = response.choices[0].message.content
            return self._parse_qa_response(content)

        except Exception as e:
            logger.error(f"Failed to generate question-answer pair: {e}")
            return None, None

    def _parse_qa_response(
        self, response: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Parse the LLM response to extract question and answer."""
        try:
            lines = response.strip().split("\n")
            question = None
            answer = None

            for line in lines:
                line = line.strip()
                if line.startswith("Question:"):
                    question = line.replace("Question:", "").strip()
                elif line.startswith("Answer:"):
                    answer = line.replace("Answer:", "").strip()

            if question and answer:
                return question, answer
            else:
                return None, None

        except Exception as e:
            logger.error(f"Failed to parse QA response: {e}")
            return None, None

    async def _test_placeholder_answer_without_context(
        self, question: str, expected_answer: str
    ) -> bool:
        """Test if placeholder LLM-A can answer the question without context."""
        # Simulate async delay
        await asyncio.sleep(0.1)

        # Simple heuristic: if question contains specific details, it's likely answerable without context
        specific_terms = ["what", "when", "where", "who", "how", "why"]
        has_specific_question = any(
            term in question.lower() for term in specific_terms
        )

        # Randomly decide for demonstration (in real implementation, this would be more sophisticated)
        import random

        return random.choice([True, False])

    async def _test_answer_without_context(
        self, question: str, expected_answer: str
    ) -> bool:
        """Test if LLM-A can answer the question without context."""
        if not self.client:
            return await self._test_placeholder_answer_without_context(
                question, expected_answer
            )

        try:
            prompt = f"""
            Answer the following question based on your general knowledge, without any specific context:

            Question: {question}

            Please provide a concise answer.
            """

            response = await self.client.chat.completions.create(
                model=self.llm_a_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer questions based on your general knowledge.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=150,
            )

            answer_without_context = response.choices[0].message.content.strip()

            # Simple similarity check (in a real implementation, you'd use more sophisticated methods)
            similarity = self._calculate_answer_similarity(
                answer_without_context, expected_answer
            )

            # If similarity is high, the question is answerable without context
            return (
                similarity > 0.7
            )  # Threshold for considering it answerable without context

        except Exception as e:
            logger.error(f"Failed to test answer without context: {e}")
            return False

    def _calculate_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate similarity between two answers (simple implementation)."""
        # Convert to lowercase and split into words
        words1 = set(answer1.lower().split())
        words2 = set(answer2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    async def validate_qa_pairs(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """Validate QA pairs by testing LLM-A's ability to answer with and without context."""
        logger.info(f"Validating {len(qa_pairs)} QA pairs")

        tasks = []
        for qa_pair in qa_pairs:
            task = self._validate_single_qa_pair(qa_pair)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and collect valid pairs
        valid_pairs = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Failed to validate QA pair: {result}")
            elif result is not None:
                valid_pairs.append(result)

        logger.info(f"Validated {len(valid_pairs)} QA pairs")
        return valid_pairs

    async def _validate_single_qa_pair(
        self, qa_pair: QAPair
    ) -> Optional[QAPair]:
        """Validate a single QA pair."""
        async with self.semaphore:
            try:
                # Test with context
                if self._is_placeholder_model(self.llm_a_model):
                    answer_with_context = (
                        await self._answer_placeholder_with_context(
                            qa_pair.question, qa_pair.source_document.transcript
                        )
                    )
                else:
                    answer_with_context = await self._answer_with_context(
                        qa_pair.question, qa_pair.source_document.transcript
                    )

                # Test without context
                if self._is_placeholder_model(self.llm_a_model):
                    answer_without_context = (
                        await self._answer_placeholder_without_context(
                            qa_pair.question
                        )
                    )
                else:
                    answer_without_context = await self._answer_without_context(
                        qa_pair.question
                    )

                # Calculate similarity scores
                context_similarity = self._calculate_answer_similarity(
                    answer_with_context, qa_pair.answer
                )
                no_context_similarity = self._calculate_answer_similarity(
                    answer_without_context, qa_pair.answer
                )

                # Update the QA pair with validation results
                qa_pair.filtered_out = no_context_similarity > 0.7

                return qa_pair

            except Exception as e:
                logger.error(f"Failed to validate QA pair: {e}")
                return None

    async def _answer_placeholder_with_context(
        self, question: str, context: str
    ) -> str:
        """Get placeholder LLM-A's answer with context."""
        await asyncio.sleep(0.1)
        return f"Based on the context: {context[:50]}... The answer is related to the provided transcript."

    async def _answer_placeholder_without_context(self, question: str) -> str:
        """Get placeholder LLM-A's answer without context."""
        await asyncio.sleep(0.1)
        return f"General knowledge answer for: {question}"

    async def _answer_with_context(self, question: str, context: str) -> str:
        """Get LLM-A's answer with context."""
        if not self.client:
            return await self._answer_placeholder_with_context(
                question, context
            )

        try:
            prompt = f"""
            Based on the following context, answer the question:

            Context: {context}

            Question: {question}

            Please provide a concise answer based only on the given context.
            """

            response = await self.client.chat.completions.create(
                model=self.llm_a_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer questions based on the provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=150,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Failed to get answer with context: {e}")
            return ""

    async def _answer_without_context(self, question: str) -> str:
        """Get LLM-A's answer without context."""
        if not self.client:
            return await self._answer_placeholder_without_context(question)

        try:
            prompt = f"""
            Answer the following question based on your general knowledge:

            Question: {question}

            Please provide a concise answer.
            """

            response = await self.client.chat.completions.create(
                model=self.llm_a_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer questions based on your general knowledge.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=150,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Failed to get answer without context: {e}")
            return ""
