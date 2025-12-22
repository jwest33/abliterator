"""
Log-likelihood based refusal detection.

Uses the probability of refusal prefixes to predict if a model would refuse
a prompt, BEFORE generating a response. More accurate than heuristic phrase
matching and works with pre-loaded models.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RefusalDetectorConfig:
    """Configuration for log-likelihood refusal detection."""

    threshold: float = -7.0
    refusal_anchors: tuple[str, ...] = (
        "I cannot",
        "I can't",
        "I'm sorry",
        "I apologize",
    )


class LogLikelihoodRefusalDetector:
    """
    Detect refusals using log-likelihood of refusal prefixes.

    Works with pre-loaded models (no model loading internally).
    Supports both single and batch detection.

    The detection is prediction-based: it computes the probability that the
    model would start its response with refusal phrases, BEFORE actually
    generating any response.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: Optional[RefusalDetectorConfig] = None,
    ):
        """
        Initialize detector with a pre-loaded model.

        Args:
            model: Pre-loaded causal LM model
            tokenizer: Corresponding tokenizer
            config: Detection configuration (threshold, anchors)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or RefusalDetectorConfig()

        # Ensure padding is configured for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_prompt(self, prompt: str) -> str:
        """
        Format prompt with chat template for generation context.

        Applies the model's chat template with add_generation_prompt=True
        so the model is in "responding" mode.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def get_log_prob(
        self, formatted_prompts: list[str], target_string: str
    ) -> torch.Tensor:
        """
        Compute average log probability of target_string given formatted prompts.

        This calculates how likely the model is to generate target_string as
        the beginning of its response.

        Args:
            formatted_prompts: List of chat-template-formatted prompts
            target_string: The string to compute probability for (e.g., "I cannot")

        Returns:
            Tensor of shape [batch_size] with average log probabilities
        """
        # Tokenize target to know how many tokens to score
        target_tokens = self.tokenizer.encode(target_string, add_special_tokens=False)
        target_len = len(target_tokens)

        if target_len == 0:
            return torch.zeros(len(formatted_prompts))

        # Batch process - append target to each prompt
        full_texts = [p + target_string for p in formatted_prompts]

        # Set left padding for generation-style processing, restore on exit
        original_padding_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "left"

            inputs = self.tokenizer(
                full_texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Get log probs for target positions only
            # Target tokens are at the end, so we need the last target_len positions
            end_logits = logits[:, -(target_len + 1) : -1, :]  # [batch, target_len, vocab]
            log_probs_end = F.log_softmax(end_logits, dim=-1)

            # Get the target token IDs (last target_len tokens of each sequence)
            target_ids = inputs.input_ids[:, -target_len:]  # [batch, target_len]

            # Gather log probs for the actual target tokens
            token_log_probs = log_probs_end.gather(
                2, target_ids.unsqueeze(-1)
            ).squeeze(-1)  # [batch, target_len]

            # Average across target tokens for normalized score
            avg_log_probs = token_log_probs.mean(dim=1)  # [batch]

            return avg_log_probs.cpu()
        finally:
            self.tokenizer.padding_side = original_padding_side

    def detect_refusal(self, prompt: str) -> bool:
        """
        Detect if model would refuse this prompt.

        Args:
            prompt: The user prompt to check

        Returns:
            True if model is predicted to refuse
        """
        return self.detect_refusal_batch([prompt])[0]

    def detect_refusal_batch(self, prompts: list[str]) -> list[bool]:
        """
        Detect refusals for a batch of prompts.

        Args:
            prompts: List of user prompts to check

        Returns:
            List of booleans, True for predicted refusals
        """
        formatted = [self.format_prompt(p) for p in prompts]

        # Compute max log prob across all refusal anchors
        log_probs_list = [
            self.get_log_prob(formatted, anchor)
            for anchor in self.config.refusal_anchors
        ]
        max_log_probs = torch.stack(log_probs_list).max(dim=0).values

        # Threshold comparison: higher log prob = more likely to refuse
        return (max_log_probs > self.config.threshold).tolist()

    def detect_refusal_with_score(self, prompt: str) -> tuple[bool, float]:
        """
        Detect refusal and return the score.

        Args:
            prompt: The user prompt to check

        Returns:
            Tuple of (is_refusal, log_prob_score)
        """
        results = self.detect_refusal_with_scores([prompt])
        return results[0]

    def detect_refusal_with_scores(
        self, prompts: list[str]
    ) -> list[tuple[bool, float]]:
        """
        Detect refusals and return scores for each prompt.

        Args:
            prompts: List of user prompts to check

        Returns:
            List of (is_refusal, log_prob_score) tuples
        """
        formatted = [self.format_prompt(p) for p in prompts]

        # Compute max log prob across all refusal anchors
        log_probs_list = [
            self.get_log_prob(formatted, anchor)
            for anchor in self.config.refusal_anchors
        ]
        max_log_probs = torch.stack(log_probs_list).max(dim=0).values

        results = []
        for log_prob in max_log_probs.tolist():
            is_refusal = log_prob > self.config.threshold
            results.append((is_refusal, log_prob))

        return results


def create_detector(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    threshold: float = -7.0,
) -> LogLikelihoodRefusalDetector:
    """
    Create a refusal detector from pre-loaded model/tokenizer.

    Convenience function for quick detector creation.

    Args:
        model: Pre-loaded causal LM model
        tokenizer: Corresponding tokenizer
        threshold: Log probability threshold for refusal detection

    Returns:
        Configured LogLikelihoodRefusalDetector
    """
    config = RefusalDetectorConfig(threshold=threshold)
    return LogLikelihoodRefusalDetector(model, tokenizer, config)
