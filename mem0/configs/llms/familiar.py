from typing import Optional

from mem0.configs.llms.base import BaseLlmConfig


class FamiliarConfig(BaseLlmConfig):
    """
    Configuration class for Familiar-specific parameters.
    Inherits from BaseLlmConfig and adds Familiar-specific settings.
    """

    def __init__(
        self,
        # Base parameters
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
    ):
        """
        Args:
            model: Familiar model to use, defaults to None
            temperature: Controls randomness, defaults to 0.1
            max_tokens: Maximum tokens to generate, defaults to 2000
            top_p: Nucleus sampling parameter, defaults to 0.1
            top_k: Top-k sampling parameter, defaults to 1
            enable_vision: Enable vision capabilities, defaults to False
            vision_details: Vision detail level, defaults to "auto"
        """
        # Initialize base parameters
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            enable_vision=enable_vision,
            vision_details=vision_details,
        )