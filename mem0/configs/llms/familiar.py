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
    ):
        """
        Args:
            model: Familiar model to use, defaults to None
        """
        # Initialize base parameters
        super().__init__(
            model=model,
        )