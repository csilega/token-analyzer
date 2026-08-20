"""
Library-level exceptions.

None of these know what HTTP is. The API layer is
responsible for catching these and translating them into the appropriate
HTTP status codes.
"""


class TokenAnalyzerError(Exception):
    """Base class for all library-raised exceptions."""


class UnsupportedModelError(TokenAnalyzerError):
    """Raised when get_adapter() is asked for a model name with no
    registered adapter."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(
            f"No adapter registered for model '{model_name}'. "
            f"See registry.SUPPORTED_MODELS for the current list."
        )