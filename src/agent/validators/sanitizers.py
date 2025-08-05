"""
Input sanitization utilities for the validation framework.

This module provides concrete implementations of sanitizers for:
- HTML/script tag removal
- Unicode normalization
- Whitespace normalization
- SQL injection prevention
- Composite sanitization
"""

import re
import unicodedata
from typing import List

from src.agent.validators.base import Sanitizer


class HtmlSanitizer(Sanitizer):
    """
    Sanitizer for removing HTML tags and preventing XSS attacks.

    This sanitizer removes:
    - Script tags and their content
    - All HTML tags while preserving text content
    - Potentially dangerous attributes
    """

    # Pattern to match script tags (case insensitive, with content)
    SCRIPT_PATTERN = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)

    # Pattern to match HTML-like tags (including closing tags and malformed ones)
    HTML_TAG_PATTERN = re.compile(r"</?[a-zA-Z][^<]*?>")

    def sanitize(self, input_data: str) -> str:
        """
        Remove HTML tags and script content from input.

        Args:
            input_data: Raw input potentially containing HTML

        Returns:
            Sanitized input with HTML tags removed
        """
        if not input_data:
            return input_data

        # First remove script tags and their content
        sanitized = self.SCRIPT_PATTERN.sub("", input_data)

        # Remove complete HTML tags
        sanitized = self.HTML_TAG_PATTERN.sub("", sanitized)

        # Handle malformed tags (starting with < followed by letters but no closing >)
        sanitized = re.sub(r"<[a-zA-Z][^<]*$", "", sanitized)  # Malformed at end
        sanitized = re.sub(
            r"<[a-zA-Z][^<>]*(?=\s)", "", sanitized
        )  # Malformed followed by space

        return sanitized


class UnicodeSanitizer(Sanitizer):
    """
    Sanitizer for Unicode normalization and control character removal.

    This sanitizer:
    - Normalizes Unicode to NFC form
    - Removes control characters
    - Preserves printable characters
    """

    def sanitize(self, input_data: str) -> str:
        """
        Normalize Unicode and remove control characters.

        Args:
            input_data: Raw input potentially containing Unicode issues

        Returns:
            Sanitized input with normalized Unicode
        """
        if not input_data:
            return input_data

        # Normalize Unicode to NFC form (canonical composed)
        sanitized = unicodedata.normalize("NFC", input_data)

        # Remove control characters (categories Cc and Cf)
        # Keep newlines and tabs for now as they might be legitimate
        sanitized = "".join(
            char
            for char in sanitized
            if not unicodedata.category(char).startswith("C")
            or char in "\n\t"  # Keep some whitespace control chars
        )

        # Remove null bytes and other dangerous control characters
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", sanitized)

        return sanitized


class WhitespaceSanitizer(Sanitizer):
    """
    Sanitizer for whitespace normalization.

    This sanitizer:
    - Collapses multiple whitespace to single spaces
    - Strips leading and trailing whitespace
    - Normalizes different types of whitespace
    """

    # Pattern to match multiple whitespace characters
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def sanitize(self, input_data: str) -> str:
        """
        Normalize whitespace in input.

        Args:
            input_data: Raw input potentially containing irregular whitespace

        Returns:
            Sanitized input with normalized whitespace
        """
        if not input_data:
            return input_data

        # Replace all whitespace sequences with single space
        sanitized = self.WHITESPACE_PATTERN.sub(" ", input_data)

        # Strip leading and trailing whitespace
        sanitized = sanitized.strip()

        return sanitized


class SqlInjectionSanitizer(Sanitizer):
    """
    Sanitizer for basic SQL injection prevention.

    This sanitizer:
    - Escapes single quotes
    - Removes or escapes dangerous SQL patterns
    - Provides basic protection (not a replacement for parameterized queries)
    """

    def sanitize(self, input_data: str) -> str:
        """
        Sanitize input to prevent basic SQL injection.

        Note: This is basic protection. Always use parameterized queries
        for proper SQL injection prevention.

        Args:
            input_data: Raw input that might be used in SQL context

        Returns:
            Sanitized input with SQL-dangerous patterns escaped
        """
        if not input_data:
            return input_data

        # Escape single quotes by doubling them
        sanitized = input_data.replace("'", "''")

        return sanitized


class CompositeSanitizer(Sanitizer):
    """
    Composite sanitizer that applies multiple sanitizers in sequence.

    This allows combining different sanitization strategies in a
    configurable pipeline.
    """

    def __init__(self, sanitizers: List[Sanitizer]):
        """
        Initialize composite sanitizer with list of sanitizers.

        Args:
            sanitizers: List of sanitizers to apply in order
        """
        self.sanitizers = sanitizers or []

    def sanitize(self, input_data: str) -> str:
        """
        Apply all sanitizers in sequence.

        Args:
            input_data: Raw input to sanitize

        Returns:
            Input sanitized by all configured sanitizers
        """
        result = input_data

        for sanitizer in self.sanitizers:
            result = sanitizer.sanitize(result)

        return result
