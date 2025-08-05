"""
Question validator for comprehensive input validation.

This module provides validation specifically for user questions to the AI agent,
including:
- Empty question detection
- Length validation
- HTML/XSS sanitization
- Malicious pattern detection
- Prompt injection detection
- Unicode normalization
"""

import re
from typing import Any

from src.agent.validators.base import BaseValidator, ValidationRule, ValidationResult
from src.agent.validators.sanitizers import (
    CompositeSanitizer,
    HtmlSanitizer,
    UnicodeSanitizer,
    WhitespaceSanitizer,
)


class EmptyQuestionRule(ValidationRule):
    """Validation rule for detecting empty questions."""

    def validate(self, value: Any) -> ValidationResult:
        """
        Check if the question is empty or contains only whitespace.

        Args:
            value: The question to validate

        Returns:
            ValidationResult indicating if question is non-empty
        """
        if value is None:
            return ValidationResult(
                is_valid=False,
                errors=["Question cannot be None"],
                warnings=[],
                sanitized_input="",
            )

        str_value = str(value).strip()
        if not str_value:
            return ValidationResult(
                is_valid=False,
                errors=["Question cannot be empty"],
                warnings=[],
                sanitized_input="",
            )

        return ValidationResult(
            is_valid=True, errors=[], warnings=[], sanitized_input=str_value
        )


class QuestionLengthRule(ValidationRule):
    """Validation rule for checking question length."""

    def __init__(self, max_length: int = 5000):
        """
        Initialize length rule with maximum allowed length.

        Args:
            max_length: Maximum allowed length for questions
        """
        self.max_length = max_length

    def validate(self, value: Any) -> ValidationResult:
        """
        Check if the question length is within allowed limits.

        Args:
            value: The question to validate

        Returns:
            ValidationResult indicating if question length is acceptable
        """
        str_value = str(value) if value is not None else ""

        if len(str_value) > self.max_length:
            return ValidationResult(
                is_valid=False,
                errors=[
                    f"Question is too long ({len(str_value)} characters). Maximum allowed: {self.max_length}"
                ],
                warnings=[],
                sanitized_input=str_value[: self.max_length],  # Truncate if needed
            )

        return ValidationResult(
            is_valid=True, errors=[], warnings=[], sanitized_input=str_value
        )


class MaliciousPatternRule(ValidationRule):
    """Validation rule for detecting malicious patterns."""

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"';\s*DROP\s+TABLE",
        r"';\s*DELETE\s+FROM",
        r"';\s*INSERT\s+INTO",
        r"';\s*UPDATE\s+",
        r"--\s*$",  # SQL comments
        r"/\*.*\*/",  # SQL block comments
        r"UNION\s+SELECT",
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"/etc/passwd",
        r"/etc/shadow",
        r"\\windows\\system32",
    ]

    # Remote code execution patterns
    RCE_PATTERNS = [
        r"\$\{jndi:",  # Log4j vulnerability pattern
        r"eval\s*\(",
        r"exec\s*\(",
        r"system\s*\(",
        r"__import__",
        r"<script[^>]*>",  # Script tags
        r"javascript:",
        r"fetch\s*\(",
    ]

    def __init__(self):
        """Initialize malicious pattern rule with compiled patterns."""
        self.patterns = []

        # Compile all patterns for efficiency
        for pattern_list in [
            self.SQL_INJECTION_PATTERNS,
            self.PATH_TRAVERSAL_PATTERNS,
            self.RCE_PATTERNS,
        ]:
            for pattern in pattern_list:
                self.patterns.append(re.compile(pattern, re.IGNORECASE))

    def validate(self, value: Any) -> ValidationResult:
        """
        Check for malicious patterns in the question.

        Args:
            value: The question to validate

        Returns:
            ValidationResult with warnings for suspicious patterns
        """
        str_value = str(value) if value is not None else ""
        warnings = []
        errors = []

        for pattern in self.patterns:
            if pattern.search(str_value):
                warnings.append(f"Suspicious pattern detected: {pattern.pattern}")

        # For highly dangerous patterns, mark as invalid
        dangerous_patterns = [
            "DROP TABLE",
            "DELETE FROM",
            "jndi:",
            "/etc/passwd",
            "<script",
        ]
        for dangerous in dangerous_patterns:
            if dangerous.lower() in str_value.lower():
                errors.append(f"Dangerous pattern detected: {dangerous}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input=str_value,
        )


class PromptInjectionRule(ValidationRule):
    """Validation rule for detecting prompt injection attempts."""

    # Common prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"forget\s+everything",
        r"system\s*:\s*you\s+are\s+now",
        r"admin\s+mode",
        r"developer\s+mode",
        r"override\s+instructions",
        r"reveal\s+secrets?",
        r"show\s+all\s+users",
        r"bypass\s+security",
        r"--\s*admin",
        r"/\*.*admin.*\*/",
    ]

    def __init__(self):
        """Initialize prompt injection rule with compiled patterns."""
        self.patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.INJECTION_PATTERNS
        ]

    def validate(self, value: Any) -> ValidationResult:
        """
        Check for prompt injection patterns in the question.

        Args:
            value: The question to validate

        Returns:
            ValidationResult with warnings for suspicious patterns
        """
        str_value = str(value) if value is not None else ""
        warnings = []

        for pattern in self.patterns:
            if pattern.search(str_value):
                warnings.append(
                    f"Potential prompt injection detected: {pattern.pattern}"
                )

        return ValidationResult(
            is_valid=True,  # Don't block, just warn
            errors=[],
            warnings=warnings,
            sanitized_input=str_value,
        )


class QuestionValidator(BaseValidator):
    """
    Comprehensive validator for user questions.

    This validator combines multiple validation rules and sanitizers to provide
    comprehensive input validation for user questions:

    - Checks for empty questions
    - Validates question length
    - Sanitizes HTML/XSS content
    - Detects malicious patterns
    - Warns about prompt injection attempts
    - Normalizes Unicode and whitespace
    """

    def __init__(
        self,
        max_length: int = 5000,
        enable_html_sanitization: bool = True,
        enable_unicode_normalization: bool = True,
        enable_whitespace_normalization: bool = True,
        enable_malicious_detection: bool = True,
        enable_prompt_injection_detection: bool = True,
    ):
        """
        Initialize question validator with configuration.

        Args:
            max_length: Maximum allowed length for questions
            enable_html_sanitization: Whether to enable HTML sanitization
            enable_unicode_normalization: Whether to enable Unicode normalization
            enable_whitespace_normalization: Whether to enable whitespace normalization
            enable_malicious_detection: Whether to enable malicious pattern detection
            enable_prompt_injection_detection: Whether to enable prompt injection detection
        """
        self.max_length = max_length

        # Initialize validation rules
        self.rules = [
            EmptyQuestionRule(),
            QuestionLengthRule(max_length),
        ]

        if enable_malicious_detection:
            self.rules.append(MaliciousPatternRule())

        if enable_prompt_injection_detection:
            self.rules.append(PromptInjectionRule())

        # Initialize sanitizer pipeline
        sanitizers = []
        if enable_html_sanitization:
            sanitizers.append(HtmlSanitizer())
        if enable_unicode_normalization:
            sanitizers.append(UnicodeSanitizer())
        if enable_whitespace_normalization:
            sanitizers.append(WhitespaceSanitizer())

        self.sanitizer = CompositeSanitizer(sanitizers)

    def validate_input(self, input_data: Any) -> ValidationResult:
        """
        Validate and sanitize a user question.

        Args:
            input_data: The question to validate

        Returns:
            ValidationResult with validation outcome and sanitized question
        """
        # Convert input to string for processing
        if input_data is None:
            str_input = ""
        else:
            str_input = str(input_data)

        # Apply sanitization first
        sanitized = self.sanitizer.sanitize(str_input)

        # Collect all validation results
        all_errors = []
        all_warnings = []
        is_valid = True

        # Apply all validation rules
        for rule in self.rules:
            result = rule.validate(str_input)  # Validate original, not sanitized
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            if not result.is_valid:
                is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            sanitized_input=sanitized,
        )

    @classmethod
    def from_config(cls, config) -> "QuestionValidator":
        """
        Create question validator from configuration.

        Args:
            config: ValidatorConfig instance

        Returns:
            Configured QuestionValidator instance
        """
        return cls(
            max_length=config.question_max_length,
            enable_html_sanitization=config.enable_html_sanitization,
            enable_unicode_normalization=config.enable_unicode_normalization,
            enable_whitespace_normalization=config.enable_whitespace_normalization,
            enable_malicious_detection=config.enable_malicious_pattern_detection,
            enable_prompt_injection_detection=config.enable_prompt_injection_detection,
        )
