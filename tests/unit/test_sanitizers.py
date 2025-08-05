"""Tests for input sanitization utilities."""


class TestHtmlSanitizer:
    """Test suite for HTML/script tag sanitization."""

    def test_should_remove_script_tags(self):
        """Should remove script tags and their content."""
        from src.agent.validators.sanitizers import HtmlSanitizer

        # Arrange
        sanitizer = HtmlSanitizer()
        input_data = "Hello <script>alert('xss')</script> world"

        # Act
        result = sanitizer.sanitize(input_data)

        # Assert
        assert result == "Hello  world"

    def test_should_remove_multiple_script_tags(self):
        """Should remove multiple script tags."""
        from src.agent.validators.sanitizers import HtmlSanitizer

        # Arrange
        sanitizer = HtmlSanitizer()
        input_data = "<script>bad1()</script>Text<script>bad2()</script>"

        # Act
        result = sanitizer.sanitize(input_data)

        # Assert
        assert result == "Text"

    def test_should_remove_html_tags(self):
        """Should remove common HTML tags."""
        from src.agent.validators.sanitizers import HtmlSanitizer

        # Arrange
        sanitizer = HtmlSanitizer()
        test_cases = [
            ("Hello <b>world</b>", "Hello world"),
            ("Click <a href='#'>here</a>", "Click here"),
            ("<div>Content</div>", "Content"),
            ("<img src='x' onerror='alert(1)'>", ""),
        ]

        # Act & Assert
        for input_data, expected in test_cases:
            result = sanitizer.sanitize(input_data)
            assert result == expected

    def test_should_handle_malformed_html(self):
        """Should handle malformed HTML gracefully."""
        from src.agent.validators.sanitizers import HtmlSanitizer

        # Arrange
        sanitizer = HtmlSanitizer()
        test_cases = [
            ("Hello <b world", "Hello "),  # Malformed tag should be removed
            ("Hello <script world", "Hello "),  # Malformed script should be removed
            (
                "Hello < world",
                "Hello < world",
            ),  # Lone < without tag structure preserved
        ]

        # Act & Assert
        for input_data, expected in test_cases:
            result = sanitizer.sanitize(input_data)
            assert result == expected

    def test_should_preserve_text_content(self):
        """Should preserve regular text content."""
        from src.agent.validators.sanitizers import HtmlSanitizer

        # Arrange
        sanitizer = HtmlSanitizer()
        input_data = "This is normal text with no HTML"

        # Act
        result = sanitizer.sanitize(input_data)

        # Assert
        assert result == input_data


class TestUnicodeSanitizer:
    """Test suite for Unicode normalization."""

    def test_should_normalize_unicode_to_nfc(self):
        """Should normalize Unicode to NFC form."""
        from src.agent.validators.sanitizers import UnicodeSanitizer

        # Arrange
        sanitizer = UnicodeSanitizer()
        # Using combining characters: é as e + combining acute accent
        input_data = "caf\u0065\u0301"  # café with combining accent

        # Act
        result = sanitizer.sanitize(input_data)

        # Assert
        assert result == "café"  # Normalized to single character

    def test_should_remove_control_characters(self):
        """Should remove control characters."""
        from src.agent.validators.sanitizers import UnicodeSanitizer

        # Arrange
        sanitizer = UnicodeSanitizer()
        input_data = "Hello\x00\x01\x02World\x7f"

        # Act
        result = sanitizer.sanitize(input_data)

        # Assert
        assert result == "HelloWorld"

    def test_should_preserve_printable_characters(self):
        """Should preserve printable characters."""
        from src.agent.validators.sanitizers import UnicodeSanitizer

        # Arrange
        sanitizer = UnicodeSanitizer()
        input_data = "Hello 世界! 123 @#$%"

        # Act
        result = sanitizer.sanitize(input_data)

        # Assert
        assert result == input_data


class TestWhitespaceSanitizer:
    """Test suite for whitespace normalization."""

    def test_should_normalize_whitespace(self):
        """Should normalize multiple whitespace to single space."""
        from src.agent.validators.sanitizers import WhitespaceSanitizer

        # Arrange
        sanitizer = WhitespaceSanitizer()
        test_cases = [
            ("Hello    world", "Hello world"),
            ("Hello\t\tworld", "Hello world"),
            ("Hello\n\nworld", "Hello world"),
            ("Hello\r\nworld", "Hello world"),
            ("  Hello   world  ", "Hello world"),
        ]

        # Act & Assert
        for input_data, expected in test_cases:
            result = sanitizer.sanitize(input_data)
            assert result == expected

    def test_should_strip_leading_trailing_whitespace(self):
        """Should strip leading and trailing whitespace."""
        from src.agent.validators.sanitizers import WhitespaceSanitizer

        # Arrange
        sanitizer = WhitespaceSanitizer()
        input_data = "   Hello world   "

        # Act
        result = sanitizer.sanitize(input_data)

        # Assert
        assert result == "Hello world"

    def test_should_handle_empty_and_whitespace_only(self):
        """Should handle empty and whitespace-only strings."""
        from src.agent.validators.sanitizers import WhitespaceSanitizer

        # Arrange
        sanitizer = WhitespaceSanitizer()
        test_cases = [
            ("", ""),
            ("   ", ""),
            ("\t\n\r", ""),
        ]

        # Act & Assert
        for input_data, expected in test_cases:
            result = sanitizer.sanitize(input_data)
            assert result == expected


class TestCompositeSanitizer:
    """Test suite for composite sanitizer that combines multiple sanitizers."""

    def test_should_apply_multiple_sanitizers_in_order(self):
        """Should apply multiple sanitizers in the specified order."""
        from src.agent.validators.sanitizers import (
            CompositeSanitizer,
            HtmlSanitizer,
            WhitespaceSanitizer,
        )

        # Arrange
        sanitizers = [HtmlSanitizer(), WhitespaceSanitizer()]
        composite = CompositeSanitizer(sanitizers)
        input_data = "  Hello <b>world</b>   <script>alert('xss')</script>  "

        # Act
        result = composite.sanitize(input_data)

        # Assert
        assert result == "Hello world"

    def test_should_handle_empty_sanitizer_list(self):
        """Should handle empty sanitizer list by returning input unchanged."""
        from src.agent.validators.sanitizers import CompositeSanitizer

        # Arrange
        composite = CompositeSanitizer([])
        input_data = "Hello world"

        # Act
        result = composite.sanitize(input_data)

        # Assert
        assert result == input_data

    def test_should_chain_sanitization_results(self):
        """Should chain the results of sanitization through each sanitizer."""
        from src.agent.validators.sanitizers import (
            CompositeSanitizer,
            HtmlSanitizer,
            UnicodeSanitizer,
            WhitespaceSanitizer,
        )

        # Arrange
        sanitizers = [HtmlSanitizer(), UnicodeSanitizer(), WhitespaceSanitizer()]
        composite = CompositeSanitizer(sanitizers)
        input_data = "  <b>Hello\x00 world</b>  "

        # Act
        result = composite.sanitize(input_data)

        # Assert
        assert result == "Hello world"


class TestSqlInjectionSanitizer:
    """Test suite for SQL injection prevention sanitization."""

    def test_should_escape_single_quotes(self):
        """Should escape single quotes to prevent SQL injection."""
        from src.agent.validators.sanitizers import SqlInjectionSanitizer

        # Arrange
        sanitizer = SqlInjectionSanitizer()
        input_data = "Robert'; DROP TABLE users;--"

        # Act
        result = sanitizer.sanitize(input_data)

        # Assert
        assert "''" in result  # Single quotes should be escaped
        assert "DROP" not in result.upper() or result.count("'") > input_data.count("'")

    def test_should_handle_common_injection_patterns(self):
        """Should sanitize common SQL injection patterns."""
        from src.agent.validators.sanitizers import SqlInjectionSanitizer

        # Arrange
        sanitizer = SqlInjectionSanitizer()
        test_cases = [
            "' OR '1'='1",
            "'; DELETE FROM users; --",
            "' UNION SELECT * FROM passwords --",
        ]

        # Act & Assert
        for input_data in test_cases:
            result = sanitizer.sanitize(input_data)
            # Should not contain dangerous patterns after sanitization
            assert result != input_data  # Should be modified
            assert "''" in result or "'" not in result  # Quotes handled
