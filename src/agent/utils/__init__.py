"""
Utilities package for the agentic AI framework.

This package contains various utility modules including:
- retry: Retry logic with exponential backoff for error recovery
"""

from typing import Any

from jinja2 import StrictUndefined, Template


def populate_template(template: str, variables: dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(
            f"Error during jinja template rendering: {type(e).__name__}: {e}"
        )


__all__ = ["populate_template"]
