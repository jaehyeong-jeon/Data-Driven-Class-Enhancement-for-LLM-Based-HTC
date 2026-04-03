"""Shim: re-exports prompt functions from the unified prompts/ package."""
from src.prompts import (
    create_ranker_prompts,
    create_pointwise_yn_prompt,
    create_pointwise_final_prompt,
)

__all__ = [
    "create_ranker_prompts",
    "create_pointwise_yn_prompt",
    "create_pointwise_final_prompt",
]
