"""Select the best symbolic solver for a problem statement.

This module only classifies a problem.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_solver.config import DEFAULT_MODEL_ID

# The selector is intentionally limited to the symbolic solver families that
# the rest of the project currently plans to orchestrate.
SolverName = Literal["asp", "prolog", "minizinc"]

# Test purposes only #
# Tests can inject this callable to avoid loading the real LLM. In production,
# the callable is built from the local Transformers runtime below.
GenerateText = Callable[[str], str]


class ProblemInput(BaseModel):
    """Strict shape expected from files in the `problems/` directory."""

    # Forbid extra fields so input files stay small and unambiguous.
    model_config = ConfigDict(extra="forbid")

    # The complete natural-language problem statement to classify.
    problem: str = Field(min_length=1)


class SolverSelection(BaseModel):
    """Strict JSON object that the LLM must return."""

    # The output contract is intentionally closed: downstream code can rely on
    # exactly these fields and no debug metadata mixed into the result.
    model_config = ConfigDict(extra="forbid")

    # The chosen symbolic solver family.
    solver: SolverName

    # A compact label for the kind of reasoning problem identified.
    problem_type: str = Field(min_length=1)

    # Model self-confidence normalized to the [0, 1] range.
    confidence: float = Field(ge=0.0, le=1.0)

    # Human-readable justification for the solver choice.
    reason: str = Field(min_length=1)


def select_solver_from_file(
    path: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    generator: GenerateText | None = None,
) -> dict[str, Any]:
    """Read a problem JSON file and select the most suitable symbolic solver."""

    # Step 1: load and validate the user-provided problem payload.
    problem_input = _read_problem_input(path)

    return select_solver(problem_input.problem, model_id=model_id, generator=generator)


def select_solver(
    problem: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    generator: GenerateText | None = None,
) -> dict[str, Any]:
    """Select the most suitable symbolic solver for a problem statement."""

    # Step 2: create the model instruction that asks for classification only.
    prompt = build_solver_selection_prompt(problem)

    # Step 3: use an injected generator for tests or build the real local LLM
    # generator lazily so importing this module never loads the model.
    raw_output = (generator or _build_transformers_generator(model_id))(prompt)

    # Step 4: enforce the strict output schema before returning data to callers.
    return parse_solver_selection(raw_output).model_dump()


def build_solver_selection_prompt(problem: str) -> str:
    """Build the instruction prompt for solver selection only."""

    return (
        "You are an agent that selects the most suitable symbolic solver for "
        "a problem, without solving it and without generating code.\n\n"
        "Choose exactly one solver from: asp, prolog, minizinc.\n\n"
        "Guidelines:\n"
        "- minizinc: combinatorial problems, optimization, finite-domain "
        "constraints, scheduling, assignment, constraint satisfaction.\n"
        "- asp: planning, declarative logic, rules, logical constraints, stable "
        "models, search for admissible sets.\n"
        "- prolog: logical inference, queries, relations, symbolic deduction, "
        "knowledge bases.\n\n"
        "Respond only with valid JSON, without Markdown and without extra text. "
        "The JSON must contain exactly these fields: solver, problem_type, "
        "confidence, reason. The solver field must be lowercase.\n\n"
        f"Problem:\n{problem}"
    )


def parse_solver_selection(raw_output: str) -> SolverSelection:
    """Parse and validate the model output as a strict solver selection."""

    # First validate the transport format: the model must return raw JSON text.
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError("Model output is not valid JSON.") from exc

    # Then validate semantic constraints: allowed solver names, confidence
    # bounds, required fields, and no extra keys.
    try:
        return SolverSelection.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Model output does not match SolverSelection schema.") from exc


def _read_problem_input(path: str | Path) -> ProblemInput:
    """Load and validate the JSON problem file."""

    # Read from disk with explicit UTF-8 so problem statements can contain
    # natural-language text consistently across environments.
    try:
        raw_payload = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read problem file: {path}") from exc

    # The file must be JSON before it can be checked against the input schema.
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Problem file is not valid JSON.") from exc

    # Pydantic enforces the exact input contract: only a non-empty `problem`.
    try:
        return ProblemInput.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Problem file must contain only a non-empty 'problem' string.") from exc


@lru_cache(maxsize=2)
def _build_transformers_generator(model_id: str) -> GenerateText:
    """Create a local Hugging Face Transformers text generator for the model."""

    # Keep heavyweight ML imports local so tests and simple imports stay fast.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformers runtime dependencies are required to load the solver selector model."
        ) from exc

    # Load tokenizer and model from the configured Hugging Face model id.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        # Prefer bfloat16 on CUDA to reduce memory use; keep float32 on CPU for
        # broad compatibility.
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    def generate(prompt: str) -> str:
        """Generate the model response for a single solver-selection prompt."""

        # Use the model's chat template so instruction-tuned Qwen variants
        # receive the system and user messages in the format they expect.
        messages = [
            {
                "role": "system",
                "content": "Respond exclusively with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize directly onto the same device as the loaded model.
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Deterministic decoding keeps solver selection repeatable and easier
        # to test. The prompt already asks for a short JSON object.
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Remove the prompt tokens so callers receive only newly generated text.
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return generate
