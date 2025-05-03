#!/usr/bin/env python3
"""
Pipeline for processing JSONL data using JSONFlow based on deepmath mapping.
This script reads raw JSONL input, maps fields, renders a prompt using a template,
and outputs a final JSONL according to the final template mapping.
"""
import os
import sys
import argparse

# Setup JSONFlow import path: add local build/lib if present
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(script_dir, 'jsonflow', 'build', 'lib')
if os.path.isdir(lib_path):
    sys.path.insert(0, lib_path)
from jsonflow.core import Pipeline, JsonOperator
from jsonflow.operators.json_ops import JsonFieldMapper
from jsonflow.io import JsonLoader, JsonSaver

DEEP_MATH_MAPPING = {
    "question": "problem",
    "final_answer": "reference",
    "r1_solution_1": "solution",
}
PROMPT_TEMPLATE = r"""
You are to act as an expert to score the solution of the problem. Please verify the user's solution process step-by-step based on the provided reference s\
olution, evaluate the correctness of its key steps, and provide a score (number of correct key steps / total number of key steps in the reference solution\
). If the reference solution is not provided, you should solve the problem first and then use your solution as the reference solution.

#### Definition of Key Steps ####

- Indispensability: If this step is skipped, subsequent derivations cannot be completed or logical gaps will occur.

- Logical Necessity: Directly impacts the final result through derivations, theorem applications, formula transformations, or core decisions.

- Domain Dependency:

- Math/Physics: Formula substitution, equation setup, theorem references (e.g., "applying the Pythagorean theorem").

- Programming: Algorithm selection (e.g., "using dynamic programming"), handling boundary conditions, critical loop/recursive logic.

- Language-based: Proposing core arguments, logical transitions in reasoning.

- Non-Key Steps: Data substitution, or text polishing that does not involve core logic.

#### Verification Process ####

Analyze the Reference Solution:

- Extract key steps from the reference solution and number them sequentially (e.g., KS1, KS2?).

- Label each step with its "indispensability" and "logical objective" (e.g., KS3: Apply the cosine theorem to calculate side length).

Compare with the User's Solution:

- Check if the user's solution includes operations logically equivalent to the reference key steps.

- Allow for differences in expression, but ensure:

1. The same principles/methods are correctly applied.

2. The same sub-problems are addressed (e.g., "taking derivatives" vs. "computing gradients").

- If the order of steps differs but the logic is consistent, verify whether it affects the final result.

#### Error Classification ####

- Critical Errors: Omission or incorrect execution of key steps (e.g., failing to check matrix invertibility) or any computational error which leads to in\
correct final results.

- Non-Critical Errors: Syntax Errors or spelling mistakes that don't affect the core logic.

#### Scoring Output ####

Score = Number of correct key steps / Total number of key steps in the reference solution (e.g., 3/5 ? 0.6).

Extra correct steps in the solution process do not deduct points but also do not add extra credit.

You must put the final score in between the special tokens: <score>...</score>.

**Example1

[Begin of the problem]$441+2(21)(19)+361=x$. Solve for $x$.[End of the problem]

[Begin of the user's solution]I want to solve for x in the equation $441+2(21)(19)+361=x$.\nI can simplify the equation by first calculating $2(21)(19)$, \
which is equal to $798$.\n?[????????????]?\n<score>2/3</score>
"""

FINAL_JSON_TEMPLATE = {
    "query": "prompt",
    "top_p": "0.01",
    "temperature": "0.6",
    "max_output_tokens": "4096",
}

class PromptOperator(JsonOperator):  # noqa: D102
    """
    Operator to render a prompt template with provided fields.
    Replaces {{problem}}, {{reference}}, {{solution}} in the template.
    """
    def __init__(self, template_str: str):
        super().__init__(name="PromptOperator", description="Render prompt template")
        self.template = template_str

    def process(self, data: dict) -> dict:
        result = data.copy()
        # simple placeholder replacement
        prompt = (
            self.template
            .replace("{{problem}}", str(data.get("problem", "")))
            .replace("{{reference}}", str(data.get("reference", "")))
            .replace("{{solution}}", str(data.get("solution", "")))
        )
        result["prompt"] = prompt
        return result

def build_field_mapper(deep_map: dict) -> JsonFieldMapper:
    """
    Build a JsonFieldMapper from deepmath mapping.
    Input mapping: source_field -> target_var
    JsonFieldMapper expects target_field -> source_path
    """
    # invert mapping: source->target to target->source
    inv = {target: source for source, target in deep_map.items()}
    return JsonFieldMapper(inv)

class FinalMapper(JsonOperator):  # noqa: D102
    """
    Operator to map processed data to final JSON template.
    Uses mapping of output key -> source key or literal value.
    """
    def __init__(self, mapping: dict):
        super().__init__(name="FinalMapper", description="Final JSON mapping operator")
        self.mapping = mapping

    def process(self, data: dict) -> dict:
        result = {}
        for out_key, val in self.mapping.items():
            # direct mapping from data if key exists
            if val in data:
                result[out_key] = data.get(val)
            else:
                # attempt numeric literal
                try:
                    if isinstance(val, str) and val.isdigit():
                        result[out_key] = int(val)
                    else:
                        result[out_key] = float(val)
                except Exception:
                    # fallback to raw string
                    result[out_key] = val
        return result


def main():  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Process JSONL input using a JSONFlow pipeline."
    )
    parser.add_argument('-i', '--input',  required=True, help="Input JSONL file path")
    parser.add_argument('-o', '--output', required=True, help="Output JSONL file path")
    args = parser.parse_args()

    field_mapper = build_field_mapper(DEEP_MATH_MAPPING)
    prompt_op     = PromptOperator(PROMPT_TEMPLATE)
    final_op      = FinalMapper(FINAL_JSON_TEMPLATE)

    # create pipeline
    pipeline = Pipeline([field_mapper, prompt_op, final_op])

    # load input JSONL
    loader = JsonLoader(args.input)
    items = loader.load()

    # process each item
    results = [pipeline.process(item) for item in items]

    # save to output JSONL
    saver = JsonSaver(args.output)
    saver.write_all(results)
