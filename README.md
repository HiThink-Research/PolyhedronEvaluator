# LLM Evaluation Framework

This directory contains a comprehensive evaluation framework for assessing Large Language Model (LLM) performance on diverse question types and answer formats. The system supports multiple data types and comparison methods to ensure accurate evaluation across various problem domains.

## Overview

The evaluation framework is designed to handle complex answer formats including:

- Multiple choice questions with single or multiple correct answers
- Numerical computations and mathematical expressions
- Text/nominal answers with normalization
- Ordered and unordered arrays
- Multi-dimensional data structures
- Mixed-type responses
- LaTeX mathematical notation

## Key Features

- **🔍 Flexible Answer Extraction**: Extracts answers from LaTeX `\boxed{}` notation and various text formats
- **🎯 Multi-Type Evaluation**: Supports different evaluation strategies based on answer type with automatic detection
- **🛠️ Robust Parsing**: Handles JSON, LaTeX tables, malformed data, and mixed text formats
- **📊 Score Computation**: Provides both binary evaluation and partial scoring for multi-part questions
- **🌍 Language Support**: Handles both English and Chinese text, including number word conversion
- **⚡ Performance Optimized**: Includes timeout handling and concurrent evaluation for large-scale assessments

## Installation

```bash
git clone https://github.com/HiThink-Research/PolyhedronEvaluator.git
pip install -r requirements.txt
```

## Quick Start

### Basic Evaluation

```python
from main import compute_score, evaluation

# Simple evaluation for binary correctness
prediction = "\\boxed{A}"
reference = "A"
result = evaluation(prediction, reference, "option")
print(result)  # True

# Numerical evaluation
result = evaluation("\\boxed{25}", "twenty-five", "numeral")
print(result)  # True

# Array evaluation
pred_array = "\\boxed{['apple', 'banana', 'cherry']}"
ref_array = "['apple', 'banana', 'cherry']"
result = evaluation(pred_array, ref_array, "ordered array")
print(result)  # True
```

### Score Computation for RL Training

```python
# Multi-part question scoring
solution = "\\boxed{A}\\boxed{100}"
ground_truth = "A====125"
eval_types = "option,numeral"
score = compute_score(solution, ground_truth, eval_types)
print(f"Score: {score}")  # 0.5 (1 answer out of 2 is correct)
```

### Advanced Examples

```python
# Complex array evaluation
solution_str = "\\boxed{[['a', 'b'], ['c', 'd']]}"
ground_truth = "[['b', 'a'], ['d', 'c']]"
score = compute_score(solution_str, ground_truth, "oua_nominal")
print(f"Ordered outer, unordered inner array score: {score}") # 1.0

# Multiple choice with multiple answers
result = evaluation("\\boxed{A, B, C}", "ABC", "multi_answers_MCQ")
print(result)  # True
```

## Evaluation Types

### Basic Types

- `"option"`: Multiple choice answers (A, B, C, etc.)
- `"nominal"`: Text-based answers with normalization
- `"numeral"`: Numerical computations and expressions

### Array Types

- `"ordered array"`: Sequence-sensitive lists
- `"unordered array"`: Order-independent sets
- `"subset"`: Subset relationship validation

### Advanced Types

- `"multi_answers_MCQ"`: Multiple correct options in multiple choice
- `"ooa_nominal"`: Ordered outer, ordered inner arrays (nominal data)
- `"oua_numeral"`: Ordered outer, unordered inner arrays (numerical data)
- `"uoa_nominal"`: Unordered outer, ordered inner arrays
- And other complex multi-dimensional combinations

## Project Structure

```
PolyhedronEvaluator/
├── main.py                    # Main evaluation logic and routing
├── extract_answer.py          # Answer extraction from model outputs
├── eval_multiple_choice.py    # Multiple choice question evaluation
├── eval_nominal.py           # Text/name-based answer evaluation
├── eval_numeral.py           # Numerical answer evaluation with SymPy
├── eval_array.py             # Array and sequence evaluation
├── eval_utils.py             # Utility functions and text processing
└── README.md                 # This documentation
```

## Core Functions

### `compute_score(solution_str, ground_truth, eval_type_text)`

Scoring function for reinforcement learning training with partial credit support.

**Parameters:**

- `solution_str`: Complete model output with potential multiple answers
- `ground_truth`: Ground truth with potential multiple parts
- `eval_type_text`: Comma-separated evaluation types

**Returns:** Float score between 0.0 and 1.0 (suitable as RL reward)

### `evaluation(prediction, ground_truth, eval_type_text)`

Strict binary evaluation for model inference assessment.

**Returns:** True if all parts are correct, False otherwise

## Text Processing Features

- **📝 Number Word Conversion**: Converts English ("twenty-five") and Chinese ("二十五") number words to digits
- **🔧 Punctuation Normalization**: Removes and normalizes punctuation across languages
- **📊 LaTeX Table Parsing**: Extracts structured data from LaTeX table formats
- **🛠️ JSON Repair**: Handles malformed JSON with automatic correction
- **🧮 Mathematical Expression Handling**: Processes complex mathematical notation

## Supported Input Formats

- Plain text answers
- LaTeX mathematical expressions with `\boxed{}` notation
- JSON arrays and objects
- LaTeX tables (`\begin{array}`, `\begin{bmatrix}`)
- Mixed format responses with multiple `\boxed{}` sections
- Comma/semicolon separated lists
- Multi-line responses

## Error Handling

The system includes robust error handling for:

- ✅ Malformed JSON structures
- ✅ Invalid LaTeX expressions
- ✅ Missing or incomplete answers
- ✅ Type conversion errors
- ✅ Parsing failures
- ✅ Timeout protection for complex symbolic computation

All evaluation functions return appropriate defaults (False/0.0) when errors occur, ensuring the evaluation pipeline continues running.

## Performance Considerations

- **⏱️ Timeout Protection**: Symbolic computations are protected by timeouts to prevent hanging
- **🔄 Concurrent Processing**: Uses ThreadPoolExecutor for performance-critical operations
- **🎯 Efficient Parsing**: Optimized parsing strategies for different data formats
- **📊 Batch Processing**: Designed to handle large-scale evaluation efficiently

## Contributing

When adding new evaluation types:

1. Implement the evaluation logic in the appropriate `eval_*.py` file
2. Add the type routing in `main.py`'s internal routing function
3. Update the documentation with the new type
4. Add comprehensive test cases to verify functionality
5. Ensure proper error handling and edge case coverage

## License

This evaluation framework is part of the [PuzzleClone](https://github.com/HiThink-Research/PuzzleClone) project and is licensed under the Apache 2.0 License.

## Acknowledgments

Our evaluation code is partially modified from [LIMO](https://github.com/GAIR-NLP/LIMO). We thank their team for their valuable contributions to the community.
