"""
LLM Evaluation Framework Main Module

This module provides the main evaluation logic and routing for assessing
Large Language Model performance on diverse question types and answer formats.
"""

import re
import json
from typing import List, Optional, Union, Any

import json_repair

try:
    from extract_answer import extract_answer
    from eval_utils import parse_latex_table
    from eval_multiple_choice import option_equal, multi_answers_MCQ
    from eval_nominal import nominal_equal
    from eval_numeral import numeral_equal
    from eval_array import (order_array_equal, unorder_array_equal, 
                           subset_array_equal, order_number_array_equal,
                           unorder_number_array_equal)
except ImportError:
    from .extract_answer import extract_answer
    from .eval_utils import parse_latex_table
    from .eval_multiple_choice import option_equal, multi_answers_MCQ
    from .eval_nominal import nominal_equal
    from .eval_numeral import numeral_equal
    from .eval_array import (order_array_equal, unorder_array_equal, 
                            subset_array_equal, order_number_array_equal,
                            unorder_number_array_equal)


def split_and_strip(text: str) -> List[str]:
    """
    Split and strip text based on common separators.
    
    Args:
        text: Input text to split and strip
        
    Returns:
        List of stripped text parts
    """
    if text.startswith("\\begin{"):
        return parse_latex_table(text)

    # Split using regex pattern for common separators
    parts = re.split(r'[,;、，；和或]', text)
    # If no splitting occurred, try backslash
    if len(parts) <= 1:
        parts = text.split("\\")
    result = [part.strip() for part in parts if part.strip()]
    return result


def is_option(ref: str) -> bool:
    """
    Check if string matches option format patterns.
    
    Supports formats:
    1. (A), (B), etc.
    2. A), B), etc.
    3. A., B., etc.
    4. A:, B:, etc.
    5. Single letter A-Z
    
    Args:
        ref: String to check
        
    Returns:
        True if string matches option format, False otherwise
    """
    if len(ref) == 1 and ref.isalpha():
        return True
    
    if len(ref) > 1:
        if ref[0] == '(' and ref[1].isalpha() and len(ref) > 2 and ref[2] == ")":
            return True
        if ref[0].isalpha() and ref[1] in '.:)':
            return True
    
    return False


def is_valid_answer_string(s: str) -> bool:
    """
    Check if string contains only valid answer characters.
    
    Args:
        s: String to validate
        
    Returns:
        True if string contains only numbers, letters, or Chinese characters
    """
    # Pattern matches numbers, English letters, or Chinese characters
    pattern = r'^[0-9a-zA-Z\u4e00-\u9fa5]+$'
    return bool(re.match(pattern, s))


def is_multiple_query(ref: str) -> Optional[List[str]]:
    """
    Check if string represents multiple queries and split them.
    
    Supports formats:
    1. A\nB\nC
    2. A B C
    3. A, B, C
    4. 1. A, 2. B, 3. C
    
    Args:
        ref: Reference string to check
        
    Returns:
        List of split queries if multiple query format detected, None otherwise
    """
    separator = ['\n', ' ', ',', ';']

    for si in separator:
        if si in ref:
            split_res = [si.strip() for si in ref.split(si)]
            if all([is_valid_answer_string(si) for si in split_res]):
                return split_res
    return None


def parseArray(ref: str, pre: str) -> tuple[List[Any], List[Any]]:
    """
    Parse array strings into structured data.
    
    Args:
        ref: Reference array string
        pre: Prediction array string
        
    Returns:
        Tuple of (reference_data, prediction_data) as lists
    """
    try:
        ref_t = json.loads(ref.replace("\\", ""))
        pre_t = json.loads(pre.replace("\\", ""))
    except (SyntaxError, ValueError):
        try:
            ref_t = json_repair.loads(ref.replace("\\", ""))
            if pre.startswith("\\begin{"):
                pre_t = parse_latex_table(pre)
            else:
                pre_t = json_repair.loads(pre.replace("\\", ""))
        except json.JSONDecodeError:
            return [], []
    if type(pre_t) == dict:
        pre_t = list(pre_t.values())
    return ref_t, pre_t


def eval_router(pre: str, ref: str, eval_type: Optional[str] = None) -> bool:
    """
    Route evaluation to appropriate evaluator based on type.
    
    Args:
        pre: Predicted answer
        ref: Reference (ground truth) answer
        eval_type: Evaluation type. Supported types:
            - "nominal": Text-based answers
            - "numeral": Numerical computations
            - "option": Multiple choice options
            - "ordered array": Sequence-sensitive lists
            - "unordered array": Order-independent sets
            - "subset": Subset relationship validation
            - "multi_answers_MCQ": Multiple correct options
            - "ooa_numeral": Ordered outer, ordered inner arrays (numerical)
            - "oua_nominal": Ordered outer, unordered inner arrays (nominal)
            - "uoa_nominal": Unordered outer, ordered inner arrays (nominal)
            
    Returns:
        True if prediction matches reference according to eval_type, False otherwise
    """
    if not pre or not ref:
        return False

    pre = pre.strip()
    ref = ref.strip()
    
    if eval_type:
        if eval_type == "option":
            return option_equal(pre, ref)
        if eval_type == "nominal":
            return nominal_equal(pre, ref) 
        if eval_type == "numeral":
            return numeral_equal(pre, ref)
        if eval_type == "multi_answers_MCQ":
            return multi_answers_MCQ(pre, ref)
        if eval_type.startswith("order"):
            ref = split_and_strip(ref)
            pre = split_and_strip(pre)
            return order_array_equal(pre, ref)
        if eval_type.startswith("unorder"):
            ref = split_and_strip(ref)
            pre = split_and_strip(pre)
            return unorder_array_equal(pre, ref)
        if eval_type.startswith("subset"):
            ref = split_and_strip(ref)
            pre = split_and_strip(pre)
            return subset_array_equal(pre, ref)
        if eval_type.startswith("ooa"):
            ref_t, pre_t = parseArray(ref, pre)
            list_sum = []
            if len(ref_t) != len(pre_t):
                return False
            if eval_type.endswith("nominal"):
                for index, _ in enumerate(ref_t):
                    list_sum.append(order_array_equal(pre_t[index], ref_t[index]))
                return all(list_sum)
            elif eval_type.endswith("numeral"):
                for index, _ in enumerate(ref_t):
                    list_sum.append(order_number_array_equal(pre_t[index], ref_t[index]))
                return all(list_sum)
        if eval_type.startswith("oua"):
            ref_t, pre_t = parseArray(ref, pre)
            list_sum = []
            if len(ref_t) != len(pre_t):
                return False
            if eval_type.endswith("nominal"):
                for index, _ in enumerate(ref_t):
                    list_sum.append(unorder_array_equal(pre_t[index], ref_t[index]))
                return all(list_sum)
            elif eval_type.endswith("numeral"):
                for index, _ in enumerate(ref_t):
                    list_sum.append(unorder_number_array_equal(pre_t[index], ref_t[index]))
                return all(list_sum)
        if eval_type.startswith("uoa"):
            ref_t, pre_t = parseArray(ref, pre)
            list_sum = []
            if len(ref_t) != len(pre_t):
                return False
            if eval_type.endswith("nominal"):
                for index, _ in enumerate(ref_t):
                    if not any(ref_t[index] == pi for pi in pre_t):
                        return False
                    else:
                        for pi in pre_t:
                            if ref_t[index] == pi:
                                list_sum.append(order_array_equal(pi, ref_t[index]))
                            else:
                                continue
                return all(list_sum)
            elif eval_type.endswith("numeral"):
                for index, _ in enumerate(ref_t):
                    if not any(ref_t[index] == pi for pi in pre_t):
                        return False
                    else:
                        for pi in pre_t:
                            if ref_t[index] == pi:
                                list_sum.append(order_number_array_equal(pi, ref_t[index]))
                            else:
                                continue
                return all(list_sum)

    # Auto-detection based on format
    if ref.startswith('[') and ref.endswith(']'):
        ref = json_repair.loads(ref)
        pre = split_and_strip(pre)
        return unorder_array_equal(pre, ref)

    if is_option(ref):
        return option_equal(pre, ref)

    ref_res = is_multiple_query(ref)
    if ref_res is not None:
        ref = ref_res
        pre = split_and_strip(pre)
        return order_array_equal(pre, ref)

    # Default to numerical evaluation
    return numeral_equal(pre, ref)


def get_n_elements(arr: List[Any], n: int, last: bool = True, fill_value: Any = None) -> List[Any]:
    """
    Get the last or first n elements from array, padding with fill_value if needed.
    
    Args:
        arr: Input array
        n: Number of elements to retrieve
        last: If True, get last n elements; if False, get first n elements
        fill_value: Value to pad with if array has fewer than n elements
        
    Returns:
        List of n elements (may include fill_value for padding)
    """
    if last:
        last_n = arr[-n:]  # Get last n
    else:
        last_n = arr[:n]  # Get first n
    
    # Pad with fill_value if needed
    if len(last_n) < n:
        last_n = last_n + [fill_value] * (n - len(last_n))
    return last_n


def compute_score(solution_str: str, ground_truth: str, eval_type_text: str):
    """
    Compute score for RL training.
    
    Args:
        solution_str: Complete model output with potential multiple answers
        ground_truth: Ground truth with potential multiple parts
        eval_type_text: Comma-separated evaluation types
        
    Returns:
        Float score between 0.0 and 1.0 (used as RL reward)
    """
    if "boxed{" in ground_truth:
        ground_truth = extract_answer(ground_truth.split("/think>")[-1].strip())[-1]

    # Calculate number of questions based on eval types
    eval_type = re.split(r'[,，]', eval_type_text)
    qtype_len = len(eval_type)

    try:
        ref_list = [ground_truth]
        if qtype_len > 1:
            separators = ["====", ";", "；", "\n"]  # Used to split ground truth
            for sep in separators:
                if sep in ground_truth:
                    ref_list = ground_truth.strip().split(sep)
                    break
        ref_list = get_n_elements(ref_list, qtype_len)

        pred_res = extract_answer(solution_str.split("/think>")[-1].strip())  # Model answers
        if len(ref_list) > len(pred_res):
            separators = ",，;； "  # Used to split predictions
            tmp = []
            for pi in pred_res:
                append_flag = True
                for sep in separators:
                    if sep in pi:
                        tmp.extend([x for x in pi.split(sep) if x])
                        append_flag = False
                        break
                if append_flag:
                    tmp.append(pi)
            pred_res = tmp

        pred_list = get_n_elements(pred_res, qtype_len)  # Get last qtype_len answers
        check_res = [eval_router(pred_list[ref_i], ref_v, eval_type[ref_i]) for ref_i, ref_v in enumerate(ref_list)]
        true_num = sum(check_res)
        
        true_num2 = 0
        if true_num == 0 and qtype_len == 1 and len(pred_res) > 1:  # If single question but multiple predictions
            pred_list = [";".join(pred_res)]
            check_res = eval_router(pred_list[0], ref_list[0], eval_type[0])
            true_num2 = sum([check_res])

        true_num3 = 0
        if true_num2 == 0:
            pred_list = get_n_elements(pred_res, qtype_len, False)  # Get first qtype_len answers
            check_res = [eval_router(pred_list[ref_i], ref_v, eval_type[ref_i]) for ref_i, ref_v in enumerate(ref_list)]
            true_num3 = sum(check_res)
        res_score = max(true_num, true_num2, true_num3) / qtype_len

    except Exception as e:
        res_score = 0.0

    return res_score


def evaluation(prediction: str, ground_truth: str, eval_type_text: str) -> bool:
    """
    Evaluate LLM prediction with strict binary result.
    
    Args:
        prediction: Model's prediction
        ground_truth: Ground truth answer
        eval_type_text: Evaluation type specification
        
    Returns:
        True if all parts are correct, False otherwise
    """
    return False if compute_score(prediction, ground_truth, eval_type_text) < 1 else True


if __name__ == "__main__":
    import time
    start_time = time.time()

    print(eval_router("[['a','b'], ['c', 'd'], ['d', 'e']]", "[['b','a'], ['d', 'c'],['d', 'e']]", eval_type="ooa_nominal"))

    solution_str = "xixihh\\boxed{[[\"6\", \"7\"], [\"8\", \"9\"], [\"7\", \"9\"], [\"6\", \"9\"]]} \\boxed{A}\n$$"
    ground_truth = "[['6', '7'], ['8', '9'], ['7', '9'], ['6', '9']]====A"
    eval_type_text = "ooa_numeral,option"

    print(compute_score(solution_str, ground_truth, eval_type_text))

    solution_str = "\\boxed{A, 12.5}\n$$"
    ground_truth = "A====125"
    eval_type_text = "option,nominal"
    print(compute_score(solution_str, ground_truth, eval_type_text))

    pre = "\\boxed{老15，老二}"
    ref = "2，15"
    print(compute_score(pre, ref, 'ordered array'))
    print(compute_score(pre, ref, 'unordered array'))

    print("--------------------")
    pre = "\\boxed{护理学,经济学,音乐学,化学,土木工程}"
    ref = "化学,音乐学,土木工程,经济学,护理学\n"
    print(compute_score(pre, ref, 'ordered array'))  # false
    print(compute_score(pre, ref, 'unordered array'))  # true

    print("--------------------")
    pre = "sdfsdf\\boxed{王桂芝};\\boxed{张凯};\\boxed{张俊};\\boxed{张玲};sdfsdf"
    ref = "王桂芝,张凯,张玲,张俊\n"
    print(compute_score(pre, ref, 'ordered array'))  # false
    print(compute_score(pre, ref, 'unordered array'))  # true

    pre = "sdfsdf\\boxed{[[55, '刘莹', 1.602], [52, '何丽', 18], [53, '雷刚', 19], [54, '许婷婷', 17]]};sdfsdf"
    ref = "[[55, '刘莹', 1.60], [52, '何丽', 18], [53, '雷刚', 19], [54, '许婷婷', 17]]\n"
    print(compute_score(pre, ref, 'ooa_nominal'))  # false

    pre = "sdfsdf\\boxed{[[\"仓鼠\"],[\"仓鼠\", \"蝾螈\"],[\"仓鼠\", \"蝾螈\"],[\"仓鼠\"],[\"仓鼠\", \"蝾螈\"],[\"仓鼠\", \"乌龟\"],[\"蝾螈\", \"变色龙\"],[\"仓鼠\", \"蝾螈\"],[\"变色龙\"]]};sdfsdf"
    ref = "[['仓鼠'], ['仓鼠', '蝾螈'], ['仓鼠', '蝾螈'], ['仓鼠'], ['仓鼠', '蝾螈'], ['仓鼠', '乌龟'], ['蝾螈', '变色龙'], ['仓鼠', '蝾螈'], ['变色龙']]\n"
    print(compute_score(pre, ref, 'oua_nominal'))  # true
    
    execution_time = time.time() - start_time
    print(f"Code execution time: {execution_time:.6f} seconds")