"""
Array Evaluation Module

This module provides functionality for evaluating array-type answers
with support for ordered, unordered, and subset comparisons.
"""

from typing import List
from functools import lru_cache

try:
    from eval_numeral import is_digit, numeral_equal
    from eval_nominal import nominal_equal
except ImportError:
    from .eval_numeral import is_digit, numeral_equal
    from .eval_nominal import nominal_equal


@lru_cache(maxsize=10000)
def is_sub_str(r, p):
    return r.strip().lower() in p.strip().lower()


def array_element_equal(pi, ri):
    if nominal_equal(pi, ri):
        return True
    if is_digit(ri):
        if numeral_equal(pi, ri):
            return True
    else:
        if is_sub_str(str(ri), str(pi)):
            return True
    return False


def unorder_array_equal(pre: list[str], ref: list[str]):
    if len(pre) != len(ref):
        return False

    ref = ref.copy()  # 防止.remove()改变lru_cache的结果
    for pi in pre:
        find_flag = False
        for ri in ref:
            if array_element_equal(pi, ri):
                find_flag = True
                ref.remove(ri)
                break
        if not find_flag:
            return False

    return True


def unorder_array_equal_soft(pre: list[str], ref: list[str]):
    n_correct = 0
    n_ref = len(ref)
    ref = ref.copy()  # 防止.remove()改变lru_cache的结果
    for pi in pre:
        for ri in ref:
            if array_element_equal(pi, ri):
                n_correct += 1
                ref.remove(ri)
                break
    return n_correct / max(len(pre), n_ref, 1)


def order_array_equal(pre: list[str], ref: list[str]):
    return len(pre) == len(ref) and all(array_element_equal(pi, ri) for pi, ri in zip(pre, ref))


def order_array_equal_soft(pre: list[str], ref: list[str]):
    n_correct = sum(array_element_equal(pi, ri) for pi, ri in zip(pre, ref))
    return n_correct / max(len(pre), len(ref), 1)


def order_number_array_equal(pre: list[str], ref: list[str]):
    return len(pre) == len(ref) and all(float(ri) - float(pi) == 0 for pi, ri in zip(pre, ref))


def order_number_array_equal_soft(pre: list[str], ref: list[str]):
    n_correct = sum(float(ri) - float(pi) == 0 for pi, ri in zip(pre, ref))
    return n_correct / max(len(pre), len(ref), 1)


def unorder_number_array_equal(pre: list[str], ref: list[str]):
    if len(pre) < len(ref):
        return False

    ref = ref.copy()  # 防止.remove()改变lru_cache的结果
    for pi in pre:
        find_flag = False
        for ri in ref:
            if float(ri) - float(pi) == 0:
                find_flag = True
                ref.remove(ri)
                break
        if not find_flag:
            return False

    return True


def unorder_number_array_equal_soft(pre: list[str], ref: list[str]):
    n_correct = 0
    n_ref = len(ref)
    ref = ref.copy()  # 防止.remove()改变lru_cache的结果
    for pi in pre:
        for ri in ref:
            if float(ri) - float(pi) == 0:
                n_correct += 1
                ref.remove(ri)
                break
    return n_correct / max(len(pre), n_ref, 1)


def subset_array_equal(pre: List[str], ref: List[str]) -> bool:
    """
    Check if predicted array is a subset of reference array.
    
    Args:
        pre: Predicted array
        ref: Reference array
        
    Returns:
        True if all elements in pre are contained in ref
    """
    for pi in pre:
        if not any([ri.lower() in pi.lower() for ri in ref]):
            return False

    return True


if __name__ == "__main__":
    test_cases = [
        {
            "pre": ["老15", "老三", "老5"],
            "ref": ["3", "15", "6"]
        },  # False False 0.0 0.6666666666666666
        {
            "pre": ["二", "(B)", "Seven"],
            "ref": ["B", "2", "7"]
        }   # False True 0.3333333333333333 1.0
    ]

    for case in test_cases:
        print(order_array_equal(case["pre"], case["ref"]), 
              unorder_array_equal(case["pre"], case["ref"]),
              order_array_equal_soft(case["pre"], case["ref"]), 
              unorder_array_equal_soft(case["pre"], case["ref"]))