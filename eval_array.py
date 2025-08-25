"""
Array Evaluation Module

This module provides functionality for evaluating array-type answers
with support for ordered, unordered, and subset comparisons.
"""

from typing import List, Union

try:
    from eval_numeral import is_digit, numeral_equal
    from eval_nominal import nominal_equal
except ImportError:
    from .eval_numeral import is_digit, numeral_equal
    from .eval_nominal import nominal_equal


def unorder_array_equal(pre: List[str], ref: List[str]) -> bool:
    """
    Compare two arrays ignoring order.
    
    Args:
        pre: Predicted array
        ref: Reference array
        
    Returns:
        True if arrays contain same elements regardless of order
    """
    if len(pre) != len(ref):
        return False

    # Create a copy of ref to avoid modifying the original
    ref_copy = ref.copy()
    
    for pi in pre:
        find_flag = False
        for ri in ref_copy:
            if nominal_equal(pi, ri):
                find_flag = True
                break
            if is_digit(ri):
                if numeral_equal(pi, ri):
                    find_flag = True
                    break
            else:
                if str(ri).strip().lower() in str(pi).strip().lower():
                    find_flag = True
                    break
        
        if find_flag:
            ref_copy.remove(ri)
        else:
            return False

    return True


def order_array_equal(pre: List[str], ref: List[str]) -> bool:
    """
    Compare two arrays considering order.
    
    Args:
        pre: Predicted array
        ref: Reference array
        
    Returns:
        True if arrays match in order and content
    """
    if len(pre) < len(ref):
        return False

    for index, ri in enumerate(ref):
        if nominal_equal(pre[index], ri):
            continue
        if is_digit(ri):
             if numeral_equal(pre[index], ri):
                continue
        else:
            if str(ri).strip().lower() in str(pre[index]).strip().lower():
                continue
        return False

    return True


def order_number_array_equal(pre: List[str], ref: List[str]) -> bool:
    """
    Compare two numeric arrays considering order.
    
    Args:
        pre: Predicted numeric array
        ref: Reference numeric array
        
    Returns:
        True if numeric arrays match in order
    """
    if len(pre) < len(ref):
        return False

    for index, ri in enumerate(ref):
        if float(ri) - float(pre[index]) != 0:
            return False

    return True


def unorder_number_array_equal(pre: List[str], ref: List[str]) -> bool:
    """
    Compare two numeric arrays ignoring order.
    
    Args:
        pre: Predicted numeric array
        ref: Reference numeric array
        
    Returns:
        True if numeric arrays contain same values regardless of order
    """
    if len(pre) < len(ref):
        return False

    for pi in pre:
        if not any([float(ri) - float(pi) == 0 for ri in ref]):
            return False

    return True


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
            "pre": ["老15", "老三"],
            "ref": ["3", "15"]
        }, 
        {
            "pre": ["二", "(B)"],
            "ref": ["B", "2"]
        }
    ]

    for case in test_cases:
        print(order_array_equal(case["pre"], case["ref"]), 
              unorder_array_equal(case["pre"], case["ref"]))