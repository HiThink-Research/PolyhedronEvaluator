"""
Multiple Choice Evaluation Module

This module provides functionality for evaluating multiple    special_chars = set(';,./、。，；：《》？:<>?\n\t\\')
    
    cleaned_chars = [char for char in pre if char not in special_chars]oice answers,
supporting both single and multiple correct answer scenarios.
"""

import re
from typing import List, Optional, Tuple


def normalize_answer(answer: str) -> List[Optional[str]]:
    """
    Normalize multiple choice answer.
    
    1. Extract letter part (e.g., "A", "B", etc.)
    2. If letter exists, return it (converted to uppercase)
    3. If no letter, remove option prefix (e.g., "A. ") and return remaining part
    
    Args:
        answer: Answer string to normalize
        
    Returns:
        List containing [option_letter, text_content] where either may be None
    """
    # Extract letter part (e.g., "A", "B", etc.)
    letter_match = re.search(r'[A-Za-z]', answer)
    option = None
    if letter_match:
        option = letter_match.group().upper()  # Convert to uppercase

    # If no letter, remove option prefix (e.g., "A. ", "B) ", etc.)
    text_answer = re.sub(r'^[A-Za-z][\.\)\s]*', '', answer).strip()
    return [option, text_answer]


def option_equal(pre: str, ref: str) -> bool:
    """
    Compare two multiple choice answers.
    
    1. Normalize both answers
    2. Compare normalized results
    
    Args:
        pre: Predicted answer
        ref: Reference answer
        
    Returns:
        True if answers match, False otherwise
    """
    pre_res = normalize_answer(pre)
    ref_res = normalize_answer(ref)
    
    if (pre_res[0] is None or ref_res[0] is None):
        pre = pre_res[1]
        ref = ref_res[1]
    else:
        pre = pre_res[0]
        ref = ref_res[0]

    if pre and ref:
        return pre == ref
    else:
        return False

def multi_answers_MCQ(pre, ref):
    """
        Compare multiple choice questions with multiple correct answers.
        
        Args:
            pre: Predicted answer string
            ref: Reference answer string
            
        Returns:
            True if all letters in reference are present in prediction
    """

    special_chars = set(r';,./、。，‘’；：“”《》？:<>?\n\t\\')
    
    cleaned_chars = [char for char in pre if char not in special_chars]

    pre = ''.join(cleaned_chars)
    pre_letters = sorted([c.lower() for c in pre if c.isalpha()])
    ref_letters = sorted(ref.lower())
    return pre_letters == ref_letters


if __name__ == "__main__":
    print(option_equal("F. 23", "F"))
    print(option_equal("A", "(A)"))
    print(option_equal("A) 33", "33"))
    print(option_equal("A) 33", "a"))
    print(multi_answers_MCQ("A、B、c", "ABC"))