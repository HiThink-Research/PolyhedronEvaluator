"""
Nominal Evaluation Module

This module provides functionality for evaluating text/name-based answers
with normalization and word number conversion support.
"""

import re
import string

try:
    from eval_utils import convert_word_number
except ImportError:
    from .eval_utils import convert_word_number


def remove_punctuation(text: str) -> str:
    """
    Remove Chinese and English punctuation from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with punctuation removed
    """
    if not isinstance(text, str):
        return str(text)
    # Define Chinese and English punctuation
    chinese_punctuation = "。，、；：？！''""（）【】《》……—～·"
    all_punctuation = string.punctuation + chinese_punctuation
    
    # Use regex to remove punctuation
    pattern = f"[{re.escape(all_punctuation)}]"
    cleaned_text = re.sub(pattern, "", text)
    
    return cleaned_text


def nominal_equal(pre: str, ref: str) -> bool:
    """
    Compare two nominal answers with normalization.
    
    Performs the following normalization steps:
    1. Remove punctuation
    2. Convert word numbers to digits
    3. Compare case-insensitively with spaces removed
    
    Args:
        pre: Predicted answer
        ref: Reference answer
        
    Returns:
        True if answers are equivalent after normalization
    """
    pre = remove_punctuation(pre)
    ref = remove_punctuation(ref)

    pre = convert_word_number(pre)
    ref = convert_word_number(ref)

    # Compare with normalized spaces and case
    if ''.join(pre.lower().split()) == ''.join(ref.lower().split()):
        return True

    return False


if __name__ == "__main__":
    print(nominal_equal("F. 23", "F"))
    print(nominal_equal("a\n ", "(A)"))
    print(nominal_equal("老3", "老三"))
    print(nominal_equal("小明妈妈", "小明"))