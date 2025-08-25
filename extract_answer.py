"""
Answer Extraction Module

This module provides functionality to extract answers from model outputs,
particularly from LaTeX boxed notation and various text formats.
"""

import re
from typing import Union, List


def extract_answer(pred_str: str, flag: str = "boxed", last: bool = False) -> any:
    """
    Extract answers from prediction string based on specified flag.
    
    Args:
        pred_str: Prediction string containing potential answers
        flag: Flag to look for (default: "boxed" for LaTeX \\boxed{} notation)
        last: If True, return only the last answer; if False, return all answers
        
    Returns:
        String (if last=True) or List of strings containing extracted answers
    """
    if not isinstance(pred_str, str):
        return "" if last else [""]
    pred_str = pred_str.replace("\u043a\u0438", "")
    pred_str = pred_str.split("/think>")[-1].strip()
    pred_res = []
    if flag in pred_str:
        for ans in pred_str.split(flag)[1:]:
            pred = ""
            if len(ans) == 0:
                return "" if last else [""]
            elif ans[0] == "{":
                stack = 1
                a = ""
                for c in ans[1:]:
                    if c == "{":
                        stack += 1
                        a += c
                    elif c == "}":
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
                if "{" in a:
                    if a.count("{") == a.count("}"):
                        if "text" in a:
                            text_pattern = r'\\text\{([^}]*)\}'
                            matches = re.findall(text_pattern, a)
                            # 12 \\text{pounds}
                            # \\text{pounds}
                            a = re.sub(text_pattern, r'\1', a)
                        else:
                            pred = a
                    else:
                        a = ""
            else:
                if "$" in ans:
                    a = ans.split("$")[0].strip()
                else:
                    a = ""
            pred = a
            # multiple line
            # pred = pred.split("\n")[0]
            pred = re.sub(r"\n\s*", "", pred)
            if pred != "" and pred[0] == ":":
                pred = pred[1:]
            if pred != "" and pred[-1] == ".":
                pred = pred[:-1]
            if pred != "" and pred[-1] == "/":
                pred = pred[:-1]
            # pred = strip_string(pred)
            pred = pred.strip()
            if pred != "":
                pred_res.append(pred)
            # breakpoint()

    if last:
        return pred_res[-1] if len(pred_res) else ""
        
    return pred_res


if __name__ == "__main__":
    print(extract_answer("\\boxed{F. xxx}"))
    print(extract_answer("The first box: \\boxed{Answer: \\text{A}}; The second box: \\boxed{Remove text wrapper \\text{B}, \\frac{1}{4}}"))
    print(extract_answer("No boxed"))