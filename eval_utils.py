"""
Evaluation Utilities Module

This module provides utility functions for text processing, number conversion,
and LaTeX table parsing to support the evaluation framework.
"""

import re
from typing import List, Any

import json_repair
from word2number import w2n
import cn2an


def convert_word_number(text: str) -> str:
    """
    Convert word numbers (both Chinese and English) to Arabic numerals.
    
    Args:
        text: Input text containing word numbers
        
    Returns:
        Text with word numbers converted to Arabic numerals
    """
    # First step: Convert Chinese numbers to Arabic numerals using cn2an
    try:
        text = cn2an.transform(text, "cn2an")
    except Exception:
        pass  # If cn2an fails, skip this step

    # Handle English numbers
    def convert_english(match):
        num_phrase = match.group(1)  # Get the number phrase part
        try:
            # Remove "and" and try conversion
            num_str = re.sub(r'\s+and\s+', ' ', num_phrase, flags=re.IGNORECASE)
            return str(w2n.word_to_num(num_str))
        except:
            return match.group(0)  # Return original text if conversion fails
    
    # Improved English number regex pattern
    # Matches: 1. Pure number word sequences (may include "and") 2. Hyphenated number forms
    english_num_pattern = r'(?<!\S)((?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion)(?:(?:\s+(?:and\s+)?(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion))+|-(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety))*)'
    
    text = re.sub(english_num_pattern, convert_english, text, flags=re.IGNORECASE)
    
    return text


def extract_begin_end_content(text: str) -> str:
    """
    Extract content between LaTeX begin/end tags.
    
    Args:
        text: Text containing LaTeX begin/end constructs
        
    Returns:
        Extracted content or empty string if no match found
    """
    # Match begin{*} ... \end{*} content, ensuring {*} part doesn't interfere
    pattern = r'begin\{([^{}]+)\}(?:{[^{}]*})?(.*?)\\end\{\1\}'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Extract matched content (second capture group) and strip whitespace
    results = [match[1].strip() for match in matches]
    if len(results):
        res = results[0].strip().strip("\\").strip()
        nest_res = extract_begin_end_content(res)
        if len(nest_res):
            return nest_res
        else:
            return res
    else:
        return ''


def parse_latex_table(latex_str: str) -> List[Any]:
    """
    Parse LaTeX table string into structured data.
    
    Args:
        latex_str: LaTeX table string to parse
        
    Returns:
        Parsed table data as list
    """
    """
    Parse LaTeX table string into structured data.
    
    Args:
        latex_str: LaTeX table string to parse
        
    Returns:
        Parsed table data as list
    """
    content = extract_begin_end_content(latex_str)
    data = []
    if len(content):
        if "\\hline" in content:
            header, content = content.split("\\hline")[:2]
        if content.startswith('['):
            data = json_repair.loads(content)
        else:
            data = []
            if "\\\\" in content:
                d_tmp = content.split("\\\\")
                for di in d_tmp:
                    if "&" in di:
                        di_da = []
                        for di_di in di.split("&"):
                            di_da.append(di_di.strip())
                        data.append(di_da)
                    else:
                        data.append(di.strip())
            else:
                if "&" in content:
                    for di in content.split("&"):
                        data.append(di.strip())
                else:
                    data.append(content.strip())
    return data


if __name__ == "__main__":
    # 我有32.78个苹果 and 21056 apples, and 23 bananas. 还有78%、16/7的梨！
    print(convert_word_number("我有三十二点七八个苹果 and twenty-one thousand and fifty-six apples, and twenty three bananas. 还有百分之七十八、七分之十六的梨！"))

    test = [
        "\\begin{array}{c|c} 编号 & 城市 \\\\ \\hline A & 柳州市 \\\\ B & 成都县 \\\\ C & 南京市 \\\\ D & 琴县 \\\\ E & 志强县 \\\\ F & 西安县 \\\\ G & 石家庄县 \\end{array}",
        "\\begin{array}{c}\n \\\\\n[[51, \"高淑珍\", 17], \\\\\n[48, \"程勇\", 16], \\\\\n[49, \"女儿\", 19], \\\\\n[50, \"段金凤\", 18]] \\\\\n \\\\\n\\end{array}",
        "\\begin{array}{c} 赵建国: 男 \\\\ 何雪: 男 \\\\ 胡畅: 女 \\\\ 毛旭: 男 \\\\ 王晶: 女 \\\\ 蒋洋: 女 \\end{array}",
        "\\begin{array}{c|c} 名字 & 性别 \\\\ \\hline 周刚 & 男 \\\\ 章燕 & 女 \\\\ 马娟 & 女 \\\\ 周瑞 & 女 \\\\ 陈旭 & 女 \\\\ 周宇 & 男 \\end{array}",
        "\\begin{array}{c} 朱凤兰 & [\"乌龟\"] \\\\ 赵鹏 & [\"乌龟\"] \\\\ 顾建平 & [\"蜜袋鼯\"] \\\\ 韩龙 & [\"乌龟\"] \\\\ 张玉梅 & [\"乌龟\", \"蜥蜴\"] \\\\ 郑玉珍 & [\"蜜袋鼯\"] \\\\ 靳亮 & [\"乌龟\", \"蜥蜴\"] \\\\ 兰建国 & [\"乌龟\"] \\\\ 徐洁 & [\"乌龟\", \"蜥蜴\"] \\\\ 夏勇 & [\"乌龟\", \"蜥蜴\"] \\end{array}",
        "\\begin{array}{c|c|c|c|c|c} A & B & C & D & E & F \\\\ \\hline 磊县 & 西宁市 & 玉珍县 & 玉华县 & 佛山市 & 永安县 \\end{array}",
        "\\begin{bmatrix} 4 & 7 & 2 & 5 & 1 & 9 & 6 & 3 & 8 \\\\ 1 & 3 & 5 & 2 & 6 & 7 & 4 & 8 & 9 \\\\ 6 & 8 & 9 & 3 & 4 & 8 & 7 & 2 & 5 \\\\ 2 & 1 & 3 & 8 & 4 & 6 & 5 & 9 & 7 \\\\ 5 & 4 & 7 & 9 & 8 & 1 & 2 & 6 & 3 \\\\ 8 & 9 & 6 & 7 & 5 & 2 & 1 & 3 & 4 \\\\ 3 & 2 & 1 & 6 & 7 & 4 & 8 & 5 & 9 \\\\ 7 & 5 & 8 & 4 & 3 & 6 & 9 & 1 & 2 \\\\ 9 & 6 & 4 & 1 & 2 & 5 & 3 & 7 & 8 \\end{bmatrix}",
        "\\begin{array}{ccccccccc} 1 & 6 & 2 & 3 & 9 & 8 & 7 & 5 & 4 \\\\ 9 & 3 & 4 & 2 & 5 & 7 & 6 & 1 & 8 \\\\ 5 & 7 & 8 & 4 & 6 & 1 & 3 & 2 & 9 \\\\ 4 & 2 & 3 & 5 & 7 & 6 & 8 & 9 & 1 \\\\ 6 & 5 & 7 & 3 & 8 & 9 & 4 & 2 & 1 \\\\ 7 & 9 & 6 & 1 & 2 & 4 & 3 & 8 & 5 \\\\ 3 & 2 & 1 & 6 & 5 & 7 & 9 & 4 & 8 \\\\ 1 & 8 & 5 & 9 & 4 & 3 & 6 & 7 & 2 \\\\ 7 & 4 & 9 & 8 & 1 & 2 & 5 & 3 & 6 \\end{array}",
        "\\begin{array}{c} A: 淑英市 \\\\ B: 呼和浩特市 \\\\ C: 瑞县 \\\\ D: 关岭市 \\\\ E: 华市 \\\\ F: 晨县 \\\\ G: 丹县 \\end{array}",
        "\\begin{array}{c} [[1, 6, 9, 2, 4, 8, 5, 3, 7], \\\\ [4, 3, 2, 1, 5, 7, 6, 8, 9], \\\\ [5, 7, 8, 3, 6, 9, 1, 2, 4], \\\\ [3, 1, 2, 6, 7, 5, 8, 9, 4], \\\\ [9, 5, 6, 8, 3, 1, 4, 7, 2], \\\\ [8, 9, 7, 4, 2, 6, 3, 1, 5], \\\\ [6, 2, 1, 7, 3, 4, 9, 5, 8], \\\\ [7, 4, 5, 9, 8, 2, 3, 6, 1], \\\\ [2, 8, 3, 5, 1, 9, 7, 4, 6]] \\end{array}",
        "\\begin{array}{ccccccccc}2 & 1 & 6 & 5 & 3 & 8 & 7 & 4 & 9 \\\\4 & 5 & 6 & 极 & 2 & 7 & 8 & 3 & 1 \\\\7 & 8 & 9 & 3 & 4 & 1 & 6 & 2 & 5 \\\\1 & 2 & 5 & 4 & 3 & 6 & 8 & 9 & 7 \\\\3 & 6 & 2 & 8 & 7 & 9 & 1 & 5 & 4 \\\\8 & 4 & 7 & 2 & 1 & 3 & 5 & 9 & 6 \\\\5 & 6 & 4 & 1 & 7 & 2 & 9 & 8 & 3 \\\\7 & 9 & 3 & 6 & 8 & 5 & 4 & 2 & 1 \\\\9 & 3 & 8 & 7 & 5 & 4 & 1 & 6 & 2\\end{array}",
        "\\begin{array}{c} \\begin{bmatrix} 5 & 8 & 3 & 6 & 2 & 9 & 4 & 1 & 7 \\\\ 1 & 9 & 7 & 3 & 5 & 4 & 6 & 8 & 2 \\\\ 9 & 6 & 1 & 4 & 3 & 8 & 7 & 2 & 5 \\\\ 2 & 1 & 4 & 5 & 7 & 6 & 3 & 9 & 8 \\\\ 4 & 6 & 8 & 7 & 9 & 3 & 2 & 5 & 1 \\\\ 7 & 4 & 9 & 1 & 8 & 2 & 5 & 3 & 6 \\\\ 6 & 3 & 2 & 8 & 4 & 1 & 9 & 5 & 7 \\\\ 8 & 1 & 5 & 3 & 7 & 5 & 9 & 6 & 2 \\\\ 3 & 7 & 5 & 2 & 6 & 9 & 8 & 4 & 1 \\end{bmatrix} \\end{array}",
        # "sdfsdf"
    ]

    # 测试解析
    for index, ti in enumerate(test):
        print(f"Test{index} ", parse_latex_table(ti))

    '''
    # Expected Results:
    Test0  [['A', '柳州市'], ['B', '成都县'], ['C', '南京市'], ['D', '琴县'], ['E', '志强县'], ['F', '西安县'], ['G', '石家庄县']]
    Test1  [[51, '高淑珍', 17], [48, '程勇', 16], [49, '女儿', 19], [50, '段金凤', 18]]
    Test2  ['赵建国: 男', '何雪: 男', '胡畅: 女', '毛旭: 男', '王晶: 女', '蒋洋: 女']
    Test3  [['周刚', '男'], ['章燕', '女'], ['马娟', '女'], ['周瑞', '女'], ['陈旭', '女'], ['周宇', '男']]
    Test4  [['朱凤兰', '["乌龟"]'], ['赵鹏', '["乌龟"]'], ['顾建平', '["蜜袋鼯"]'], ['韩龙', '["乌龟"]'], ['张玉梅', '["乌龟", "蜥蜴"]'], ['郑玉珍', '["蜜袋鼯"]'], ['靳亮', '["乌龟", "蜥蜴"]'], ['兰建国', '["乌龟"]'], ['徐洁', '["乌龟", "蜥蜴"]'], ['夏勇', '["乌龟", "蜥蜴"]']]
    Test5  ['磊县', '西宁市', '玉珍县', '玉华县', '佛山市', '永安县']
    Test6  [['4', '7', '2', '5', '1', '9', '6', '3', '8'], ['1', '3', '5', '2', '6', '7', '4', '8', '9'], ['6', '8', '9', '3', '4', '8', '7', '2', '5'], ['2', '1', '3', '8', '4', '6', '5', '9', '7'], ['5', '4', '7', '9', '8', '1', '2', '6', '3'], ['8', '9', '6', '7', '5', '2', '1', '3', '4'], ['3', '2', '1', '6', '7', '4', '8', '5', '9'], ['7', '5', '8', '4', '3', '6', '9', '1', '2'], ['9', '6', '4', '1', '2', '5', '3', '7', '8']]
    Test7  [['1', '6', '2', '3', '9', '8', '7', '5', '4'], ['9', '3', '4', '2', '5', '7', '6', '1', '8'], ['5', '7', '8', '4', '6', '1', '3', '2', '9'], ['4', '2', '3', '5', '7', '6', '8', '9', '1'], ['6', '5', '7', '3', '8', '9', '4', '2', '1'], ['7', '9', '6', '1', '2', '4', '3', '8', '5'], ['3', '2', '1', '6', '5', '7', '9', '4', '8'], ['1', '8', '5', '9', '4', '3', '6', '7', '2'], ['7', '4', '9', '8', '1', '2', '5', '3', '6']]
    Test8  ['A: 淑英市', 'B: 呼和浩特市', 'C: 瑞县', 'D: 关岭市', 'E: 华市', 'F: 晨县', 'G: 丹县']
    Test9  [[1, 6, 9, 2, 4, 8, 5, 3, 7], [4, 3, 2, 1, 5, 7, 6, 8, 9], [5, 7, 8, 3, 6, 9, 1, 2, 4], [3, 1, 2, 6, 7, 5, 8, 9, 4], [9, 5, 6, 8, 3, 1, 4, 7, 2], [8, 9, 7, 4, 2, 6, 3, 1, 5], [6, 2, 1, 7, 3, 4, 9, 5, 8], [7, 4, 5, 9, 8, 2, 3, 6, 1], [2, 8, 3, 5, 1, 9, 7, 4, 6]]
    Test10  [['2', '1', '6', '5', '3', '8', '7', '4', '9'], ['4', '5', '6', '极', '2', '7', '8', '3', '1'], ['7', '8', '9', '3', '4', '1', '6', '2', '5'], ['1', '2', '5', '4', '3', '6', '8', '9', '7'], ['3', '6', '2', '8', '7', '9', '1', '5', '4'], ['8', '4', '7', '2', '1', '3', '5', '9', '6'], ['5', '6', '4', '1', '7', '2', '9', '8', '3'], ['7', '9', '3', '6', '8', '5', '4', '2', '1'], ['9', '3', '8', '7', '5', '4', '1', '6', '2']]
    Test11  [['5', '8', '3', '6', '2', '9', '4', '1', '7'], ['1', '9', '7', '3', '5', '4', '6', '8', '2'], ['9', '6', '1', '4', '3', '8', '7', '2', '5'], ['2', '1', '4', '5', '7', '6', '3', '9', '8'], ['4', '6', '8', '7', '9', '3', '2', '5', '1'], ['7', '4', '9', '1', '8', '2', '5', '3', '6'], ['6', '3', '2', '8', '4', '1', '9', '5', '7'], ['8', '1', '5', '3', '7', '5', '9', '6', '2'], ['3', '7', '5', '2', '6', '9', '8', '4', '1']]
    '''