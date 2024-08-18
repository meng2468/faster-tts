# Copyright 2023, YOUDAO
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# an2cn
"""
This code is modified from https://github.com/Ailln/cn2an.
"""

from typing import Union
from warnings import warn

"""
This code is modified from https://github.com/Ailln/cn2an.
"""

NUMBER_CN2AN = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "壹": 1,
    "幺": 1,
    "二": 2,
    "贰": 2,
    "两": 2,
    "三": 3,
    "叁": 3,
    "四": 4,
    "肆": 4,
    "五": 5,
    "伍": 5,
    "六": 6,
    "陆": 6,
    "七": 7,
    "柒": 7,
    "八": 8,
    "捌": 8,
    "九": 9,
    "玖": 9,
}
UNIT_CN2AN = {
    "十": 10,
    "拾": 10,
    "百": 100,
    "佰": 100,
    "千": 1000,
    "仟": 1000,
    "万": 10000,
    "亿": 100000000,
}
UNIT_LOW_AN2CN = {
    10: "十",
    100: "百",
    1000: "千",
    10000: "万",
    100000000: "亿",
}
NUMBER_LOW_AN2CN = {
    0: "零",
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
}
NUMBER_UP_AN2CN = {
    0: "零",
    1: "壹",
    2: "贰",
    3: "叁",
    4: "肆",
    5: "伍",
    6: "陆",
    7: "柒",
    8: "捌",
    9: "玖",
}
UNIT_LOW_ORDER_AN2CN = [
    "",
    "十",
    "百",
    "千",
    "万",
    "十",
    "百",
    "千",
    "亿",
    "十",
    "百",
    "千",
    "万",
    "十",
    "百",
    "千",
]
UNIT_UP_ORDER_AN2CN = [
    "",
    "拾",
    "佰",
    "仟",
    "万",
    "拾",
    "佰",
    "仟",
    "亿",
    "拾",
    "佰",
    "仟",
    "万",
    "拾",
    "佰",
    "仟",
]
STRICT_CN_NUMBER = {
    "零": "零",
    "一": "一壹",
    "二": "二贰",
    "三": "三叁",
    "四": "四肆",
    "五": "五伍",
    "六": "六陆",
    "七": "七柒",
    "八": "八捌",
    "九": "九玖",
    "十": "十拾",
    "百": "百佰",
    "千": "千仟",
    "万": "万",
    "亿": "亿",
}
NORMAL_CN_NUMBER = {
    "零": "零〇",
    "一": "一壹幺",
    "二": "二贰两",
    "三": "三叁仨",
    "四": "四肆",
    "五": "五伍",
    "六": "六陆",
    "七": "七柒",
    "八": "八捌",
    "九": "九玖",
    "十": "十拾",
    "百": "百佰",
    "千": "千仟",
    "万": "万",
    "亿": "亿",
}

class An2Cn(object):
    def __init__(self) -> None:
        self.all_num = "0123456789"
        self.number_low = NUMBER_LOW_AN2CN
        self.number_up = NUMBER_UP_AN2CN
        self.mode_list = ["low", "up", "rmb", "direct"]

    def an2cn(self, inputs: Union[str, int, float] = None, mode: str = "low") -> str:
        """阿拉伯数字转中文数字

        :param inputs: 阿拉伯数字
        :param mode: low 小写数字，up 大写数字，rmb 人民币大写，direct 直接转化
        :return: 中文数字
        """
        if inputs is not None and inputs != "":
            if mode not in self.mode_list:
                raise ValueError(f"mode 仅支持 {str(self.mode_list)} ！")

            # 将数字转化为字符串，这里会有Python会自动做转化
            # 1. -> 1.0 1.00 -> 1.0 -0 -> 0
            if not isinstance(inputs, str):
                inputs = self.__number_to_string(inputs)

            # 数据预处理：
            # 1. 繁体转简体
            # 2. 全角转半角
            #inputs = preprocess(inputs, pipelines=[
            #    "traditional_to_simplified",
            #    "full_angle_to_half_angle"
            #])

            # 检查数据是否有效
            self.__check_inputs_is_valid(inputs)

            # 判断正负
            if inputs[0] == "-":
                sign = "负"
                inputs = inputs[1:]
            else:
                sign = ""

            if mode == "direct":
                output = self.__direct_convert(inputs)
            else:
                # 切割整数部分和小数部分
                split_result = inputs.split(".")
                len_split_result = len(split_result)
                if len_split_result == 1:
                    # 不包含小数的输入
                    integer_data = split_result[0]
                    if mode == "rmb":
                        output = self.__integer_convert(integer_data, "up") + "元整"
                    else:
                        output = self.__integer_convert(integer_data, mode)
                elif len_split_result == 2:
                    # 包含小数的输入
                    integer_data, decimal_data = split_result
                    if mode == "rmb":
                        int_data = self.__integer_convert(integer_data, "up")
                        dec_data = self.__decimal_convert(decimal_data, "up")
                        len_dec_data = len(dec_data)

                        if len_dec_data == 0:
                            output = int_data + "元整"
                        elif len_dec_data == 1:
                            raise ValueError(f"异常输出：{dec_data}")
                        elif len_dec_data == 2:
                            if dec_data[1] != "零":
                                if int_data == "零":
                                    output = dec_data[1] + "角"
                                else:
                                    output = int_data + "元" + dec_data[1] + "角"
                            else:
                                output = int_data + "元整"
                        else:
                            if dec_data[1] != "零":
                                if dec_data[2] != "零":
                                    if int_data == "零":
                                        output = dec_data[1] + "角" + dec_data[2] + "分"
                                    else:
                                        output = int_data + "元" + dec_data[1] + "角" + dec_data[2] + "分"
                                else:
                                    if int_data == "零":
                                        output = dec_data[1] + "角"
                                    else:
                                        output = int_data + "元" + dec_data[1] + "角"
                            else:
                                if dec_data[2] != "零":
                                    if int_data == "零":
                                        output = dec_data[2] + "分"
                                    else:
                                        output = int_data + "元" + "零" + dec_data[2] + "分"
                                else:
                                    output = int_data + "元整"
                    else:
                        output = self.__integer_convert(integer_data, mode) + self.__decimal_convert(decimal_data, mode)
                else:
                    raise ValueError(f"输入格式错误：{inputs}！")
        else:
            raise ValueError("输入数据为空！")

        return sign + output

    def __direct_convert(self, inputs: str) -> str:
        _output = ""
        for d in inputs:
            if d == ".":
                _output += "点"
            else:
                _output += self.number_low[int(d)]
        return _output

    @staticmethod
    def __number_to_string(number_data: Union[int, float]) -> str:
        # 小数处理：python 会自动把 0.00005 转化成 5e-05，因此 str(0.00005) != "0.00005"
        string_data = str(number_data)
        if "e" in string_data:
            string_data_list = string_data.split("e")
            string_key = string_data_list[0]
            string_value = string_data_list[1]
            if string_value[0] == "-":
                string_data = "0." + "0" * (int(string_value[1:]) - 1) + string_key
            else:
                string_data = string_key + "0" * int(string_value)
        return string_data

    def __check_inputs_is_valid(self, check_data: str) -> None:
        # 检查输入数据是否在规定的字典中
        all_check_keys = self.all_num + ".-"
        for data in check_data:
            if data not in all_check_keys:
                raise ValueError(f"输入的数据不在转化范围内：{data}！")

    def __integer_convert(self, integer_data: str, mode: str) -> str:
        if mode == "low":
            numeral_list = NUMBER_LOW_AN2CN
            unit_list = UNIT_LOW_ORDER_AN2CN
        elif mode == "up":
            numeral_list = NUMBER_UP_AN2CN
            unit_list = UNIT_UP_ORDER_AN2CN
        else:
            raise ValueError(f"error mode: {mode}")

        # 去除前面的 0，比如 007 => 7
        integer_data = str(int(integer_data))

        len_integer_data = len(integer_data)
        if len_integer_data > len(unit_list):
            raise ValueError(f"超出数据范围，最长支持 {len(unit_list)} 位")

        output_an = ""
        for i, d in enumerate(integer_data):
            if int(d):
                output_an += numeral_list[int(d)] + unit_list[len_integer_data - i - 1]
            else:
                if not (len_integer_data - i - 1) % 4:
                    output_an += numeral_list[int(d)] + unit_list[len_integer_data - i - 1]

                if i > 0 and not output_an[-1] == "零":
                    output_an += numeral_list[int(d)]

        output_an = output_an.replace("零零", "零").replace("零万", "万").replace("零亿", "亿").replace("亿万", "亿") \
            .strip("零")

        # 解决「一十几」问题
        if output_an[:2] in ["一十"]:
            output_an = output_an[1:]

        # 0 - 1 之间的小数
        if not output_an:
            output_an = "零"

        return output_an

    def __decimal_convert(self, decimal_data: str, o_mode: str) -> str:
        len_decimal_data = len(decimal_data)

        if len_decimal_data > 16:
            warn(f"注意：小数部分长度为 {len_decimal_data} ，将自动截取前 16 位有效精度！")
            decimal_data = decimal_data[:16]

        if len_decimal_data:
            output_an = "点"
        else:
            output_an = ""

        if o_mode == "low":
            numeral_list = NUMBER_LOW_AN2CN
        elif o_mode == "up":
            numeral_list = NUMBER_UP_AN2CN
        else:
            raise ValueError(f"error mode: {o_mode}")

        for data in decimal_data:
            output_an += numeral_list[int(data)]
        return output_an

import re
from pypinyin import pinyin, lazy_pinyin, Style
import jieba
import string
from pypinyin_dict.phrase_pinyin_data import cc_cedict
import re
import argparse
from string import punctuation
import numpy as np
from g2p_en import G2p
import os

ROOT_DIR = os.path.dirname(os.path.abspath("__file__"))

re_english_word = re.compile('([^\u4e00-\u9fa5]+|[ \u3002\uff0c\uff1f\uff01\uff1b\uff1a\u201c\u201d\u2018\u2019\u300a\u300b\u3008\u3009\u3010\u3011\u300e\u300f\u2014\u2026\u3001\uff08\uff09\u4e00-\u9fa5]+)', re.I)
def g2p_cn_en(text, g2p, lexicon):
    # Our policy dictates that if the text contains Chinese, digits are to be converted into Chinese.
    text=tn_chinese(text)
    parts = re_english_word.split(text)
    parts=list(filter(None, parts))
    tts_text = ["<sos/eos>"]
    chartype = ''
    text_contains_chinese = contains_chinese(text)
    for part in parts:
        if part == ' ' or part == '': continue
        if re_digits.match(part) and (text_contains_chinese or chartype == '') or contains_chinese(part):
            if chartype == 'en':
                tts_text.append('eng_cn_sp')
            phoneme = g2p_cn(part).split()[1:-1]
            chartype = 'cn'
        elif re_english_word.match(part):
            if chartype == 'cn':
                if "sp" in tts_text[-1]:
                    ""
                else:
                    tts_text.append('cn_eng_sp')
            phoneme = get_eng_phoneme(part, g2p, lexicon, False).split()
            if not phoneme :
                # tts_text.pop()
                continue
            else:
                chartype = 'en'
        else:
            continue
        tts_text.extend( phoneme )

    tts_text=" ".join(tts_text).split()
    if "sp" in tts_text[-1]:
        tts_text.pop()
    tts_text.append("<sos/eos>")

    return " ".join(tts_text)

def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = re.search(pattern, text)
    return match is not None

# Chinese frontend
cc_cedict.load()
re_special_pinyin = re.compile(r'^(n|ng|m)$')
def split_py(py):
    tone = py[-1]
    py = py[:-1]
    sm = ""
    ym = ""
    suf_r = ""
    if re_special_pinyin.match(py):
        py = 'e' + py
    if py[-1] == 'r':
        suf_r = 'r'
        py = py[:-1]
    if py == 'zi' or py == 'ci' or py == 'si' or py == 'ri':
        sm = py[:1]
        ym = "ii"
    elif py == 'zhi' or py == 'chi' or py == 'shi':
        sm = py[:2]
        ym = "iii"
    elif py == 'ya' or py == 'yan' or py == 'yang' or py == 'yao' or py == 'ye' or py == 'yong' or py == 'you':
        sm = ""
        ym = 'i' + py[1:]
    elif py == 'yi' or py == 'yin' or py == 'ying':
        sm = ""
        ym = py[1:]
    elif py == 'yu' or py == 'yv' or py == 'yuan' or py == 'yvan' or py == 'yue ' or py == 'yve' or py == 'yun' or py == 'yvn':
        sm = ""
        ym = 'v' + py[2:]
    elif py == 'wu':
        sm = ""
        ym = "u"
    elif py[0] == 'w':
        sm = ""
        ym = "u" + py[1:]
    elif len(py) >= 2 and (py[0] == 'j' or py[0] == 'q' or py[0] == 'x') and py[1] == 'u':
        sm = py[0]
        ym = 'v' + py[2:]
    else:
        seg_pos = re.search('a|e|i|o|u|v', py)
        sm = py[:seg_pos.start()]
        ym = py[seg_pos.start():]
        if ym == 'ui':
            ym = 'uei'
        elif ym == 'iu':
            ym = 'iou'
        elif ym == 'un':
            ym = 'uen'
        elif ym == 'ue':
            ym = 've'
    ym += suf_r + tone
    return sm, ym


chinese_punctuation_pattern = r'[\u3002\uff0c\uff1f\uff01\uff1b\uff1a\u201c\u201d\u2018\u2019\u300a\u300b\u3008\u3009\u3010\u3011\u300e\u300f\u2014\u2026\u3001\uff08\uff09]'


def has_chinese_punctuation(text):
    match = re.search(chinese_punctuation_pattern, text)
    return match is not None
def has_english_punctuation(text):
    return text in string.punctuation

# with thanks to KimigaiiWuyi in https://github.com/netease-youdao/EmotiVoice/pull/17.
# Updated on November 20, 2023: EmotiVoice now incorporates cn2an (https://github.com/Ailln/cn2an) for number processing.
re_digits = re.compile('(\d[\d\.]*)')
def number_to_chinese(number):
    an2cn = An2Cn()
    result = an2cn.an2cn(number)

    return result

def tn_chinese(text):
    parts = re_digits.split(text)
    words = []
    for part in parts:
        if re_digits.match(part):
            words.append(number_to_chinese(part))
        else:
            words.append(part)
    return ''.join(words)

def g2p_cn(text):
    res_text=["<sos/eos>"]
    seg_list = jieba.cut(text)
    for seg in seg_list:
        if seg == " ": continue
        seg_tn = tn_chinese(seg)
        py =[_py[0] for _py in pinyin(seg_tn, style=Style.TONE3,neutral_tone_with_five=True)]

        if any([has_chinese_punctuation(_py) for _py in py])  or any([has_english_punctuation(_py) for _py in py]):
            res_text.pop()
            res_text.append("sp3")
        else:
            
            py = [" ".join(split_py(_py)) for _py in py]
            
            res_text.append(" sp0 ".join(py))
            res_text.append("sp1")
    #res_text.pop()
    res_text.append("<sos/eos>")
    return " ".join(res_text)

# English frontend
def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def get_eng_phoneme(text, g2p, lexicon, pad_sos_eos=True):
    """
    english g2p
    """
    filters = {",", " ", "'"}
    phones = []
    words = list(filter(lambda x: x not in {"", " "}, re.split(r"([,;.\-\?\!\s+])", text)))

    for w in words:
        if w.lower() in lexicon:
            
            for ph in lexicon[w.lower()]:
                if ph not in filters:
                    phones += ["[" + ph + "]"]

            if "sp" not in phones[-1]:
                phones += ["engsp1"]
        else:
            phone=g2p(w)
            if not phone:
                continue

            if phone[0].isalnum():
                
                for ph in phone:
                    if ph not in filters:
                        phones += ["[" + ph + "]"]
                    if ph == " " and "sp" not in phones[-1]:
                        phones += ["engsp1"]
            elif phone == " ":
                continue
            elif phones:
                phones.pop() # pop engsp1
                phones.append("engsp4")
    if phones and "engsp" in phones[-1]:
        phones.pop()

    # mark = "." if text[-1] != "?" else "?"
    if pad_sos_eos:
        phones = ["<sos/eos>"] + phones + ["<sos/eos>"]
    return " ".join(phones)
    