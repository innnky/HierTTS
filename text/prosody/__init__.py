import re

from .prosody_txt import init_model, run_auto_labels

class Seg:
    def __init__(self, word):
        self.word = word
    def __repr__(self):
        return "'"+self.word+"'"
def split_chinese_string(s):
    # 匹配中文字符和数字标记
    pattern =re.compile(r'[\u4e00-\u9fa5]+|[^\u4e00-\u9fa5]+')
    return pattern.findall(s)

def get_labels(model, text):
    labels = run_auto_labels(model, text)
    labels = labels.replace("\n", "")
    labels = re.split("#\d", labels)
    res = []
    for label in labels:
        res += [Seg(i) for i in split_chinese_string(label)]
    return res

cache_model = None
def get_seg(text):
    global cache_model
    if cache_model == None:
        cache_model = init_model()
    return get_labels(cache_model, text)