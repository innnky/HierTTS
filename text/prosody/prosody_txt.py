from .benepar import parse_chart
from . import treebanks
from . import seq_with_label
import os
import re

punctuation_list = ['，', '。', '、', '；', '：', '？', '！', '“', '”', '‘', '’', '—', '…', '（', '）', '《', '》']


def data_pre_processing(x):
    x = re.sub('——', '—', x)
    x = re.sub('……', '…', x)
    return x


def separate_each_character(x):
    x_list = []
    for i in x:
        if i in punctuation_list:
            i = '(' + i + ' ' + i + ')'
            x_list.append(i)
        else:
            i = '(' + 'n' + ' ' + i + ')'
            x_list.append(i)
    x = ''.join(x_list)
    return x


def seq2tree(x):
    tree = '(' + 'TOP' + ' ' + '(' + 'S' + ' ' + x + ')' + ')'
    return tree


def init_model(model_path='text/prosody/pretrained_SpanPSP_Databaker.pt'):

    parser = parse_chart.ChartParser.from_trained(model_path)
    # parser.cuda()
    return parser


def run_auto_labels(parser, line):
    output_dir_yl1 = os.path.join('text/prosody/temp1.txt')
    line_list = re.split('([，。！？,、；：！])', line.strip())

    if os.path.exists(output_dir_yl1):
        os.remove(output_dir_yl1)
    for line in line_list:
        if line == '':
            continue
        line = data_pre_processing(line)
        line = separate_each_character(line)
        line = seq2tree(line)
        with open(output_dir_yl1, 'a', encoding='utf-8') as t:
            t.write(line + '\n')
    test_treebank = treebanks.load_trees(output_dir_yl1)
    test_predicted = parser.parse(
        test_treebank.without_gold_annotations(),
        subbatch_max_tokens=500,
    )
    return seq_with_label.output(test_predicted)

