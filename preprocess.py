import os

import tgt

from preprocessors.utils import get_alignment_word_boundary

from text.symbols import symbols
from text.zh_frontend import get_seg
# print(symbols)
preprocessed_out_path = "norm_preprocessed_hifi16_bert/paimon"
tgt_base_path = "/Volumes/Extend/下载/vispeech/mfa_temp/textgrids/zh/folder1"
wav_base_path = "/Volumes/Extend/下载/vispeech/mfa_temp/wavs/zh/folder1"
transcription_path = "/Volumes/Extend/下载/vispeech/data/zh/paimon/transcription_raw.txt"

os.makedirs(preprocessed_out_path, exist_ok=True)
os.makedirs(f"{preprocessed_out_path}/alignment", exist_ok=True)
os.makedirs(f"{preprocessed_out_path}/audio", exist_ok=True)
os.makedirs(f"{preprocessed_out_path}/energy", exist_ok=True)
os.makedirs(f"{preprocessed_out_path}/f0", exist_ok=True)
os.makedirs(f"{preprocessed_out_path}/spec", exist_ok=True)

lab_dict ={}
for line in open(transcription_path).readlines():
    name, _, txt =line.strip().split("|")
    lab_dict[name] = txt

for tgt_path in os.listdir(tgt_base_path):
    if tgt_path.endswith("TextGrid"):
        name = tgt_path.replace(".TextGrid", "")
        if os.path.exists(f"{wav_base_path}/{name}.wav"):
            textgrid = tgt.io.read_textgrid(f"{tgt_base_path}/{tgt_path}")
            phone_tier = textgrid.get_tier_by_name("phones")
            word_tier = textgrid.get_tier_by_name("words")
            res = get_alignment_word_boundary(phone_tier, word_tier, 16000, 200, "Chinese")
            txt = lab_dict[name]
            print(get_seg(txt))
            print(res)
            break





