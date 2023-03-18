import os
import shutil

import librosa
import soundfile
import tqdm
from multiprocessing import Pool
from text.zh_frontend import zh_to_phonemes
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
from multiprocessing import cpu_count

zh_dict = [i.strip() for i in open("text/zh_dict.dict").readlines()]
zh_dict = {i.split("\t")[0]: i.split("\t")[1] for i in zh_dict}

reversed_zh_dict = {}
all_zh_phones = set()
for k, v in zh_dict.items():
    reversed_zh_dict[v] = k
    [all_zh_phones.add(i) for i in v.split(" ")]

# print(all_zh_phones)

def phones_to_pinyins(phones):
    pinyins = ''
    accu_ph = []
    for ph in phones:
        accu_ph.append(ph)
        if ph not in all_zh_phones:
            # print(ph)
            assert len(accu_ph) == 1
            # pinyins += ph
            accu_ph = []
        elif " ".join(accu_ph) in reversed_zh_dict.keys():
            pinyins += " " + reversed_zh_dict[" ".join(accu_ph)]
            accu_ph = []
    assert  accu_ph==[]
    return pinyins.strip()
# print(phones_to_pinyins(['n', 'i2', 'uang3', 'n', 'a3', 'r5', 'r5', 'z', 'ou3', 'ia5']))

def process_text(line):
    id_, text = line.strip().split("|")
    phones = zh_to_phonemes(text)
    phones = phones_to_pinyins(phones)
    # phones = phones.replace(",", "").replace(".", "").replace("!", "").replace("?", "").replace("…", "").replace("#", "")
    return (id_, phones)

if __name__ == '__main__':
    wav_base_dir = "/Volumes/Extend/AI/tts数据集/dataset/large_pretrain/biaobei"
    transcription_path = "/Volumes/Extend/下载/BZNSYP/000001-010000.txt"
    spk="biaobei"
    lang = "zh"
    with ProcessPoolExecutor(max_workers=1) as executor:
        os.makedirs(f"mfa_temp/wavs/{lang}/{spk}", exist_ok=True)
        lines = open(transcription_path).readlines()
        futures = [executor.submit(process_text, line) for line in lines]
        for x in tqdm.tqdm(as_completed(futures), total=len(lines)):
            id_, phones = x._result
            if os.path.exists(f"{wav_base_dir}/{id_}.wav"):
                wav, sr = librosa.load(f"{wav_base_dir}/{id_}.wav", sr=44100)
                soundfile.write(f"mfa_temp/wavs/{lang}/{spk}/{id_}.wav", wav, sr)
                with open(f"mfa_temp/wavs/{lang}/{spk}/{id_}.txt", "w") as o:
                    o.write(phones + "\n")

    print("rm -rf ./mfa_temp/temp; mfa train mfa_temp/wavs/zh mfa_temp/zh_dict.dict mfa_temp/zh_model.zip mfa_temp/textgrids/zh --overwrite -t ./mfa_temp/temp")
    # print("rm -rf ./mfa_temp/temp; mfa train mfa_temp/wavs/ja/ mfa_temp/ja_dict.dict mfa_temp/model.zip mfa_temp/textgrids/ja --clean --overwrite -t ./mfa_temp/temp -j 5")
