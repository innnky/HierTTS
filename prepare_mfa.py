import os
import shutil

import librosa
import soundfile
import tqdm
from multiprocessing import Pool
from text.zh_frontend import zh_to_pinyin
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
from multiprocessing import cpu_count

# print(phones_to_pinyins(['n', 'i2', 'uang3', 'n', 'a3', 'r5', 'r5', 'z', 'ou3', 'ia5']))

def process_text(line):
    id_, text = line.strip().split("|")
    phones = zh_to_pinyin(text)
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
