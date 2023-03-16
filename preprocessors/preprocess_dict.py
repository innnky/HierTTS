path = "/Volumes/Extend/下载/vispeech/mfa_temp/zh_dictold.dict"
out = "/Volumes/Extend/下载/vispeech/mfa_temp/zh_dict.dict"
with open(out, "w") as f:
    for line in open(path).readlines():
        a = line.split(" ")[0]
        b = " ".join(line.split(" ")[1:])
        f.write(f"{a}\t{b}")