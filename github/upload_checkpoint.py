# python merge_files.py source_folder target_file delete_old_files

# python merge_files.py generated out_file.json yup

import sys
import json
import os

if __name__ == "__main__":
    source_folder = sys.argv[1]
    out_file = sys.argv[2]
    delete_old = sys.argv[3].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']

    res = []
    for f in [f for f in os.listdir(source_folder) if f.endswith('.json')]:
        with open(f"{source_folder}/{f}", encoding='utf8') as ff:
            res += json.load(ff)

    with open(out_file, "w", encoding='utf8') as f:
        json.dump(res, f, ensure_ascii=False)

    if delete_old:
        [os.remove(f"{source_folder}/{x}") for x in os.listdir(source_folder)]