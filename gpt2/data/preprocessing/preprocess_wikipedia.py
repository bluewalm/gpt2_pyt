# Copyright (c) 2024, BLUEWALM. All rights reserved. 
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

import sys
import argparse
import subprocess
import unicodedata

try:
    import nltk
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk
finally:
    nltk.download('punkt_tab')

try:
    from wikinlp.downloader import Downloader
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/possible-worlds-research/wikinlp.git"])
finally:
    from wikinlp.downloader import Downloader


def is_sentence(text: str):
    if len(text) < 2:
        return False
    if (text[-1] != '\n') or (text[-2] not in '!?.'):  # not a sentence 
        return False
    if "and ." in text:  # missing reference 
        return False
    return True


def clean_text(text):
    # normalize unicode characters 
    normalized = unicodedata.normalize('NFKD', text)
    # remove non-ascii characters 
    ascii_text = normalized.encode('ASCII', 'ignore').decode('ASCII')
    # remove empty parantheses
    ascii_text = ascii_text.replace(" (  ) ", " ")
    ascii_text = ascii_text.replace(" [  ] ", " ")
    ascii_text = ascii_text.replace(" {  } ", " ")
    ascii_text = ascii_text.replace(" ( ) ", " ")
    ascii_text = ascii_text.replace(" [ ] ", " ")
    ascii_text = ascii_text.replace(" { } ", " ")
    return ascii_text


def prepare_wikipedia_dataset(language, n_dump_files):
    print("INFO: Running WikiNLP downloader & preprocessing.... this will take a while. ")
    wikinlp = Downloader(language)
    filepaths = wikinlp.mk_wiki_data(n_dump_files=n_dump_files, 
                                tokenize=True, 
                                lower=True, 
                                doctags=False)
    for fp in filepaths:
        print("INFO: Preprocessing file : ", fp)
        with open(fp, "r") as f:
            lines = f.readlines()
        tokenized_lines = []
        for line in lines:
            if not is_sentence(line):
                continue
            line = clean_text(line)
            tokenized_lines.append(line)
        with open(fp, "w") as f:
            f.writelines(tokenized_lines)
    return filepaths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='wikipedia_preprocessing',
        description='A script to conveniently download wikipedia dump files and preprocess them for training. ')
    # arguments 
    parser.add_argument('--n_dump_files', type=int, default=1,
                     help='Number of wikipedia dump files to download. If not specified, it will download all of them. ')
    parser.add_argument('--language', type=str, default='en',
                     help='The language of the wikipedia dump file to be downloaded and preprocessed. ')
    args = parser.parse_args()
    assert args.n_dump_files > 0
    # start the script 
    filepaths = prepare_wikipedia_dataset(language=args.language, 
                                    n_dump_files=args.n_dump_files)
    if not filepaths:
        raise Exception("preprocessing failure")
    print()
    print("Corpus is located in the following files: ")
    for fp in filepaths:
        print(fp)
    print()
    print("USAGE : select some files for training, others for evaluation. ")
    print("Pass the selected filepaths to the trainer via the appropriate arguments, respectively. ")
    print()

