# -*- coding: utf-8 -*-

import os
import sys

def poems_extract(poems_file):
    with open(poems_file, 'r') as f:
        texts = f.readlines()
    i = 0
    poems = []
    puncs = [',', '.', ' ']
    while i < len(texts):
        if texts[i].startswith('<http'):
            poem = []
            flag = False
            i += 3
            while i < len(texts):
                text = texts[i].strip()
                if len(text) < 2:
                    if flag:
                        poem = ''.join(poem)
                        if (poem.find('<') == -1) and (poem.find('（') == -1):
                            poems.append(poem)
                        break
                    else:
                        i += 1
                else:
                    for w in text:
                        if w not in puncs:
                            poem.append(w)
                    flag = True
                    i += 1
        else:
            i += 1
    return poems

def save_with_bos_and_eos(poems, out_file):
    with open(out_file, 'w') as out:
        for poem in poems:
            words_list = ['<BOS>'] + [ww for ww in poem] + ['<EOS>']
            out.write(' '.join(words_list) + '\n')


if __name__ == '__main__':
    poems = poems_extract('../corpus/poetry.txt')
    save_with_bos_and_eos(poems, '../corpus/extracted_poems.txt')
