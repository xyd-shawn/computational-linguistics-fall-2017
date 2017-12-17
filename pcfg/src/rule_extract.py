# -*- coding: utf-8 -*-

import os
import sys

import operator
import pickle
from collections import Counter

from utils import *


class rulesExtract(object):
    '''
    对一些已经做好parse的句子，读取它们parse tree的标准文本形式的文件。
    利用极大似然估计生成相应的规则。
    '''
    def __init__(self, parse_sentences_file):
        # 初始化，加载文件
        parse_sentences = self.load(parse_sentences_file)
        self.tags = dict()    # non-terminals, 值为对应的出现次数
        self.phrasal_rules = dict()    # phrasal_rules, 值为对应的出现次数
        self.lexical_rules = dict()    # lexical_rules, 值为对应的出现次数
        for ps in parse_sentences:
            if not self.extract_rules(ps):
                print("There is something wrong with '%s'" % (ps))

    def load(self, parse_sentences_file):
        # 加载已经parse过的文本
        with open(parse_sentences_file, 'r') as f:
            res = f.readlines()
        res = [s.strip() for s in res]
        return res

    def extract_rules(self, ps):
        # 对一个parse tree提取里面的phrasal rule和lexical rule
        t, prs, lrs = {}, {}, {}    # corresponding to tags, phrasal_rules, lexical_rules
        nst, pr = [], []    # nst: the stack of tags; pr: record the position of '(', ')' or ' '
        nlb = 0    # 栈里剩下的左括号数目，即未匹配的左括号数目
        for i, ss in enumerate(ps):
            if ss in ['(', ')', ' ']:
                pr.append(i)
        for i in range(len(pr)):
            if ps[pr[i]] == '(':
                nlb += 1
                if i > 0:
                    if ps[pr[i - 1]] == ' ':
                        return False    # 左括号左邻符号不应该是空格，函数返回假
                    elif ps[pr[i - 1]] == '(':
                        nst.append(ps[(pr[i - 1] + 1):pr[i]])
                        add_dict(t, nst[-1], 1)
            elif ps[pr[i]] == ')':
                if (i == 0) or (nlb == 0):
                    return False    # 已无与之对应的左括号
                else:
                    nlb -= 1
                    if ps[pr[i - 1]] == ')':
                        if len(nst) < 3:    # 栈内元素不能形成(Ni->Nj Nk)
                            return False
                        else:
                            Nk = nst.pop()
                            Nj = nst.pop()
                            Ni = nst[-1]
                            gram = Ni + ' # ' + Nj + ' ' + Nk
                            add_dict(prs, gram, 1)
                    elif ps[pr[i - 1]] == ' ':
                        w = (ps[(pr[i - 1] + 1):pr[i]]).lower()
                        if len(nst) == 0:    # 不能形成(Ni->w)
                            return False
                        else:
                            gram = nst[-1] + ' # ' + w
                            add_dict(lrs, gram, 1)
                    else:
                        return False    # ')'左邻符号不能为'('
            else:
                if (i > 0) and (ps[pr[i - 1]] == '('):    # 空格必须是左括号的右邻
                    nst.append(ps[(pr[i - 1] + 1):pr[i]])
                    add_dict(t, nst[-1], 1)
                else:
                    return False
        if nlb > 0:    # 有左括号未被匹配
            return False
        else:
            self.tags = dict(Counter(self.tags) + Counter(t))
            self.phrasal_rules = dict(Counter(self.phrasal_rules) + Counter(prs))
            self.lexical_rules = dict(Counter(self.lexical_rules) + Counter(lrs))
            return True

    def check_rules(self):
        # 检查提取的规则是否符合CNF条件
        merge_tags = dict()
        for key in self.phrasal_rules.keys():
            pos = key.find('#')
            tag = key[:(pos - 1)]
            add_dict(merge_tags, tag, self.phrasal_rules[key])
        for key in self.lexical_rules.keys():
            pos = key.find('#')
            tag = key[:(pos - 1)]
            add_dict(merge_tags, tag, self.lexical_rules[key])
        if operator.eq(self.tags, merge_tags):
            return True
        else:
            return False

    def mle(self):
        # 极大似然估计
        self.prob_phrasal_rules = dict.fromkeys(self.phrasal_rules, 0)
        self.prob_lexical_rules = dict.fromkeys(self.lexical_rules, 0)
        for key in self.prob_phrasal_rules.keys():
            pos = key.find('#')
            tag = key[:(pos - 1)]
            self.prob_phrasal_rules[key] = self.phrasal_rules[key] / self.tags[tag]
        for key in self.prob_lexical_rules.keys():
            pos = key.find('#')
            tag = key[:(pos - 1)]
            self.prob_lexical_rules[key] = self.lexical_rules[key] / self.tags[tag]

    def output(self):
        print('Nonterminal')
        for key in self.tags.keys():
            print(key, self.tags[key])
        print('Phrasal rules')
        for key in self.prob_phrasal_rules.keys():
            print(key, self.prob_phrasal_rules[key])
        print('Lexical rules')
        for key in self.prob_lexical_rules.keys():
            print(key, self.prob_lexical_rules[key])

    def save(self, out_file):
        # 保存规则模型
        total_rules = dict(Counter(self.prob_phrasal_rules) + Counter(self.prob_lexical_rules))
        if out_file.endswith('.pkl'):
            out = open(out_file, 'wb')
            pickle.dump(total_rules, out)
            out.close()
        else:
            out = open(out_file, 'w')
            sorted_data = sorted(total_rules.items(), key=lambda v: v[1], reverse=True)
            for key, value in sorted_data:
                out.write('%s # %.5f\n' % (key, value))
            out.close()

if __name__ == '__main__':
    ruex = rulesExtract('../test/test.txt')
    ruex.mle()
    if ruex.check_rules():
        print('Right rules!')
        ruex.output()
    else:
        print('Wrong rules!')
    ruex.save('../test/rules.txt')
