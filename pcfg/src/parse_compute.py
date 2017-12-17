# -*- coding: utf-8 -*-

import os
import sys

import pickle
import copy

from utils import *


class parseCompute(object):
    '''
    加载进一个规则，生成一种符合CNF的语法。
    对需要分析的句子，寻找最可能的parse tree，计算句子相应的内向概率和外向概率，并输出。
    '''
    def __init__(self, rules_file):
        # 初始化，加载规则模型
        self.phrasal_rules, self.lexical_rules = self.load(rules_file)
        print('Successfully loading the rules!')
        if self.check_rules():
            print('The rules satisfy CNF.')
            print('lexical_rules:')
            for key, value in self.lexical_rules.items():
                print(key, value)
            print('phrasal_rules:')
            for key, value in self.phrasal_rules.items():
                print(key, value)
        else:
            print("The rules don't satisfy CNF.")

    def load(self, rules_file):
        # 对给定的规则文件，加载文件，将规则划分为lexical rules和phrasal rules
        phrasal_rules = dict()
        lexical_rules = dict()
        if rules_file.endswith('.pkl'):
            with open(rules_file, 'rb') as f:
                rules = pickle.load(f)
            for key in rules.keys():
                pos = key.find('#')
                if len(key[(pos + 1):].strip().split()) == 1:
                    lexical_rules[key] = rules[key]
                else:
                    phrasal_rules[key] = rules[key]
        else:
            with open(rules_file, 'r') as f:
                rules = f.readlines()
            for rule in rules:
                temp = rule.strip().split('#')
                key = temp[0] + '#' + temp[1]
                key = key.strip()
                value = float(temp[2].strip())
                if len(temp[1].strip().split()) == 1:
                    lexical_rules[key] = value
                else:
                    phrasal_rules[key] = value
        return phrasal_rules, lexical_rules

    def check_rules(self):
        # 检查规则是否满足CNF条件
        tags = {}
        for key in self.phrasal_rules.keys():
            tag = key.split('#')[0].strip()
            add_dict(tags, tag, self.phrasal_rules[key])
        for key in self.lexical_rules.keys():
            tag = key.split('#')[0].strip()
            add_dict(tags, tag, self.lexical_rules[key])
        flag = True
        for tag, value in tags.items():
            if abs(value - 1) > 1e-3:
                print("Tag %s doesn't satisfy CNF." % (tag))
                flag = False
        return flag

    def compute_parse_cyk(self, sentence):
        # 利用CYK方法寻找句子最有可能的parse tree
        n = len(sentence)
        cyk_tags, cyk_prob, cyk_pos = [], [], []
        res_tags, res_prob, res_pos = [], [], []
        for i in range(n):    # 底层，用lexical_rules
            word = sentence[i].lower()
            info_tags, info_prob, info_pos = [], [], []
            for key in self.lexical_rules.keys():
                temp = key.split('#')
                Ni = temp[0].strip()
                ww = temp[1].strip()
                if word == ww:
                    info_tags.append(Ni)
                    info_prob.append(self.lexical_rules[key])
                    info_pos.append(i)
            res_tags.append(info_tags)
            res_prob.append(info_prob)
            res_pos.append(info_pos)
        cyk_tags.append(res_tags)
        cyk_prob.append(res_prob)
        cyk_pos.append(res_pos)
        for i in range(1, n):
            res_tags, res_prob, res_pos = [], [], []
            for j in range(n - i):
                info_tags, info_prob, info_pos = [], [], []
                for k in range(j, j + i):
                    for key in self.phrasal_rules.keys():
                        temp = key.split('#')
                        Ni = temp[0].strip()
                        temp1 = temp[1].strip().split()
                        Nj, Nk = temp1[0], temp1[1]
                        flagj, flagk = False, False
                        if Nj in cyk_tags[k - j][j]:
                            flagj = True
                            indj = cyk_tags[k - j][j].index(Nj)
                        if Nk in cyk_tags[i + j - k - 1][k + 1]:
                            flagk = True
                            indk = cyk_tags[i + j - k - 1][k + 1].index(Nk)
                        if flagj and flagk:
                            cur_prob = cyk_prob[k - j][j][indj] * cyk_prob[i + j - k - 1][k + 1][indk] * self.phrasal_rules[key]
                            if Ni in info_tags:    # 同一种词性要选取大的
                                ind = info_tags.index(Ni)
                                if info_prob[ind] < cur_prob:    # most likely tree
                                    info_prob[ind] = cur_prob
                                    info_pos[ind] = [k, indj, indk]
                            else:
                                info_tags.append(Ni)
                                info_prob.append(cur_prob)
                                info_pos.append([k, indj, indk])
                res_tags.append(info_tags)
                res_prob.append(info_prob)
                res_pos.append(info_pos)
            cyk_tags.append(res_tags)
            cyk_prob.append(res_prob)
            cyk_pos.append(res_pos)
        return cyk_tags, cyk_prob, cyk_pos

    def to_normal_form(self, sentence, cyk_tags, cyk_prob, cyk_pos):
        # 将parse tree转化标准的文本形式
        sf = ''
        n = len(sentence)
        top_tag, top_prob, top_pos = 0, 0, -1
        if len(cyk_tags[n - 1][0]) == 0:    # 规则下未形成parse tree
            return top_prob, sf
        for i in range(len(cyk_tags[n - 1][0])):
            if cyk_prob[n - 1][0][i] > top_prob:
                top_tag = cyk_tags[n - 1][0][i]
                top_prob = cyk_prob[n - 1][0][i]
                top_pos = cyk_pos[n - 1][0][i]
        if top_tag != 'S':    # 概率最大的不是一个句子
            return top_prob, sf
        sf = '(S'
        ll, lr = 0, top_pos[0]
        rl, rr = top_pos[0] + 1, n - 1
        indj, indk = top_pos[1], top_pos[2]
        sf = self.recursive(ll, lr, rl, rr, indj, indk, sentence, cyk_tags, cyk_pos, sf)
        sf += ')'
        return top_prob, sf

    def recursive(self, ll, lr, rl, rr, indj, indk, sentence, cyk_tags, cyk_pos, sf):
        # 递归得到左子树和右子树
        if (lr >= ll) and (rr >= rl):    # 每个节点必定包含左右子节点
            if lr == ll:
                sf += '(' + cyk_tags[0][ll][indj] + ' ' + sentence[ll] + ')'
                sf = self.recursive(ll, lr - 1, rl, rr, indj, indk, sentence, cyk_tags, cyk_pos, sf)
            elif lr > ll:
                sf += '(' + cyk_tags[lr - ll][ll][indj]
                sf = self.recursive(ll, cyk_pos[lr - ll][ll][indj][0], cyk_pos[lr - ll][ll][indj][0] + 1, lr,
                            cyk_pos[lr - ll][ll][indj][1], cyk_pos[lr - ll][ll][indj][2], sentence, cyk_tags, cyk_pos, sf)
                sf += ')'
            if rr == rl:
                sf += '(' + cyk_tags[0][rl][indk] + ' ' + sentence[rl] + ')'
                sf = self.recursive(ll, lr, rl, rr - 1, indj, indk, sentence, cyk_tags, cyk_pos, sf)
            elif rr > rl:
                sf += '(' + cyk_tags[rr - rl][rl][indk]
                sf = self.recursive(rl, cyk_pos[rr - rl][rl][indk][0], cyk_pos[rr - rl][rl][indk][0] + 1, rr,
                            cyk_pos[rr - rl][rl][indk][1], cyk_pos[rr - rl][rl][indk][2], sentence, cyk_tags, cyk_pos, sf)
                sf += ')'
        return sf

    def compute_inside_prob(self, sentence):
        # bottom-up compute inside probability
        n = len(sentence)
        in_tags, in_prob = [], []
        res_tags, res_prob = [], []
        for i in range(n):    # 底层，用lexical_rules
            word = sentence[i].lower()
            info_tags, info_prob = [], []
            for key in self.lexical_rules.keys():
                temp = key.split('#')
                Ni = temp[0].strip()
                ww = temp[1].strip()
                if word == ww:
                    info_tags.append(Ni)
                    info_prob.append(self.lexical_rules[key])
            res_tags.append(info_tags)
            res_prob.append(info_prob)
        in_tags.append(res_tags)
        in_prob.append(res_prob)
        for i in range(1, n):
            res_tags, res_prob = [], []
            for j in range(n - i):
                info_tags, info_prob = [], []
                for key in self.phrasal_rules.keys():
                    temp = key.split('#')
                    Ni = temp[0].strip()
                    temp1 = temp[1].strip().split()
                    Nj, Nk = temp1[0], temp1[1]
                    for k in range(j, j + i):
                        flagj, flagk = False, False
                        if Nj in in_tags[k - j][j]:
                            flagj = True
                            indj = in_tags[k - j][j].index(Nj)
                        if Nk in in_tags[i + j - k - 1][k + 1]:
                            flagk = True
                            indk = in_tags[i + j - k - 1][k + 1].index(Nk)
                        if flagj and flagk:
                            cur_prob = in_prob[k - j][j][indj] * in_prob[i + j - k - 1][k + 1][indk] * self.phrasal_rules[key]
                            if Ni in info_tags:    # 同一种词性累加(区别于CYK求最大parse tree)
                                ind = info_tags.index(Ni)
                                info_prob[ind] += cur_prob
                            else:
                                info_tags.append(Ni)
                                info_prob.append(cur_prob)
                res_tags.append(info_tags)
                res_prob.append(info_prob)
            in_tags.append(res_tags)
            in_prob.append(res_prob)
        return in_tags, in_prob

    def compute_outside_prob(self, in_tags, in_prob):
        # top-down compute outside probability
        n = len(in_tags)
        out_prob = copy.deepcopy(in_prob)
        for i in range(len(in_tags[n - 1][0])):
            if in_tags[n - 1][0][i] == 'S':
                out_prob[n - 1][0][i] = 1.0
            else:
                out_prob[n - 1][0][i] = 0.0
        for i in range(n - 2, -1, -1):
            for j in range(n - i):
                for k in range(len(in_tags[i][j])):
                    Nj = in_tags[i][j][k]
                    out_prob[i][j][k] = 0
                    for u in range(j + i + 1, n):    # 向右遍历
                        for v in range(len(in_tags[u - j - i - 1][j + i + 1])):
                            Ng = in_tags[u - j - i - 1][j + i + 1][v]
                            for q in range(len(in_tags[u - j][j])):
                                Nf = in_tags[u - j][j][q]
                                rule = Nf + ' # ' + Nj + ' ' + Ng
                                if rule in self.phrasal_rules:
                                    cur_prob = out_prob[u - j][j][q] * self.phrasal_rules[rule] * in_prob[u - j - i - 1][j + i + 1][v]
                                    out_prob[i][j][k] += cur_prob
                    for u in range(j):    # 向左遍历
                        for v in range(len(in_tags[j - 1 - u][u])):
                            Ng = in_tags[j - 1 - u][u][v]
                            for q in range(len(in_tags[j + i - u][u])):
                                Nf = in_tags[j + i - u][u][q]
                                rule = Nf + ' # ' + Ng + ' ' + Nj
                                if rule in self.phrasal_rules:
                                    cur_prob = out_prob[j + i - u][u][q] * self.phrasal_rules[rule] * in_prob[j - 1 - u][u][v]
                                    out_prob[i][j][k] += cur_prob
        return out_prob

    def sentence_analysis(self, sentence, out_file):
        if isinstance(sentence, str):
            sentence = sentence.strip().split()
        out = open(out_file, 'w')
        cyk_tags, cyk_prob, cyk_pos = self.compute_parse_cyk(sentence)
        top_prob, sf = self.to_normal_form(sentence, cyk_tags, cyk_prob, cyk_pos)
        out.write(sf + '\n')
        out.write('%.8f\n' % (top_prob))
        in_tags, in_prob = self.compute_inside_prob(sentence)
        out_prob = self.compute_outside_prob(in_tags, in_prob)
        n = len(sentence)
        for i in range(n):
            for j in range(n - i):
                for k in range(len(in_tags[i][j])):
                    ss = '%s # %d # %d # %.8f # %.8f\n' % (in_tags[i][j][k], j + 1, j + i + 1, in_prob[i][j][k], out_prob[i][j][k])
                    out.write(ss)
        out.close()


if __name__ == '__main__':
    pc = parseCompute('../test/rules.txt')
    sent = 'A boy with a telescope saw a girl'
    pc.sentence_analysis(sent, '../test/sentence_analysis.txt')
