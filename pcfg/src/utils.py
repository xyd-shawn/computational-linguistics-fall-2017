# -*- coding: utf-8 -*-

import os
import sys


def add_dict(dic, key, num):
    if key in dic:
        dic[key] += num
    else:
        dic[key] = num
