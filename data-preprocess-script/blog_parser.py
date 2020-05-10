# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import os
import os.path

path = '/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/blogs/'
files = [x.name for x in os.scandir(path)]


def parse_client(file):
    fileName = path + file

    tree = ET.parse(fileName)

    textList = []

    for element in tree.iterfind('post'):
        textList.append(element.text.strip())

    return textList

print(parse_client(files[0]))
