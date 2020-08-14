import os
import gc
import re
import csv
from nltk.tokenize import sent_tokenize
from collections import OrderedDict

gc.disable()

def gen_csvs():
    source_path = 'en'
    target_path = 'fr'

    global_client_text = {}
    pattern = re.compile('(.*)ID=(.*?) (.*)NAME="(.*?)"(.*)')

    speaker_order = {}

    def clean_file(path, file, language='english'):
        client_text = OrderedDict()
        speaker_order = []
        order_to_name = {}

        with open(os.path.join(path, file), 'r') as fin:
            lines = fin.readlines()

            last_speaker = ''
            last_speaker_id = ''
            last_text = []

            for line in lines:
                line = line.strip()
                # a new start of speaker
                if 'SPEAKER ID' in line:
                    # add current text to dict
                    if len(last_text) > 0 and len(last_speaker) > 0:

                        if last_speaker_id not in client_text:
                            client_text[last_speaker_id] = []

                        client_text[last_speaker_id] += last_text
                        speaker_order.append(last_speaker_id)
                        order_to_name[last_speaker_id] = last_speaker

                    try:
                        matched_items = pattern.findall(line)[0]
                        last_speaker = matched_items[3].strip()
                        last_speaker_id = matched_items[1].strip()
                        last_text = []
                    except Exception:
                        print(file)
                        print(line)
                        assert(1==0)

                elif line != '<P>' and len(line) > 0:
                    if len(line) > 4 and line[0] == '(' and line[3] == ')':
                        line = line[4:]

                    last_text.append(sent_tokenize(line, language=language))

        return client_text, speaker_order, order_to_name

    source_files = set([x for x in os.listdir(source_path)])
    target_files = [x for x in os.listdir(target_path) if x in source_files]

    # handle the source files
    for fidx, file in enumerate(target_files):
        if '.txt' in file:
            source_dict, source_speaker_order, sorder_to_name = clean_file(source_path, file)
            target_dict, target_speaker_order, torder_to_name = clean_file(target_path, file, language='french')

            ordered_keys = list(source_dict.keys())
            target_keys = list(target_dict.keys())

            for speaker_id in ordered_keys:
                if speaker_id in target_dict:

                    if len(source_dict[speaker_id]) != len(target_dict[speaker_id]):
                            print(file)
                            print(speaker_id, len(source_dict[speaker_id]))
                            print(speaker_id, len(target_dict[speaker_id]))
                            print('==================# of paragraphs mismatch')
                            #assert(2==3)
                            continue
                    else:
                        for sidx in range(len(source_dict[speaker_id])):
                            if len(source_dict[speaker_id][sidx]) != len(target_dict[speaker_id][sidx]):
                                print(file)
                                print(speaker_id, len(source_dict[speaker_id][sidx]))
                                print(speaker_id, len(target_dict[speaker_id][sidx]))
                                print('==================# of Sentence mismatch')
                                #assert(2==3)
                            else:
                                speaker_name = sorder_to_name[speaker_id]
                                if speaker_name not in global_client_text:
                                    global_client_text[speaker_name] = {'source': [], 'target':[]}

                                global_client_text[speaker_name]['source'] += source_dict[speaker_id][sidx]
                                global_client_text[speaker_name]['target'] += target_dict[speaker_id][sidx]

            print('Done {} files'.format(fidx+1))

    sorted_keys = sorted(list(global_client_text.keys()), reverse=True, key=lambda k:len(global_client_text[k]['source']))

    print(len(global_client_text))

    with open('summary_'+source_path, 'w') as fout:
        for k in sorted_keys:
            fout.writelines(str(len(global_client_text[k]['source'])) + '\n')

    threshold = 20
    current_id = 0

    with open('training.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ClientId", "Source", "Target"])
        
        for client in global_client_text:
            # Take as training set
            if threshold <= len(global_client_text[client]['source']) <= 4000:
                current_id += 1

                temp = []
                for idx in range(len(global_client_text[client]['source'])):
                    temp.append([current_id, global_client_text[client]['source'][idx], global_client_text[client]['target'][idx]])
                writer.writerows(temp)

    with open('testing.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ClientId", "Source", "Target"])
        
        for client in global_client_text:
            # Take as training set
            if len(global_client_text[client]['source']) < threshold or len(global_client_text[client]['source']) > 4000:
                current_id += 1

                temp = []
                for idx in range(len(global_client_text[client]['source'])):
                    temp.append([current_id, global_client_text[client]['source'][idx], global_client_text[client]['target'][idx]])
                writer.writerows(temp)

def write_to_txts(prefix):
    # write to source, target files
    with open(prefix+'.csv', 'r') as fin:
        reader = csv.reader(fin)

        source_texts = []
        target_texts = []

        # split source/target
        for line in reader:
            if reader.line_num == 1:
                continue

            source = line[1]
            target = line[2]

            source_texts.append(source)
            target_texts.append(target)

        # write to files
        with open(prefix+'.source', 'w') as fout:
            for line in source_texts:
                line = line.strip()
                fout.writelines(line+'\n')

        with open(prefix+'.target', 'w') as fout:
            for line in target_texts:
                line = line.strip()
                fout.writelines(line+'\n')

write_to_txts('training')
write_to_txts('testing')

