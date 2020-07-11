import csv
from PIL import Image
import math

def isIntersection(cords_a, cords_b):
    [xmin_a, xmax_a, ymin_a, ymax_a] = cords_a
    [xmin_b, xmax_b, ymin_b, ymax_b] = cords_b

    intersect_flag = True

    minx = max(xmin_a , xmin_b)
    miny = max(ymin_a , ymin_b)

    maxx = min(xmax_a , xmax_b)
    maxy = min(ymax_a , ymax_b)
    if minx > maxx or miny > maxy:
        intersect_flag = False

    return intersect_flag

def crop_img(cords, img_id, label, db_type):
    img = Image.open(db_type + '_raw/' + img_id + '.jpg')
    sizes = img.size

    x_min = int(math.floor(sizes[0] * cords[0]))
    x_max = int(math.floor(sizes[0] * cords[1]))

    y_min = int(math.floor(sizes[1] * cords[2]))
    y_max = int(math.floor(sizes[1] * cords[3]))

    if x_min >= x_max or y_min >= y_max:
        return

    cropped = img.crop((x_min, y_min, x_max, y_max))  # (left, upper, right, lower)

    if cropped.size[0] < 10 or cropped.size[1] < 10:
        return

    cropped = cropped.convert('RGB')
    cropped.save(db_type + '/' + img_id + '__' + label.replace('/','_') + '.jpg', quality=95)

db_type = 'train'
file_name = './boxes/' + db_type + ".csv"
csvFile = open(file_name, 'r')
reader = csv.reader(csvFile)

result = {}
img_dict = {}
last_img_name = ''

count = 0
line_n = 0

for item in reader:
    if reader.line_num == 1:
        continue

    line_n += 1
    # take the full image
    if sum([abs(int(x)) for x in item[8:13]]) != 0:
        continue

    img_id = item[0]

    if last_img_name != img_id:
        last_img_name = img_id
        img_dict = {}

    label = item[2]

    # get the boundary of images
    [x_min, x_max, y_min, y_max] = [float(x) for x in item[4:8]]
    cords = [x_min, x_max, y_min, y_max]

    # update the cord to avoid overlap
    flag = False

    if label in img_dict:
        for prev_cords in img_dict[label]:
            flag = isIntersection(cords, prev_cords)

            if flag:
                break
    else:
        img_dict[label] = []

    if flag:
        continue

    img_dict[label].append(cords)

    # crop the image given cords
    try:
        crop_img(cords, img_id, label, db_type)
        count += 1

    except Exception as e:
        print(e)

    if count % 1000 == 0:
        print(count, '\t', line_n)

