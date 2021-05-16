import csv, os
import pickle

db_types = ['train', 'test', 'validation']
img_author_map = {}

for db_type in db_types:
  file_name = './authors/' + db_type + ".csv"
  csvFile = open(file_name, 'r')
  reader = csv.reader(csvFile)

  for item in reader:
      if reader.line_num == 1:
          continue

      img_id = item[0]
      author_id = item[6].strip()

      img_author_map[img_id] = author_id

# load the images
image_ids = []
full_images = []
for db_type in db_types:
  #image_ids.append([])

  imgFiles = os.scandir(db_type)
  for imgFile in imgFiles:
    imgFile = imgFile.name
    full_images.append(os.path.join(db_type, imgFile))
    image_ids.append(imgFile.split('_')[0].strip())

img_to_author = {}
author_count = 0

author_to_imgs = {}
author_sets = set()
author_ids = {}

for img in image_ids:
  author = img_author_map[img]

  if author not in author_sets:
    author_sets.add(author)
    author_ids[author] = author_count
    author_to_imgs[author_count] = 0
    author_count += 1

  author_id = author_ids[author]
  img_to_author[img] = author_id
  author_to_imgs[author_id] += 1

for idx, img in enumerate(image_ids):
  if author_to_imgs[img_to_author[img]] < 16 and 'test' not in full_images[idx]:
    # move to testing set
    os.system('mv {} ./test/'.format(full_images[idx]))
  elif author_to_imgs[img_to_author[img]] >= 16 and 'train' not in full_images[idx]:
    os.system('mv {} ./train/'.format(full_images[idx]))

  if idx % 5000 == 0:
    print(idx)
# # dump information
# sorted_author_keys = sorted(list(author_to_imgs.keys()), key=lambda k:author_to_imgs[k], reverse=True)

# with open('author_img_count', 'w') as fout:
#   count = 0
#   for line in sorted_author_keys:
#     count += author_to_imgs[line]
#     fout.writelines(str(author_to_imgs[line]) + '\t' + str(count) + '\n')

# with open('img_to_author_map', 'wb') as fout:
#   pickle.dump(img_to_author, fout, -1)

# print("Total images: {}, clients: {}".format(len(image_ids), author_count))
