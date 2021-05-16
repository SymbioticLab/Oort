import json
import gc
import csv

gc.disable()
data = []

file = 'yelp_academic_dataset_review.json'
with open(file, 'r') as fin:
    lines = fin.readlines()

for line in lines:
    try:
        data.append(json.loads(line))
    except Exception:
        print('Error....')

sum_by_user = {}
sum_by_score = {}
cnt = 0

text_by_user = {}

print(len(data))
for review in data:
    user_id = review['user_id']
    score = review['stars']
    review_txt = review['text'].strip().lower()

    if user_id not in text_by_user:
        text_by_user[user_id] =[]

    text_by_user[user_id].append([int(score), review_txt])

    sum_by_user[user_id] = sum_by_user.get(user_id, 0) + 1
    sum_by_score[score] = sum_by_score.get(score, 0) + 1

    cnt += 1

    if cnt % 50000 == 0:
        print(cnt)

current_id = 0
threshold = 16

# consider as training set
with open('training.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ClientId", "Score", "Text"])
    
    for client in text_by_user:
        if len(text_by_user[client]) >= threshold:
            current_id += 1

            for idx in range(len(text_by_user[client])):
                text_by_user[client][idx] = [current_id]+text_by_user[client][idx]

            writer.writerows(text_by_user[client])

# consider as training set
with open('testing.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ClientId", "Score", "Text"])
    
    for client in text_by_user:
        if len(text_by_user[client]) < threshold:
            current_id += 1

            for idx in range(len(text_by_user[client])):
                text_by_user[client][idx] = [current_id]+text_by_user[client][idx]

            writer.writerows(text_by_user[client])

# sorted_keys = sorted(list(sum_by_user.keys()), reverse=True, key=lambda k: sum_by_user[k])
# with open('sum_by_user', 'w') as fout:
#     for name in sorted_keys:
#         fout.writelines(str(sum_by_user[name]) + '\n')

# print(sum_by_score)

