import csv, random, math


def load_dataset(filename, split, training_set=[], test_set=[], ATTR_NUM=0):
    with open(filename) as csv_file:
        rows = csv.reader(csv_file)
        next(rows)
        dataset = list(rows)
        for d in dataset:
            for j in range(ATTR_NUM):
                d[j] = float(d[j])

            d[ATTR_NUM] = True if d[ATTR_NUM] == 'true' else False
            if random.random() < split:
                training_set.append(d)
            else:
                test_set.append(d)
        return (training_set, test_set)


def euclidean(a, b, ATTR_NUM):
    distance = 0
    for i in range(ATTR_NUM):
        distance += pow((a[i] - b[i]), 2)
    return math.sqrt(distance)


def get_neighbors(instance, training_set, k, ATTR_NUM):
    distances=[]
    for i in training_set:
        distances.append((i, euclidean(i, instance, ATTR_NUM)))
    distances.sort(key=lambda x: x[1])
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0][ATTR_NUM])
    return neighbors


def get_response(neighbors):
    t=sum(neighbors)
    f=len(neighbors)-t
    if f == max(t, f):
        return False
    else:
        return True