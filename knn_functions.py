import csv
import math


def euclidean(a, b, ATTR_NUM):
    distance = 0
    for i in range(ATTR_NUM):
        distance += pow((a[i] - b[i]), 2)
    return math.sqrt(distance)


def get_neighbors(instance, training_set, k, ATTR_NUM, adaptive=False):
    distances = []
    for i in training_set:
        dist = euclidean(i, instance, ATTR_NUM)
        if adaptive:
            radius = get_min_radius(i, training_set, ATTR_NUM)
            dist = dist / (radius if radius != 0.0 else 0.000000000001)
        distances.append((i, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0][ATTR_NUM])
    return neighbors


def get_min_radius(y, training_set, ATTR_NUM):
    min_radius = math.inf
    for t in training_set:
        if y[-1] == t[-1]:
            continue
        min_radius = min(min_radius, euclidean(y, t, ATTR_NUM))
    return min_radius


def get_response(neighbors, weighted=False):
    if weighted:
        ts = []
        fs = []
        for i in range(len(neighbors)):
            if neighbors[i]:
                ts.append(len(neighbors)-i)
            else:
                fs.append(len(neighbors)-i)

        if sum(fs) >= sum(ts):
            return False
        else:
            return True
    else:
        t = sum(neighbors)
        f = len(neighbors)-t
        if f == max(f, t):
            return False
        else:
            return True


def get_accuracy(results):
    correct = 0
    if not results:
        return 0
    for r in results:
        if r[0][-1] == r[1]:
            correct += 1
    res = correct/len(results)
    return res
