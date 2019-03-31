import csv, math, json, time
import knn_functions as knn
import matplotlib.pyplot as plt
from playsound import playsound

ATTR_NUM = 21
csvs = ['cm1', 'kc2']
ks = [1, 2, 3, 5, 7, 9, 11, 13, 15]


def k_fold_dataset(l, k):
    res = []
    for i in range(0, len(l), k):
        res.append(l[i:i+k])
    return res


meta = {}
for data in csvs:
    meta[data] = {}
    for k in ks:
        with open(data + '.csv') as csv_file:
            rows = csv.reader(csv_file)
            next(rows)
            dataset = list(rows)
            print('len: ' + str(len(dataset)))
            for d in dataset:
                for j in range(ATTR_NUM):
                    d[j] = float(d[j])

                d[-1] = True if d[-1] == 'true' else False

        folded_dataset = k_fold_dataset(dataset, math.ceil(len(dataset)/10))

        normal_knn_accuracy = 0
        weighted_knn_accuracy = 0
        adaptive_knn_accuracy = 0

        normal_total_time = 0
        weighted_total_time = 0
        adaptive_total_time = 0

        fold = 0
        for i in range(len(folded_dataset)):
            fold += 1
            test_set = folded_dataset[i]
            training_set = []
            for j in range(len(folded_dataset)):
                if j != i:
                    training_set += folded_dataset[j]

            normal_results = []
            weighted_results = []
            adaptive_results = []
            it = 0
            for t in test_set:
                it += 1
                print('fold[{:3}]::test[{:3}]'.format(fold, it), end='\r')
                aux_start_w = time.time()

                start_normal = time.time()
                normal_neighbors=knn.get_neighbors(t, training_set, k, ATTR_NUM)

                aux_end_w = time.time()-aux_start_w

                normal_res = knn.get_response(normal_neighbors)
                normal_results.append((t, normal_res))

                normal_total_time += time.time()-start_normal

                ####################### END NORMAL ##########################

                start_w = time.time() - aux_end_w

                weighted_res = knn.get_response(normal_neighbors, weighted=True)
                weighted_results.append((t, weighted_res))

                weighted_total_time += time.time()-start_w

                ####################### END WEIGHT ##########################

                start_a = time.time()

                adaptive_neighbors = knn.get_neighbors(t, training_set, k, ATTR_NUM, adaptive=True)
                adaptive_res = knn.get_response(adaptive_neighbors)
                adaptive_results.append((t, adaptive_res))

                adaptive_total_time += time.time()-start_a

                ####################### END ADAPTIVE ########################

            normal_knn_accuracy += knn.get_accuracy(normal_results)
            weighted_knn_accuracy += knn.get_accuracy(weighted_results)
            adaptive_knn_accuracy += knn.get_accuracy(adaptive_results)

        meta[data][k] = {
            'n_acc': normal_knn_accuracy/len(folded_dataset),
            'w_acc': weighted_knn_accuracy/len(folded_dataset),
            'a_acc': adaptive_knn_accuracy/len(folded_dataset),
            'n_time': round(normal_total_time*1000),
            'w_time': round(weighted_total_time*1000),
            'a_time': round(adaptive_total_time*1000)
        }
        print('')
        print('K-NN [{}]: {:3.2f}% :: {:3.0f}s'.format(k, 100.0*normal_knn_accuracy /
                                                       len(folded_dataset), normal_total_time), end='\n\n')
        print('WK-NN[{}]: {:3.2f}% :: {:3.0f}s'.format(k, 100.0*weighted_knn_accuracy /
                                                       len(folded_dataset), weighted_total_time), end='\n\n')
        print('AK-NN[{}]: {:2.2f}% :: {:3.0f}s'.format(k, 100.0*adaptive_knn_accuracy /
                                                       len(folded_dataset), adaptive_total_time), end='\n\n')

with open('metadata.json', 'w') as jsonfile:
    json.dump(meta, jsonfile)

playsound('Still Dre.mp3')
