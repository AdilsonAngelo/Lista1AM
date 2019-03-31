from matplotlib import pyplot as plt
import json

with open('metadata.json') as json_file:
    metadata=json.load(json_file)

for dataset in metadata:
    n_acc=([], [])
    w_acc=([], [])
    a_acc=([], [])
    n_time=([], [])
    w_time=([], [])
    a_time=([], [])
    for k_value, data in metadata[dataset].items():
        n_acc[0].append(int(k_value))
        n_acc[1].append(data['n_acc']*100)
        w_acc[0].append(int(k_value))
        w_acc[1].append(data['w_acc']*100)
        a_acc[0].append(int(k_value))
        a_acc[1].append(data['a_acc']*100)
        n_time[0].append(int(k_value))
        n_time[1].append(data['n_time'])
        w_time[0].append(int(k_value))
        w_time[1].append(data['w_time'])
        a_time[0].append(int(k_value))
        a_time[1].append(data['a_time'])

    plt.plot(n_acc[0], n_acc[1], '--', label='knn')
    plt.plot(w_acc[0], w_acc[1], '-', label='knn weight')
    plt.plot(a_acc[0], a_acc[1], '-.', label='knn adaptive')

    plt.legend(bbox_to_anchor=(.45, 1.25), loc='upper left', borderaxespad=0.)
    
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')

    plt.subplots_adjust(top=.75)
    plt.show()

    ####################################################
    plt.plot(n_time[0], n_time[1], '--', label='knn')
    plt.plot(w_time[0], w_time[1], '-', label='knn weight')
    plt.plot(a_time[0], a_time[1], '-.', label='knn adaptive')

    plt.legend(bbox_to_anchor=(.45, 1.25), loc='upper left', borderaxespad=0.)
    
    plt.xlabel('k')
    plt.ylabel('Time (s)')

    plt.subplots_adjust(top=.75)
    plt.show()