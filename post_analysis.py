import os
import numpy as np
import matplotlib.pyplot as plt

path = '.log/'
acc_dict = {}

for file in os.listdir(path):
    dataset, net, drop_or_not, layers = settings = file.split('.txt')[0].split('_')
    text = open(path+file).readlines()
    acc_arr = []

    for line in text:
        if 'Epoch:' in line and 'v_time' in line:
            acc = float(line.split('acc_val: ')[1].split(' cur')[0])
            acc_arr.append(acc)
    if acc_arr != []:
        acc_max = max(acc_arr)
        acc_dict[file] = acc_max  # add to acc max dict

    # # draw the results
    # max_ind = acc_arr.index(acc_max)
    # x_range = np.arange(0,max_ind+200)
    # step = 1
    # epoch = [i*step for i in range(0,(max_ind+20)//step)]
    # y = [acc_arr[i*step] for i in range(0,(max_ind+20)//step)]
    # l = plt.plot(epoch,y,'r--')
    # plt.plot(epoch,y,'b+-')
    # plt.title(file[:-4])
    # plt.xlabel('Epoch')
    # plt.ylabel('acc')
    # plt.legend()
    # plt.savefig('.imgs/'+file[:-4]+'.png')

print(acc_dict)
