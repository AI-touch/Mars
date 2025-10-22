import io

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
import matplotlib.colors as colors


import matplotlib.pyplot as plt
# import palettable
# from palettable.cmocean.sequential import Amp_3 as col
# from palettable.cartocolors.sequential import Teal_3 as col
import numpy as np
import pandas as pd
# 支持中文
from pylab import *
from matplotlib.font_manager import FontProperties

mpl.rcParams['font.sans-serif'] = ['SimHei']


def plotmatrix(absvalue):
    labels = {1: 'Neutral',
              2: 'Calm',
              3: 'Happy',
              4: 'Sad',
              5: 'Angry',
              6: 'Fearful',
              7: 'Disgusted',
              8: 'Surprised'
              }

    fig = plt.figure(figsize=(9, 8.5))
    ax = fig.gca()
    width = np.shape(absvalue)[1]
    height = np.shape(absvalue)[0]
    ax.grid(False)
    #     ax.tick_params(axis = 'both', which = 'minor', labelsize = 10)

    # 设置横纵坐标的名称以及对应字体格式
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    bounds = [0, 0.1, 0.5, 1]
    #     cmap = col.mpl_colormap
    cmap = colors.ListedColormap(['#FFF3E3', '#F98556', '#8C0000'])
    norm = colors.BoundaryNorm(bounds, cmap.N)
    res = plt.imshow(np.array(absvalue), cmap=cmap, interpolation='nearest', norm=norm)
    #     res = plt.imshow(np.array(conf_mat_pro), cmap=plt.cm.Blues, interpolation='nearest')
    for i, row in enumerate(absvalue):
        for j, c in enumerate(row):
            if c > 0:
                plt.text(j - 0.4, i + .15, c, font2)

    plt.tick_params(axis='both', size=0, labelsize=17)

    _ = plt.xticks(range(8), [l for l in labels.values()], rotation=35, fontproperties='Times New Roman', size=32)
    _ = plt.yticks(range(8), [l for l in labels.values()], fontproperties='Times New Roman', size=32)
    plt.tight_layout()
    plt.show()

    png1 = io.BytesIO()
    fig.savefig(png1, format="svg", dpi=600)
    fig.savefig('confuse/audio.svg', format='svg', bbox_inches='tight')

if __name__ == '__main__':
    path = 'result/marsloss_1'
    # path = 'result-fl'

    all_y = np.load(path + '/y_true.npy', allow_pickle=True)
    all_y_pred = np.load(path +'/test_pre_v.npy', allow_pickle=True)

    # 计算准确率（Accuracy）
    acc = accuracy_score(all_y, all_y_pred)

    # 计算F1-score
    # 默认情况下，f1_score计算的是每个类别的F1-score的平均值（macro-average）
    # 如果您想计算加权平均（weighted-average）或其他类型的F1-score，请参考sklearn文档调整参数
    f1 = f1_score(all_y, all_y_pred, average='macro')

    print("Accuracy:", acc)
    print("F1-score:", f1)


    cm = confusion_matrix(all_y, all_y_pred)
    # print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.transpose(cm)
    cm = [[round(j, 2) for j in cm[i]] for i in range(len(cm))]
    print(cm)
    plotmatrix(cm)