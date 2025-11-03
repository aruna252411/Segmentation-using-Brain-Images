import matplotlib
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from itertools import cycle


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'CSO', 'BWO', 'MBO', 'EBOA', 'PROPOSED']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(2):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='CSO')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='BWO')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='MBO')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='EBOA')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='EEBOA')
        plt.xlabel('No. of Iteration', fontsize=14, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontsize=14, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.legend(loc=1, prop={'weight': 'bold', 'size': 12})
        plt.savefig("./New_Results/Conv_%s.png" % (i + 1))
        plt.show()


import seaborn as sns


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    no_of_Dataset = 1
    for n in range(no_of_Dataset):
        ax = plt.subplot()
        cm = confusion_matrix(np.asarray(Actual[n].argmax(axis=1)), np.asarray(Predict[n].argmax(axis=1)))
        sns.heatmap(cm, annot=True, fmt='g',
                    ax=ax)
        path = "./New_Results/Confusion_%s.png" % (n + 1)
        plt.title("Accuracy")
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig(path)
        plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['Resnet', 'Inception', 'MobileNet', 'Densenet', 'MDE-LSTM']
    for a in range(2):  # For 5 Datasets
        # Actual = np.load('Target.npy', allow_pickle=True).astype('int')
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        perc = round(Actual.shape[0] * 0.50)
        Actual = Actual[perc:, :]
        colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])  # "aqua",
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel("True Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='#1d3557')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#1d3557')
        plt.title("ROC Curve")
        plt.legend(loc="lower right", prop={'weight': 'bold', 'size': 12})
        path1 = "./New_Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


def plot_results():
    eval = np.load('./Old_file/Evaluate_all.npy', allow_pickle=True)
    eval1 = eval
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    # Graph_Terms = [0]
    Graph_Terms = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    Algorithm = ['TERMS', 'MBO', 'CSO', 'COS', 'RPO', 'PROPOSED']
    Classifier = ['TERMS', 'Resnet', 'Inception', 'MobileNet', 'Densenet', 'PROPOSED']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]
        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Learnperc - Dataset', i + 1, 'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)
    # K_fold = [1, 2, 3, 4, 5]
    ActivationFun = ['Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='#EE2C2C', width=0.15, label="Resnet")
            ax.bar(X + 0.15, Graph[:, 6], color='#1874CD', width=0.15, label="Inception")
            ax.bar(X + 0.30, Graph[:, 7], color='#FF1493', width=0.15, label="MobileNet")
            ax.bar(X + 0.45, Graph[:, 8], color='#CAFF70', width=0.15, label="Densenet")
            ax.bar(X + 0.60, Graph[:, 4], color='k', width=0.15, label="MDE-LSTM")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                       ncol=3, fancybox=True, shadow=True)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                       ncol=3, fancybox=True, shadow=True, prop={'weight': 'bold', 'size': 12})
            plt.xticks(X + 0.20, ('Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid'), fontsize=14, fontweight='bold',
                       color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontsize=14, fontweight='bold', color='k')
            path1 = "./Results/Dataset_%s_%s_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()


def plot_seg_results():
    Eval_all = np.load('./Old_file/Eval_all1.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD', 'VARIANCE']
    Algorithm = ['TERMS', 'MBO', 'CSO', 'COS', 'RPO', 'PROPOSED']
    Methods = ['TERMS', 'Resnet', 'Inception', 'MobileNet', 'Densenet', 'PROPOSED']
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]

        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        # stats = np.zeros((value_all[0].shape[1] + 9, value_all.shape[0] - 9, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(stats.shape[2])

            fig = plt.figure()
            ax = fig.add_axes([0.14, 0.12, 0.7, 0.7])
            ax.bar(X + 0.00, stats[i, 0, :], color='blue', width=0.15, label="MBO-ASMUnet")
            ax.bar(X + 0.15, stats[i, 1, :], color='#FF34B3', width=0.15, label="CSO-ASMUnet")
            ax.bar(X + 0.30, stats[i, 2, :], color='lawngreen', width=0.15, label="COS-ASMUnet")
            ax.bar(X + 0.45, stats[i, 3, :], color='teal', width=0.15, label="RPO-ASMUnet")
            ax.bar(X + 0.60, stats[i, 4, :], color='k', width=0.15, label="MRPO-ASMUnet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
                       ncol=3, fancybox=True, shadow=True, prop={'weight': 'bold', 'size': 12})
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'), fontsize=14, fontweight='bold', color='k')
            plt.xlabel('Statisticsal Analysis', fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[i - 4], fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontsize=14, fontweight='bold', color='k')
            # plt.legend(loc=1)
            path1 = "./New_Results/Dataset_%s_%s_alg.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.14, 0.12, 0.7, 0.7])
            ax.bar(X + 0.00, stats[i, 5, :], color='#66CD00', width=0.15, label="UNet")
            ax.bar(X + 0.15, stats[i, 6, :], color='#EE30A7', width=0.15, label="Res-Unet")
            ax.bar(X + 0.30, stats[i, 7, :], color='#1C86EE', width=0.15, label="Trans-UNet")
            ax.bar(X + 0.45, stats[i, 8, :], color='#FF7F24', width=0.15, label="Swin-mobileUnet")
            ax.bar(X + 0.60, stats[i, 4, :], color='k', width=0.15, label="MRPO-ASMUnet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
                       ncol=3, fancybox=True, shadow=True, prop={'weight': 'bold', 'size': 12})
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'), fontsize=14, fontweight='bold', color='k')
            plt.xlabel('Statisticsal Analysis', fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[i - 4], fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontsize=14, fontweight='bold', color='k')
            # plt.legend(loc=1)
            path1 = "./New_Results/Dataset_%s_%s_met.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()


from matplotlib.colors import ListedColormap


def Confusion():
    for n in range(2):
        Actual = np.load('Actual_' + str(n + 1) + '.npy', allow_pickle=True).astype(np.int32)
        Predict = np.load('Predict_' + str(n + 1) + '.npy', allow_pickle=True).astype(np.int32)
        # class_1 = ['Early Blight', 'Healthy', 'Late Blight']
        # class_2 = ['Black \n Sigatoka', 'Bract \n Mosaic \n Virus', 'Healthy', 'Insect \n Pest', 'Moko', 'Panama',
        #            'Yellow \n Sigatoka']
        # classes = [class_1, class_2]
        Confusion_matrix = metrics.confusion_matrix(Actual.argmax(axis=1), Predict.argmax(axis=1))
        fig = plt.figure(figsize=(10, 7))
        fig.canvas.manager.set_window_title('Confusion Matrix')
        ax = fig.add_axes([0.12, 0.18, 0.7, 0.7])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=Confusion_matrix)
        # colors = ['#f1e3e3', '#8e1d49', '#040d14']  # Use colors from the image, for example
        # cmap = ListedColormap(colors)
        cm_display.plot(ax=ax, cmap='Reds_r')
        for labels in cm_display.text_.ravel():
            labels.set_fontsize(16)  # Set desired font size
        path = "./Results/Confusion_%s.png" % (n + 1)
        plt.title("Confusion Matrix")
        plt.ylabel('Actual', fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.xlabel('Predicted', fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=13, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=13, fontweight='bold', color='k')
        plt.savefig(path)
        plt.show()


if __name__ == '__main__':
    plotConvResults()
    plot_results()
    Plot_ROC_Curve()
    plot_seg_results()
    Confusion()
