import math

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from Model.model_2048.Resnet18 import ResNet18     # TODO
import matplotlib.pyplot as plt
import os
import random
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

class_names=["BPSK","QPSK","8PSK","16QAM","32QAM","64QAM","GMSK","OQPSK","FQPSK","ARTM","SOQPSK","FM","PM",
             "16APSK","32APSK","BPSK_FM","BPSK_PM","QPSK_FM","QPSK_PM","FSK_PM","FQPSK_PM","SOQPSK_PM"]
snr_levels = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(7)

class IQDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as hdf:
            self.IQ_data = hdf['IQ_data'][:]
            self.class_ = hdf['class'][:]
            self.snr = hdf['snr'][:]  # 信噪比

    def __len__(self):
        return len(self.IQ_data)

    def __getitem__(self, idx):
        iq_data = self.IQ_data[idx]
        label = self.class_[idx]
        snr = self.snr[idx]
        # 转置操作
        iq_data = iq_data.transpose()
        return iq_data, label, snr

def plot_confusion_matrix(cm, classes, title, normalize=True, cmap=plt.cm.GnBu, filename='confusion_matrix.pdf'):   ##画混淆矩阵

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        if cm[i, j] > 0.995:
            plt.text(j, i, '{:.2f}'.format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else 'black')
        if cm[i, j] > 0.05 and cm[i, j] < 0.995:
            plt.text(j, i, '.' + '{:.2f}'.format(cm[i, j], fmt).split('.')[1], horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else 'black' )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, filename))
    plt.close()

def test_and_plot_cm(model, test_loader, device):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    snr_cm_data = {snr: {'preds': [], 'labels': []} for snr in snr_levels}

    with torch.no_grad():
        for inputs, labels, snrs in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for snr in snr_levels:
                indices = (snrs == snr)
                snr_cm_data[snr]['preds'].extend(preds[indices].cpu().numpy())
                snr_cm_data[snr]['labels'].extend(labels[indices].cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    np.save(os.path.join(result_folder, f'{model_name}_cm.npy'), cm)
    plot_confusion_matrix(cm, class_names, f'Overall Confusion Matrix', filename=f'{model_name}_overall_cm.pdf')


    for snr in snr_levels:
        cm = confusion_matrix(snr_cm_data[snr]['labels'], snr_cm_data[snr]['preds'])
        plot_confusion_matrix(cm, class_names, f'Confusion Matrix for SNR {snr}', filename=f'{model_name}_cm_snr_{snr}.pdf')

def test(model, test_loader, device):
    model.to(device)
    model.eval()
    class_acc = {class_name: {snr: [] for snr in snr_levels} for class_name in class_names}
    overall_acc = {snr: [] for snr in snr_levels}

    with torch.no_grad():
        for inputs, labels, snrs in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for snr in snr_levels:
                indices = (snrs == snr)
                snr_preds = preds[indices]
                snr_labels = labels[indices]
                if len(snr_labels) > 0:
                    overall_acc[snr].append(accuracy_score(snr_labels.cpu(), snr_preds.cpu()))

                for class_idx, class_name in enumerate(class_names):
                    class_indices = (labels[indices] == class_idx)
                    if class_indices.any():
                        class_preds = preds[indices][class_indices]
                        class_labels = labels[indices][class_indices]
                        acc = accuracy_score(class_labels.cpu(), class_preds.cpu())
                        class_acc[class_name][snr].append(acc)

    class_avg_acc = {class_name: [np.mean(accs) if accs else 0 for snr, accs in acc.items()] for class_name, acc in class_acc.items()}
    overall_avg_acc = [np.mean(overall_acc[snr]) if overall_acc[snr] else 0 for snr in snr_levels]
    return class_avg_acc, overall_avg_acc

if __name__ == '__main__':

    result_folder = './ResNet18'   # TODO
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    batch_size = 128
    # load data
    file_path = './data/TSML_GNU_2048.h5'  ######## TODO
    dataset = IQDataset(file_path)


    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



    # load model
    model = ResNet18() #TODO
    model_name = 'ResNet18'
    model.load_state_dict(torch.load('./result/2048/ResNet18_2048.pth'))

    class_avg_acc, overall_avg_acc = test(model, test_loader, device)

    np.save(os.path.join(result_folder, f'{model_name}_class_avg_acc.npy'), class_avg_acc)
    np.save(os.path.join(result_folder, f'{model_name}_overall_avg_acc.npy'), overall_avg_acc)

    plt.figure(figsize=(15, 10))
    for class_name, snr_acc in class_avg_acc.items():
        plt.plot(snr_levels, snr_acc, label=class_name)
    plt.xlabel('SNR')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy for Each Class at Different SNR Levels')
    plt.legend()
    plt.savefig(os.path.join(result_folder, f'{model_name}_class_accuracy_snr.png'))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(snr_levels, overall_avg_acc, label='Overall Accuracy')
    plt.xlabel('SNR')
    plt.ylabel('Accuracy')
    plt.title('Overall Classification Accuracy at Different SNR Levels')
    plt.legend()
    plt.savefig(os.path.join(result_folder, f'{model_name}_overall_accuracy_snr.png'))
    plt.show()

    test_and_plot_cm(model, test_loader, device)

