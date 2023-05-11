import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Plot the train&test loss and accuracy trendline acorss epochs
def PlotEpoch(test_acc_list, test_loss_list, train_acc_list, train_loss_list):

    n = len(train_loss_list)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(list(range(n)), train_acc_list, label='Train')
    ax[0].plot(list(range(n)), test_acc_list, label='Test')
    ax[0].set_title('Accuracy')
    ax[0].legend()
    ax[1].plot(list(range(n)), train_loss_list, label='Train')
    ax[1].plot(list(range(n)), test_loss_list, label='Test')
    ax[1].set_title('Loss')
    ax[1].legend()

# Get the predicted labels and the true labels of the current model
def GetPredictions(model, testloader, device):
    model.eval()

    all_pred=[]
    all_label=[]
    for i, (x, y) in enumerate(testloader):

        x = x.to(device)

        pred = model(x)
        pred=torch.argmax(pred,dim=1)


        all_pred+=pred.cpu().tolist()
        all_label+=y.tolist()
    
    return all_pred,all_label

# Show the images where the model predicted the labels wrong
def ShowMistakes(pred, label, valid_set):
    indices = np.where(np.array(label)!= np.array(pred))[0]

    for idx in indices:
        print(idx)
        X,label=valid_set.__getitem__(idx)
        print('pred',pred[idx])

        print('truth',label)
        transform=transforms.ToPILImage()
        X=transform(X).show()

# Display the confusion matrix of the current predictions
def ShowConfusionMatrix(pred, label):
    conf_mat = confusion_matrix(label,pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


