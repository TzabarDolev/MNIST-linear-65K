import torch
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn


class OurLogisticRegression(nn.Module):
    """NN model"""
    def __init__(self, inpt_size, n_classes):
        super(OurLogisticRegression, self).__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(inpt_size, 75),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=75),

            # Layer 2
            nn.Linear(75, 50),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=50),

            # Layer 3
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=30),

            # Layer 4
            nn.Linear(30, n_classes),
            nn.LogSoftmax()
        )

    def forward(self, x):
        out = self.net(x)
        return out


def visualize(n, train_err, test_err, train_loss, test_loss, title, file_name, save_plot):
    """create graphs for train and test error and loss"""
    f, (ax1, ax2) = plt.subplots(1, 2)
    # f.suptitle(title)

    ax1.plot(range(n), train_loss, test_loss)
    ax1.set_yticks(np.arange(0, 0.6, 0.05))
    ax1.set_title("Loss")
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend(["train", "test"])
    ax1.set_ylim(bottom=0, top=0.6)
    ax1.grid()

    ax2.plot(range(n), train_err, test_err)
    ax2.set_title("Error")
    ax2.set_yticks(np.arange(0, 0.6, 0.05))
    ax2.set_ylabel('Error')
    ax2.set_xlabel('Epochs')
    ax2.legend(["train", "test"])
    ax2.set_ylim(bottom=0)
    ax2.grid()

    f.tight_layout()
    plt.show()
    if save_plot:
        f.savefig(file_name)
    return


def testing_model(model, loader, criteria, optimizer, name):
    """testing the model"""
    correct = 0
    total = 0
    test_loss = 0
    for i, (images, labels) in enumerate(loader):
        images = images.view(-1, 28 * 28)

        if name == "train":
            """TRAIN"""
            optimizer.zero_grad()
            outputs = model(images)  # forward on the model for this batch
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()
            test_loss += loss
        else:
            """TEST"""
            outputs = model(images)
            test_loss += criteria(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += int((predicted == labels).sum())

    """results per epoch"""
    avg_loss = test_loss/(i+1)
    avg_err = 1 - (correct / total)

    return avg_err, avg_loss


"""counting network parameters"""
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def myround(x, base=5):
    return int(base * round(x/base))


def main():

    """problem parameters"""
    num_classes = 10
    input_size = 784

    """Hyper parameters"""
    num_epochs = 80
    learning_rate = 0.005625
    batch_size = 150
    criterion_method = nn.CrossEntropyLoss()

    save_flag = True  # save plot or not

    """download the MNIST train/test dataset"""
    train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False)

    """Data Loader- allows iterating over our data in batches"""
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    """initializing the model"""
    model = OurLogisticRegression(input_size, num_classes)
    print("Trainig Neural Net with %d parameters, lr: %f, batch size: %d"
          % (get_n_params(model), learning_rate, batch_size))

    """Loss and Optimizer"""
    criterion = criterion_method
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    """Training the model"""
    test_error = []
    train_error = []
    train_loss = []
    test_loss = []
    max_epoch = 0
    for epoch in range(num_epochs):
        """TRAIN"""
        avg_train_err, avg_train_loss = testing_model(model, train_loader, criterion, optimizer, "train")
        train_error.append(avg_train_err)
        train_loss.append(avg_train_loss)
        # print('Epoch: [%d/%d], TRAIN: Avg.Loss: %0.4f, Avg.Err: %f'
        #       % (epoch + 1, num_epochs, avg_train_loss, avg_train_err))

        """TEST"""
        """testing the model -TEST"""
        avg_test_err, avg_test_loss = testing_model(model, test_loader, criterion, optimizer, "test")
        test_error.append(avg_test_err)
        test_loss.append(avg_test_loss)
        # print('Epoch: [%d/%d], TEST: Avg.Loss: %0.4f, Avg.Err: %f'
        #       % (epoch + 1, num_epochs, avg_test_loss, avg_test_err))
        print('Epoch: [%d/%d], TEST Accuracy: %0.4f'
              % (epoch + 1, num_epochs, 1-avg_test_err))

        if 1-avg_test_err > max_epoch:
            max_epoch = 1-avg_test_err
            if max_epoch > 0.98:
                torch.save(model.state_dict(), 'acc_'+str(int(max_epoch*10000))+'.pkl')
                print("Max Accuracy: %f" % max_epoch)

    print("Max Accuracy: %f" % max_epoch)
    print("")

    """Get Graphs"""
    hyper_params = "Batch-Size: " + str(batch_size) + ", #-Epochs: " + str(num_epochs) + \
                   ", Learning-Rate: " + str(learning_rate)
    file_name = (str(batch_size)+"-"+str(num_epochs)+"-"+str(learning_rate)).replace(".","")+".png"
    visualize(num_epochs, train_error, test_error, train_loss, test_loss, hyper_params, file_name, save_flag)

    """save the model"""
    torch.save(model.state_dict(), 'model.pkl')

    return {'loss': -max_epoch, 'status': STATUS_OK}


if __name__ == "__main__":
    main()
