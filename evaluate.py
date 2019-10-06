from main import *


def evaluate():
    batch_size = 500

    """load test-set and test-loader"""
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    """load the model"""

    model = OurLogisticRegression(784, 10)
    model.load_state_dict(torch.load('model.pkl'))
    model.eval()  # important

    regularization = nn.CrossEntropyLoss()

    """testing"""
    correct = 0
    total = 0
    test_loss = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.view(-1, 28 * 28)
        outputs = model(images)
        test_loss += regularization(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += int((predicted == labels).sum())

    """results per epoch"""
    avg_acc = 100 * (correct / total)
    # avg_loss = test_loss/(i+1)

    print("TEST: Avg.Accuracy: %0.2f%%" % avg_acc)
    # print("Loss: %f" % avg_loss)

    return avg_acc


if __name__ == "__main__":
    acc = evaluate()
    print('')
