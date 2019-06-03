import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

has_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if has_cuda else "cpu")
torch.backends.cudnn.endbale = has_cuda

advection_type = "ABC"
root_folder = "~/Programming/testingStuff/particleAdvection/Samples/" + advection_type + "/"
input_folder = "Input_Images/"
output_folder = "Output_Images/"

class Convoluation(nn.Module):
    def __init__(self,
            n_epochs=1,
            batch_size_train=64,
            batch_size_test = 1000,
            learning_rate = 0.1,
            momentum = 0.5,
            log_interval = 10,
            ):
        super(Convoluation,self).__init__()
        self.n_epochs = n_epochs
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.log_interval = log_interval
        optimize()

        self.conv1 = nn.Conv2d(1,10, kernel_size=5)
        self.fc1 = nn.Linear(a,y)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = x.view(-1,a)
        x = self.fc1(x)
        return F.log_softmax(x)

    def optimize(self):
        self.optimizer = optim.SGD(self.parameters(), self.learning_rate, self.momentum)
        #self.optimizer = optim.Adam(self.parameters(), self.learning_rate, self.momentum)

    def train_network(self,epoch):
        self.train()
        optimizer = self.optimizer
        for batch_idx, (data,target) in enumerate(train_data):
            optimizer.zero_grad()
            output = 0 # Something with data
            loss = F.nil_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                # Fix training data
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(something.dataset), 
                    100.*batch_idx / len(training_data), loss.item()))
                train_losses.appen(loss.items())
                # Training data
                train_counter.append((batch_idx*64) + (epoch-1)*len(training_data)))

    def test_network(self):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            # Test data
            for data, target in test_data:
                output = 0 # Something with data
                test_loss += F.nill_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                # Test data
                correct += pred.eq(test_data)
        # Test data
        test_loss /= len(test_data)
        test_losses.append(test_loss)
        # Test data
        print('\nTest set: Avg loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_dataset), 100.*correct/len(test_dataset)))

    def run(self):
        self.test_network()
        for epoch in range(1, self.n_epochs+1):
            self.train_network(epoch)
            self.test_network()


def main():
    conv = Convoluation()
    print(conv)

if __name__ == '__main__':
    main()

