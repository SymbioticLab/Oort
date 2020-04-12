import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import pickle
import h5py as h5
import sys
sys.path.append("..")
from utils.stackoverflow import *

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        output = F.sigmoid(self.linear(x))
        return output




def create_logistic_model(vocab_tokens_size, vocab_tags_size):
    """Logistic regression to predict tags of StackOverflow.

    Args:
        vocab_tokens_size: Size of token vocabulary to use.
        vocab_tags_size: Size of token vocabulary to use.

    """
    model = LogisticRegression(vocab_tokens_size, vocab_tags_size)
    return model




def train(vocab_tokens_size=10000, vocab_tags_size=500):
    # Hyper-parameters 
    input_size = 28 * 28    # 784
    num_classes = 10
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001

    # Logistic regression model
    model = create_logistic_model(vocab_tokens_size, vocab_tags_size)


    train_dataset = stackoverflow('/users/xzhu/tag/stackoverflow_tf/', True)
    test_dataset = stackoverflow('/users/xzhu/tag/stackoverflow_tf/', False)
    #print(train_dataset.__getitem__(0)[0].size())

    # Data loader (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)

    # Loss and optimizer
    # nn.CrossEntropyLoss() computes softmax internally
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (text, labels) in enumerate(train_loader):

            # Forward pass
            outputs = model(text)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # # Test the model
    # # In test phase, we don't need to compute gradients (for memory efficiency)
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in test_loader:
    #         images = images.reshape(-1, input_size)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum()

    #     print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))



train()
