import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from Util.Confusion_Matrix_Generator import ConfusionMatrixGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MusicData(Dataset):
    def __init__(self, test=False):
        self.test = test
        df = pd.read_csv('music_cleaned.csv')
        X = df.iloc[:, 1:-1].to_numpy()
        y = df['genre'].to_numpy()
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(torch.tensor(X, dtype=torch.float32).view(-1, 6), torch.tensor(y, dtype=torch.long), test_size=0.3)
        self.classes = ['Country', 'Hip-Hop']

        if self.test:
            self.len = self.test_labels.shape[0]
        else:
            self.len = self.train_labels.shape[0]

    def __getitem__(self, item):
        if self.test:
            return self.test_data[item], self.test_labels[item]
        return self.train_data[item], self.train_labels[item]

    def __len__(self):
        return self.len

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.in_to_h1 = nn.Linear(6, 36)
        self.h1_to_h2 = nn.Linear(36, 250)
        self.h2_to_h3 = nn.Linear(250, 4)
        self.h3_to_out = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))
        x = F.dropout(x, 0.2)
        x = F.relu(self.h1_to_h2(x))
        x = F.relu(self.h2_to_h3(x))
        return self.h3_to_out(x)

def train(epochs=40, batch_size=16, lr=0.01, validation_frequency=5):
    data = MusicData()
    loader = DataLoader(data, batch_size=batch_size)
    classifier = Classifier()

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for _, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = classifier(x)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % validation_frequency == 0:
            with torch.no_grad():
                predictions = torch.argmax(classifier(data.test_data.to(device)), dim=1)
                correct = (predictions == data.test_labels.to(device)).sum().item()
                print(f"Accuracy on validation set for epoch {epoch + 1}: {correct / len(data.test_labels):.4f}")

    return classifier, loader

classifier, loader = train(epochs=50)

test_loader = DataLoader(MusicData(test=True))
cm_generator = ConfusionMatrixGenerator(model=classifier, test_loader=test_loader, device=device)
cm_generator.generate_confusion_matrix()