import numpy as np
import torch
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class EnergyConsumptionData(Dataset):
    def __init__(self):
        # https://www.kaggle.com/datasets/samxsam/household-energy-consumption

        def eval_consumption(consumption):
            if consumption <= 6.0:
                return 0
            elif 6.0 < consumption <= 10.40:
                return 1
            elif 10.40 < consumption <= 14.8:
                return 2
            else:
                return 3

        df = pd.read_csv("household_energy_consumption.csv")
        df['Has_AC'] = df['Has_AC'].replace({'Yes': 1, 'No': 0}).astype(int)
        df = df[['Avg_Temperature_C', 'Has_AC', 'Energy_Consumption_kWh']]
        df['Avg_Temperature_C'] -= np.average(df['Avg_Temperature_C'], axis=0)
        df['Avg_Temperature_C'] /= np.std(df['Avg_Temperature_C'], axis=0)

        data = df[['Avg_Temperature_C', 'Has_AC']].to_numpy()
        labels = df['Energy_Consumption_kWh'].apply(eval_consumption).to_numpy()

        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            torch.tensor(np.asarray(data).flatten(), dtype=torch.float).view(-1, 2),
            torch.tensor(labels, dtype=torch.int64),
            test_size=0.2)
        self.len = len(self.train_labels)

    def __getitem__(self, item):
        return self.train_data[item], self.train_labels[item]

    def __len__(self):
        return self.len

class EnergyConsumptionDataPredictor(nn.Module):
    def __init__(self):
        # Call the constructor of the super class
        super(EnergyConsumptionDataPredictor, self).__init__()

        self.in_to_h1 = nn.Linear(2, 6)
        self.h1_to_h2 = nn.Linear(6, 4)
        self.h2_to_h3 = nn.Linear(4, 4)

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))
        x = F.relu(self.h1_to_h2(x))
        out = F.relu(self.h2_to_h3(x))
        return out

def trainNN(epochs=5, batch_size=16, lr=0.001, display_test_acc=False):
    # load dataset
    mnist = EnergyConsumptionData()

    # create data loader
    mnist_loader = DataLoader(mnist, batch_size=batch_size, drop_last=True, shuffle=True)

    # determine which device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create CNN
    number_classify = EnergyConsumptionDataPredictor().to(device)
    print(f"Total parameters: {sum(param.numel() for param in number_classify.parameters())}")

    # loss function
    ce_loss = nn.CrossEntropyLoss()

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(number_classify.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        for _, data in enumerate(tqdm(mnist_loader)):
            x, y = data

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = number_classify(x)

            #output = output.reshape(-1, 4)
            loss = ce_loss(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0
        if display_test_acc:
            with torch.no_grad():
                test_out = number_classify(mnist.test_data.to(device))#.reshape(-1, 4)
                predictions = torch.argmax(test_out, dim=1)  # Get the prediction
                correct = (predictions == mnist.test_labels.to(device)).sum().item()
                print(f"Accuracy on test set: {correct / len(mnist.test_labels):.4f}")


trainNN(epochs=5, display_test_acc=True)