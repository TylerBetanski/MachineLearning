import numpy as np
import torch
import pandas as pd
from nltk import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class HousePriceData(Dataset):
    def __init__(self):
        # https://www.kaggle.com/datasets/romanahmed2024/house-price-prediction

        df = pd.read_csv("house_price_regression_dataset.csv")
        data = df[['Feature1', 'Feature2', 'Feature5', 'Feature6', 'Feature8', 'Feature10']].copy()
        data[df.select_dtypes("int64").columns] = data[df.select_dtypes("int64").columns].astype("float64")
        float_cols = data.select_dtypes("float64").columns

        data[float_cols] -= np.average(data[float_cols], axis=0)
        data[float_cols] /= np.average(data[float_cols], axis=0)

        data = data.to_numpy()
        labels = df['Price'].to_numpy()

        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            torch.tensor(np.asarray(data).flatten(), dtype=torch.float).view(-1, 6),
            torch.tensor(labels, dtype=torch.float),
            test_size=0.2)
        self.len = len(self.train_labels)

    def __getitem__(self, item):
        return self.train_data[item], self.train_labels[item]

    def __len__(self):
        return self.len

class HousePricePredictor(nn.Module):
    def __init__(self):
        # Call the constructor of the super class
        super(HousePricePredictor, self).__init__()

        self.in_to_h1 = nn.Linear(6, 12)
        self.h1_to_h2 = nn.Linear(12, 4)
        self.h2_to_h3 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))
        x = F.relu(self.h1_to_h2(x))
        out = F.relu(self.h2_to_h3(x))
        return out

def trainNN(epochs=5, batch_size=16, lr=0.001, display_test_acc=False):
    # load dataset
    house = HousePriceData()

    # create data loader
    house_loader = DataLoader(house, batch_size=batch_size, drop_last=True, shuffle=True)

    # determine which device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create CNN
    predictor = HousePricePredictor().to(device)
    print(f"Total parameters: {sum(param.numel() for param in predictor.parameters())}")

    # loss function
    mse_loss = nn.MSELoss(reduction='sum')

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        for _, data in enumerate(tqdm(house)):
            x, y = data

            x = x.to(device)

            y = y.detach().clone().reshape(-1)
            y = y.to(device)

            optimizer.zero_grad()

            output = predictor(x)

            loss = mse_loss(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0
        if display_test_acc:
            with torch.no_grad():
                test_out = predictor(house.test_data.to(device))#.reshape(-1, 4)
                predictions = torch.argmax(test_out, dim=1).to("cpu")  # Get the prediction
                true_labels = np.array(house.test_labels)
                acc = np.average((true_labels-np.array(predictions))**2)
                print(f"Accuracy on test set: {acc:.4f}")


trainNN(epochs=5, display_test_acc=True)