import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import os.path
from tqdm import tqdm

class CaptchaData(Dataset):
    # https://www.kaggle.com/datasets/fournierp/captcha-version-2-images
    def __init__(self):
        labels = []
        data = []
        for root, dirs, files in os.walk('samples', topdown=False):
            for name in files:
                if name.endswith('.png'):
                    file_path = f"samples/{name}"
                    image = Image.open(file_path).convert('L')

                    captcha = name.removesuffix('.png')
                    if len(set(captcha)) != len(captcha):
                        labels.append(1)
                    else:
                        labels.append(0)

                    file_data = np.asarray(image.getdata(), dtype=np.uint8).flatten()
                    data.append(file_data)

        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            torch.tensor(np.asarray(data).flatten() / 255.0, dtype=torch.float).view(-1, 1, 50, 200),
            torch.tensor(labels, dtype=torch.float),
            test_size=0.2)
        self.len = len(self.train_labels)

    def __getitem__(self, item):
        return self.train_data[item], self.train_labels[item]

    def __len__(self):
        return self.len

class CaptchaDuplicateClassifier(nn.Module):
    def __init__(self):
        super(CaptchaDuplicateClassifier, self).__init__()

        self.in_to_h1 = nn.Conv2d(1, 9, kernel_size=3, padding=(1,1)) # 1->9 Channels, 50x200->50x200
        # Maxpool2D 9->9 channels, 50x200->25x100
        # Pad 1 to end of Dim0 9->9 channels, 25x100->26x100

        self.h1_to_h2 = nn.Conv2d(9, 27, kernel_size=5, padding=(2,2)) #9->27 Channels, 26x100->26x100
        # Maxpool2D 27->27 channels, 26x100->13x50
        # Pad 1 to end of Dim0 27->27 channels, 13x50->14x50

        self.h2_to_h3 = nn.Linear(27 * 14 * 50, 40)
        self.h3_to_h4 = nn.Linear(40, 15)
        self.h4_to_out = nn.Linear(15, 1)

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))
        x = F.max_pool2d(x, (2,2))
        x = F.pad(x, (0, 0, 0, 1))

        x = F.relu(self.h1_to_h2(x))
        x = F.max_pool2d(x, (2,2))
        x = F.pad(x, (0, 0, 0, 1))

        x = torch.flatten(x, 1)
        x = F.relu(self.h2_to_h3(x))
        x = F.relu(self.h3_to_h4(x))
        out = F.sigmoid(self.h4_to_out(x))
        out = torch.flatten(out)

        return out

def train(epochs=5, batch_size=16, lr=0.001, display_test_acc=False):
    captcha = CaptchaData()
    captcha_loader = DataLoader(captcha, batch_size=batch_size, drop_last=True, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    classifier = CaptchaDuplicateClassifier().to(device)
    print(f"Total Parameters: {sum(param.numel() for param in classifier.parameters())}")

    cross_entropy = nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        for _, data in enumerate(tqdm(captcha_loader)):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = classifier(x)

            loss = cross_entropy(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0
        if display_test_acc:
            with torch.no_grad():
                raw_predictions = classifier(captcha.test_data.to(device))
                predictions = torch.empty(raw_predictions.shape[0])
                for i in range(0, raw_predictions.shape[0]):
                    predictions[i] = round(raw_predictions[i].item())
                correct = (predictions == captcha.test_labels.to(device)).sum().item()
                print(f"Accuracy on test set: {correct / len(captcha.test_labels):.4f}")

train(epochs=20, display_test_acc=True)