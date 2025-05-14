import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ConfusionMatrixGenerator:
    def __init__(self, model, test_loader, device):
        """
        Initialize the ConfusionMatrixGenerator.

        Args:
            model (torch.nn.Module): The trained model.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
            device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def generate_confusion_matrix(self):
        """
        Generate and display the confusion matrix for the test dataset.
        """
        self.model.eval()  # Set the model to evaluation mode
        all_preds = []
        all_labels = []

        # Collect predictions and true labels
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # Use dataset instance for label names
        class_names = self.test_loader.dataset.classes
        cm = confusion_matrix(all_labels, all_preds, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.tight_layout()
        plt.show()