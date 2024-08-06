import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle

# Vérifier si CUDA est disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Définition du modèle
class SimpleCNN(nn.Module):
    def __init__(self, num_filters, kernel_size, fc_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=kernel_size, stride=1, padding=1)
        self.flatten = nn.Flatten()

        # Calculer dynamiquement la taille de la sortie après les convolutions et le pooling
        self.fc_input_size = self._get_conv_output_size(num_filters, kernel_size)
        self.fc1 = nn.Linear(self.fc_input_size, fc_size)
        self.fc2 = nn.Linear(fc_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    # Créer un tenseur de test pour passer à travers les couches convolutionnelles
    def _get_conv_output_size(self, num_filters, kernel_size):
        x = torch.rand(1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.numel()

# Chargement des données
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Hyperparamètres pour la grid search
param_grid = {
    'num_filters': [32, 64],
    'kernel_size': [3, 5],
    'fc_size': [128, 256],
    'learning_rate': [0.01, 0.001]
}

num_epochs = 5

# Fonction d'entraînement et d'évaluation
def train_and_evaluate(params):
    model = SimpleCNN(params['num_filters'], params['kernel_size'], params['fc_size']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_recalls = []
    val_recalls = []
    train_f1s = []
    val_f1s = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        # Parcours des batches d'entraînement
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Déplacer les données sur le GPU
            optimizer.zero_grad()  # Réinitialiser les gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Calculer la perte
            loss.backward()  # Calculer les gradients
            optimizer.step()  # Mettre à jour les poids du modèle
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct / total)
        train_recalls.append(recall_score(all_labels, all_preds, average='macro'))
        train_f1s.append(f1_score(all_labels, all_preds, average='macro'))

        # Évaluation sur l'ensemble de validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_losses.append(val_loss / len(test_loader))
        val_accuracies.append(correct / total)
        val_recalls.append(recall_score(all_labels, all_preds, average='macro'))
        val_f1s.append(f1_score(all_labels, all_preds, average='macro'))

        print(f'Epoch {epoch+1}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}, '
              f'Training Accuracy: {train_accuracies[-1]}, Validation Accuracy: {val_accuracies[-1]}, '
              f'Training Recall: {train_recalls[-1]}, Validation Recall: {val_recalls[-1]}, '
              f'Training F1 Score: {train_f1s[-1]}, Validation F1 Score: {val_f1s[-1]}')

    return model, train_losses, val_losses, train_accuracies, val_accuracies, train_recalls, val_recalls, train_f1s, val_f1s

# Fonction pour sauvegarder le modèle
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Fonction pour charger le modèle
def load_model(model_class, path, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.to(device)
    return model

# Grid search
best_params = None
best_accuracy = 0
best_train_losses, best_val_losses, best_train_accuracies, best_val_accuracies = None, None, None, None
best_train_recalls, best_val_recalls, best_train_f1s, best_val_f1s = None, None, None, None

worst_params = None
worst_accuracy = 1
worst_train_losses, worst_val_losses, worst_train_accuracies, worst_val_accuracies = None, None, None, None
worst_train_recalls, worst_val_recalls, worst_train_f1s, worst_val_f1s = None, None, None, None

all_results = []

for params in ParameterGrid(param_grid):
    print(f"Testing params: {params}")
    model, train_losses, val_losses, train_accuracies, val_accuracies, train_recalls, val_recalls, train_f1s, val_f1s = train_and_evaluate(params)
    final_accuracy = val_accuracies[-1]
    all_results.append((params, final_accuracy))
    if final_accuracy > best_accuracy:
        best_accuracy = final_accuracy
        best_params = params
        best_train_losses = train_losses
        best_val_losses = val_losses
        best_train_accuracies = train_accuracies
        best_val_accuracies = val_accuracies
        best_train_recalls = train_recalls
        best_val_recalls = val_recalls
        best_train_f1s = train_f1s
        best_val_f1s = val_f1s
        save_model(model, 'best_model.pth')
    if final_accuracy < worst_accuracy:
        worst_accuracy = final_accuracy
        worst_params = params
        worst_train_losses = train_losses
        worst_val_losses = val_losses
        worst_train_accuracies = train_accuracies
        worst_val_accuracies = val_accuracies
        worst_train_recalls = train_recalls
        worst_val_recalls = val_recalls
        worst_train_f1s = train_f1s
        worst_val_f1s = val_f1s
        save_model(model, 'worst_model.pth')

# Sauvegarder les meilleurs et pires hyperparamètres et résultats
with open('best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)
with open('worst_params.pkl', 'wb') as f:
    pickle.dump(worst_params, f)
with open('best_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)
    
print(f"Best params: {best_params} with accuracy: {best_accuracy}")
print(f"Worst params: {worst_params} with accuracy: {worst_accuracy}")
    
# Visualisation des résultats de la meilleure et de la pire configuration
model_best = load_model(SimpleCNN, 'best_model.pth', best_params['num_filters'], best_params['kernel_size'], best_params['fc_size'])
model_worst = load_model(SimpleCNN, 'worst_model.pth', worst_params['num_filters'], worst_params['kernel_size'], worst_params['fc_size'])

epochs = range(1, num_epochs + 1)

# Graphiques pour le meilleur modèle
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

axs[0, 0].plot(epochs, best_train_losses, 'g', label='Best Training loss')
axs[0, 0].plot(epochs, best_val_losses, 'b', label='Best Validation loss')
axs[0, 0].set_title('Best Model Training and Validation Loss')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

axs[0, 1].plot(epochs, best_train_accuracies, 'g', label='Best Training accuracy')
axs[0, 1].plot(epochs, best_val_accuracies, 'b', label='Best Validation accuracy')
axs[0, 1].set_title('Best Model Training and Validation Accuracy')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].legend()

axs[1, 0].plot(epochs, best_train_recalls, 'g', label='Best Training recall')
axs[1, 0].plot(epochs, best_val_recalls, 'b', label='Best Validation recall')
axs[1, 0].set_title('Best Model Training and Validation Recall')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Recall')
axs[1, 0].legend()

axs[1, 1].plot(epochs, best_train_f1s, 'g', label='Best Training F1 Score')
axs[1, 1].plot(epochs, best_val_f1s, 'b', label='Best Validation F1 Score')
axs[1, 1].set_title('Best Model Training and Validation F1 Score')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('F1 Score')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# Graphiques pour le pire modèle
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

axs[0, 0].plot(epochs, worst_train_losses, 'r', label='Worst Training loss')
axs[0, 0].plot(epochs, worst_val_losses, 'm', label='Worst Validation loss')
axs[0, 0].set_title('Worst Model Training and Validation Loss')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

axs[0, 1].plot(epochs, worst_train_accuracies, 'r', label='Worst Training accuracy')
axs[0, 1].plot(epochs, worst_val_accuracies, 'm', label='Worst Validation accuracy')
axs[0, 1].set_title('Worst Model Training and Validation Accuracy')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].legend()

axs[1, 0].plot(epochs, worst_train_recalls, 'r', label='Worst Training recall')
axs[1, 0].plot(epochs, worst_val_recalls, 'm', label='Worst Validation recall')
axs[1, 0].set_title('Worst Model Training and Validation Recall')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Recall')
axs[1, 0].legend()

axs[1, 1].plot(epochs, worst_train_f1s, 'r', label='Worst Training F1 Score')
axs[1, 1].plot(epochs, worst_val_f1s, 'm', label='Worst Validation F1 Score')
axs[1, 1].set_title('Worst Model Training and Validation F1 Score')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('F1 Score')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

from PIL import Image

# Charger une image de test
image_path = "1290.png"
image = Image.open(image_path).convert('L')
image = image.resize((28, 28))
image = transform(image).unsqueeze(0).to(device)

# Faire une prédiction
model_best.eval()
with torch.no_grad():
    output = model_best(image)
    _, predicted = torch.max(output, 1)

# Afficher l'image et la prédiction
plt.imshow(image.cpu().squeeze(), cmap='gray')
plt.title(f'Predicted: {predicted.item()}')
plt.show()
