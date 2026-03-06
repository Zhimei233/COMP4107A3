import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# PyTorch dataset for the UWaveGestureLibrary dataset
class UWaveGestureLibraryDataset(torch.utils.data.Dataset):
  
  _ONE_HOT = torch.eye(8, dtype=torch.float32)

  def __init__(self, dataset_filepath):
    # dataset_filepath is the full path to a file containing data
    super().__init__()
    self.samples = []   

    with open(dataset_filepath, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(':')
            # parts[0..2] → x, y, z axes;  
            # parts[3] → label
            x_vals = [float(v) for v in parts[0].split(',')]
            y_vals = [float(v) for v in parts[1].split(',')]
            z_vals = [float(v) for v in parts[2].split(',')]
            label  = int(float(parts[3]))          # 1-based

            x_tensor = torch.tensor(
                [x_vals, y_vals, z_vals], dtype=torch.float32
            )                                      # shape: (3, 315)
            y_tensor = self._ONE_HOT[label - 1]    # shape: (8,)

            self.samples.append((x_tensor, y_tensor))
    # Return nothing    

  def __len__(self):
    num_samples = len(self.samples)
    # num_samples is the total number of samples in the dataset
    return num_samples


  def __getitem__(self, index):
    # index is the index of the sample to be retrieved
    # x is one sample of data
    # y is the label associated with the sample
    x, y = self.samples[index]
    return x, y


# A function that creates a cnn model to predict which class a sequence corresponds to
def u_wave_gesture_library_cnn_model(training_data_filepath):
  # training_data_filepath is the full path to a file containing the training data
  
  dataset = UWaveGestureLibraryDataset(training_data_filepath)

  # Split dataset, 80% train, 20% validation
  train_size = int(0.8 * len(dataset))
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

  class CNNModel(nn.Module):
    def __init__(self):
      super().__init__()

      # Define a simple CNN architecture suitable for the input shape (3, 315)
      self.conv = nn.Sequential(
        nn.Conv1d(3, 16, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.Conv1d(16, 32, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool1d(2)
      )

      self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 75, 64),
        nn.ReLU(),
        nn.Linear(64, 8)
      )

    # Define the forward pass of the CNN
    def forward(self, x):
      x = self.conv(x)
      x = self.fc(x)
      return x

  model = CNNModel()

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  epochs = 10

  for epoch in range(epochs):
    model.train()

    for x, y in train_loader:
      labels = torch.argmax(y, dim=1)

      optimizer.zero_grad()

      outputs = model(x)
      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()

  def evaluate(loader):
    model.eval()
    correct = 0
    total = 0

    # Evaluate the model on the given data loader and return the accuracy
    with torch.no_grad():
      for x, y in loader:
        labels = torch.argmax(y, dim=1)

        outputs = model(x)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total if total > 0 else 0

  training_performance = evaluate(train_loader)
  validation_performance = evaluate(val_loader)

  # model is a trained cnn model to predict which class a sequence corresponds to
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set
  return model, training_performance, validation_performance


# A function that creates an rnn model to predict which class a sequence corresponds to
def u_wave_gesture_library_rnn_model(training_data_filepath):
  # training_data_filepath is the full path to a file containing the training data
  
  dataset = UWaveGestureLibraryDataset(training_data_filepath)

  # Split dataset, 80% train, 20% validation
  train_size = int(0.8 * len(dataset))
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

  class RNNModel(nn.Module):
    def __init__(self):
      super().__init__()

      self.rnn = nn.RNN(
        input_size=3,
        hidden_size=64,
        batch_first=True
      )

      self.fc = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 8)
      )

    def forward(self, x):
      # x shape: (batch, 3, 315)
      # -> (batch, 315, 3)
      x = x.permute(0, 2, 1)  

      out, _ = self.rnn(x)

      # take last timestep
      out = out[:, -1, :]

      out = self.fc(out)
      return out

  model = RNNModel()

  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  epochs = 10

  # Training
  for epoch in range(epochs):
    model.train()

    for x, y in train_loader:
      labels = torch.argmax(y, dim=1)

      optimizer.zero_grad()

      outputs = model(x)
      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()

  def evaluate(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
      for x, y in loader:
        labels = torch.argmax(y, dim=1)

        outputs = model(x)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total if total > 0 else 0

  training_performance = evaluate(train_loader)
  validation_performance = evaluate(val_loader)
  # model is a trained rnn model to predict which class a sequence corresponds to
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set
  return model, training_performance, validation_performance
