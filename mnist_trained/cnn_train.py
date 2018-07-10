import torch
import torch.nn as nn
import torchvision
import mnist as m
import numpy as np

# Do we have a fancy GPU?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Parameters
num_epochs = 100
num_classes = 10	# i.e. the digits
batch_size = 300	# How many samples per batch use
learning_rate = 0.001

# Load MNIST data into 
X_train = []
X_test = []
Y_train = []
Y_test = []

for element in m.read("training"):
	Y_train.append(element[0])
	X_train.append(element[1])

for element in m.read("testing"):
	Y_test.append(element[0])
	X_test.append(element[1])

X_train = np.expand_dims(np.asarray(X_train, dtype=np.float32),axis=1)
X_test = np.expand_dims(np.asarray(X_test, dtype=np.float32),axis=1)
Y_train = np.asarray(Y_train, dtype=np.int64)
Y_test = np.asarray(Y_test, dtype=np.int64)

train_set = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
test_set = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True) # We want to random sample the data at each epoch
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = batch_size, shuffle = False) # No need in this case

# C.N.N. (with two convolutional layers)
class CNN(nn.Module): # In Pytorch, our models must subclass this class
	def __init__(self, num_classes = 10):
		# First of all, respect the dependencies!
		super(CNN, self).__init__()
		# Layer 1:
		# > 2d convolution with 16 different channel-outs,
		#   since the kernel size is 5, we need a zero-padding
		#   of 2 pixels.
		# > Batch Normalization step over our batch of 100 samples.
		# > Filter only positive values with ReLU activation function.
		# > Maximal Pooling Map with 2x2 kernel and stride of 2.
		# Layer 2:
		# > 2d convolution, each one with 2 different 
		#   outputs (32 final out-channels).
		# > Batch Normalization.
		# > ReLU activation function.
		# > Maximal 2x2 Pooling Map.
		# Final Stage:
		# > Linear classificator over 7 * 7 * 32 elements to 10 classes.
		#   N.B. remember that the MNIST database is made of 28x28px images,
		#   therefore, after two 2x2 pooling maps we will have 7x7px images.
		self.layer1 = nn.Sequential( # A sequential container which can contain other 'nn' modules
			nn.Conv2d(1, 16, kernel_size = 5, stride = 1, padding = 2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2)
			)
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2)
			)
		self.final_stage = nn.Linear(7 * 7 * 32, num_classes)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1)
		out = self.final_stage(out)
		return out

# Create and load model over (if any) GPU
model = CNN(num_classes).to(device)

# Define Error function and Optimizer function
error_function = nn.CrossEntropyLoss()
optimizer_function = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Training
n_steps = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)

		# Forward pass the image
		output = model(images)
		error = error_function(output, labels)

		# Backward propagation and optimization
		optimizer_function.zero_grad() # Clear gradient infos
		error.backward() # Re-compute backpropagation
		optimizer_function.step() # Single optimization step
		
		if (i+1) % 100 == 0:
			print("Epoch corrente [{}/{}], Step [{}/{}], Errore: {:.4f}".format(epoch + 1, num_epochs, i+1, n_steps, error.item()))

# Testing
model.eval()
with torch.no_grad(): # Disable gradient calculation to save pretious computational time
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, prediction = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (prediction == labels).sum().item()
	print("Accuratezza della CNN su di un test di 10000 immagini: {} %".format(100 * correct / total))

# Saving
torch.save(model, "cnn.pt")