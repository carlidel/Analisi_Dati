import os
import image_tools as img
import sudoku_extractor as sud
import sudoku_solver as sol
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import numpy as np
import pickle

# Comparing function:
def compare_solution(prediction, reality):
    correct = 0
    wrong = 0
    for i in range(len(prediction)):
        if prediction[i] != 0:
            if prediction[i] == reality[i]:
                correct += 1
            else:
                wrong += 1
                print("Errore! Previsto {}, ma era un {}.".format(prediction[i], reality[i]))
    return (correct + wrong, correct, wrong)


# In order to reload the previously trained CNN, we must have the model class definition in the program scope!
class CNN(nn.Module):
    def __init__(self, num_classes = 10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
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

# Let's load our models!

# Since the CNN is already trained, we can work on the CPU this time
device = torch.device('cpu')

# Create and load model over CPU
cnn_mnist = torch.load('cnn_mnist.pt', map_location=lambda storage, loc: storage)
cnn_74k = torch.load('cnn_74k.pt', map_location=lambda storage, loc: storage)

# Load SVMs
svm_mnist = pickle.load(open("svm_mnist.sav", 'rb'))
svm_74k = pickle.load(open("svm_74k.sav", 'rb'))

# Load selected and trained TPot model
tpot_mnist = pickle.load(open("tpot_mnist.sav", 'rb'))

# Load images
path = Path("img/")
items =  [f for f in os.listdir(path) if f.endswith('.jpg')]

# Load correct known numbers
solution = []
sol_paths = [f for f in os.listdir(path) if f.endswith('.txt')]
for sol_path in sol_paths:
    file = open(path / sol_path)
    s = [s for s in file][0]
    solution.append([int(c) for c in s])

# Now solve ALL the sudokus!
j = -1

total_numbers = 0
wrong_cnn_mnist = 0
wrong_cnn_74k = 0
wrong_svm_mnist = 0
wrong_svm_74k = 0
wrong_tpot_mnist = 0
perfect_sudokus = 0

for image in items:
    print("Si lavora ora su: " + image)
    perfect_sudoku = False
    j = j + 1
    
    if not os.path.exists(path / image.replace(".jpg","") / ""):
        os.makedirs(path / image.replace(".jpg","") / "")
    img.path = path / image.replace(".jpg","") / ""
    sudoku = sud.sudoku_extractor(path / image)

    # CNN MNIST

    numbers = []

    for row in sudoku.cells:
        for cell in row:
            if np.count_nonzero(cell) == 0:
                numbers.append(0)
            else:
                # Shrink it!
                res_cell = cv2.resize(cell, (28,28), interpolation = cv2.INTER_AREA)
                res_cell = res_cell / np.max(res_cell) * 255.
                # Value it!
                tensor = torch.from_numpy(res_cell.astype(np.float32))
                tensor.unsqueeze_(0)
                tensor.unsqueeze_(0)
                tensor = tensor.to(device)
                outputs = cnn_mnist(tensor)
                _, prediction = torch.max(outputs.data, 1)
                numbers.append(int(prediction[0]))
    for i in range(0,81,9):
        print(numbers[i:i+9])

    total, correct, wrong = compare_solution(numbers, solution[j])
    print("CNN_MNIST: {}/{}".format(correct, total))
    wrong_cnn_mnist += wrong

    if total == correct:
        print("Predetto tutto correttamente! La soluzione è:")
        sol.sudoku_solver([[numbers[i*9 + j] for j in range(9)] for i in range(9)])
        perfect_sudoku = True

    print("")

    total_numbers += total

    # CNN 74k

    numbers = []

    for row in sudoku.cells:
        for cell in row:
            if np.count_nonzero(cell) == 0:
                numbers.append(0)
            else:
                # Shrink it!
                res_cell = cv2.resize(cell, (28,28), interpolation = cv2.INTER_AREA)
                res_cell = res_cell / np.max(res_cell) * 255.
                # Value it!
                tensor = torch.from_numpy(res_cell.astype(np.float32))
                tensor.unsqueeze_(0)
                tensor.unsqueeze_(0)
                tensor = tensor.to(device)
                outputs = cnn_74k(tensor)
                _, prediction = torch.max(outputs.data, 1)
                numbers.append(int(prediction[0]))
    for i in range(0,81,9):
        print(numbers[i:i+9])

    total, correct, wrong = compare_solution(numbers, solution[j])
    print("CNN_74k: {}/{}".format(correct, total))
    wrong_cnn_74k += wrong

    if total == correct:
        print("Predetto tutto correttamente! La soluzione è:")
        sol.sudoku_solver([[numbers[i*9 + j] for j in range(9)] for i in range(9)])
        perfect_sudoku = True

    print("")

    # TPOT MNIST

    numbers = []

    for row in sudoku.cells:
        for cell in row:
            if np.count_nonzero(cell) == 0:
                numbers.append(0)
            else:
                # Shrink it!
                res_cell = cv2.resize(cell, (14,14), interpolation = cv2.INTER_AREA)
                res_cell = res_cell / np.max(res_cell) * 255.
                # Value it!
                prediction = tpot_mnist.predict(res_cell.flatten().reshape(1,-1))
                numbers.append(prediction[0])
    for i in range(0,81,9):
        print(numbers[i:i+9])

    total, correct, wrong = compare_solution(numbers, solution[j])
    print("TPOT_MNIST: {}/{}".format(correct, total))
    wrong_tpot_mnist += wrong

    if total == correct:
        print("Predetto tutto correttamente! La soluzione è:")
        sol.sudoku_solver([[numbers[i*9 + j] for j in range(9)] for i in range(9)])
        perfect_sudoku = True
     
    print("")

    # SVM MNIST

    numbers = []

    for row in sudoku.cells:
        for cell in row:
            if np.count_nonzero(cell) == 0:
                numbers.append(0)
            else:
                # Shrink it!
                res_cell = cv2.resize(cell, (28,28), interpolation = cv2.INTER_AREA)
                res_cell = res_cell / np.max(res_cell)
                # Value it!
                prediction = svm_mnist.predict(res_cell.flatten().reshape(1,-1))
                numbers.append(prediction[0])
    for i in range(0,81,9):
        print(numbers[i:i+9])

    total, correct, wrong = compare_solution(numbers, solution[j])
    print("SVM_MNIST: {}/{}".format(correct, total))
    wrong_svm_mnist += wrong

    if total == correct:
        print("Predetto tutto correttamente! La soluzione è:")
        sol.sudoku_solver([[numbers[i*9 + j] for j in range(9)] for i in range(9)])
        perfect_sudoku = True
     
    print("")
    
    # SVM 74k

    numbers = []

    for row in sudoku.cells:
        for cell in row:
            if np.count_nonzero(cell) == 0:
                numbers.append(0)
            else:
                # Shrink it!
                res_cell = cv2.resize(cell, (28,28), interpolation = cv2.INTER_AREA)
                res_cell = res_cell / np.max(res_cell)
                # Value it!
                prediction = svm_74k.predict(res_cell.flatten().reshape(1,-1))
                numbers.append(prediction[0])
    for i in range(0,81,9):
        print(numbers[i:i+9])

    total, correct, wrong = compare_solution(numbers, solution[j])
    print("SVM_74k: {}/{}".format(correct, total))
    wrong_svm_74k += wrong

    if total == correct:
        print("Predetto tutto correttamente! La soluzione è:")
        sol.sudoku_solver([[numbers[i*9 + j] for j in range(9)] for i in range(9)])
        perfect_sudoku = True
     
    print("")

    if perfect_sudoku:
        perfect_sudokus += 1

print("Numeri totali da riconoscere: " + str(total_numbers))
print("Errori CNN_MNIST: " + str(wrong_cnn_mnist))
print("Errori CNN_74k: " + str(wrong_cnn_74k))
print("Errori SVM_MNIST: " + str(wrong_svm_mnist))
print("Errori SVM_74k: " + str(wrong_svm_74k))
print("Errori TPOT_MNIST: " + str(wrong_tpot_mnist))
print("Sudoku Perfettamente risolti: {}/{}".format(perfect_sudokus, j+1))
