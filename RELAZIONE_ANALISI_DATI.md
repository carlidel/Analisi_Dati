# Sudoku Grabber
## Elaborato per il Corso di Analisi Immagini e Analisi Dati - parte di Analisi Dati
### Carlo Emilio Montanari
#### a.a. 2017/18, Università di Bologna - Corso di Laurea Magistrale in Applied Physics

## Scopo del progetto

Il presente lavoro si pone come obiettivo la costruzione di 3 diversi classificatori di immagini rappresentanti singole cifre, che verranno poi utilizzati sulle celle numeriche dei Sudoku precedentemente estratti con i metodi di sogliatura discussi nella relazione riguardante Analisi Immagini.

Si costruiscono, si addestrano e si testano le performance dei 3 seguenti classificatori:

1. Support Vector Machine.
2. Convolutional Neural Network.
3. Pipeline specifica selezionata tramite la libreria Python `tpot`.

## Dipendenze del progetto

Il progetto è in linguaggio Python3 e richiede le seguenti librerie:

1. `numpy`
2. `cv2`
3. `sklearn`
4. `pytorch` (con supporto CUDA 9 per l'esecuzione del training della CNN)
5. `tpot`

## Esecuzione del progetto

È possibile eseguire separatamente gli script di training dei classificatori eseguendo gli script contenuti nelle cartelle:

1. `mnist_trained` per l'allenamento su dataset MNIST.
2. `chars74k_trained` per l'allenamento su dataset Chars74k.

Mentre, per eseguire il risolutore di Sudoku con tutti i classificatori addestrati, è sufficiente eseguire lo script `main.py`.

## Note preliminari sui dataset di training

Per addestrare i classificatori nel riconoscere le cifre, si è fatto uso in primo luogo del database standard MNIST di numeri scritti a mano nella sua forma originale (immagini di 28x28 pixel a 255 livelli di intensità). Per il tale scopo è stato usato lo script `mnist.py` per rendere facile la lettura dei file del dataset. Si è inoltre mantenuta la separazione predefinita tra sottoinsieme di training e sottoinsieme di testing. (training elements: 60'000, test elements: 10'000)

Oltre al MNIST, si è voluto anche addestrare classificatori su parte del dataset di caratteri stampati a macchina Chars74K ([](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)), avendo però cura di ridimensionare i campioni ad alta risoluzione alla dimensione di 28x28 pixel, in modo da non dover ristrutturare totalmente i classificatori usati. (training elements: 9'000, test elements: 1'000)

## Analisi dei classificatori utilizzati

Si analizzano e si commentano nel dettaglio i classificatori utilizzati ed il loro training nel dettaglio.

### Support Vector Machine

Si fa uso drll'implementazione standard della SVM contenuta nella libreria `sklearn.svm`. Questo classificatore, ponendo il problema della ricerca di iperpiani di separazione ottimali in forma di una funzione obiettivo senza minimi locali, risulta adatto nella classificazione di problemi su molte dimensioni (e.g. classificazione di immagini di dimensioni limitate in cui l'intensità di ogni pixel viene trattato come feature).

Inoltre, impostando funzioni di kernel di vario tipo, è possibile proiettare il problema iniziale su di uno spazio di dimensione superiore senza significativi costi computazionali e andando così ad ottenere separazioni basate su superfici non lineari.

Si prende come modello di SVM la seguente:
```python
svm_model = SVC(probability = False, kernel = "rbf", C = 2.8, gamma = .0073, cache_size = 10000)
```
ossia, una Support Vector Machine che non faccia previsioni probabilistiche ma solo classificazioni singole e che utilizzi come kernel una Radial Baisis Function [](rel_img/rbf_eq.png) con `gamma` molto basso (ergo, bassa dispersione) e una tolleranza di casi mal classificati `C` poco al disopra del valore di default 1 (onde evitare overfitting sul database di training).

È da evidenziare inoltre che, nell'utilizzo di questo classificatore, tutti i valori delle intensità dei pixel delle immagini vengono riscalati a valori float da 0 ad 1, in quanto l'implementazione della SVM è strutturata per operare su questo intervallo di valori delle features. (le SVM non sono invarianti per cambiamenti di scala!)

Successivamente all'addestramento del classificatore tramite la funzione `fit`, la SVM viene salvata su disco tramite la libreria standard `pickle` per l'utilizzo nel programma finale.

### Convolutional Neural Network

Per l'implementazione della CNN è stata utilizzata la libreria `pytorch` che, oltre ad usare una comoda sintassi basata su tensori e programmazione a oggetti, permette l'addestramento delle reti neurali parallelizzato su schede grafiche Nvidia supportanti il linguaggio CUDA.

Come struttura della rete neurale, si è optato per 2 livelli di convoluzione ed un livello finale di collegamento neurale lineare semplice. Nel dettaglio, i 2 livelli di convoluzione sono costituiti da:

1. Un livello di convoluzione bidimensionale con dimensione del kernel 5x5 (tale che nel primo livello vada da 1 a 16 canali distinti di immagini, mentre nel secondo da 16 a 32 canali).
2. Un livello di normalizzazione bidimensionale a livello di batch (maggiori dettagli sulla notazione batch e epoch più avanti).
3. Un livello di passaggio per la funzione di attivazione ReLU, che scarta i valori negativi ottenuti fino a questo punto.
4. Un livello finale di pooling massimale con un maschera di kernel 2x2 ed un valore di stride pari a 2.

Questi due livelli di convoluzione portano una immagine di 28x28 pixel a diventare 32 immagini distinte di dimensione 7x7 pixel, le quali verranno in fine appiattite lungo un unico vettore di features di dimensione 7x7x32. Questo vettore è quanto viene poi collegato alle 10 classi di cifre tramite mappa lineare.

Per l'addestramento della CNN, viene fatto uso delle utilità fornite da `Pytorch` nella classe `DataLoader`, che permettono di sottoporre comodamente più volte l'intero dataset mescolato alla rete neurale.

Si sottopone quindi la rete a molteplici `epoch` nelle quali la CNN viene sottoposta all'intero dataset. Questo dataset viene inoltre somministrato a blocchi `batch` di più elementi in parallelo (da cui il termine "normalizzazione lungo tutta la batch").

Come funzione di errore si opta per la definizione `CrossEntropyLoss`, basata sulle definizioni classiche di entropia dell'informazione ed indicata nella documentazione di `pytorch` come la più adatta a classificazioni multi-classe. Insieme a questa viene usata come funzione di ottimizzazione `torch.optim.Adam`, una funzione adattiva migliorata di gradient descent anch'essa suggerita dalla documentazione di `pytorch` per l'addestramento di CNN.

Ad ogni batch sottoposta alla CNN, si calcola quindi la backpropagation dell'errore lungo tutta la rete e si esegue uno step di ottimizzazione la cui dimensione è definita dal `learning_rate` nella funzione `torch.optim.Adam`.

Alla fine del training, la rete addestrata viene salvata con la funzione standard `torch.save` di salvataggio su disco per l'utilizzo nel programma finale.

Si riportano i punti più importanti dello script di addestramento della rete:
```python
# Da cnn_train.py

import torch
import torch.nn as nn
import torchvision
import mnist as m
import numpy as np

# Do we have a fancy GPU?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Parameters
num_epochs = 100
num_classes = 10    # i.e. the digits
batch_size = 300    # How many samples per batch use
learning_rate = 0.001

# [...]

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True) # We want to random sample the data at each epoch
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = batch_size, shuffle = False) # No need in this case

# C.N.N. (with two convolutional layers)
class CNN(nn.Module): # In Pytorch, our models must subclass this class
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

# [...]
```

### Classificatore selezionato con TPOT ([Link al paper](https://dl.acm.org/citation.cfm?id=2908918))

TPOT è un Tree-Based Pipeline Optimization Tool open source realizzato per rendere più semplice ed accessibile la costruzione di classificatori ottimizzati. TPOT, tramite algoritmi genetici ed evoluzionistici, cerca di ricavare da un dato dataset il "miglior" preprocessing sulle features insieme al "miglior" algoritmo classificatore possibile, selezionando in automatico frazioni di training e testing dal dataset fornitogli. Si illustra in breve il funzionamento del sistema ed i classificatori ottenuti per i due dataset.

#### Funzionamento di TPOT

1. **Pipeline Operators**. Prendendone le implementazioni dalla libreria `scikit-learn`, TPOT considera molteplici funzioni di preprocessing, decomposition, feature selection e di modelli di classificazione come "operatori" con cui assemblare pipeline di classificatori più o meno complessi.
2. **Assemblaggio Tree-based di Pipelines**. Questi operatori vengono quindi assemblati in strutture ad albero simili a quella rappresentata nella seguente figura:
[](rel_img/tpot.jpg)
I dataset vengono quindi sottoposti in molteplice copia ai vari alberi prodotti ed i risultati finali vengono testati per poter attribuire un giudizio finale all'intera pipeline (TPOT si preoccupa di suo nel suddividere il dataset fornitogli in parte di training e parte di testing).
3. **Evoluzione di Pipeline ad albero**. Le varie pipeline vengono infine sottoposte all'algoritmo genetico implementato nella libreria Python `DEAP`, dove, su di ogni generazione, vengono selezionate le pipeline migliori, eseguita una mutazione ed un crossover sulle varie componenti delle pipeline e, infine, eseguito un 3-way tournament tra tutte le pipeline restanti, scartando di volta in volta la pipeline peggiore nel torneo. Alla fine dei cicli generazionali, viene stampata la pipeline migliore direttamente in formato Python script.

#### Utilizzo di TPOT

Dopo aver caricato ed impostato il dataset in un formato opportuno, per utilizzare TPOT ed ottenere un classificatore sono sufficienti le seguenti righe di codice

```python
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=3)
tpot.fit(X_train, Y_train)
tpot.export('tpot_num_pipeline.py')
```

È da sottolineare tuttavia che questo sistema ad evoluzione genetica di pipeline richiede un tempo di addestramento estremamente lungo per dataset con numero consistente di features (diversi giorni per una immagine di 28x28 pixel con ogni pixel trattato come feature).

Inoltre, se le feature risultano troppe, gran parte degli operatori proposti da TPOT risultano inadatti e vengono scartati in automatico prima del termine della loro esecuzione in una pipeline in seguito ad un `time limit exceeded`.

Alla luce di ciò, TPOT è stato utilizzato solamente sul dataset MNIST dopo che questo è stato ridimensionato da immagini di 28x28 pixel a immagini di 14x14 pixel. La pipeline risultante vincitrice è la seguente:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.9651667280513239
exported_pipeline = RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.15000000000000002, min_samples_leaf=1, min_samples_split=18, n_estimators=100)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
```

Che, opportunamente adattata al formato effettivo del dataset in MNIST e fittata, può venire salvata con `pickle` ed utilizzata nel programma finale.

## Il programma finale

Lo script finale `main.py` fa uso di tutti e 5 i classificatori addestrati e salvati offline e si ricollega alla pipeline di elaborazione di foto di Sudoku, implementata precedentemente nella parte di progetto di Analisi Immagini. Nel dettaglio, lo script compie le seguenti operazioni:

1. Carica i 5 classificatori.
2. Carica le interpretazioni corrette (scritte a mano) dei 13 Sudoku esempio le cui foto sono depositate nella cartella `img`.
3. Per ciascuna delle 13 immagini:
    1. Esegue la pipeline di elaborazione immagini trattata nella parte di Analisi Immagini, che restituisce le 81 celle elaborate e separate.
    2. Utilizza, uno dopo l'altro, i 5 classificatori sulle 81 celle per valutarne le performance, confrontandone i risultati con la soluzione nota scritta a mano.
    3. SE l'intero Sudoku è stato interpretato correttamente, viene preso nota del successo e viene eseguito l'algoritmo ricorsivo di risoluzione.
    4. SE l'interpretazione del Sudoku presenta errori, vengono stampate le cifre interpretate male e viene preso nota del numero di fallimenti del classificatore.
4. Alla fine di tutto, vengono stampate le performance dei 5 classificatori rispetto a tutte le cifre interpretabili.

### (semplice) Soluzione ricorsiva del Sudoku

Per risolvere un Sudoku è sufficiente (anche se non troppo elegante in termini di tempi di calcolo) implementare un algoritmo ricorsivo. Questo algoritmo è stato implementato nello script `sudoku_solver.py`
```python
def find_next_cell_to_fill(grid, i, j):
    for x in range(i,9):
        for y in range(j,9):
            if grid[x][y] == 0:
                return x,y
    for x in range(0,9):
        for y in range(0,9):
            if grid[x][y] == 0:
                return x,y
    return -1,-1

def is_valid(grid, i, j, e):
    if all([e != grid[i][x] for x in range(9)]): # Test row
        if all([e != grid[x][j] for x in range(9)]): # Test column
            # In what 3x3 cell are we?
            secTopX, secTopY = 3 *(i//3), 3 *(j//3) 
            for x in range(secTopX, secTopX+3):
                for y in range(secTopY, secTopY+3): # Test 3x3 cell
                    if grid[x][y] == e:
                        return False
            return True
    return False

def sudoku_solver(grid, i=0, j=0):
    """
    This is the recursive solving function.
    """
    i,j = find_next_cell_to_fill(grid, i, j)
    if i == -1:
        # At this point we have reached the solution
        for line in grid:
            print(line)
        return True
    for e in range(1,10):
        if is_valid(grid,i,j,e):
            grid[i][j] = e
            # Recursive call here
            if sudoku_solver(grid, i, j):
                return True
            # No solution here, execute backtracking
            grid[i][j] = 0
    # Fail, no solution at all
    return False
```

## Risultati e commenti finali

Il programma è stato fatto girare su 13 foto diverse di Sudoku (le stesse 13 foto usate nella trattazione di Analisi Immagini). Queste 13 foto di Sudoku contenevano un totale di 349 caselle con numero da classificare. Le performance ottenute dai classificatori sono state le seguenti:

|Classificatore|Errori Commessi|Percentuale letta correttamente|
|--|:--:|--:|
|CNN addestrata su MNIST|22/349|93.70%|
|CNN addestrata su Char74k|10/349|97.13%|
|SVM addestrata su MNIST|104/349|70.20%|
|SVM addestrata su Char74k|41/349|88.25%|
|TPOT pipeline su MNIST|117/349|66.47%|

In totale, solamente 8 Sudoku su 13 sono stati interpretati correttamente da almeno uno dei classificatori.

Queste percentuali di riconoscimento risultano essere significativamente inferiori rispetto a quelle ottenute in fase di tesing post training (~98.5% nelle CNN, ~96% nelle SVM e nella pipeline TPOT).

Nel complesso, solamente le CNN hanno ottenuto performance di classificazione un minimo accettabili, ciò evidenzia come una rete convoluzionale risulta essere la soluzione più rapida e diretta per ottenere performance consistenti.

Le SVM, al contrario, hanno ottenuto risultati drammaticamente più scarsi, che però molto probabilmente possono essere migliorati di molto utilizzando parametri diversi e più severi per `C` e `gamma` (la cosa però richiederebbe molto più tempo di calcolo rispetto all'affinamento di una CNN già molto performante).

La pipeline su TPOT, invece, ha ottenuto i risultati peggiori, al punto che la costruzione automatizzata di pipeline sembra non essere proprio una strategia di classificazione degna di essere seguita.

Oltre a questo, si può osservare come in generale le performance siano risultate migliori nell'utilizzo del dataset Char74k rispetto al dataset MNIST, nonostante le dimensioni molto più consistenti di quest'ultimo. Ciò evidenzia come le differenze tra numeri scritti a mano e numeri stampati a macchina non siano per niente trascurabili durante il training di un classificatore. Questo si osserva anche qualitativamente nell'analisi delle cifre più sbagliate dai classificatori addestrati su MNIST: gran parte delle cifre classificate male sono state delle confusioni "lecite" tra 1 e 7, 7 e 1, 5 e 6, 6 e 5.

Questo porta a pensare che un dataset meglio costruito di caratteri stampati a macchina può permettere il raggiungimento di performance molto migliori nella classificazione.

### Possibili miglioramenti

Con le SVM, si possono compiere studi più approfonditi con diversi kernel e diversi parametri di tolleranza, anche se difficilmente si possono raggiungere risultati tanto performanti quanto quelli di una CNN.

Con le CNN, invece, un percorso da esplorare per migliorare le performance potrebbe essere quello di compiere operazioni di data augmentation sul dataset Char74k, in modo da ottenere per esempio una rete neurale più robusta e capace di riconoscere cifre distorte, sfocate o leggermente ruotate. Ciò potrebbe anche permettere di rendere comparabile in qualità il dataset Char74k al dataset MNIST.