# Exercícios de Machine Learning e Deep Learning: Cleveland Heart Disease com benchmarking dos algoritmos SVM-kNN-DECISION-TREE e o uso de uma CNN para os dados Fashion-MNIST

Os algoritmos a seguir foram desenvolvidos ao longo da disciplina de Reconhecimento de Padrões, com o objetivo de aplicar os conceitos abordados no Mestrado em Engenharia de Controle e Automação. Este repositório contém implementações de algoritmos de Machine Learning e Deep Learning, incluindo CNNs, SVMs, k-NN, Árvore de Decisão, aplicados aos conjuntos de dados Fashion-MNIST e Cleveland Heart Disease.

## Introdução  ℹ️

Este repositório contém o código fonte de todos os exercícios abaixo em Python para implementações de diferentes algoritmos de Machine Learning e Deep Learning. Os exercícios foram desenvolvidos usando bibliotecas como scikit-learn, TensorFlow e Keras.

* Execute o notebook Python para explorar os diferentes exercícios.

**Código**: `knn_cleveland.py`

## Exercício 1: k-NN 📏

Implementação do algoritmo k-NN para classificação no conjunto de dados Cleveland Heart Disease.

### Matriz de Confusão

![Matriz de Confusão k-NN](![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/90378938-9dd6-4b93-9147-2644900888b7)
)

### Acurácia

![Acurácia k-NN](![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/2612f3f3-2f65-4122-9c06-c4b54098b603)
)

## Exercício 2: Árvore de Decisão 🌳

Implementação de uma Árvore de Decisão para classificação no conjunto de dados Cleveland Heart Disease.

### Matriz de Confusão

![Matriz de Confusão Árvore de Decisão](![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/90562983-0e74-47a0-8698-984cf553d351)
)

### Acurácia

![Acurácia Árvore de Decisão](![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/ae44155d-e3dc-4240-9650-e51d349a50d0)
)

## Exercício 3: SVM 📈

Implementação de SVMs com kernels Linear e RBF para classificação no conjunto de dados Cleveland Heart Disease.

### Matriz de Confusão (Linear)

![Matriz de Confusão SVM Linear](![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/c6b6341a-0970-43bb-a3d0-6f2571ac9ace)
)

### Matriz de Confusão (RBF)

![Matriz de Confusão SVM RBF](![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/39ae39cc-6e7f-46c6-b49a-377293f9a0fd)
)

### Curva de Aprendizado SVM RBF

![Curva de Aprendizado SVM RBF](![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/cf82114f-2668-4288-9d12-c3b6b8e19c83)
)

## Comparação dos Resultados 🆚

| Algoritmo         | Parâmetros | Acurácia Teste (80-20) |
|-------------------|------------|------------------------|
| k-NN (k=3)        | k=3        | 0.82                   |
| k-NN (k=5)        | k=5        | 0.80                   |
| k-NN (k=7)        | k=7        | 0.80                   |
| Árvore de Decisão | -          | 0.74                   |
| SVM Linear        | Linear     | 0.79                   |
| SVM RBF           | RBF        | 0.80                   |


| Modelo           | Parâmetros  | Acurácia Média |
|------------------|-------------|----------------|
| k-NN (k=3)       | k=3         | 0.83           |
| k-NN (k=5)       | k=5         | 0.82           |
| k-NN (k=7)       | k=7         | 0.82           |
| Árvore de Decisão| -           | 0.70           |
| SVM Linear       | Linear      | 0.83           |
| SVM RBF          | RBF         | 0.83           |

Comparando esses resultados com os obtidos anteriormente com a partição 80-20, observamos que, em geral, as acurácias médias com validação cruzada tendem a ser melhores. Isso ocorre porque a validação cruzada utiliza múltiplas divisões dos dados, garantindo uma avaliação mais robusta do desempenho do modelo.

## Exercício 4: CNN 🖼️

Construção de uma Rede Neural Convolucional (CNN) para classificação no conjunto de dados Fashion-MNIST.

### Modelo

## Modelo da CNN

```python
model = models.Sequential([
    # Camadas convolutivas e de pooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Camada Flatten para transição para uma camada densa (fully-connected)
    layers.Flatten(),
    
    # Camadas densas (fully-connected)
    layers.Dense(64, activation='relu'),
    
    # Camada de saída
    layers.Dense(10, activation='softmax')
]
```

## Gráficos de Acurácia e Perda 📊

![Acurácia e Perda](![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/00212801-ed7c-49bf-a36c-ef9734b73b6e)
)

## Conclusão 🎯

Após o treinamento, o modelo alcançou uma boa acurácia na base de teste, com o valor de `Acurácia na base de teste: 0.9057999849319458` . Além disso, os gráficos de acurácia e perda durante o treinamento mostram o desempenho do modelo ao longo das épocas.

O modelo pode ser ajustado e otimizado com diferentes arquiteturas de rede, hiperparâmetros e técnicas de regularização para melhorar ainda mais o desempenho na classificação de imagens do Fashion-MNIST.
