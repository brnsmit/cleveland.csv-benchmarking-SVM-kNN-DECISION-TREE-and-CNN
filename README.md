# Exercícios de Machine Learning e Deep Learning: Cleveland Heart Disease com benchmarking dos algoritmos SVM-kNN-DECISION-TREE e o uso de uma CNN para os dados Fashion-MNIST

Os algoritmos a seguir foram desenvolvidos ao longo da disciplina de Reconhecimento de Padrões, com o objetivo de aplicar os conceitos abordados no Mestrado em Engenharia de Controle e Automação. Este repositório contém implementações de algoritmos de Machine Learning e Deep Learning, incluindo CNNs, SVMs, k-NN, Árvore de Decisão, aplicados aos conjuntos de dados Fashion-MNIST e Cleveland Heart Disease.

## Introdução  ℹ️

Este repositório contém o código fonte de todos os exercícios abaixo em Python para implementações de diferentes algoritmos de Machine Learning e Deep Learning. Os exercícios foram desenvolvidos usando bibliotecas como scikit-learn, TensorFlow e Keras.

* Execute o notebook Python para explorar os diferentes exercícios.

**Código**: `RECPAD_L2.ipynb`

## Exercício 1: k-NN 📏

Implementação do algoritmo k-NN para classificação no conjunto de dados Cleveland Heart Disease.

### Matriz de Confusão

![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/c434ba10-6730-4a2a-8558-3a2481fce04f)

### Acurácia

![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/e497c295-033a-4007-9a3f-0a748434ed4b)

## Exercício 2: Árvore de Decisão 🌳

Implementação de uma Árvore de Decisão para classificação no conjunto de dados Cleveland Heart Disease.

### Matriz de Confusão

![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/e8913f0f-de75-40d7-95a7-1db92962bbfa)

### Acurácia

![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/0b88d64a-5179-434e-a020-eb30dbb85103)

## Exercício 3: SVM 📈

Implementação de SVMs com kernels Linear e RBF para classificação no conjunto de dados Cleveland Heart Disease.

### Matriz de Confusão (Linear)

![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/5eb3c126-9201-448f-9b68-b8d09d1c5ff7)

### Matriz de Confusão (RBF)

![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/24a0d9e2-cfed-49ab-9d10-1141b94cfa5f)

### Curva de Aprendizado SVM RBF

![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/1c0e4119-36f0-4472-b98b-80ea722a2edd)

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

![image](https://github.com/brnsmit/cleveland.csv-benchmarking-SVM-kNN-DECISION-TREE-and-CNN/assets/168189996/2fe5a533-db57-464f-b938-bb114afd73d1)

## Conclusão 🎯

Após o treinamento, o modelo alcançou uma boa acurácia na base de teste, com o valor de `Acurácia na base de teste: 0.9057999849319458` . Além disso, os gráficos de acurácia e perda durante o treinamento mostram o desempenho do modelo ao longo das épocas.

O modelo pode ser ajustado e otimizado com diferentes arquiteturas de rede, hiperparâmetros e técnicas de regularização para melhorar ainda mais o desempenho na classificação de imagens do Fashion-MNIST.
