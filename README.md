## [World Happiness Prediction](https://github.com/Ran-Liao/Adv-Machine-Learning/blob/main/World%20Happiness%20Project.ipynb)

### Objective 
What makes the citizens of one country more happy than the citizens of other countries? Do variables measuring perceptions of corruption, GDP, maintaining a healthy lifestyle, or social support associate with a country's happiness ranking? We would like to predict hapiness rankings using the United Nation's World Happiness Rankings country level data 

### Data
There are 88 rows in training set

#### Features
Country or region	
GDP per capita	
Social support	
Healthy life expectancy	
Freedom to make life choices	
Generosity	
Perceptions of corruption	
name	
region	
sub-region	
Terrorist_attacks
#### Target
Happiness score

### Modeling
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

#### Best Model
Gradient Boosting Classifier(learning_rate=1.3, max_depth=1, random_state=0)

|Metrics    |        |
|-----------|--------|
| Accuracy  | 58.89% |
| F1 score  | 57.65% |
| Precision | 65.60% |
| Recall    | 59.5%  |

#### Feature Importance
| Feature                                         | Importance |
|-------------------------------------------------|------------|
| num__GDP per capita                             | 0.501039   |
| num__Social support                             | 0.149897   |
| num__Healthy life expectancy                    | 0.065391   |
| num__Freedom to make life choices               | 0.005091   |
| num__Generosity                                 | 0.089089   |
| num__Perceptions of corruption                  | 0.006805   |
| num__Terrorist_attacks                          | 0.063233   |
| cat__sub-region_Eastern Asia                    | 0.020859   |
| cat__sub-region_Latin America and the Caribbean | 0.059937   |
| cat__sub-region_South-eastern Asia              | 0.032341   |

## [Covid X-rays Prediction](https://github.com/Ran-Liao/Adv-Machine-Learning/blob/main/COVID_Hackathon_Model_Submission_Notebook_281_29.ipynb)

### Objective
Predict whether a person contracted Covid, Pneumonia or neither given their x-rays images of lungs

### Data
The data contains x-rays images for 3616 people diagnosed with Covid, 1345 people diagnosed with pneumonia, and 10192 normal people without Covid or Pneumonia

### Application
With such a model, it potentially helps:

Early detection and treatment: A predictive model could help in the early detection of COVID-19, pneumonia, and other respiratory illnesses, allowing for prompt treatment and preventing the spread of disease.

Screening precision: A predictive model with goog predictive power can be used for screening purposes to identify individuals who may have COVID-19 or pneumonia and require further evaluation.

Misdiagnosis reduction: A predictive model with goog predictive power can add to the disagnosis toolkit that help differentiate between COVID-19 and other respiratory illnesses, such as pneumonia, which could lead to better disease management and outcomes
At the same time, multiple groups of individuals could potentially be benefited by it:

Healthcare providers could be assited in making more accurate diagnoses and treatment decisions
Patients with such illness could be detected and treated on the timely basis
Researchers could use it to study population with Covid or Pneumonia against normal population to optimize future prevention and treatment strategies

### Model
- VGG16 (with/without batch normalization)
- Squeezenet (with/without batch normalization)
- ResNet (transfer learning)
- Inception (transfer learning)

#### Best Model
VGG16 with batch normalization

| Layer (type)                                                      | Output Shape         | Param # |
|-------------------------------------------------------------------|----------------------|---------|
| conv2d_94 (Conv2D)                                                | (None, 192, 192, 32) | 896     |
| conv2d_95 (Conv2D)                                                | (None, 192, 192, 32) | 1056    |
| batch_normalization_94 (Batch Normalization)                      | (None, 192, 192, 32) | 128     |
| max_pooling2d_4 (MaxPooling2D)                                    | (None, 96, 96, 32)   | 0       |
| conv2d_96 (Conv2D)                                                | (None, 96, 96, 64)   | 18496   |
| conv2d_97 (Conv2D)                                                | (None, 96, 96, 64)   | 4160    |
| batch_normalization_95 (Batch Normalization)                      | (None, 96, 96, 64)   | 256     |
| max_pooling2d_5 (MaxPooling2D)                                    | (None, 48, 48, 64)   | 0       |
| conv2d_98 (Conv2D)                                                | (None, 48, 48, 128)  | 73856   |
| conv2d_99 (Conv2D)                                                | (None, 48, 48, 128)  | 16512   |
| batch_normalization_96 (Batch Normalization)                      | (None, 48, 48, 128)  | 512     |
| max_pooling2d_6 (MaxPooling2D)                                    | (None, 24, 24, 128)  | 0       |
| conv2d_100 (Conv2D)                                               | (None, 24, 24, 512)  | 590336  |
| conv2d_101 (Conv2D)                                               | (None, 24, 24, 512)  | 262656  |
| batch_normalization_97 (Batch Normalization)                      | (None, 24, 24, 512)  | 2048    |
| max_pooling2d_7 (MaxPooling2D)                                    | (None, 12, 12, 512)  | 0       |
| flatten (Flatten)                                                 | (None, 73728)        | 0       |
| dense_2 (Dense)                                                   | (None, 3)            | 221187  |
|                                                                   |                      |         |
| Total params: 1,192,099                                           |                      |         |
| Trainable params: 1,190,627                                       |                      |         |
| Non-trainable params: 1,472                                       |                      |         |

|Metrics    |        |
|-----------|--------|
| Accuracy  | 89.46% |
| F1 score  | 89.45% |
| Precision | 89.79% |
| Recall    | 89.46% |

## [Stanford Sentiment Treebank - Movie Review Classification Competition](https://github.com/Ran-Liao/Adv-Machine-Learning/blob/main/movie%2Bclassification%2Bnotebook.ipynb)

### Objective
Predict whether a movie review is positive or negative in sentiment

### Data
The SST-2 dataset is a benchmark dataset for sentiment analysis. It contains a collection of movie reviews with a binary label indicating whether the review is positive or negative. There are approximately 8,000 reviews in total, which are split into training and test sets. 

### Application
Building a predictive model using the SST-2 dataset can be practically useful for a variety of applications, such as:

1. Product/Service Performance: Companies and producers can use such models to automatically classify reviews of their products or services as positive or negative, allowing them to identify areas for improvement and respond to customer feedback.

2. Marketing and Investment: Market researchers and directors/producers can use such models to analyze customer sentiment towards specific movies or brands, helping them to identify market trends.

3. Personalization: Companies can use such models to identify consumers' preference and sentiment toward a certain movie or category of movies along with user information in order to provide personalized streaming services and high quality recommendation

### Model
- Conv1d with Word Embedding
- LSTMs with Word Embedding 
- Bidirectional LSTM with Word Embedding 
- LSTMs with pre-trained glove embedding

#### Best Model
LSTMs with word embedding

| Layer (type)                                                      | Output Shape   | Param # |
|-------------------------------------------------------------------|----------------|---------|
| embedding_7 (Embedding)                                           | (None, 40, 16) | 160000  |
| lstm_2 (LSTM)                                                     | (None, 40, 32) | 6272    |
| lstm_3 (LSTM)                                                     | (None, 32)     | 8320    |
| flatten_5 (Flatten)                                               | (None, 32)     | 0       |
| dense_8 (Dense)                                                   | (None, 2)      | 66      |
|                                                                   |                |         |
| Total params: 174,658                                             |                |         |
| Trainable params: 174,658                                         |                |         |
| Non-trainable params: 0                                           |                |         |

|Metrics    |        |
|-----------|--------|
| Accuracy  | 81.67% |
| F1 score  | 81.65% |
| Precision | 81.80% |
| Recall    | 81.66% |
