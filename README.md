# README for Predicting Student Grades Based on ChatGPT Interactions

This repository contains the code and methodology for a machine learning project aimed at predicting student homework grades from their interactions with ChatGPT. The dataset comprises ChatGPT history in HTML format and corresponding CSV files with student scores.

## Project Overview

### Data Description

- **ChatGPT Histories**: Collection of 122 HTML files containing student interactions with ChatGPT.
- **Student Scores**: A CSV file with the final grades of students.
- **Testing_ChatGPT_Data**: Collection of 188 HTML files containing student interactions with ChatGPT, utilized for testing our machine learning model.

### Code File Description

- **Project Codebase**: The project codebase encompasses the model training steps, as well as the testing procedure, conducted using a new testing document titled `testing_chatgpt_data`.

### Objective

The project's goal is to predict the points students received for individual homework questions and their total grade using a two-step machine learning pipeline that involves clustering and a neural network.

## Methodology

### Preprocessing

1. **Cosine Similarity Scoring**: Calculated between user prompts and homework questions to obtain similarity scores for each question.
2. **Dataset Splitting with Stratification**: Attention was given to 11 rare classes in the dataset, which were first mapped to fewer categories before splitting.
3. **Dataframe Creation**: Separate dataframes for training (`x_train`, `y_train`) and testing (`x_test`, `y_test`) were prepared.

### Clustering Algorithm for Estimating Points

#### Rationale for Clustering

Due to the limited and complex nature of the dataset, it was challenging to derive features and relationships directly from the ChatGPT interactions. We lacked specific labels for the points received from each homework question. To address this, we employed an unsupervised technique, K-means clustering, to estimate these points.

#### Implementation

- **K-Means Clustering**: Applied to each question column to assign cluster labels to each student's response.
- **Feature Formulation**: Cluster centers for each question were used to formulate new features, including the points received per question and a total calculated grade.
- **Insight Enhancement**: This approach significantly enhanced the dataset's insights, enabling the subsequent application of a neural network model for total grade prediction.

### Clustering Visualizations: Illustrative Cluster Analysis

The visualizations provided below represent the clustering outcomes for two distinct homework questions to illustrate our clustering approach. These visualizations are key to understanding how students' responses to each question are grouped based on the similarity of their interaction with ChatGPT.

#### Visualization for `Q_1`
![image](https://github.com/Invadel/CS_412_Machine_Learning/assets/120125253/8cf13eb2-3cfb-4cf3-b6c0-0a171470f01f)

In the first plot, we see the clustering results for `Q_1`. The data points are scattered across the plot and color-coded according to the cluster they have been assigned to. The red 'X' marks represent the centroid of each cluster, signifying the average location of all the points within that cluster. This clustering may help us to infer common patterns in student responses and estimate the typical score associated with each pattern.

#### Visualization for `Q_2`
![image](https://github.com/Invadel/CS_412_Machine_Learning/assets/120125253/b54a6442-a283-43ae-9c81-afa99d3e16c8)

Similarly, the second plot shows the results for `Q_2`. The clusters here may indicate different levels of understanding or different approaches taken by students in their ChatGPT interactions regarding the second question. The cluster centers are prominently displayed, illustrating the mean score of the responses within each cluster.

In these visualizations, the clusters should be interpreted horizontally, as they are formed based on the normalized scores for questions. Each cluster is characterized by a range of scores, and the horizontal spread reflects the diversity of student responses within similar scoring brackets. The data points in each cluster are not determined by their position along the x-axis, which is merely an index, but by their closeness in score value, which is indicated on the y-axis. This horizontal reading of clusters aligns with our analytical focus on understanding score distributions and patterns in student interactions for each question.

### Neural Network Model for Grade Prediction

#### Inputs

- Includes metrics like user prompts, error counts, entropy, average characters in prompts and responses, points received per question, and the calculated total grade.
- Altogether, the neural network model was trained using 27 inputs.

#### Architecture

- Custom neural network with residual blocks, designed for regression tasks.

#### Training and Evaluation

- Trained over a maximum of 200 epochs with mean squared error loss, incorporating early stopping to prevent overfitting and enhance training efficiency.
- **Early Stopping**: Training is halted if the test loss does not improve for a predefined patience period (e.g., 10 consecutive epochs), ensuring optimal model performance without unnecessary computation. With early stopping, overfitting is also prevented. 
- Performance during training was evaluated using accuracy, precision, and recall metrics.

#### Testing and Performance Evaluation

- **Testing Dataset Preparation**: The testing dataset was also clustered using the cluster labels and centers learned from the training dataset.
- **Model Testing**: The neural network model was then tested with this prepared testing dataset.
- **Evaluation**: The performance of the model was evaluated based on how well it predicted the total grades, with particular attention to the model's ability to generalize from training to unseen data.

## Regression Plot Analysis

### Visualization of Prediction Accuracy

##### Prediction Accurracy without Sigmoid mapping:

![image](https://github.com/Invadel/CS_412_Machine_Learning/assets/120125253/5227f482-61fe-4d4c-a564-8579d2b3d1e3)

##### Prediction Accurracy with Sigmoid mapping (Forces inputs to the range \[0, 100]):

![image](https://github.com/Invadel/CS_412_Machine_Learning/assets/105321074/5113769e-5176-4249-a9f1-f8a6ab4da42e)

The regression plot above is a critical tool for visualizing the accuracy of the neural network model's predictions. It illustrates the relationship between the actual grades of the students (`y_true`) and the grades predicted by the model (`y_pred`).

#### Key Features of the Plot:

- **Data Points**: Each blue dot represents an individual student's data, plotting their actual grade against the grade predicted by the model.
- **Best Fit Line**: The red line indicates the best fit for these data points, showing the trend of the predictions in comparison to the actual grades.
- **Shaded Area**: The pink shaded region around the line represents the confidence interval for the regression estimate, giving a visual indication of the prediction's precision.

#### Interpretation:

- **Alignment**: The closer the blue dots are to the red line, the more accurate the predictions are.
- **Distribution**: Ideally, the points should be evenly distributed around the line of best fit, indicating consistent accuracy across the range of grades.
- **Outliers**: Points far from the red line can be considered outliers and may require further investigation to understand why the prediction was inaccurate.

#### Conclusion:

This regression plot is a testament to the model's performance, showing a strong correlation between predicted and actual grades, which is a positive indicator of the model's effectiveness. However, the presence of any outliers or a wide confidence interval could suggest areas for further model refinement.

## Performance Metrics

### Discussion of Error Metrics

In the pursuit of refining our predictive model's accuracy, we have dedicated considerable effort to surpass the benchmarks set by our baseline model, which was constructed using a decision tree regressor. This foundational model yielded a Mean Squared Error (MSE) score of 575.88, a useful metric but one that suggested substantial room for improvement.

By transitioning to a more sophisticated neural network architecture that leverages residual blocks — specifically tailored for regression tasks — our model has demonstrated a marked enhancement in performance. The evidence of this lies in the error metrics derived from our rigorous evaluation:

- **Mean Absolute Error (MAE)**: At 14.36, this metric indicates the average deviation of the predicted grades from their true values, showcasing a tight average error margin which is indicative of the model's precision.
- This metric is further improved by the sigmoid activation function at the output, reducing the Mean Absolute Error (MAE) of the model to 5.44.
- **Mean Square Error (MSE)**: The MSE of 378.84, a significant reduction from the baseline's 575.88, underscores a substantial increase in the model's predictive accuracy and its ability to minimize the error squared sum over the dataset.
- This metric is further improved by the sigmoid activation function at the output, reducing the Mean Squared Error (MSE) of the model to 54.
- **Root Mean Square Error (RMSE)**: The RMSE, which stands at 19.46, provides us with a standard deviation of the prediction errors, quantifying how much the predictions deviate from the actual values. The lower RMSE, as compared to the baseline model, demonstrates the model's enhanced predictive consistency.
- This metric is further improved by the sigmoid activation function at the output, reducing the Root Mean Square Error (RMSE) to 7.35%.
- **Mean Absolute Percentage Error (MAPE)**: With a MAPE of 15.25%, we gain insight into the relative prediction error in percentage terms, which further confirms the improved accuracy of our neural network model.
- This metric is further improved by the sigmoid activation function at the output, reducing the Mean Absolute Percentage Error (MAPE) to 5.97%.

These refined metrics not only reflect the model's robustness but also its improved alignment with the intricate patterns inherent in the data. Our neural network model has not only risen to the challenge but set a new standard of performance, providing us with a more reliable and precise tool for grade prediction.

## Neural Network Code

The neural network is implemented using PyTorch. It includes data preparation, model definition, training, and evaluation steps, with a focus on performance metrics and regression plot visualization.

## Usage

1. **Data Preparation**: Format your data as described in the Data Description section.
2. **Model Training**: Run the neural network code to train the model on your data.
3. **Model Evaluation**: Evaluate the model's performance using the provided metrics.
4. **Prediction**: Use the trained model to predict student grades based on new ChatGPT interaction data.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

# BERT Model for Analyzing ChatGPT Assistant Responses

This section of the repository details our approach to utilizing a BERT (Bidirectional Encoder Representations from Transformers) model for analyzing ChatGPT assistant responses. Our goal was to harness the power of BERT's language understanding capabilities to extract meaningful insights from the assistant's responses in student interactions. This method stands as an alternative approach to our main clustering and neural network pipeline.

## Overview of the BERT Model Approach

### Data Utilization

- **Scope**: Only the assistant responses extracted from the 122 HTML files containing student interactions with ChatGPT were used.
- **Objective**: To analyze and understand the nuances and patterns in the ChatGPT responses which could correlate with student grades.

### Data Preprocessing

- **Tokenizer**: Utilized BERT's tokenizer to process the assistant responses, converting them into a format suitable for the model.
- **Data Split**: The dataset was divided into training, validation, and testing sets with an 80-10-10 split, ensuring a comprehensive evaluation of the model's performance.

### Model Training

- **Pre-Trained BERT Model**: We leveraged a pre-trained BERT model, capitalizing on its existing language processing capabilities.
- **Epochs**: The model was trained over 100 epochs, ensuring sufficient learning while avoiding overfitting.
- **Learning Rate**: Set at 1e-5, this learning rate was chosen to balance the speed of convergence with the stability of the training process.

### Performance Evaluation

- **Graphical Analysis**: The results of the model training have been visualized in graphs, offering a clear and easily interpretable depiction of the model's performance over the course of the training. These graphs are presented below.
- **Metrics**: Standard evaluation metrics were employed to gauge the effectiveness of the model in processing and interpreting the assistant responses. The Mean Squared Error (MSE) on the validation set with the BERT Model was 287.2.

![WhatsApp Image 2024-01-19 at 23 49 49](https://github.com/Invadel/CS_412_Machine_Learning/assets/120125253/6e46bd58-ecac-4e3e-acc4-278c4a0b437e)

![WhatsApp Image 2024-01-19 at 23 52 44](https://github.com/Invadel/CS_412_Machine_Learning/assets/120125253/4f232a28-47b5-49b4-8604-f879a7615085)

## Acknowledgments and Contributions

This project is the culmination of collective efforts and insightful collaboration among a group of dedicated individuals, each bringing their unique expertise and enthusiasm to the table:

- **Selin Ceydeli**: Implementation of the K-means clustering algorithm to derive new features from the dataset and to determine the points awarded for each question, Plots to illustrate the clustering approach, Preparing the README file
- **Mert Dursun**: Implementation of the Neural Network model inspired by ResNet, BERT Implementation, Plots for evaluation and visual interpretation
- **Canberk Tahıl**: BERT Implementation, Employing BERT-based visualizations for model evaluation and interpretation
- **Barış Tekdemir**: Focused on solving technical issues throughout the project
- **Onur Sezen**: Annotation of the code snippets for better clarity, Focused on solving technical issues throughout the project

We are immensely grateful for the diverse perspectives and skills each contributor brought to this project. Their dedication to excellence is evident in every line of code, every analysis, and the successful outcomes we achieved.

---

### Detailed Explanation of Approaches and Algorithms

#### Motivation Behind Clustering Approach

The clustering approach was crucial due to the unique challenges posed by the dataset:
- **Limited Data**: The dataset's limited scope made it difficult to extract meaningful features directly.
- **Complexity of ChatGPT Interactions**: The intricate nature of conversational data added to the complexity.
- **Absence of Specific Labels**: We lacked detailed labels for individual question scores.

K-means clustering was chosen for its efficiency in grouping similar data points, thereby enabling us to estimate scores for each question based on the cluster characteristics.

#### Benefits of the Clustering Approach

- **Feature Enhancement**: It allowed us to create meaningful features from an otherwise limited dataset.
- **Unsupervised Learning Advantage**: As an unsupervised technique, it bypassed the need for labeled data for individual question scores.
- **Foundation for Neural Network**: The features derived from clustering (cluster centers, estimated points) provided a solid base for the neural network model.

#### Neural Network Model: A Step Forward

After clustering, the neural network model capitalized on the newly formulated features to predict total grades. This two-step approach synergized unsupervised and supervised learning techniques, utilizing the strengths of each to overcome data limitations.

#### Testing and Validation

- **Cluster-Based Testing**: By applying the same clustering labels and centers to the testing dataset, we ensured consistency and relevance in testing.
- **Comprehensive Model Evaluation**: The model's success was measured not just in its ability to predict but also in how it generalized to unseen data.

### Conclusion and Future Work

This project exemplifies the effective combination of clustering and neural network techniques in a challenging data environment. Future work could explore more complex clustering algorithms, deeper neural network architectures, or alternative supervised learning methods to enhance prediction accuracy and model robustness.

---

*This README provides a comprehensive guide to understanding the methodologies, rationales, and technical details of the project. It serves as both a documentation and a starting point for those interested in exploring the intersection of machine learning and educational data analysis.*
