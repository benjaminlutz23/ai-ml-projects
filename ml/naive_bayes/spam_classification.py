# Benjamin Lutz
# Spam Classification Using Gaussian Naive Bayes 

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# --------------------------------------------------------------------------------------------

# Load the dataset from the CSV file
print("Loading the Spambase dataset from 'data/spambase.data'... \n")
data = pd.read_csv('data/spambase.data', header=None)

# Extract features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset into training and test sets. "stratify" parameter ensures that we maintain the original proportion 
# of training to test from the original dataset
print("Splitting the dataset into training and test sets... \n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# Ensure each set has about 2,300 instances
X_train = X_train.iloc[:2300]
y_train = y_train.iloc[:2300]
X_test = X_test.iloc[:2300]
y_test = y_test.iloc[:2300]

# Verify the split
print(f'Training set: {len(X_train)} instances, Spam: {sum(y_train)}, Not-spam: {len(y_train) - sum(y_train)}')
print(f'Test set: {len(X_test)} instances, Spam: {sum(y_test)}, Not-spam: {len(y_test) - sum(y_test)} \n')

# Save the training and test sets to CSV files
# Create the data directory if it does not exist
os.makedirs('data', exist_ok=True)

print("Saving the training and test sets to CSV files...")
X_train.to_csv('data/X_train.csv', index=False, header=False)
y_train.to_csv('data/y_train.csv', index=False, header=False)
X_test.to_csv('data/X_test.csv', index=False, header=False)
y_test.to_csv('data/y_test.csv', index=False, header=False)
print("Training and test sets saved in the 'data' directory as 'X_train.csv', 'y_train.csv', 'X_test.csv', and 'y_test.csv' \n")


# -------------------------------------------------------------------------------------------

# Compute the prior probability for each class
n_spam = sum(y_train)  # Number of spam emails
n_total = len(y_train)  # Total number of emails
n_not_spam = n_total - n_spam  # Number of not-spam emails

prior_spam = n_spam / n_total  # Proportion of spam emails
prior_not_spam = n_not_spam / n_total  # Proportion of not-spam emails

print(f'Prior probability of spam (P(1)): {prior_spam}')
print(f'Prior probability of not spam (P(0)): {prior_not_spam} \n')

# Compute the mean and standard deviation for each feature given each class
mean_std = {
    'spam': {'mean': [], 'std': []},
    'not_spam': {'mean': [], 'std': []}
}

# For each feature
for feature in range(X_train.shape[1]):
    # Compute mean and standard deviation for spam class
    spam_feature_values = X_train[y_train == 1].iloc[:, feature]
    spam_mean = spam_feature_values.mean()
    spam_std = spam_feature_values.std()
    if spam_std == 0:
        spam_std = 0.0001
    mean_std['spam']['mean'].append(spam_mean)
    mean_std['spam']['std'].append(spam_std)
    
    # Compute mean and standard deviation for not-spam class
    not_spam_feature_values = X_train[y_train == 0].iloc[:, feature]
    not_spam_mean = not_spam_feature_values.mean()
    not_spam_std = not_spam_feature_values.std()
    if not_spam_std == 0:
        not_spam_std = 0.0001
    mean_std['not_spam']['mean'].append(not_spam_mean)
    mean_std['not_spam']['std'].append(not_spam_std)

# Print the mean and standard deviation for each class for verification
print("Mean and Standard Deviation for spam class: \n")
print("Means:", mean_std['spam']['mean'], "\n")
print("Standard Deviations:", mean_std['spam']['std'], "\n\n")

print("Mean and Standard Deviation for not-spam class: \n")
print("Means:", mean_std['not_spam']['mean'], "\n")
print("Standard Deviations:", mean_std['not_spam']['std'], "\n\n")



# --------------------------------------------------------------------------------------------

# Define Gaussian PDF
def gaussian_pdf(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

# Predict function (I added an epsilon to avoid division by zero)
def predict(X, epsilon=1e-10):
    y_pred = []
    # For each instance
    for i in range(X.shape[0]):
        # Compute the log of the likelihood of the instance being spam and not spam
        spam_likelihood = np.log(prior_spam + epsilon)
        not_spam_likelihood = np.log(prior_not_spam + epsilon)
        
        # For each feature
        for feature in range(X.shape[1]):
            # Compute the likelihood of the feature value given each class
            spam_prob = gaussian_pdf(X.iloc[i, feature], mean_std['spam']['mean'][feature], mean_std['spam']['std'][feature])
            not_spam_prob = gaussian_pdf(X.iloc[i, feature], mean_std['not_spam']['mean'][feature], mean_std['not_spam']['std'][feature])
            
            # Add the log likelihood to the total likelihood
            spam_likelihood += np.log(spam_prob + epsilon)
            not_spam_likelihood += np.log(not_spam_prob + epsilon)
        
        # Predict the class with the highest likelihood (argmax)
        if spam_likelihood > not_spam_likelihood:
            y_pred.append(1)
        else:
            y_pred.append(0)
    
    return np.array(y_pred)

# Predict on the test set
y_pred = predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy} \n')
print(f'Precision: {precision} \n')
print(f'Recall: {recall} \n')
print('Confusion Matrix:')
print(conf_matrix)
print("\n")
