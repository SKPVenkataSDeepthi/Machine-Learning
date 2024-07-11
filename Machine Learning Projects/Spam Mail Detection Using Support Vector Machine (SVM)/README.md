This project is about building a spam mail detection system using a Support Vector Machine (SVM) classifier that can effectively classify emails as spam or ham. The key steps include data preprocessing, feature extraction, model training, and evaluation.

Steps Involved
1. Import Libraries
2. Download NLTK Stopwords:
   * This ensures that the necessary NLTK data files (stopwords) are available for text preprocessing.
3. Load Dataset:
   * The dataset is loaded from a CSV file and a quick inspection of the first few rows is done to understand its structure.
4. Data Preprocessing Function:
   * Text Cleaning: Remove non-alphabetic characters.
   * Lowercasing: Convert text to lowercase.
   * Tokenization and Stopword Removal: Split text into words and remove common stopwords.
   * Stemming: Reduce words to their root form using the SnowballStemmer.
   * Joining Words: Combine words back into a single string.
5. Apply Preprocessing to the Dataset:
   * The preprocessing function is applied to the text data in the dataset to prepare it for feature extraction.
6. Feature Extraction using TF-IDF:
   * TF-IDF Vectorization: Transform the text data into numerical features using the Term Frequency-Inverse Document Frequency (TF-IDF) method. The number of features is limited to 3000.
7. Convert Labels to Binary:
   * The labels in the dataset are converted to binary format where 'spam' is 1 and 'ham' is 0.
8. Split Data into Training and Testing Sets:
   * The data is split into training and testing sets with 80% of the data used for training and 20% for testing. The random_state parameter ensures reproducibility.
9. Train SVM model:
    * An SVM model with a linear kernel is trained using the training data.
10. Make Predictions:
    * The trained SVM model is used to make predictions on the test data.
11. Evaluate the model:
    * The accuracy and classification report (precision, recall, F1-score) of the model are printed to evaluate its performance.

