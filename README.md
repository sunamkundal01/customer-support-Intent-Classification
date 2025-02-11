# Customer Support Intent Classification

This project implements an intent classification model using BERT for customer support. It is structured to facilitate training, inference, and dataset management.

## Project Structure

- **data/**: Contains the dataset used for training the model.
  - `customer_intent.json`: The dataset file.

- **models/**: Contains the model definition.
  - `bert_classifier.py`: Defines the BERT classifier architecture and methods for training and evaluation.

- **training/**: Contains the training scripts.
  - `train.py`: Script for loading the dataset, initializing the model, and training.

- **inference/**: Contains scripts for making predictions.
  - `predict.py`: Loads the trained model and processes input data for predictions.

- **utils/**: Contains utility functions for dataset processing and configuration.
  - `dataset.py`: Functions for loading, preprocessing, and splitting the dataset.
  - `config.py`: Configuration settings including hyperparameters and file paths.

- `main.py`: Entry point for inference, orchestrating model loading and prediction.

- `requirements.txt`: Lists the required dependencies for the project.

## Setup Instructions

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd intent_classification
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   Ensure that the `customer_intent.json` file is correctly formatted and placed in the `data/` directory.

4. Train the model:
   Run the training script:
   ```sh
   python training/train.py
   ```

5. Make predictions:
   Use the inference script:
   ```sh
   python inference/predict.py
   ```

## Usage

To use the model for making predictions, simply run the `main.py` file and enter the input text when prompted:
```sh
python main.py
```
Enter the text you want to classify, and the model will output the predicted intent. Type "exit" to stop the program.

## Applications and Real-Life Usage

Intent classification models like this one have a wide range of applications in real-life scenarios, particularly in customer support and service automation. Some of the key applications include:

1. **Automated Customer Support**: Automatically classify customer queries and route them to the appropriate department or provide instant responses using predefined templates.
2. **Chatbots**: Enhance chatbot capabilities by accurately understanding and responding to customer intents, leading to more effective and efficient customer interactions.

By implementing this intent classification model, businesses can improve their customer support processes, reduce response times, and enhance overall customer satisfaction.