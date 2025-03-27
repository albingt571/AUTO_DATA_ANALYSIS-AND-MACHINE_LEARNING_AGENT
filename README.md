# ML Agent Project

## Overview
This project is a machine learning workflow automation tool that leverages a language model (LLM) to assist in data cleaning, exploratory data analysis (EDA), feature engineering, and model training. The project is built using Python and Streamlit for the user interface.

## Features
- *Data Cleaning*: Automatically identifies and removes unwanted columns, handles missing values, and removes duplicates.
- *Exploratory Data Analysis (EDA)*: Generates comprehensive visualizations and statistical summaries.
- *Feature Engineering*: Automatically engineers features based on the dataset and target column.
- *Model Training*: Trains and evaluates machine learning models, providing a summary of the best model and its metrics.

## File Structure

mlagent/
├── .env
├── app.py
├── config.py
├── requirements.txt
├── database/
│   └── db.py
├── files_and_models/
│   ├── eda_output/
│   ├── processed_data/
│   ├── saved_models/
│   └── uploads/
├── functions/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── storage.py
└── workflows/
    └── rag.py


## Installation
1. Clone the repository:
   bash
   git clone https://github.com/yourusername/mlagent.git
   cd mlagent
   
2. Set up a virtual environment:
   bash
   python -m venv env
   
3. Activate the virtual environment:
   - On Windows:
     bash
     .\env\Scripts\activate
     
   - On macOS/Linux:
     bash
     source env/bin/activate
     
4. Install the required dependencies:
   bash
   pip install -r requirements.txt
   
5. Create a .env file in the root directory and add your environment variables:
   plaintext
   # Example environment variables
   GROQ_API_KEY=your_groq_api_key
   POSTGRESQL_URL=your_postgresql_url
   

## PostgreSQL URL Format
The PostgreSQL URL should be formatted as follows:

postgresql://username:password@hostname:port/database_name

- *username*: Your PostgreSQL username.
- *password*: Your PostgreSQL password.
- *hostname*: The host where your PostgreSQL server is running (e.g., localhost or an IP address).
- *port*: The port number on which PostgreSQL is listening (default is 5432).
- *database_name*: The name of the database you want to connect to.

### Example
plaintext
postgresql://myuser:mypassword@localhost:5432/mydatabase


## Usage

1. Run the Streamlit app:
   bash
   streamlit run app.py
   
2. Upload your dataset and follow the steps in the app:
   - *Step 1*: Upload CSV file.
   - *Step 2*: Remove unwanted columns.
   - *Step 3*: Clean the data.
   - *Step 4*: Perform EDA.
   - *Step 5*: Select the target column.
   - *Step 6*: Engineer features.
   - *Step 7*: Train and evaluate models.
   - *Step 8*: Store summaries in the database.
   - *Step 9*: Download cleaned data and trained models.
   - *Step 10*: Interact with the chatbot for additional insights.

## Documentation
For detailed documentation on each function and workflow, refer to the respective Python files in the functions/ and workflows/ directories.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
