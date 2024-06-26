import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import threading
import time
# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Add this line to enable CORS for all routes

# Function to connect to the database using SQLAlchemy
def connect_to_database(config):
    try:
        db_url = f"postgresql://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        engine = create_engine(db_url, connect_args={"sslmode": config["sslmode"]})
        conn = engine.connect()
        print("Connected to database successfully")
        return conn
    except Exception as e:
        print("Unable to connect to the database:", e)
        return None

# Configuration for the "development" environment
development_config = {
    "username": "postgres",
    "password": "skillsmatch1234",
    "database": "skillsmatch1",
    "host": "skillsmatch1.c12ygwcqcl5k.us-east-1.rds.amazonaws.com",
    "port": 5432,
    "sslmode": "require",
}

# Call the function with the development configuration
conn = connect_to_database(development_config)

# Function to fetch candidates data from the database
def fetch_candidates_data(conn):
    try:
        candidates_query = """SELECT * FROM public."Candidates" ORDER BY candidate_id;"""
        candidates_data = pd.read_sql(candidates_query, conn)
        
        print("Fetched candidates data successfully")
        return candidates_data
    except Exception as e:
        print("Error fetching candidates data:", e)
        return None

# Function to fetch jobs data from the database
def fetch_jobs_data(conn):
    try:
        jobs_query = """SELECT j.*, c.company_name, c.company_email 
                        FROM public."Jobs" j
                        JOIN public."Companies" c ON j."companyHR_id" = c."companyHR_id"
                        ORDER BY job_id;"""
        jobs_data = pd.read_sql(jobs_query, conn)
        
        print("Fetched jobs data successfully")
        return jobs_data
    except Exception as e:
        print("Error fetching jobs data:", e)
        return None

# Function to read data from database
def read_data_from_database():
    conn = connect_to_database(development_config)
    if conn:
        candidates_data = fetch_candidates_data(conn)
        jobs_data = fetch_jobs_data(conn)
        conn.close()  # Close the database connection
        return candidates_data, jobs_data
    else:
        return None, None

# Load datasets using database queries
candidates_data, jobs_data = read_data_from_database()

# Ensure data is loaded successfully
if candidates_data is None or jobs_data is None:
    print("Error loading data from the database. Exiting...")
    exit()

# Combine text data for CountVectorizer
candidate_texts = candidates_data['skills'] + " " + candidates_data['preferredJobTitle'] + " " + \
                 candidates_data['location'] + " " + candidates_data['education_level'] + " " + \
                 candidates_data['experience'].astype(str) + " " + candidates_data['preferredJobType'] + " " + \
                 candidates_data['work_preference'] + " " + candidates_data['softSkills']
job_texts = jobs_data['skills_required'] + " " + jobs_data['job_title'] + " " + jobs_data['job_location'] + " " + \
            jobs_data['education_required'] + " " + jobs_data['work_experience_required'].astype(str) + " " + \
            jobs_data['job_type'] + " " + jobs_data['work_type'] + " " + jobs_data['soft_skills_required']

# Replace NaN values with empty strings in candidate and job texts
candidate_texts.fillna('', inplace=True)
job_texts.fillna('', inplace=True)

# CountVectorizer without max_features parameter
count_vectorizer = CountVectorizer(stop_words='english')
candidate_vectors = count_vectorizer.fit_transform(candidate_texts)
job_vectors = count_vectorizer.transform(job_texts)

# Ensure that both candidate and job vectors have the same number of features
assert candidate_vectors.shape[1] == job_vectors.shape[1], "Number of features must match"

# Define the deep learning model architecture for candidate-job recommendation
model = Sequential([
    Dense(512, input_shape=(candidate_vectors.shape[1],), activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(jobs_data), activation='softmax')  # Output layer with softmax activation for multi-class classification
])

def train_model_periodically():
    epochs = 10  # Number of epochs for each training iteration
    interval_seconds = 3600  # Interval in seconds between training iterations
    while True:
        print("Starting training iteration...", flush=True)
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}...", flush=True)
            # Add your training code here
            history = model.fit(x=candidate_vectors.toarray(), y=candidates_data.index, epochs=1, batch_size=128, verbose=1)
            # Optionally save the model weights
        
        print("Training completed for this iteration.", flush=True)
        
        # Sleep for a certain period before the next training iteration
        print(f"Waiting for {interval_seconds} seconds before the next training iteration...", flush=True)
        time.sleep(interval_seconds)


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for candidate-job recommendation with EarlyStopping
history = model.fit(x=candidate_vectors.toarray(), y=candidates_data.index, epochs=10, batch_size=128)

# Calculate cosine similarity between candidates and jobs
cosine_sim_matrix = cosine_similarity(candidate_vectors, job_vectors)

# Function to recommend jobs for a candidate using cosine similarity
def recommend_jobs(candidate_id, top_n=7):
    candidate_index = candidates_data[candidates_data['candidate_id'] == candidate_id].index
    if len(candidate_index) == 0:
        return []  # Candidate ID not found
    candidate_cosine_sim = cosine_sim_matrix[candidate_index]
    top_indices = np.argsort(candidate_cosine_sim[0])[-top_n:]
    return top_indices

# # Mocked function to recommend jobs for a candidate using cosine similarity
# def recommend_jobs(candidate_id, top_n=7):
#     total_jobs = len(jobs_data)
#     if top_n > total_jobs:
#         top_n = total_jobs  # Ensure not to recommend more jobs than available
#     return np.random.choice(total_jobs, top_n, replace=False)

# Function to recommend jobs for a given candidate ID via API
@app.route('/recommend_jobs/<int:candidate_id>', methods=['GET'])
def recommend_jobs_api(candidate_id):
    # Fetch candidate data from the database using candidate_id
    candidate_query = f"""SELECT * FROM public."Candidates" WHERE candidate_id = {candidate_id};"""
    print(candidate_query)
    candidate_data = pd.read_sql(candidate_query, conn)

    if candidate_data.empty:
        response = jsonify({'error': 'Candidate not found'})
        response.status_code = 404
        return response

    # Combine candidate data for CountVectorizer
    candidate_text = candidate_data['skills'] + " " + candidate_data['preferredJobTitle'] + " " + \
                     candidate_data['location'] + " " + candidate_data['education_level'] + " " + \
                     candidate_data['experience'].astype(str) + " " + candidate_data['preferredJobType'] + " " + \
                     candidate_data['work_preference'] + " " + candidate_data['softSkills']

    # Replace NaN values with empty strings
    candidate_text.fillna('', inplace=True)

    # Transform candidate text using CountVectorizer
    candidate_vector = count_vectorizer.transform(candidate_text)

    recommended_indices = recommend_jobs(candidate_id)
    recommended_jobs = []
    for job_id in recommended_indices:
        job_info = jobs_data.iloc[job_id]
        recommended_job = {
            'job_title': str(job_info['job_title']),
            'skills_required': str(job_info['skills_required']),
            'job_location': str(job_info['job_location']),
            'education_required': str(job_info['education_required']),
            'work_experience_required': int(job_info['work_experience_required']),
            'job_type': str(job_info['job_type']),
            'work_type': str(job_info['work_type']),
            'soft_skills_required': str(job_info['soft_skills_required']),
            'company_name': str(job_info['company_name']),
            'company_email': str(job_info['company_email'])
        }
        recommended_jobs.append(recommended_job)
    response = jsonify({'recommended_jobs': recommended_jobs})
    response.status_code = 200
    return response

if __name__ == '__main__':
    # Start the training thread
    training_thread = threading.Thread(target=train_model_periodically)
    training_thread.start()

    # Run the Flask application
    app.run(port=2003, host='0.0.0.0')