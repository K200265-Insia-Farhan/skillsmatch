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

# Configuration for the "development" environment
development_config = {
    "username": "postgres",
    "password": "skillsmatch1234",
    "database": "skillsmatch1",
    "host": "skillsmatch1.c12ygwcqcl5k.us-east-1.rds.amazonaws.com",
    "port": 5432,
    "sslmode": "require",
}

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

# Function to fetch courses data from the database
def fetch_courses_data(conn):
    try:
        course_query = """SELECT * FROM public."Courses";"""
        course_data = pd.read_sql(course_query, conn)

        print("Fetched courses data successfully")
        return course_data
    except Exception as e:
        print("Error fetching courses data:", e)
        return None

# Function to read data from database
def read_data_from_database():
    conn = connect_to_database(development_config)
    if conn:
        candidates_data = fetch_candidates_data(conn)
        courses_data = fetch_courses_data(conn)
        conn.close()  # Close the database connection
        return candidates_data, courses_data
    else:
        return None, None

# Load datasets using database queries
candidates_data, courses_data = read_data_from_database()

# Ensure data is loaded successfully
if candidates_data is None or courses_data is None:
    print("Error loading data from the database. Exiting...")
    exit()

# Combine text data for CountVectorizer
candidate_texts = candidates_data['skills'] + " " + candidates_data['preferredJobTitle'] + " " + candidates_data['location']+ " " + candidates_data['education_level']+ " " + candidates_data['experience'].astype(str)+ " " + candidates_data['preferredJobType']+ " " + candidates_data['work_preference']+ " " + candidates_data['softSkills']
courses_texts = courses_data['course_title'] + " " + courses_data['short_intro'] + " " +courses_data['category']+ " " +courses_data['sub_category']

# Replace NaN values with empty strings in candidate and job texts
candidate_texts.fillna('', inplace=True)
courses_texts.fillna('', inplace=True)

# CountVectorizer without max_features parameter
count_vectorizer = CountVectorizer(stop_words='english')
candidate_vectors = count_vectorizer.fit_transform(candidate_texts)
course_vectors = count_vectorizer.transform(courses_texts)

# Ensure that both candidate and job vectors have the same number of features
assert candidate_vectors.shape[1] == course_vectors.shape[1], "Number of features must match"

# Define the deep learning model architecture for candidate-course recommendation
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
    Dense(len(courses_data), activation='softmax')  # Output layer with softmax activation for multi-class classification
])

# Define a function for training the model with periodic progress
def train_model_periodically():
    epochs = 10  # Number of epochs for each training iteration
    interval_seconds = 3600  # Interval in seconds between training iterations
    while True:
        print("Training model...", flush=True)
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}...", flush=True)
            # Add your training code here
            history = model.fit(x=candidate_vectors.toarray(), y=candidates_data.index, epochs=1, batch_size=128, verbose=1)
            # Optionally save the model weights after training
        
        print("Training completed for this iteration.", flush=True)
        
        # Sleep for a certain period before the next training iteration
        print(f"Waiting for {interval_seconds} seconds before the next training iteration...", flush=True)
        time.sleep(interval_seconds)
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for candidate-course recommendation with EarlyStopping
try:
    history = model.fit(x=candidate_vectors.toarray(), y=candidates_data.index, epochs=10, batch_size=128)
except Exception as e:
    print("Error training the model:", e)

# Calculate cosine similarity between candidates and courses
cosine_sim_matrix = cosine_similarity(candidate_vectors, course_vectors)
recommended_urls = {}
# Initialize recommended indices dictionary to keep track of recommended courses for each candidate
recommended_indices = {}
# Function to recommend courses for a candidate using cosine similarity
def recommend_courses(candidate_id, top_n=15):
    candidate_index = candidates_data[candidates_data['candidate_id'] == candidate_id].index
    if len(candidate_index) == 0:
        return []  # Candidate ID not found
    candidate_cosine_sim = cosine_sim_matrix[candidate_index]
    # Get indices of top courses based on cosine similarity
    top_indices = np.argsort(candidate_cosine_sim[0])[-top_n:]
    
    # Filter out courses with duplicate URLs
    unique_indices = []
    unique_urls = set()
    for idx in top_indices:
        course_url = courses_data.loc[idx, 'URL']
        if course_url not in unique_urls:
            unique_indices.append(idx)
            unique_urls.add(course_url)
            if len(unique_indices) == top_n:
                break
    
    # Update recommended_urls for this candidate
    recommended_urls[candidate_id] = recommended_urls.get(candidate_id, []) + [courses_data.loc[idx, 'URL'] for idx in unique_indices]
    return unique_indices





# Function to recommend courses for a given candidate ID via API
@app.route('/recommend/<int:candidate_id>', methods=['GET'])
def recommend_course_api(candidate_id):
    # Connect to the database
    conn = connect_to_database(development_config)
    if conn is None:
        response = jsonify({'error': 'Unable to connect to the database'})
        response.status_code = 500
        return response
    
    try:
        # Fetch candidate data from the database using candidate_id
        candidate_query = f"""SELECT * FROM public."Candidates" WHERE candidate_id = {candidate_id};"""
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

        # Recommend courses for the given candidate
        recommended_indices = recommend_courses(candidate_id)

        recommended_courses = []
        for course_id in recommended_indices:
            course_info = courses_data.iloc[course_id]
            recommended_course = {
                'course_title': str(course_info['course_title']),
                'short_intro': str(course_info['short_intro']),
                'category': str(course_info['category']),
                'sub_category': str(course_info['sub_category']),
                'course_url': str(course_info['URL'])  # Include course URL in the response
            }
            recommended_courses.append(recommended_course)

        response = jsonify({'recommended_courses': recommended_courses})
        response.status_code = 200
        return response
    except Exception as e:
        response = jsonify({'error': str(e)})
        response.status_code = 500
        return response

        

if __name__ == '__main__':
    # Start the training thread
    training_thread = threading.Thread(target=train_model_periodically)
    training_thread.start()

    # Run the Flask application
    app.run(port=2004, host='0.0.0.0')