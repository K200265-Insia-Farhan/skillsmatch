import psycopg2
import csv

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="skillsmatch",
    host="skillsmatch.cjkwoye4kdui.us-east-1.rds.amazonaws.com",
    port="5432"  # Use the correct port number here
)

# Create a cursor object
cursor = conn.cursor()

# Read data from CSV and insert into database
with open('jobs.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Prepare the INSERT statement
        insert_query = """
        INSERT INTO public."Jobs" 
        (job_title, skills_required, job_location, education_required, work_experience_required, job_type, work_type, soft_skills_required,"companyHR_id") VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

        # Execute the INSERT statement
        cursor.execute(insert_query, (
            row['job_title'],
            row['skills_required'],
            row['job_location'],
            row['education_required'],
            row['work_experience_required'],
            row['job_type'],
            row['work_type'],
            row['soft_skills_required'],
            row['companyHR_id']
        ))

# Commit changes and close connection
conn.commit()
conn.close()