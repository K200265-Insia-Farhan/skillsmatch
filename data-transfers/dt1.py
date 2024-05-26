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

with open('candidates.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Prepare the INSERT statement
        insert_query = """
            INSERT INTO public."Candidates" (
                firstname,
                lastname,
                email,
                password,
                skills,
                location,
                education_level,
                "preferredJobTitle",
                "preferredJobType",
                experience,
                work_preference,
                "softSkills"
            ) VALUES (
                 %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s
            )
        """

        # Execute the INSERT statement
        cursor.execute(insert_query, (
            row['firstname'],
            row['lastname'],
            row['email'],
            row['password'],
            row['skills'],
            row['location'],
            row['education'],
            row['preferredJobTitle'],  # Map 'preferredJobTitle' from CSV to 'job_preferred' in DB
            row['preferredJobType'],
            row['experience'],
            row['preferredWorkType'],
            row['softSkills']
        ))

# Commit changes and close connection
conn.commit()
conn.close()