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

with open('courses.csv', 'r', encoding='utf-8') as file:  # Specify encoding here
    reader = csv.DictReader(file)
    for row in reader:
        # Prepare the INSERT statement
        insert_query = """
            INSERT INTO public."Courses"(
            course_title, 
            "URL", 
            short_intro, 
            category, 
            sub_category, 
            course_type, 
            language, 
            subtitle_language, 
            skills, 
            instructors, 
            rating, 
            number_of_views, 
            duration, 
            site)
	        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s,%s)
        """

        # Execute the INSERT statement
        cursor.execute(insert_query, (
            row['course_title'],
            row['URL'],
            row['short_intro'],
            row['category'],
            row['sub_category'],  # Corrected column name here
            row['course_type'],
            row['language'],
            row['subtitle_language'],  
            row['skills'],
            row['instructors'],
            row['rating'],
            row['number_of_views'],
            row['duration'],
            row['site']
        ))

# Commit changes and close connection
conn.commit()
conn.close()
