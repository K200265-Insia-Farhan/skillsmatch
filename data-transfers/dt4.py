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

with open('company.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Prepare the INSERT statement
        insert_query = """
            INSERT INTO public."Companies"
            (
            company_name, 
            company_location, 
            company_email, 
            company_website, 
            "companyHR_id")
	        VALUES ( %s, %s, %s, %s, %s)
        """

        # Execute the INSERT statement
        cursor.execute(insert_query, (
            row['company_name'],
            row['company_location'],
            row['company_email'],
            row['company_website'],
            row['companyHR_id']
        ))

# Commit changes and close connection
conn.commit()
conn.close()