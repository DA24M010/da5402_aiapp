import psycopg2
from psycopg2 import sql
import os
def create_feedback_table():
    try:
        # Connect to PostgreSQL
        connection = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "postgres"),
            port=os.environ.get("POSTGRES_PORT", "5432"),
            database=os.environ.get("POSTGRES_DB", "your_database_name"),
            user=os.environ.get("POSTGRES_USER", "your_user"),
            password=os.environ.get("POSTGRES_PASSWORD", "your_password")
        )

        
        cursor = connection.cursor()

        # SQL command to create the feedback_images table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS feedback_images (
            id SERIAL PRIMARY KEY,
            image_data BYTEA NOT NULL,
            label INT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # Execute the query
        cursor.execute(create_table_query)
        connection.commit()

        print("Table 'feedback_images' created successfully!")

    except Exception as error:
        print(f"Error creating table: {error}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Call the function to create the table
create_feedback_table()
