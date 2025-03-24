import pyodbc
from typing import List
import os
from models.knowledge import Knowledge


def load_knowledge_from_file(directory_path) -> List[Knowledge]:
    knowledge: List[Knowledge] = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            knowledge.append(Knowledge.load(file_path))

    return knowledge


def load_knowledge_from_mssql(connection_string: str, query: str) -> List[Knowledge]:
    knowledge: List[Knowledge] = []
    try:
        # Establish connection to the MSSQL database
        connection = pyodbc.connect(connection_string)
        cursor = connection.cursor()

        # Execute the query
        cursor.execute(query)
        rows = cursor.fetchall()

        # Map rows to QuestionAnswer objects
        for row in rows:
            # Assuming the database schema matches the QuestionAnswer fields
            knowledge.append(Knowledge(*row))

    except pyodbc.Error as e:
        print(f"Error connecting to MSSQL database: {e}")
    finally:
        # Ensure the connection is closed
        if 'connection' in locals():
            connection.close()

    return knowledge