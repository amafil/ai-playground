import pyodbc
from typing import List
from models.question import Question


def mock_helpdesk_messages():
    result = []
    # This is a mock function. Replace it with the actual implementation
    # of the function that retrieves messages from the helpdesk channel.
    result.append(
        "here is the first request mock from the user. Use this method to retrieve the messages from the helpdesk channel."
    )
    return result


def load_questions(connection_string: str, query: str):
    knowledge: List[Question] = []
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
            knowledge.append(Question(*row))

    except pyodbc.Error as e:
        print(f"Error connecting to MSSQL database: {e}")
    finally:
        # Ensure the connection is closed
        if "connection" in locals():
            connection.close()

    return knowledge
