    #Ingestion to Database
    currDate = start_date

    try:
        # Connect to an existing database
        connection = psycopg2.connect(user="pvahbfuxwqhvpq",
                                    password="3837ad2efc075df162ec73cc54d80e55b1aff7a1098b0eb5916502107f4b97bb",
                                    host="ec2-34-194-40-194.compute-1.amazonaws.com",
                                    port="5432",
                                    database="dcrh9u79n7rtsa")

        # Create a cursor to perform database operations
        cursor = connection.cursor()
        
        # Executing a SQL query to insert datetime into table
        
    #     read_file = open('pre_processed_data.json')
    #     data = json.load(read_file)
        
        cursor.execute("INSERT INTO json_table (json_string, timestamp) VALUES (%s, %s)", (data, currDate))
        connection.commit()
        print("1 item inserted successfully")

    #     # Executing a SQL query
    #     cursor.execute("SELECT json_string FROM json_table;")
    #     # Fetch result
    #     record = cursor.fetchone()
    #     print("You are connected to - ", record, "\n")

    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    return "Log---Scraper completed data collection"