import pandas as pd
import numpy as np
import re
import pyLDAvis
from sqlalchemy import create_engine
from IPython.core.display import display, HTML

#Retrieve HTML from database
# Create an engine instance
engine = create_engine('postgresql+psycopg2://pvahbfuxwqhvpq:3837ad2efc075df162ec73cc54d80e55b1aff7a1098b0eb5916502107f4b97bb@ec2-34-194-40-194.compute-1.amazonaws.com/dcrh9u79n7rtsa')
# Connect to PostgreSQL server
dbConnection = engine.connect()
# Read data from PostgreSQL database table and load into a DataFrame instance
dataFrame = pd.read_sql("select * from \"html_table\"", dbConnection)
pd.set_option('display.expand_frame_repr', False)
# Close the database connection
dbConnection.close()

#Display HTML
html_str = dataFrame.iloc[0,0]
display(HTML(html_str))