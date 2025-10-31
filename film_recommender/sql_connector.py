'''
sql_connector.py
'''

import logging
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

logger_sqlconnector = logging.getLogger('sql_connector')  # TODO: change back to __name__


class SQLConnector:

    def __init__(self, hostname, port, database, username, password):
        self.hostname = hostname
        self.port = port
        self.database = database
        self.username = username
        self.password = password

    def get_connection(self):
        try:
            conn = psycopg2.connect(
                host=self.hostname,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                sslmode='require'
            )

            logger_sqlconnector.info("Successfully established database connection.")

            return conn
        
        except Exception:
            logger_sqlconnector.exception("Error occurred in establishing database connection.")
            raise

    def read_sql_table(self, table_name, sort_on=None):
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.hostname,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                sslmode='require'
            )

            if sort_on:
                query = f"SELECT * FROM {table_name} ORDER BY {sort_on}"
            else:
                query = f"SELECT * FROM {table_name}"

            df = pd.read_sql(query, conn)
            logger_sqlconnector.info(f"Successfully read in '{table_name}'. Shape of df: {df.shape}")
            return df

        except Exception:
            logger_sqlconnector.exception(f"Error occurred in reading table '{table_name}'.")
            raise

        finally:
            if conn:
                conn.close()

    def alter_sql_table(self, query):
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.hostname,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                sslmode='require'
            )
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()
            cur.close()
            logger_sqlconnector.info(f"Successfully executed query:\n{query}")

        except Exception:
            logger_sqlconnector.exception(f"Error occurred in executing query:\n{query}")
            raise

        finally:
            if conn:
                conn.close()

    def append_df_to_sql_table(self, df, table_name):
        engine = None
        try:
            engine = create_engine(
                f"postgresql+psycopg2://{self.username}:{self.password}@{self.hostname}:{self.port}/{self.database}"
            )
            df.to_sql(table_name, engine, if_exists='append', index=False)
            logger_sqlconnector.info(f"Data successfully appended to '{table_name}'.")

        except Exception:
            logger_sqlconnector.exception(f"Error occurred in appending data to '{table_name}'.")
            raise

        finally:
            if engine:
                engine.dispose()

    def reset_sql_table(self, table, table_creation_date):
        query = f'''
        DELETE FROM {table}
        WHERE created_at > '{table_creation_date}';
        '''
        try:
            self.alter_sql_table(query)
            logger_sqlconnector.info(f"SQL table '{table}' has been reset!")
        except Exception:
            logger_sqlconnector.exception(f"Error occurred in resetting SQL table '{table}'.")
            raise
