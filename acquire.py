import pandas as pd
import numpy as np
import env
import os


def get_titanic_data(database,user=env.user, password=env.password, host=env.host):
    '''
    Grabs titanic data from codeup mySql database if does not currently exist in users' system.
    Returns as dataframe.
    '''
    query = 'select * from passengers'
    connection = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    df = pd.read_sql(query, connection)
    return df

def get_iris_data(database,user=env.user, password=env.password, host=env.host):
    '''
    Grabs iris data from codeup mySql database if does not currently exist in users' system.
    Returns as dataframe.
    '''
    query = 'select * from species s join measurements m using (species_id)'
    connection = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    df = pd.read_sql(query, connection)
    return df


def get_telco_data(database,user=env.user, password=env.password, host=env.host):
    '''
    Grabs telco data from codeup mySql database.
    Returns as dataframe.
    '''
    query ='''
    SELECT *
    FROM customer_details AS cd
    LEFT OUTER JOIN customer_subscriptions AS cs ON cd.customer_id = cs.customer_id
    LEFT OUTER JOIN internet_service_types AS ist ON cs.internet_service_type_id = ist.internet_service_type_id
    LEFT OUTER JOIN payment_types AS pt ON cp.payment_type_id = pt.payment_type_id
    LEFT OUTER JOIN contract_types AS ct ON cc.contract_type_id = ct.contract_type_id;
    '''
    connection = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    df = pd.read_sql(query, connection)
    return df

