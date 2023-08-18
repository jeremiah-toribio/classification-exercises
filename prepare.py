import pandas as pd
import numpy as np
import env
import acquire
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def prep_iris():
    '''
    Pulls and prepares iris data. Drops a few columns, alters name of 'species' column and adds an additional number
    value associated by species type.
    '''

    iris = acquire.get_iris_data('iris_db')
    iris = iris.drop(columns=['species_id', 'measurement_id'])
    iris = iris.rename(columns={'species_name':'species'})
    iris_dummies = pd.get_dummies(iris['species']).astype('int')
    iris = pd.concat([iris, iris_dummies], axis=1)
    return iris

def prep_titanic():
    '''
    Pulls data from mySql db and assigns as titanic.
    This function will drop duplicate columns and encode the rest of the categorical columns.
    '''
    # acquire titanic
    titanic = acquire.get_titanic_data('titanic_db')
    # drop duplicate or not useful information
    titanic = titanic.drop(columns=['class','embark_town','deck'])
    # handle nulls
    titanic['age'] = titanic['age'].fillna(titanic.age.mean()).round()
    titanic['embarked'] = titanic['embarked'].fillna(value='S')
    # encode categories
    dummy_sex = pd.get_dummies(titanic['sex'], drop_first=True).astype(int)
    dummy_embarked = pd.get_dummies(titanic[['embarked']],drop_first=True).astype(float)
    dummy = pd.concat([dummy_sex,dummy_embarked], axis = 1)
    titanic = pd.concat([titanic,dummy], axis=1)
    return titanic

def prep_telco():
    '''
    Pulls data from mySql server and drops duplicate columns and values (keeps 1 of needed)
    encodes all categorical data and drops columns that are unnecessary as a by product of 
    new dummy columns.
    '''
    # pulling data from mysql using get_telco_data
    telco = acquire.get_telco_data('telco_churn')
    telco = telco.drop_duplicates()
    # removing duplicate columns
    telco = telco.loc[:,~telco.columns.duplicated()].copy()
    #encoding categorical type data
    dummy_df = pd.get_dummies(telco[['multiple_lines','online_security','online_backup','payment_type',
                                'contract_type', 'tech_support','streaming_tv','streaming_movies','device_protection']],dtype=int ,drop_first=True)

    telco['partner_binary'] = pd.get_dummies(telco['partner'], dtype=int, drop_first=True)
    telco['dependents_binary'] = pd.get_dummies(telco['dependents'], dtype=int,drop_first=True)
    telco['phone_service_binary'] = pd.get_dummies(telco['phone_service'], dtype=int, drop_first=True)
    telco['gender_binary'] = pd.get_dummies(telco['gender'], dtype=int, drop_first=True)
    telco['paperless_billing_binary'] = pd.get_dummies(telco['paperless_billing'], dtype=int, drop_first=True)
    telco['churn_binary'] = pd.get_dummies(telco['churn'], dtype=int, drop_first=True)
    telco = pd.concat([telco, dummy_df], axis=1)

    telco['total_charges'] = telco['total_charges'].str.replace(' ','0').astype('float')
    # dropping extra columns after encoding
    telco = telco.drop(columns=['online_security_No internet service',
                    'online_security_No internet service','online_backup_No internet service',
                    'tech_support_No internet service','streaming_tv_No internet service','streaming_movies_No internet service','device_protection_No internet service',
                    'tech_support','device_protection'])
    # 'multiple_lines','online_security', 'online_backup','streaming_tv','streaming_movies',
    return telco
    

def splitter(df,target='churn'):
    '''
    Returns
    Train, Validate, Test
    '''
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])

    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train[target])

    return train, validate, test