
import os
import random as rand
import numpy as np

import pandas as pd
import regex as re
import torch


def normalize(df, columns):
    result = df.copy()
    for column in columns:
        mu = df[column].mean(axis=0)
        sigma = df[column].std(axis=0)
        assert sigma != 0
        result[column] = (df[column] - mu) / sigma
    return result


def preprocess_adult(df, protected_group, target):
    df = df.drop("fnlwgt", axis=1)

    numerical_columns = ["age", "educational-num", "capital-gain", "capital-loss",
                         "hours-per-week"]
    df = normalize(df, numerical_columns)

    df['Class-label'] = [1 if v == 1 else 0 for v in df['Class-label']]

    mapped_sex_values = df.sex.map({"Male": 0, "Female": 1})
    df.loc[:, "sex"] = mapped_sex_values

    # make race binary
    def race_map(value):
        if value != "White":
            return (1)
        return (0)

    mapped_race_values = df.race.map(race_map)
    df.loc[:, "race"] = mapped_race_values

    categorical = df.columns.tolist()
    for column in numerical_columns:
        categorical.remove(column)
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

    #df = sample_by_group_ratios(group_ratios, df, seed)

    # convert to one-hot vectors
    categorical_non_binary = ["workclass", "education", "marital-status", "occupation",
                              "relationship", "native-country"]
    df = pd.get_dummies(df, columns=categorical_non_binary)

    return df


def preprocess_dutch(df, protected_group, target):
    mapped_sex_values = df.sex.map({"male": 0, "female": 1})
    df.loc[:, "sex"] = mapped_sex_values

    # note original dataset has values {0,1,9} for prev_res_place, but all samples with 9 are underage, hence get dropped
    mapped_prev_res_place_values = df.prev_residence_place.map({1: 0, 2: 1})
    df.loc[:, "prev_residence_place"] = mapped_prev_res_place_values

    categorical = df.columns.to_list()
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

    # convert categorical unprotected features to one-hot vectors
    if target in categorical:
        categorical.remove(target)
    if "sex" in categorical:
        categorical.remove("sex")  # binary
    if "prev_res_place" in categorical:
        categorical.remove("prev_res_place")  # binary

    #df = sample_by_group_ratios(group_ratios, df, seed)

    df = pd.get_dummies(df, columns=categorical)

    return df


def preprocess_bank(df, protected_group, target):
    numerical_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    if protected_group in numerical_columns:
        numerical_columns.remove(protected_group)
    df = normalize(df, numerical_columns)

    df['y'] = [1 if v == 'yes' else 0 for v in df['y']]

    df['marital'] = ['Married' if v == 'married' else 'Non-Married' for v in df['marital']]
    mapped_marital_values = df.marital.map({'Married': 1, 'Non-Married': 0})
    df.loc[:, "marital"] = mapped_marital_values

    df['default'] = [1 if v == 'yes' else 0 for v in df['default']]
    df['housing'] = [1 if v == 'yes' else 0 for v in df['housing']]
    df['loan'] = [1 if v == 'yes' else 0 for v in df['loan']]

    categorical = df.columns.to_list()
    for column in numerical_columns:
        categorical.remove(column)
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

    #df = sample_by_group_ratios(group_ratios, df, seed)

    # convert to one-hot vectors
    categorical_non_binary = ["job", "education", "contact", "month", "poutcome"]
    df = pd.get_dummies(df, columns=categorical_non_binary)

    return df


def preprocess_credit(df, protected_group, target):

    numerical_columns = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4",
                         "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4",
                         "PAY_AMT5", "PAY_AMT6"]
    if protected_group in numerical_columns:
        numerical_columns.remove(protected_group)
    df = normalize(df, numerical_columns)

    df['SEX'] = ['Male' if v == 1 else 'Female' for v in df['SEX']]
    mapped_sex_values = df.SEX.map({"Male": 0, "Female": 1})
    df.loc[:, "SEX"] = mapped_sex_values

    categorical = df.columns.to_list()
    for column in numerical_columns:
        categorical.remove(column)
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

    #df = sample_by_group_ratios(group_ratios, df, seed)

    # convert to one-hot vectors
    categorical_non_binary = ["EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df = pd.get_dummies(df, columns=categorical_non_binary)

    return df


def preprocess_compas(df, protected_group, target):
    new_columns = ["age_cat", "race", "sex", "priors_count", "c_charge_degree", "score_text", "v_score_text",
                   "two_year_recid"]
    df = df[new_columns]

    # 让预测为1成为好事
    mapped_two_year_recid_values = df.two_year_recid.map({1: 0, 0: 1})
    df.loc[:, "two_year_recid"] = mapped_two_year_recid_values

    numerical_columns = ["priors_count"]
    if protected_group in numerical_columns:
        numerical_columns.remove(protected_group)
    df = normalize(df, numerical_columns)

    # 去掉了太多的数据
    df = df[(df['race'] == 'African-American') | (df['race'] == "Caucasian")]
    df['race'] = ['Black' if v == 'African-American' else "White" for v in df['race']]
    # 1是弱势群体
    mapped_race_values = df.race.map({"White": 0, "Black": 1})
    df.loc[:, "race"] = mapped_race_values

    df['sex'] = [1 if v == 'Female' else 0 for v in df['sex']]
    df['c_charge_degree'] = [1 if v == 'F' else 0 for v in df['c_charge_degree']]

    categorical = df.columns.to_list()
    for column in numerical_columns:
        categorical.remove(column)
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

    #df = sample_by_group_ratios(group_ratios, df, seed)

    # convert to one-hot vectors
    categorical_non_binary = ["age_cat", "score_text", "v_score_text"]
    df = pd.get_dummies(df, columns=categorical_non_binary)

    return df


def preprocess_law(df, protected_group, target):
    numerical_columns = ["decile1b", "decile3", "lsat", "ugpa", "zfygpa", "zgpa"]
    if protected_group in numerical_columns:
        numerical_columns.remove(protected_group)
    df = normalize(df, numerical_columns)

    mapped_race_values = df.race.map({"White": 0, "Non-White": 1})
    df.loc[:, "race"] = mapped_race_values

    categorical = df.columns.to_list()
    for column in numerical_columns:
        categorical.remove(column)
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

    #df = sample_by_group_ratios(group_ratios, df, seed)

    # convert to one-hot vectors
    categorical_non_binary = ["fam_inc", "tier"]
    df = pd.get_dummies(df, columns=categorical_non_binary)

    return df






