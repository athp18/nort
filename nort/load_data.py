import pandas as pd
import numpy as np


def get_scorer(df):
    """Return the scorer name automatically."""
    return df.columns.get_level_values("scorer")[0]


def get_bodyparts(df):
    """Return list of bodyparts automatically."""
    return df.columns.get_level_values("bodyparts").unique()


def create_clean_df(df):
    """
    Flatten DLC DataFrame to columns: bodypart_x, bodypart_y, bodypart_likelihood
    """
    scorer = get_scorer(df)
    bodyparts = get_bodyparts(df)

    data = {}
    for bp in bodyparts:
        for key in ["x", "y", "likelihood"]:
            data[f"{bp}_{key}"] = df[scorer][bp][key].values

    clean_df = pd.DataFrame(data)
    clean_df.index.name = "frame"
    return clean_df


def combine_sessions(df_dict):
    """
    Combine sessions (dict of name: dataframe) into one DataFrame.
    """
    for session, df in df_dict.items():
        df["session"] = session
    return pd.concat(df_dict.values(), ignore_index=True)
