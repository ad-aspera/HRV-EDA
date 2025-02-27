import pandas as pd

def make_decile_column(data: pd.DataFrame, column: str):
    """For each id, splits the data into deciles based on the column
    decile column name: {column}_decile"""
    data[f"{column}_decile"] = data.groupby("id")[column].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False)
    )
    return data