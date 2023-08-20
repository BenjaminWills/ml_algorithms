from sklearn.preprocessing import LabelEncoder


def encode_string_columns(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":  # Check if the column has string values
            df[col] = le.fit_transform(
                df[col]
            )  # Encode unique string values with numbers
    return df
