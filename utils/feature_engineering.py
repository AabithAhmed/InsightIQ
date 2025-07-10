from sklearn.preprocessing import LabelEncoder

def apply_feature_engineering(df):
    label_encoders = {}

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    def tenure_group(tenure):
        if tenure <= 12:
            return '0-1 year'
        elif tenure <= 24:
            return '1-2 years'
        elif tenure <= 48:
            return '2-4 years'
        elif tenure <= 60:
            return '4-5 years'
        else:
            return '5+ years'

    df['tenure_group'] = df['tenure'].apply(tenure_group)
    le = LabelEncoder()
    df['tenure_group'] = le.fit_transform(df['tenure_group'])
    label_encoders['tenure_group'] = le

    return df, label_encoders
