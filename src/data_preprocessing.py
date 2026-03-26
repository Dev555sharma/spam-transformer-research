import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path=r"C:\Users\DELL\spam-transformer-research\data\raw\spam.csv"):
    df1 = pd.read_csv(path, encoding='latin-1')

    # Keep only useful columns
    df1 = df1[['v1', 'v2']]
    df1.columns = ['label', 'text']

    # Convert labels to numeric
    df1['label'] = df1['label'].map({'ham': 0, 'spam': 1})

    return df1

def load_emails_v2(path=r'C:\Users\DELL\spam-transformer-research\data\raw\emails_V2.csv'):
    # 1. Load the dataset
    df2 = pd.read_csv(path)


    # 2. Clean the text (Remove "Subject: " from the first column)
    # Using the index [0] ensures it works even if the name changes
    first_col = df2.columns[0]
    df2[first_col] = df2[first_col].str.replace('Subject: ', '', regex=False)

    # 3. Swap the columns (Col 2 becomes 1, Col 1 becomes 2)
    cols = list(df2.columns)
    cols[0], cols[1] = cols[1], cols[0]
    df2 = df2[cols]

    

    # 4. Rename to match the 'v1', 'v2' structure of your first function
    # Note: Ensure these keys match your actual CSV headers
    df2 = df2.rename(columns={'spam': 'label'})

    #remove null values
    df2 = df2.dropna()

    return df2


def basic_eda(df):
    print("\n===== BASIC DATA ANALYSIS =====")

    # Class distribution
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    print("\nClass Ratio:")
    print(df['label'].value_counts(normalize=True))

    # Text length
    df['text_length'] = df['text'].apply(len)

    print("\nText Length Stats:")
    print(df['text_length'].describe())

    print("\nSample Data:")
    print(df.sample(5))


def clean_data(df):
    # Minimal cleaning (important for BERT)
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.strip()

    return df


def split_data(df, test_size=0.2):
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=42
    )
    return train_df, test_df


def save_data(train_df, test_df):
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)


if __name__ == "__main__":
    df1 = load_data()
    df2 = load_emails_v2()

    # MERGE: use concat to stack them vertically
    # ignore_index=True ensures the new dataset has a fresh 0 to N index
    df = pd.concat([df1, df2], axis=0, ignore_index=True)

    basic_eda(df)

    df = clean_data(df)

    train_df, test_df = split_data(df)

    save_data(train_df, test_df)

    print("\n✅ Data preprocessing complete. Files saved in data/processed/")
