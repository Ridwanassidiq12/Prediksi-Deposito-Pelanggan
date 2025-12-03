import pandas as pd

def feature_engineering(data_ml):
    REQUIRED_COLUMNS = ['housing', 'loan', 'emp.var.rate', 'cons.conf.idx', 
                        'previous', 'campaign', 'age', 'month', 'pdays']

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in data_ml.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut hilang dari input data: {missing_cols}")

    # Pastikan tidak ada kolom 'level_0' dan 'index' di dataframe
    data_ml = data_ml.copy()
    data_ml = data_ml.drop(columns=[col for col in ['index', 'level_0'] if col in data_ml.columns]) # Hapus kolom index yang mungkin tertinggal dalam feature_engineering()
    # Hapus kolom 'level_0' atau 'index' jika ada
    data_ml = data_ml.loc[:, ~data_ml.columns.isin(['level_0', 'index'])]

    # Feature engineering
    data_ml['loan_burden'] = ((data_ml['housing'] == 'yes') & 
                                   (data_ml['loan'] == 'yes')).astype(int)
    
    data_ml['economic_stability'] = data_ml['emp.var.rate'] + data_ml['cons.conf.idx']
    data_ml['contact_count'] = data_ml['previous'] + data_ml['campaign']

    bins = [16, 25, 35, 50, 98]
    labels = ["Muda", "Dewasa Muda", "Dewasa", "Lansia"]
    data_ml['age_group'] = pd.cut(data_ml['age'], bins=bins, labels=labels, include_lowest=True).astype(str)

    month_to_season = {
        "mar": "spring", "apr": "spring", "may": "spring",
        "jun": "summer", "jul": "summer", "aug": "summer",
        "sep": "fall", "oct": "fall", "nov": "fall",
        "dec": "winter", "jan": "winter", "feb": "winter"
    }
    data_ml['season'] = data_ml['month'].str.lower().map(month_to_season)

    data_ml['pdays_group'] = data_ml['pdays'].apply(lambda x: 'No' if x == 999 else 'Yes')

    return data_ml
