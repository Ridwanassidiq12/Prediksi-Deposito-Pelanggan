import pandas as pd

def feature_engineering(data_ml_drop):
    data_ml_drop = data_ml_drop.copy()
    data_ml_drop['loan_burden'] = ((data_ml_drop['housing']=='yes') & (data_ml_drop['loan']=='yes')).astype(int)
    data_ml_drop['economic_stability'] = data_ml_drop['emp.var.rate'] + data_ml_drop['cons.conf.idx']
    data_ml_drop['contact_count'] = data_ml_drop['previous'] + data_ml_drop['campaign']

    bins = [16, 25, 35, 50, 98]
    labels = ["Muda","Dewasa Muda","Dewasa","Lansia"]
    data_ml_drop['age_group'] = pd.cut(data_ml_drop['age'], bins=bins, labels=labels, include_lowest=True).astype(str)

    month_to_season = {
        "mar":"spring","apr":"spring","may":"spring",
        "jun":"summer","jul":"summer","aug":"summer",
        "sep":"fall","oct":"fall","nov":"fall",
        "dec":"winter","jan":"winter","feb":"winter"
    }
    data_ml_drop['season'] = data_ml_drop['month'].str.lower().map(month_to_season)

    data_ml_drop['pdays_group'] = data_ml_drop['pdays'].apply(lambda x: 'No' if x==999 else 'Yes')

    return data_ml_drop
