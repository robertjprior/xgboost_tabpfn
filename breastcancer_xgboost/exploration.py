import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport


df = pd.read_excel('data/Breast Cancer Prediction_Datasets_training.xlsx')
profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file("reports/train_description_report.html")
df.head()
df.describe()