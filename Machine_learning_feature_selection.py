import pandas as pd
import numpy as np
from sklearn.feature_selection import  SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder


df = pd.DataFrame({"feature1" : ["a","b","c","a","b"],
                   "feature2" : ["x","y","x","x","y"],
                   "feature3" : ["p","q","p","q","p"],
                   "target":[0,1,0,1,0]})

l_encoder = LabelEncoder()

"""Check screen shot"""
df_new = df.apply(l_encoder.fit_transform)
print(df_new)

x = df_new.drop("target",axis=1)
y = df_new['target']

chi_square_feature_selection = SelectKBest(chi2, k=2)
best_feature = chi_square_feature_selection.fit_transform(x,y)

selected_features = chi_square_feature_selection.get_support(indices=True)
print(selected_features)

final_df = pd.DataFrame(best_feature,columns=x.columns[chi_square_feature_selection.get_support(indices=True)])
print(final_df)