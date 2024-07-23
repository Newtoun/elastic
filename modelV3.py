import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df_lambdamart = pd.read_csv('Beta.csv')
print(df_lambdamart.isnull().sum())


label_encoder_query = LabelEncoder()
df_lambdamart['query_id'] = label_encoder_query.fit_transform(df_lambdamart['query'])
print(df_lambdamart['query_id'])
label_encoder_title = LabelEncoder()

X = df_lambdamart[['query_id', 'reviews', 'boughtInLastMonth']]
y = df_lambdamart['relevancia']

scaler = StandardScaler()
X[['reviews']] = scaler.fit_transform(X[['reviews']])

group = df_lambdamart.groupby('query_id').size().to_frame('size')['size'].tolist()

unique_queries = df_lambdamart['query_id'].unique()
print(unique_queries)
train_queries, test_queries = train_test_split(unique_queries, test_size=0.2)

train_mask = df_lambdamart['query_id'].isin(train_queries)
test_mask = df_lambdamart['query_id'].isin(test_queries)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

group_train = df_lambdamart[train_mask].groupby('query_id').size().to_frame('size')['size'].tolist()
group_test = df_lambdamart[test_mask].groupby('query_id').size().to_frame('size')['size'].tolist()

#Treinar o modelo LambdaMART

train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
test_data = lgb.Dataset(X_test, label=y_test, group=group_test, reference=train_data)

params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'learning_rate': 0.1,
    'num_leaves': 100,
    'min_data_in_leaf': 40,
    'feature_fraction': 0.9,
    'force_col_wise': True
}

model = lgb.train(params, train_data, valid_sets=[train_data, test_data], num_boost_round=100)
#model.save_model('lambdamart_modelV2.txt')
