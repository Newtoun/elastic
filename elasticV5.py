from flask import Flask, request, jsonify
from flask_cors import CORS
from elasticsearch import Elasticsearch
import pandas as pd
from elasticsearch.helpers import bulk
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
CORS(app)

es = Elasticsearch("http://localhost:9200")

index_name = "beta_products"

settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1,
        "analysis": {
            "analyzer": {
                "my_custom_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "query": {
                "type": "text",
                "analyzer": "my_custom_analyzer"
            },
            "title": {
                "type": "text",
                "analyzer": "my_custom_analyzer"
            },
            "reviews": {
                "type": "integer"
            },
            "relevancia": {
                "type": "integer"
            },
            "price": {
                "type": "float"
            },
            "boughtInLastMonth": {
                "type": "integer"
            }
        }
    }
}

if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=settings)
    
    df = pd.read_csv("teste.csv")
    df = df.dropna()
    bulk_data = []
    for i, row in df.iterrows():
        doc = row.to_dict()
        bulk_data.append({
            "_index": index_name,
            "_id": i,
            "_source": doc
        })
    bulk(es, bulk_data)
    es.indices.refresh(index=index_name)

df = pd.read_csv("teste.csv")
df = df.dropna()
label_encoder_query = LabelEncoder()
df['query_id'] = label_encoder_query.fit_transform(df['query'])
scaler = StandardScaler()
df[['reviews', 'price']] = scaler.fit_transform(df[['reviews', 'price']])

model = lgb.Booster(model_file='lambdamart_modelV2.txt')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')

    must_matches = [{"match": {"title":query}}]
    
    es_query = {
        "bool": {
            "should": must_matches
        }
    }
    
    print("Elasticsearch Query:", es_query)
    
    resp = es.search(
        index=index_name,
        query=es_query,
        size=100
    )

    es_results = [hit['_source'] for hit in resp['hits']['hits']]
    print(es_results)
  
    if not es_results:
        print('vazio')
        return jsonify([])

    es_df = pd.DataFrame(es_results)
    es_df['query'] = query  

    es_df['query_id'] = label_encoder_query.transform([query])[0]

    es_df[['reviews', 'price']] = scaler.transform(es_df[['reviews', 'price']])

    X_es = es_df[['query_id', 'reviews', 'boughtInLastMonth']]

    es_df['predicted_relevance'] = model.predict(X_es)

    es_df = es_df.sort_values(by='predicted_relevance', ascending=False)

    results = es_df.head(10).to_dict(orient='records')

    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
