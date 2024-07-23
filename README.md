# Elastic-basic
Primeiro é preciso rodar o docker do elastic sem a necessidade de credenciais
```bash
docker run --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.7.0
```
Na pasta server instale as dependências
```bash
python3 -m pip install pandas==1.4.3 notebook==6.3.0 elasticsearch==8.7.0
```

Ainda na pasta server copie o arquivo amazon_products.csv


Rode o arquivo elastic.py
```bash
python3 elastic.py
```

Teste uma consulta no servidor. No seu navegador coloque essa rota
```bash
http://localhost:5000/search?q=smart%20TV
```
