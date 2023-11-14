import glob
import json
import os
import sys

from SPARQLWrapper import JSON, SPARQLWrapper

input_file = sys.argv[1]
output_file = sys.argv[2]

sparql = SPARQLWrapper("https://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)

query_prefix = """
    SELECT *
    WHERE {
        ?subject <http://xmlns.com/foaf/0.1/isPrimaryTopicOf>"""
query_suffix = """.
        ?subject ?relation ?object
    }
"""
# LIMIT 3
output = []
entity = os.path.basename(input_file).split(".")[0]
print(entity)
dbpedia_uri = "<http://en.wikipedia.org/wiki/" + entity + ">"
sparql_query = query_prefix + " " + dbpedia_uri + query_suffix
sparql.setQuery(sparql_query)
try:
    ret = sparql.queryAndConvert()
    output.extend(ret["results"]["bindings"])
except Exception as e:
    print(e)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, sort_keys=True)
