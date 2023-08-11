from langchain.vectorstores import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
import os

HF_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"
ELASTIC_URL = "http://localhost:9200/"
ELASTIC_INDEX = "genshinpedia_small"
DB_FILES = "en_datasets"
ADD_FILES = False

def load_sentences_hf_model():
    print("Preparing Huggingface embedding setup...")
    return HuggingFaceEmbeddings(model_name=HF_SENTENCE_TRANSFORMER_MODEL)


def add_files_to_es():
    hf_embdd = load_sentences_hf_model()
    elastic_db = ElasticVectorSearch(embedding=hf_embdd, elasticsearch_url=ELASTIC_URL, index_name=ELASTIC_INDEX)
    print("Connected to ES Local DB")
    print("Start adding files...")
    for file in os.listdir(DB_FILES):
        file_path = DB_FILES + "/" + file
        if not os.path.isfile(file_path):
            continue
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter("\n", chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        #list_of_docs.append(docs)
        elastic_db.add_documents(docs)
        print("Finish adding {}".format(file))
    print("Finish mapping files")
        
# TODO Mover a UnitTests
def ask_question(query):
    hf_embdd = load_sentences_hf_model()
    elastic_db = ElasticVectorSearch(embedding=hf_embdd, elasticsearch_url=ELASTIC_URL, index_name=ELASTIC_INDEX)
    return elastic_db.similarity_search(query)
    
        
if __name__ == "__main__":
    # Configuracion necesaria para ejecutar Elastic Search des
    os.system("sudo sysctl -w vm.max_map_count=262144")
    
    # Ejecutamos el flujo principal
    if ADD_FILES:
        add_files_to_es()
    
    # Ejecutamos una busqueda de prueba
    print(ask_question("What disease does Collei have?"))
    
    