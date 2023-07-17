import os 
import gradio as gr
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import pandas as pd

os.environ["OPENAI_API_KEY"] = "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" # Définir la clé API OpenAI


index = None # Index est une variable globale qui contient l'index des documents chargés


################## Fonctions d'ingestion ##################

def load_pdf(tempfile):  # input_file
    global index, index_creator 
    print(tempfile.name) # affiche le nom du fichier
    loader = PyPDFLoader(tempfile.name)  # Charger le fichier PDF
    if index is None:  # Si l'index n'est pas chargé
        index_creator = VectorstoreIndexCreator( # Créer un index
            vectorstore_cls=Chroma, # Utiliser Chroma comme vectorstore
            embedding=OpenAIEmbeddings(), # Utiliser OpenAI comme embedding
            text_splitter=CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100) # Utiliser TikToken comme text_splitter. Chunk_size est le nombre de caractères par chunk (morceau de texte), chunk_overlap est le nombre de caractères en commun entre deux chunks.
        )
        index = index_creator.from_loaders([loader]) # Créer l'index à partir du loader
    else:
        index.vectorstore.add_documents(loader.load()) # Ajouter les documents du loader à l'index
    resp = index.vectorstore._collection.get(include=["documents"]) # Récupérer les documents de l'index
    print(resp)
    return f' there are {len(resp["ids"])} pages in the index'  # Afficher le nombre de pages dans l'index



################## Fonctions de traitement ##################

### Fonction de recherche. Prend en entrée la question et renvoie la réponse et les sources.
def query_with_source(question): 
    print("------------------Query-----------------")
    global index
    if index is None:
        return "index not loaded","index not loaded"
    res = index.vectorstore.similarity_search(question) # Recherche de similarité
    return format_source(res)

### Fonction de question/réponse. Prend en entrée la question et renvoie la réponse et les sources.
def llm_question(question, chain_type_): 
    print("------------------QUESTION------------------")
    global index
    if index is None:
        return "index not loaded", "index not loaded"
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type=chain_type_, retriever=index.vectorstore.as_retriever(), return_source_documents=True) # Créer un modèle de question/réponse avec le module OpenAI
    res = qa({"query": question})
    print(res)
    return res['result'], format_source(res['source_documents'])

### Fonction de résumé. Prend en entrée le chain_type et renvoie le résumé. chain_type_ est le type de chaîne de résumé à utiliser.
def llm_summarize( chain_type_): 
    print("------------------SUMMARIZE------------------")
    global index
    if index is None:
        return "index not loaded", "index not loaded"
    lchain = load_summarize_chain(llm=OpenAI(), chain_type=chain_type_) # Créer une chaîne de résumé avec le module OpenAI
    # qa = RetrievalQA.from_chain(lchain, retriever=index.vectorstore.as_retriever(), return_source_documents=True)
    val = lchain.run(index.vectorstore) # Exécuter la chaîne de résumé
    print(val)
    # for i in val:
    #     print(i)
    return val

### Fonction pour formater les sources. Prend en entrée les sources et renvoie un DataFrame.
def format_source(sources): 
    columns = ["source", "page", "contenu"]
    liste = []
    for i in sources:
        source = i.metadata["source"]
        # only keep last element ion path
        source = source.split("/")[-1]
        temp = [source, i.metadata['page'], i.page_content] # i.page_content est le contenu de la page
        liste.append(temp)
    df = pd.DataFrame(liste, columns=columns) 
    print(df)
    return gr.update(value=df) # Mettre à jour le DataFrame de sortie



################## Interface graphique ##################

with gr.Blocks() as demo: # Créer une interface graphique
    gr.Markdown("#Assistant rédaction de DTJ")

### Créer un onglet "ingestion"
    with gr.Tab("ingestion"): 
        gr.Markdown("## ajout de document") 
        # input_file = gr.File( type="file")
        # output_text = gr.Textbox("texte", lines=10, readonly=True)
        # input_file.change(fn = load_pdf, output = output_text)
        app = gr.Interface(fn=load_pdf, inputs="file", outputs="textbox")

### Créer un onglet "recherche"
    with gr.Tab("recherche"): 
        gr.Markdown("## Recherche")
        input_text = gr.Textbox("query", lines=10, interactive=True)
        submit_button = gr.Button("submit", variant="primary")
        answer_output = gr.DataFrame(columns=["source", "page", "contenu"])
        submit_button.click(fn=query_with_source, inputs=input_text, outputs=answer_output)

    # with gr.Tab("Summarize"):
    #     gr.Markdown("## Summarize")
    #     chain_type_selector = gr.Radio(["stuff", "map_reduce","map_rerank", "refine"], value="map_reduce", label="chain type")
    #     submit_button = gr.Button("submit", variant="primary")
    #     answer_output = gr.Textbox("summary text", label="summary")
    #     submit_button.click(fn=llm_summarize, inputs=chain_type_selector, outputs=answer_output)

### Créer un onglet "question"
    with gr.Tab("Question"): 
        gr.Markdown("## Query the document with a question")
        input_text = gr.Textbox("what is the document about ? answer in French.", lines=10, interactive=True)
        chain_type_selector = gr.Radio(["stuff", "map_reduce","map_rerank", "refine"], value="map_reduce", label="chain type")
        submit_button = gr.Button("submit", variant="primary")
        answer_output = gr.Textbox("answer")
        source_output = gr.DataFrame(columns=["source", "page", "contenu"])
        submit_button.click(fn=llm_question, inputs=[input_text, chain_type_selector], outputs=[answer_output, source_output])


demo.launch(share=True)