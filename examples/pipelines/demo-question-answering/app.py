import logging
import sys
import click
import pathway as pw
import yaml
from dotenv import load_dotenv
from pathway.udfs import DiskCache
from pathway.xpacks.llm.question_answering import BaseRAGQuestionAnswerer
from pathway.stdlib.indexing import BruteForceKnnFactory
from pathway.xpacks.llm import embedders, llms, parsers, splitters
from pathway.xpacks.llm.document_store import DocumentStore

# Set your Pathway license key here to use advanced features.
pw.set_license_key("demo-license-key-with-telemetry")

# Set up basic logging to capture key events and errors.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load environment variables (e.g., API keys) from the .env file.
load_dotenv()

# Function to handle data sources
def data_sources(source_configs) -> list[pw.Table]:
    sources = []
    for source_config in source_configs:
        if source_config["kind"] == "local":
            source = pw.io.fs.read(
                **source_config["config"],
                format="binary",
                with_metadata=True,
            )
            sources.append(source)
    return sources

# Command-line interface (CLI) function to run the app with a specified config file.
@click.command()
@click.option("--config_file", default="config.yaml", help="Config file to be used.")
def run(config_file: str = "config.yaml"):
    # Load the configuration from the YAML file.
    with open(config_file) as config_f:
        configuration = yaml.safe_load(config_f)

    llm = llms.LiteLLMChat(model="gemini/gemini-pro", cache_strategy=DiskCache())


    parser = parsers.UnstructuredParser()

    text_splitter = splitters.TokenCountSplitter(max_tokens=400)

    embedding_model = "avsolatorio/GIST-small-Embedding-v0"

    embedder = embedders.SentenceTransformerEmbedder(
        embedding_model,
        call_kwargs={"show_progress_bar": False}
    )


    index = BruteForceKnnFactory(embedder=embedder)

    # Host and port configuration for running the server.
    host_config = configuration["host_config"]
    host, port = host_config["host"], host_config["port"]

    # Initialize the vector store for storing document embeddings in memory.
    # This vector store updates the index dynamically whenever the data source changes
    # and can scale to handle over a million documents.
    doc_store = DocumentStore(
            *data_sources(configuration["sources"]),
            splitter=text_splitter,
            parser=parser,
            retriever_factory=index,
        )

    # Create a RAG (Retrieve and Generate) question-answering application.
    rag_app = BaseRAGQuestionAnswerer(llm=llm, indexer=doc_store)

    # Build the server to handle requests at the specified host and port.
    rag_app.build_server(host=host, port=port)

    # Run the server with caching enabled, and handle errors without shutting down.
    rag_app.run_server(with_cache=True, terminate_on_error=False)

# Entry point to execute the app if the script is run directly.
if __name__ == "__main__":
    run()
