import os

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CONTACT = "priem-fcs@hse.ru"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "data", "index")
CORPUS_PATH = os.path.join(BASE_DIR, "data", "text.txt")

# In this repository the dataset is in the project root
QA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "fcs_hse_qa_dataset.json"))

# Models
EMBEDDER = "intfloat/multilingual-e5-base"
RERANKER = "BAAI/bge-reranker-v2-m3"
GENERATOR = "Qwen/Qwen2.5-1.5B-Instruct"

# Fast launch mode
# Set USE_LLM=False to avoid downloading/running Qwen.
# Set USE_RERANKER=False to avoid downloading/running the reranker.
USE_LLM = False
USE_RERANKER = False

# Retrieval settings
TOP_N = 20
TOP_K = 5
REL_THRESHOLD = 0.25
MAX_NEW_TOKENS = 256
