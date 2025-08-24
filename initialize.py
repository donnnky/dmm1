"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import sys
import logging
import unicodedata
from uuid import uuid4
from logging.handlers import TimedRotatingFileHandler

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


import constants as ct


############################################################
# 設定関連
############################################################
# .env を読み込む（ローカル用）
try:
    load_dotenv()
except Exception:
    pass

# Cloud では Secrets を優先
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if "USER_AGENT" in st.secrets:
    os.environ["USER_AGENT"] = st.secrets["USER_AGENT"]
else:
    os.environ.setdefault(
        "USER_AGENT",
        "Mozilla/5.0 (compatible; StreamlitBot/1.0; +https://streamlit.io)"
    )


############################################################
# 関数定義
############################################################
def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    initialize_session_state()
    initialize_session_id()
    initialize_logger()
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)

    logger = logging.getLogger(ct.LOGGER_NAME)
    if logger.hasHandlers():
        return

    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8",
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, "
        f"session_id={getattr(st.session_state, 'session_id', '-')}: %(message)s"
    )
    log_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    logger = logging.getLogger(ct.LOGGER_NAME)
    if "retriever" in st.session_state:
        return

    try:
        # 0) APIキー前提チェック
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")

        # 1) データ読み込み
        docs_all = load_data_sources()

        # 2) 正規化（Windows 対策）
        for doc in docs_all:
            doc.page_content = adjust_string(doc.page_content)
            for k in list(doc.metadata.keys()):
                doc.metadata[k] = adjust_string(doc.metadata[k])

        # 3) チャンク
        splitter = CharacterTextSplitter(
            chunk_size=getattr(ct, "CHUNK_SIZE", 500),
            chunk_overlap=getattr(ct, "CHUNK_OVERLAP", 50),
            separator="\n",
        )
        chunks = splitter.split_documents(docs_all)

        if not chunks:
            raise RuntimeError(
                "No documents were loaded. Check RAG_TOP_FOLDER_PATH and WEB_URL_LOAD_TARGETS."
            )

        # 4) 埋め込み & ベクターストア
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(chunks, embedding=embeddings)

        # 5) Retriever
        top_k = getattr(ct, "RETRIEVER_TOP_K", getattr(ct, "RAG_TOP_K", 3))
        st.session_state.retriever = db.as_retriever(search_kwargs={"k": top_k})

        logger.info("Retriever initialized. docs=%d, chunks=%d, k=%d",
                    len(docs_all), len(chunks), top_k)

    except Exception:
        logger.exception("initialize_retriever failed")
        raise


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み
    Returns: list[Document]
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # 1) ローカルファイル
    docs_all = []
    top = getattr(ct, "RAG_TOP_FOLDER_PATH", None)
    if top and os.path.exists(top):
        recursive_file_check(top, docs_all)
    else:
        logger.warning(f"RAG_TOP_FOLDER_PATH not found: {top}. Skipping local files.")

    # 2) Web ページ
    web_docs_all = []
    urls = getattr(ct, "WEB_URL_LOAD_TARGETS", [])
    if not isinstance(urls, (list, tuple)):
        urls = []

    # User-Agent（secrets > env > 既定値）
    ua = os.getenv(
        "USER_AGENT",
        "Mozilla/5.0 (compatible; StreamlitBot/1.0; +https://streamlit.io)"
    )

    for web_url in urls:
        try:
            # langchain-community のバージョン差異に合わせて生成を試行
            try:
                loader = WebBaseLoader(web_paths=[web_url], header_template={"User-Agent": ua})
            except TypeError:
                loader = WebBaseLoader(web_url, header_template={"User-Agent": ua})

            web_docs = loader.load()
            web_docs_all.extend(web_docs)
        except Exception as e:
            logger.warning(f"Web load failed: {web_url} ({e})")

    docs_all.extend(web_docs_all)
    return docs_all


def recursive_file_check(path, docs_all):
    """
    フォルダを再帰的にたどってファイルを読み込む
    """
    if os.path.isdir(path):
        for name in os.listdir(path):
            full = os.path.join(path, name)
            recursive_file_check(full, docs_all)
    else:
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み
    """
    ext = os.path.splitext(path)[1]
    if ext in ct.SUPPORTED_EXTENSIONS:
        loader = ct.SUPPORTED_EXTENSIONS[ext](path)
        docs = loader.load()
        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    """
    if not isinstance(s, str):
        return s

    if sys.platform.startswith("win"):
        s = unicodedata.normalize("NFC", s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s

    return s