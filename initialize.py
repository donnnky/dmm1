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
from langchain_community.vectorstores import Chroma

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
    """
    RAG の Retriever を作成
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    if "retriever" in st.session_state:
        return

    try:
        # 1) データ読み込み
        docs_all = load_data_sources()

        # 2) Windows 対策（文字正規化）
        for doc in docs_all:
            doc.page_content = adjust_string(doc.page_content)
            for key in list(doc.metadata.keys()):
                doc.metadata[key] = adjust_string(doc.metadata[key])

        # 3) 埋め込みモデル
        embeddings = OpenAIEmbeddings()

        # 4) チャンク分割
        splitter = CharacterTextSplitter(
            chunk_size=ct.CHUNK_SIZE,
            chunk_overlap=ct.CHUNK_OVERLAP,
            separator="\n",
        )
        chunks = splitter.split_documents(docs_all)

        # 5) ベクターストア
        db = Chroma.from_documents(chunks, embedding=embeddings)

        # 6) Retriever
        st.session_state.retriever = db.as_retriever(
            search_kwargs={"k": ct.RETRIEVER_TOP_K}
        )

        logger.info(
            "Retriever initialized. docs=%d, chunks=%d, k=%d",
            len(docs_all), len(chunks), ct.RETRIEVER_TOP_K
        )

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

    Returns:
        読み込んだ通常データソース
    """
    # データソース（ファイル由来）
    docs_all = []
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    # Webページの読み込み
    web_docs_all = []
    header = {
        "User-Agent": os.getenv(
            "USER_AGENT",
            "Mozilla/5.0 (compatible; StreamlitBot/1.0; +https://streamlit.io)"
        )
    }

    for web_url in ct.WEB_URL_LOAD_TARGETS:
        try:
            # langchain の引数差異に対応
            try:
                loader = WebBaseLoader(web_paths=[web_url], header_template=header)
            except TypeError:
                loader = WebBaseLoader(web_url, header_template=header)

            web_docs = loader.load()
            web_docs_all.extend(web_docs)

        except Exception as e:
            logging.getLogger(ct.LOGGER_NAME).warning(
                f"Web load failed: {web_url} ({e})"
            )

    # ファイル由来 + Web由来 を結合
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