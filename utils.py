"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # モードによってLLMから回答を取得する用のプロンプトを変更
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # LLMへのリクエストとレスポンス取得
    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response

from typing import Optional, Any, Dict

def get_pdf_page_number(meta: Dict[str, Any]) -> Optional[int]:
    """
    ドキュメント metadata から PDF のページ番号を取り出す。
    - "page" や "page_index" は 0 始まりの実装が多いので +1 補正
    - "page_number" は 1 始まりの実装が多いので補正なし
    - "loc" に入ってくる実装もあるため救済
    見つからなければ None を返す
    """
    if not isinstance(meta, dict):
        return None

    # 直接のキー候補
    direct_keys_zero_based = ("page", "page_index")
    direct_keys_one_based = ("page_number",)

    for k in direct_keys_zero_based:
        if k in meta:
            try:
                p = int(meta[k])
                return p + 1 if p >= 0 else None
            except Exception:
                pass

    for k in direct_keys_one_based:
        if k in meta:
            try:
                p = int(meta[k])
                return p if p >= 1 else None
            except Exception:
                pass

    # loc にネストされるパターンの救済
    loc = meta.get("loc")
    if isinstance(loc, dict):
        # よくある候補をチェック
        for k in ("page_number", "page", "pageIndex", "pageNumber"):
            if k in loc:
                try:
                    p = int(loc[k])
                    # loc.page は 0 始まりの実装がある
                    if k.lower() in ("page", "pageindex"):
                        return p + 1 if p >= 0 else None
                    return p if p >= 1 else None
                except Exception:
                    pass

    return None


def format_source_with_page_if_pdf(meta: Dict[str, Any]) -> str:
    """
    表示用の「ソース文字列」を作る。
    PDF かつページが取得できたら「（ページNo.X）」を付与。
    """
    src = meta.get("source") or meta.get("file_path") or meta.get("path") or ""
    if isinstance(src, str) and src.lower().endswith(".pdf"):
        p = get_pdf_page_number(meta)
        if p:
            return f"{src}（ページNo.{p}）"
    return src
