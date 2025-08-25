"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
import utils
import constants as ct

def _label_with_page_if_pdf(path, page):
    """
    表示ラベルを作る。PDF かつページ番号があれば「（ページNo.X）」を付ける。
    多くのローダーで page は 0 始まりなので +1 して人間向け表記にする。
    """
    try:
        if isinstance(path, str) and path.lower().endswith(".pdf") and page is not None:
            p = int(page)
            if p >= 0:
                return f"{path}（ページNo.{p + 1}）"
    except Exception:
        pass
    return path

############################################################
# 関数定義
############################################################

def display_app_title():
    """
    タイトル表示
    """
    st.markdown(f"## {ct.APP_NAME}")


def display_select_mode():
    """
    回答モードのラジオボタンを表示which python
    """
    # 回答モードを選択する用のラジオボタンを表示
    col1, col2 = st.columns([100, 1])
    with col1:
        # 「label_visibility="collapsed"」とすることで、ラジオボタンを非表示にする
        st.session_state.mode = st.radio(
            label="",
            options=[ct.ANSWER_MODE_1, ct.ANSWER_MODE_2],
            label_visibility="collapsed"
        )


def display_initial_ai_message():
    """
    AIメッセージの初期表示
    """
    with st.chat_message("assistant"):
        # 「st.success()」とすると緑枠で表示される
        st.markdown("こんにちは。私は社内文書の情報をもとに回答する生成AIチャットボットです。上記で利用目的を選択し、画面下部のチャット欄からメッセージを送信してください。")

        # 「社内文書検索」の機能説明
        st.markdown("**【「社内文書検索」を選択した場合】**")
        # 「st.info()」を使うと青枠で表示される
        st.info("入力内容と関連性が高い社内文書のありかを検索できます。")
        # 「st.code()」を使うとコードブロックの装飾で表示される
        # 「wrap_lines=True」で折り返し設定、「language=None」で非装飾とする
        st.code("【入力例】\n社員の育成方針に関するMTGの議事録", wrap_lines=True, language=None)

        # 「社内問い合わせ」の機能説明
        st.markdown("**【「社内問い合わせ」を選択した場合】**")
        st.info("質問・要望に対して、社内文書の情報をもとに回答を得られます。")
        st.code("【入力例】\n人事部に所属している従業員情報を一覧化して", wrap_lines=True, language=None)


def display_conversation_log():
    """
    会話ログの一覧表示
    """
    # 会話ログのループ処理
    for message in st.session_state.messages:
        # 「message」辞書の中の「role」キーには「user」か「assistant」が入っている
        with st.chat_message(message["role"]):

            # ユーザー入力値の場合、そのままテキストを表示するだけ
            if message["role"] == "user":
                st.markdown(message["content"])
            
            # LLMからの回答の場合
            else:
                # 「社内文書検索」の場合、テキストの種類に応じて表示形式を分岐処理
                if message["content"]["mode"] == ct.ANSWER_MODE_1:
                    
                    # ファイルのありかの情報が取得できた場合（通常時）の表示処理
                    if not "no_file_path_flg" in message["content"]:
                        # ==========================================
                        # ユーザー入力値と最も関連性が高いメインドキュメントのありかを表示
                        # ==========================================
                        # 補足文の表示
                        st.markdown(message["content"]["main_message"])

                        # 参照元のありかに応じて、適したアイコンを取得
                        icon = utils.get_source_icon(message['content']['main_file_path'])
                        main_page = message['content'].get('main_page_number')
                        label = _label_with_page_if_pdf(message['content']['main_file_path'], main_page)
                        st.success(label, icon=icon)

                        # ==========================================
                        # ユーザー入力値と関連性が高いサブドキュメントのありかを表示
                        # ==========================================
                        if "sub_message" in message["content"]:
                            # 補足メッセージの表示
                            st.markdown(message["content"]["sub_message"])

                            # サブドキュメントのありかを一覧表示
                            for sub_choice in message["content"]["sub_choices"]:
                                # 参照元のありかに応じて、適したアイコンを取得
                                icon = utils.get_source_icon(sub_choice['source'])
                                label = _label_with_page_if_pdf(sub_choice['source'], sub_choice.get('page_number'))
                                st.info(label, icon=icon)
                    # ファイルのありかの情報が取得できなかった場合、LLMからの回答のみ表示
                    else:
                        st.markdown(message["content"]["answer"])
                
                # 「社内問い合わせ」の場合の表示処理
                else:
                    # LLMからの回答を表示
                    st.markdown(message["content"]["answer"])

                    # 参照元のありかを一覧表示
                    if "file_info_list" in message["content"]:
                        # 区切り線の表示
                        st.divider()
                        # 「情報源」の文字を太字で表示
                        st.markdown(f"##### {message['content']['message']}")
                        # ドキュメントのありかを一覧表示
                        for item in message["content"]["file_info_list"]:
                            if isinstance(item, dict):
                                icon = utils.get_source_icon(item["path"])
                                st.info(item["label"], icon=icon)
                            else:
                                # 互換：古い形式（文字列だけ）でも動くように
                                icon = utils.get_source_icon(item)
                                st.info(item, icon=icon)



def display_search_llm_response(llm_response):
    """
    「社内文書検索」モードにおけるLLMレスポンスを表示
    """
    if llm_response.get("context") and llm_response.get("answer") != ct.NO_DOC_MATCH_ANSWER:
        # メイン文書
        main_doc = llm_response["context"][0]
        main_file_path = main_doc.metadata.get("source")
        main_page_number = main_doc.metadata.get("page")

        main_message = "入力内容に関する情報は、以下のファイルに含まれている可能性があります。"
        st.markdown(main_message)

        icon = utils.get_source_icon(main_file_path)
        label = _label_with_page_if_pdf(main_file_path, main_page_number)
        st.success(label, icon=icon)

        # サブ候補
        sub_choices = []
        seen_paths = set()
        if main_file_path:
            seen_paths.add(main_file_path)

        for document in llm_response["context"][1:]:
            sub_file_path = document.metadata.get("source")
            if not sub_file_path or sub_file_path in seen_paths:
                continue
            seen_paths.add(sub_file_path)

            sub_page_number = document.metadata.get("page")
            if sub_page_number is not None:
                sub_choices.append({"source": sub_file_path, "page_number": sub_page_number})
            else:
                sub_choices.append({"source": sub_file_path})

        if sub_choices:
            sub_message = "その他、ファイルありかの候補を提示します。"
            st.markdown(sub_message)
            for sub in sub_choices:
                icon = utils.get_source_icon(sub["source"])
                label = _label_with_page_if_pdf(sub["source"], sub.get("page_number"))
                st.info(label, icon=icon)

        # 再描画用データ
        content = {
            "mode": ct.ANSWER_MODE_1,
            "main_message": main_message,
            "main_file_path": main_file_path,
        }
        if main_page_number is not None:
            content["main_page_number"] = main_page_number
        if sub_choices:
            content["sub_message"] = sub_message
            content["sub_choices"] = sub_choices

    else:
        st.markdown(ct.NO_DOC_MATCH_MESSAGE)
        content = {
            "mode": ct.ANSWER_MODE_1,
            "answer": ct.NO_DOC_MATCH_MESSAGE,
            "no_file_path_flg": True,
        }

    return content



def display_contact_llm_response(llm_response):
    """
    「社内問い合わせ」モードにおけるLLMレスポンスを表示
    """
    # 本文
    st.markdown(llm_response.get("answer", ""))

    content = {"mode": ct.ANSWER_MODE_2, "answer": llm_response.get("answer", "")}

    # 情報源（該当なしのときは何もしない）
    if llm_response.get("answer") != ct.INQUIRY_NO_MATCH_ANSWER:
        st.divider()
        message = "情報源"
        st.markdown(f"##### {message}")

        seen_paths = set()
        file_info_list = []

        for document in (llm_response.get("context") or []):
            meta = getattr(document, "metadata", {}) or {}
            file_path = meta.get("source") or meta.get("file_path")
            if not file_path or file_path in seen_paths:
                continue

            page_number = meta.get("page")
            label = _label_with_page_if_pdf(file_path, page_number)
            icon = utils.get_source_icon(file_path)
            st.info(label, icon=icon)

            seen_paths.add(file_path)
            file_info_list.append({"path": file_path, "label": label})

        content["message"] = message
        content["file_info_list"] = file_info_list

    return content

