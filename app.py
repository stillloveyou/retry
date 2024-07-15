import streamlit as st
import openai
import os
import pytesseract
import cv2
import numpy as np
from models.appliance_classifier import classify_appliance
from dotenv import load_dotenv
from PIL import Image
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples


##llm
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever

def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    
    return dictionary_chain


def get_rag_chain():
    llm = get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    system_prompt = (
        "당신은 전기세 절약 전문가입니다. 사용자가 업로드한 이미지를 활용해 목표전기세를 맞추기 위해 도와주세요."
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    
    return conversational_rag_chain


def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    tax_chain = {"input": dictionary_chain} | rag_chain
    ai_response = tax_chain.stream(
        {
            "question": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )
    return ai_response

## app.py

st.set_page_config(layout="wide", page_title="전기세 절약", page_icon="⚡")

st.title("⚡ 전기세 절약 웹")
st.caption("사진을 업로드하고 전기세 절약 플랜을 받으세요!")

load_dotenv()
openai.api_key = os.getenv('')

# 가전 제품 종류 선택
appliance_type = st.selectbox("가전 제품 종류를 선택하세요", ["에어컨", "세탁기", "냉장고", "전자레인지", "오븐", "선풍기"])

# 전력 정보 사진 업로드
uploaded_file = st.file_uploader(f"{appliance_type}의 전력 정보 사진을 업로드하세요", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="업로드한 이미지", use_column_width=True)

    # 이미지 파일 경로 설정
    image_path = os.path.join("uploaded_image." + uploaded_file.name.split(".")[-1])
    img.save(image_path)

    with st.spinner("이미지에서 전력 정보 추출 중..."):
        # 이미지를 numpy 배열로 변환
        img_cv = cv2.imread(image_path)
        # 이미지를 회색조로 변환
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # 이미지를 이진화
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        # 이미지를 확대하여 인식률 증가
        resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # 전처리된 이미지로 텍스트 추출
        text = pytesseract.image_to_string(resized, lang='kor+eng')
        st.write(f"추출된 전력 정보: {text}")

    power_consumption = st.number_input("소비전력 (W)를 입력하세요", min_value=0, step=10)

    if st.button("전기세 절약 플랜 생성"):
        with st.spinner("플랜 생성 중..."):
            prompt = f"다음 {appliance_type}의 소비전력 정보를 바탕으로 한달 전기세 절약 플랜을 작성해 주세요: 소비전력: {power_consumption}W"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant that helps people save electricity."},
                    {"role": "user", "content": prompt}
                ]
            )
            plan = response['choices'][0]['message']['content'].strip()
            st.write("전기세 절약 플랜:")
            st.write(plan)


if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])



if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})

