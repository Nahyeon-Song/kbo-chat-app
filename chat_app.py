import boto3
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_aws import ChatBedrock
import asyncio
import time

import glob

st.set_page_config(
    page_title="KBO 리그 규정 문의",
    page_icon="⚾",
    layout="centered"
)

# Bedrock 설정
region = "us-east-1"  # 사용하는 리전으로 변경

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=region,
    # aws_access_key_id=st.secrets["aws_access_key_id"],
    # aws_secret_access_key=st.secrets["aws_secret_access_key"]
)

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

model_kwargs =  { 
    "max_tokens": 1024,
    "temperature": 0.9,
    "top_p": 1.0
}

claude_3_client = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

# 시스템 프롬프트 설정
system_prompt = (
    "당신은 KBO 리그 규정에 대해 답변하는 어시스턴트입니다. "
    "반드시 주어진 컨텍스트 내용만을 기반으로 답변해야 합니다. "
    "사용자가 특정 용어를 KBO 접두사 없이 물어봐도, KBO 리그 관련 내용으로 이해하고 답변하세요. "
    "만약 컨텍스트에 관련 내용이 전혀 없다면 다음과 같이 답변하세요: "
    "'죄송합니다. 해당 내용은 KBO 리그 규정에서 찾을 수 없습니다.'\n\n"
    "컨텍스트에 부분적인 정보만 있는 경우, 규정에 있는 내용만 답변하고 "
    "나머지 부분은 규정에서 찾을 수 없다고 명시해주세요.\n\n"
    "질문에 대한 내용이 여러 군데 있을 경우 모두 찾아서 답변해주세요.\n\n"
    "Context: {context}"
)

@st.cache_resource
def initialize_rag():
    # 문서 로드
    md_files = glob.glob("documents/md/final/*.md")
    
    all_docs = []
    for file_path in md_files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
    
    # 마크다운 헤더 정의
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    
    # 마크다운 헤더 기반 분할기 생성
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # 모든 문서에 대해 헤더 기반 분할 적용
    all_splits = []
    for doc in all_docs:
        # 마크다운 헤더 기반으로 분할
        header_splits = markdown_splitter.split_text(doc.page_content)
        all_splits.extend(header_splits)
    
    # 재귀적 분할 설정
    chunk_size = 750
    chunk_overlap = 150
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # 최종 문서 리스트
    final_docs = []
    
    # 각 헤더 섹션을 더 작게 분할하고 메타데이터 유지
    for doc in all_splits:
        smaller_docs = text_splitter.create_documents(
            texts=[doc.page_content],
            metadatas=[doc.metadata]
        )
        final_docs.extend(smaller_docs)
    
    # 임베딩 및 벡터 스토어 설정
    embeddings = BedrockEmbeddings(region_name=region)
    vectorstore=FAISS.from_documents(final_docs, embeddings)
    retriever = VectorStoreRetriever(
        vectorstore=vectorstore,
        search_type="mmr",  # MMR 검색 사용
        search_kwargs={
            "k": 8,  # 검색할 문서 수
            "fetch_k": 20,  # 초기 검색 문서 수
            "lambda_mult": 0.8  # 다양성과 관련성 사이의 균형 (0: 다양성, 1: 관련성)
        }
    )
    
    # Chain 설정
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(claude_3_client, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return chain, retriever

# 비동기 처리를 위한 함수
async def async_retrieve(retriever, query):
    # 비동기 환경에서 동기 함수 실행
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: retriever.invoke(query))

def get_answer(query, retriever, chain):
    # 동기화를 위한 시간 지연 추가
    time.sleep(0.5)  # 0.5초 지연으로 검색 결과가 완전히 준비되도록 함
    
    # 검색 결과를 명시적으로 완료하고 변수에 저장
    try:
        # 동기 방식으로 검색 실행 (asyncio 사용 불가능한 경우)
        docs = retriever.invoke(query)
        
        # 디버깅을 위한 로깅 추가
        doc_count = len(docs) if docs else 0
        st.session_state.debug_info = f"검색된 문서 수: {doc_count}"
        print(f"EC2/로컬 디버깅 - 검색된 문서 수: {doc_count}, 타임스탬프: {time.time()}")
        
        # 검색 결과가 없는 경우 처리
        if not docs or doc_count == 0:
            return "죄송합니다. 해당 내용은 KBO 리그 규정에서 찾을 수 없습니다."
        
        # 검색된 문서 내용 확인 (디버깅용)
        for i, doc in enumerate(docs):
            print(f"문서 {i+1} 미리보기: {doc.page_content[:100]}...")
        
        # 스트리밍을 위한 빈 컨테이너 생성
        message_placeholder = st.empty()
        full_response = ""
        
        # 응답 생성 시작 표시
        message_placeholder.markdown("검색 완료, 응답 생성 중..." + "▌")
        time.sleep(0.5)  # 응답 생성 전 추가 지연
        
        # 스트리밍 응답 처리
        for chunk in chain.stream({"input": query}):
            if "answer" in chunk:
                full_response += chunk["answer"]
                # 현재까지의 응답을 표시
                message_placeholder.markdown(full_response + "▌")
        
        # 최종 응답 표시 (커서 제거)
        message_placeholder.markdown(full_response)
        return full_response
        
    except Exception as e:
        error_msg = f"오류 발생: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return f"응답 생성 중 오류가 발생했습니다: {str(e)}"

# Streamlit UI
st.title("KBO 리그 규정 문의")

# 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# RAG 초기화
chain, retriever = initialize_rag()

# 이전 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("KBO 리그 규정에 대해 궁금한 점을 물어보세요"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 어시스턴트 응답
    with st.chat_message("assistant"):
        answer = get_answer(prompt, retriever, chain)
        st.session_state.messages.append({"role": "assistant", "content": answer})
