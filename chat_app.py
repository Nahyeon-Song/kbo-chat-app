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
import os

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
    # 문서를 명시적인 순서로 로드
    md_files = [
        "documents/md/final/2025_리그규정_cleaned_header.md",
        "documents/md/final/crop_2025_야구규칙.md"
    ]
    
    print(f"로드할 마크다운 파일 수: {len(md_files)}")
    
    # 파일 존재 여부 확인
    for file_path in md_files:
        if not os.path.exists(file_path):
            print(f"경고: 파일이 존재하지 않습니다 - {file_path}")
    
    all_docs = []
    for file_path in md_files:
        try:
            print(f"파일 로드 중: {file_path}")
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            all_docs.extend(docs)
            print(f"로드된 파일: {file_path}, 문서 수: {len(docs)}")
        except Exception as e:
            print(f"파일 로드 오류 {file_path}: {e}")
            st.error(f"Error loading {file_path}: {e}")
    
    print(f"총 로드된 문서 수: {len(all_docs)}")
    
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
    
    print(f"헤더 기반 분할 후 문서 수: {len(all_splits)}")
    
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

    print(f"\n최종 청크 수: {len(final_docs)}")
    print(f"평균 문서 크기: {sum(len(doc.page_content) for doc in final_docs) / len(final_docs):.0f} 문자")
    
    # 임베딩 및 벡터 스토어 설정
    print("\n임베딩 시작...")
    embeddings = BedrockEmbeddings(region_name=region, model_id="amazon.titan-embed-text-v1")
    
    # 임베딩 테스트
    test_text = "KBO 리그 규정"
    test_embedding = embeddings.embed_query(test_text)
    print(f"테스트 임베딩 차원: {len(test_embedding)}")
    print(f"임베딩 샘플: {test_embedding[:5]}...")
    
    print("\n벡터 스토어 생성 중...")
    # vectorstore = FAISS.from_documents(final_docs, embeddings)
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print(f"벡터 스토어 생성 완료, 인덱스 크기: {vectorstore.index.ntotal}")
    
    # 검색 테스트
    test_query = "외국인 선수 등록"
    print(f"\n테스트 쿼리로 검색: '{test_query}'")
    test_results = vectorstore.similarity_search(test_query, k=2)
    print(f"검색 결과 수: {len(test_results)}")
    for i, doc in enumerate(test_results):
        print(f"결과 {i+1} 미리보기: {doc.page_content[:100]}...")
    
    retriever = VectorStoreRetriever(
        vectorstore=vectorstore,
        search_type="mmr",  # MMR 검색 사용
        search_kwargs={
            "k": 8,  # 검색할 문서 수
            "fetch_k": 20,  # 초기 검색 문서 수
            "lambda_mult": 0.8  # 다양성과 관련성 사이의 균형 (0: 다양성, 1: 관련성)
        }
    )
    
    # 리트리버 테스트
    print(f"\n리트리버로 테스트 쿼리 검색: '{test_query}'")
    retriever_results = retriever.invoke(test_query)
    print(f"리트리버 결과 수: {len(retriever_results)}")
    for i, doc in enumerate(retriever_results):
        print(f"리트리버 결과 {i+1} 미리보기: {doc.page_content[:100]}...")
    
    # Chain 설정
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(claude_3_client, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return chain, retriever

def get_answer(query, retriever, chain):
    try:
        # 1. 명시적인 검색 단계
        print(f"검색 시작 - 타임스탬프: {time.time()}, 쿼리: '{query}'")
        
        # 검색 결과를 변수에 확실히 저장
        search_results = retriever.invoke(query)
        
        # 검색 결과 확인 및 로깅
        doc_count = len(search_results) if search_results else 0
        print(f"검색 완료 - 문서 수: {doc_count}, 타임스탬프: {time.time()}")
        
        # 검색된 문서 내용 상세 출력
        for i, doc in enumerate(search_results):
            print(f"문서 {i+1}:")
            print(f"  내용 미리보기: {doc.page_content[:150]}...")
            print(f"  메타데이터: {doc.metadata}")
            if hasattr(doc, 'score'):
                print(f"  유사도 점수: {doc.score}")
        
        # 검색 결과가 없는 경우
        if not search_results or doc_count == 0:
            return "죄송합니다. 해당 내용은 KBO 리그 규정에서 찾을 수 없습니다."
        
        # 2. UI 업데이트
        message_placeholder = st.empty()
        message_placeholder.markdown("검색 완료, 응답 생성 중..." + "▌")
        
        # 3. 검색 결과를 명시적으로 전달하는 새로운 체인 생성
        # 기존 체인을 재사용하지 않고 새로운 체인 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        print(query)
        print(prompt)
        
        # 문서 체인 생성 (검색 결과를 직접 전달)
        doc_chain = create_stuff_documents_chain(claude_3_client, prompt)
        
        # 4. 응답 생성 (검색 결과를 직접 전달)
        print(f"응답 생성 시작 - 타임스탬프: {time.time()}")
        
        # 검색 결과를 명시적으로 컨텍스트에 포함
        full_response = ""
        
        # 스트리밍 응답 처리
        for chunk in doc_chain.stream({"input": query, "context": search_results}):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        print(f"응답 생성 완료 - 타임스탬프: {time.time()}")
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
