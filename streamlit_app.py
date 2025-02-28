# -------------------------------
# Imports for the RAG Chatbot
# -------------------------------
from langchain.docstore.document import Document

from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Weaviate
import weaviate
from openai import OpenAI

# -------------------------------
# Credentials and Endpoints
# -------------------------------
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import matplotlib.colors as mcolors  # 색상 처리를 위해 추가
import plotly.express as px
from data_preprocessing import preprocess_data
import random
import altair as alt
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster
import io
import numpy as np
import time
import json
import networkx as nx
from pyvis.network import Network
from streamlit.components.v1 import html

st.set_page_config(
    page_title="개발자 채용 공고 트렌드", 
    layout="wide", 
    page_icon="💻", 
    initial_sidebar_state="collapsed"
)

# 기본 헤더 숨기기 위한 스타일 추가
header_style = """
    <style>
        /* 기본 Streamlit 헤더 숨기기 */
        .css-1v3fvcr {display: none;} /* 헤더 숨기기 (버튼, 로고 등) */
        /* 기본 Streamlit 헤더 숨기기 */
        header {visibility: hidden;}
        /* 기본 메뉴 숨기기 */
        #MainMenu {visibility: hidden;}
        /* 푸터 숨기기 */
        footer {visibility: hidden;}

        .header {
            background-color: white;
            padding: 60px 20px;  /* 위아래 padding을 좁혀서 높이 줄이기 */
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2); /* 위쪽에 그림자 추가 */
            z-index: 999;
        }
        .header .logo {
            display: flex;
            justify-content: flex-start;  /* 왼쪽 정렬 */
            align-items: center;
            position: absolute;
            top: 0px;  /* 위에서 조금 아래로 */
            left: 80px;  /* 왼쪽에서 살짝 오른쪽으로 */
        }
        .header img {
            width: 160px;  /* 이미지 크기 조정 */
        }
        .header .menu {
            display: flex;
            gap: 20px;  /* 메뉴 항목 간 간격 */
            position: absolute;
            top: 42px;  /* 이미지 아래로 위치 */
            left: 240px;  /* 이미지 오른쪽으로 위치 */
            font-size: 20px;
            color: #000000;
        }
        .header .menu a {
            text-decoration: none;
            color: #000000;
            font-weight: bold;
            transition: color 0.3s;
        }
        .header .menu a:hover {
            color: #808080;
        }
        .banner-container {
            position: relative;
            width: 100%;
            text-align: center;
        }
        .banner img {
            width: 100%;
            max-height: 250px;
            object-fit: cover;
        }
        .banner-text {
            position: absolute;
            top: 40%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 40px;
            font-weight: bold;
            white-space: nowrap;
        }
        .banner-subtext {
            position: absolute;
            top: 70%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 18px;
            font-weight: normal;
            white-space: nowrap;
        }
        .multiselect-container {
            margin-top: 10px; /* 배너와의 간격 */
        }
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #7760FC !important;
            color: white !important;
            border-radius: 5px;
            padding: 5px 10px;
        }
        .divider {
            margin-top: 20px;
            border-top: 1px solid rgba(211, 211, 211, 0.5);
            margin-bottom: 20px;
        }

        div[data-baseweb="tab-list"] button {
            color: black !important; /* 기본 텍스트 색상 */
            border-bottom: 2px solid transparent !important; /* 기본 아래 선 숨기기 */
            padding-bottom: 5px !important;
            transition: none !important; /* 애니메이션 효과 제거 */
        }

        /* 선택된 탭 스타일 */
        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #7760FC !important;  /* 선택된 탭의 텍스트 색상 */
            border-bottom: 3px solid #7760FC !important;  /* 선택된 탭 아래 선 색상 */
            margin-bottom: -2px !important; /* 하단에 남는 빨간색 제거 */
            transition: none !important; /* 애니메이션 효과 제거 */
        }

        /* 기본 애니메이션 효과를 제거하여 색상이 느리게 변경되는 문제 해결 */
        div[data-baseweb="tab-highlight"] {
            background-color: #7760FC !important; /* 애니메이션 색상을 탭과 동일하게 변경 */
        }

        body { background-color: white; }
        .title { font-size: 28px; font-weight: bold; margin-bottom: 0px; }
        .subtitle { font-size: 14px; color: grey; margin-top: 2px; }
        .growth-box {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 18px;
            font-weight: bold;
        }
        .button { background-color: #8A5CF6; color: white; font-size: 16px; padding: 10px; border-radius: 10px; text-align: center; display: inline-block; }
        .growth-icon { font-size: 20px; vertical-align: middle; margin-right: 5px; }
        .growth-positive { background-color: #FDECEC; color: #E14C4C; }
        .growth-negative { background-color: #E3F2FD; color: #1976D2; }
        .rounded-box { width: 100%; border-radius: 10px; padding: 10px; background-color: #F8F8F8; margin-bottom: 20px; position: relative; }
        /* 'help' style */
        .help-text {
            position: absolute;
            right: 10px;
            bottom: 10px;
            font-size: 12px;
            color: grey;
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.3s ease-in-out;
        }
        
        .rounded-box:hover .help-text {
            visibility: visible;
            opacity: 1;
        }
    </style>

"""
st.markdown(header_style, unsafe_allow_html=True)



@st.cache_resource
def init_vector_store():
    # 1. Create a Weaviate client
    client = weaviate.Client(
        url=weaviate_url,
        auth_client_secret=weaviate.auth.AuthApiKey(api_key=weaviate_api_key)
    )

    # 2. Initialize the embeddings
    embedding = OpenAIEmbeddings(api_key=openai_api_key)

    # 3. Define the index name and text key
    index_name = "JobDetails"
    text_key = "page_content"

    # 4. Create the vector store
    vector_store = Weaviate(
        client,
        index_name=index_name,
        text_key=text_key,
        embedding=embedding,
        by_text=False
    )

    # 5. (Optional) Load and index documents if you need to do it here
    #    Or you can do it outside once, and not repeat it every time.
    documents = []
    with open("/Users/aria/Downloads/combined_output.json", "r", encoding="utf-8") as f:
        jobs = json.load(f)
        for job in jobs:
            page_content = (
                f"Job Name: {job['Job_name']}\n"
                f"Company: {job['Company_name']}\n"
                f"Region: {job['Region']}\n"
                f"Company Keywords: {job['Company_keyword']}\n"
                f"Job Description Keywords: {job['JD_keyword']}\n"
                f"Requirements Keywords: {job['Requirements_keyword']}\n"
                f"Career Field: {job['career_field']}"
            )
            metadata = {
                "Company_name": job["Company_name"],
                "Region": job["Region"],
                "career_field": job["career_field"],
            }
            documents.append(Document(page_content=page_content, metadata=metadata))

    vector_store.add_documents(documents)
    return vector_store


@st.cache_resource
def get_friendli_client():
    # Create a Friendli client using the OpenAI interface.
    client = OpenAI(
        base_url="https://inference.friendli.ai/v1",
        api_key=friendli_token
    )
    return client



# -------------------------------
# Chat function (streaming)
# -------------------------------
def chat_function(message, history):
    """
    Given a user message and chat history (a list of (user, bot) tuples),
    search for relevant documents, prepare the context, and stream the response.
    This function yields incremental output.
    """
    vector_store = init_vector_store()
    friendli_client = get_friendli_client()

    # Retrieve relevant documents (k=3)
    relevant_docs = vector_store.similarity_search(message, k=3)
    retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Prepare the conversation messages for the LLM.
    messages = [{"role": "system", "content": f"Context from relevant job postings:\n{retrieved_context}"}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    # Request a streaming completion.
    chat_stream = friendli_client.chat.completions.create(
        model="meta-llama-3.1-70b-instruct",
        messages=messages,
        temperature=0.7,
        stream=True
    )

    answer = ""
    for chunk in chat_stream:
        delta = chunk.choices[0].delta
        content = delta.content if delta.content is not None else ""
        answer += content
        yield answer
#####################################################################################
def average_colors(color_list):
    """ 여러 개의 HEX 색상을 받아서 평균 색상을 반환 """
    rgb_values = [[int(c[i:i+2], 16) for i in (1, 3, 5)] for c in color_list]
    avg = [sum(channel) // len(channel) for channel in zip(*rgb_values)]
    return f'#{avg[0]:02X}{avg[1]:02X}{avg[2]:02X}'

def N_average_colors(color_list, weight_list):
    """ 여러 개의 HEX 색상과 그에 대한 가중치를 받아서 색상을 보간하여 반환 """
    rgb_values = [[int(c[i:i+2], 16) for i in (1, 3, 5)] for c in color_list]
    weighted_rgb = [sum(c * w for c, w in zip(channel, weight_list)) // sum(weight_list) for channel in zip(*rgb_values)]
    return f'#{weighted_rgb[0]:02X}{weighted_rgb[1]:02X}{weighted_rgb[2]:02X}'

# 세션 상태에 color_scheme 초기화
if 'color_scheme' not in st.session_state:
    st.session_state.color_scheme = {
        "Back end": "#21DDB8",    # Teal
        "Front end": "#00C7F2",   # Yellow
        "AI": "#695CFB",          # Red-Orange
        "Data": "#FFC246",        # Deep Red
    }

# Big Data/AI 색상 = Data와 AI 색상의 중간색
color_DataAI= average_colors([
    st.session_state.color_scheme["Data"], 
    st.session_state.color_scheme["AI"]
])

# ALL 색상 = 4개 직종의 평균색
color_all = average_colors([
    st.session_state.color_scheme["Back end"],
    st.session_state.color_scheme["Front end"],
    st.session_state.color_scheme["AI"],
    st.session_state.color_scheme["Data"]
])

# 사이드바에 색상 선택기 추가
st.sidebar.title("색상 설정")
for job_category in st.session_state.color_scheme.keys():
    st.session_state.color_scheme[job_category] = st.sidebar.color_picker(
        f"{job_category} 색상",
        st.session_state.color_scheme[job_category]
    )

# Plotly 차트에 사용할 색상 리스트
PLOTLY_COLORS = list(st.session_state.color_scheme.values())

# CSV 파일 읽기 (사용자가 업로드한 파일로 바꿔주세요)
file_path = "data/tech_keywords_combined.csv"
data = pd.read_csv(file_path)

# 예시로 사용자가 선택한 기술 리스트
selected_skills = ["Python", "AWS", "React", "SQL", "Node.js"]

# 직무별 가중치 계산 함수
def calculate_job_fitness(selected_skills, data):
    job_fitness = {'AI': 0, 'Back end': 0, 'Front end': 0, 'Data': 0}
    total_scores = {'AI': 0, 'Back end': 0, 'Front end': 0, 'Data': 0}

    # 모든 기술에 대한 총합 계산 (고정된 기준을 위해)
    for index, row in data.iterrows():
        total_scores['AI'] += row['AI']
        total_scores['Back end'] += row['Back end']
        total_scores['Front end'] += row['Front end']
        total_scores['Data'] += row['Data']

    for skill in selected_skills:
        skill_data = data[data['Keyword'] == skill]
        if not skill_data.empty:
            job_fitness['AI'] += skill_data['AI'].values[0]
            job_fitness['Back end'] += skill_data['Back end'].values[0]
            job_fitness['Front end'] += skill_data['Front end'].values[0]
            job_fitness['Data'] += skill_data['Data'].values[0]

    # 비율 계산 (전체 기술 대비 %)
    for key in job_fitness.keys():
        if total_scores[key] > 0:
            job_fitness[key] = (job_fitness[key] / total_scores[key]) * 100

    return job_fitness


# 로컬 이미지 경로
image_path = "/Users/gyun/Desktop/서울대/시각화/시각화/stream_Job/image/wanted.png"

# 배너 이미지 경로
banner_image_path = "/Users/gyun/Desktop/서울대/시각화/시각화/stream_Job/image/banner.png"

logo_image_path = "/Users/gyun/Desktop/서울대/시각화/시각화/stream_Job/image/logo.png"

# 로컬 이미지를 Base64로 변환하는 함수
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()



# Base64로 변환된 이미지 경로
base64_image = image_to_base64(image_path)
banner_image = image_to_base64(banner_image_path)

# 헤더에 로컬 이미지 추가
st.markdown(f"""
    <div class="header">
        <div class="logo">
            <!-- 이미지 클릭 가능한 링크 (현재 창에서 열림) -->
            <a href="https://www.wanted.co.kr/" target="_self">
                <img src="data:image/png;base64,{base64_image}">
            </a>
        </div>
        <div class="menu">
            <a href="https://www.wanted.co.kr/wdlist" target="_self">≡채용</a>
            <a href="https://www.wanted.co.kr/events" target="_self">커리어</a>
            <a href="https://social.wanted.co.kr/community" target="_self">소셜</a>
            <a href="https://www.wanted.co.kr/cv/list" target="_self">이력서</a>
            <a href="https://www.wanted.co.kr/gigs/experts" target="_self">프리랜서</a>
            <a href="https://www.wanted.co.kr/" target="_self">더보기</a>
        </div>
    </div>
    <div class="banner-container">
        <img class="banner" src="data:image/png;base64,{banner_image}">
        <div class="banner-text">학습 기술을 기반으로<br>직군/기술 추천을 확인해 보세요!</div>
        <div class="banner-subtext">학습 하신 기술을 기반으로 상세 직군을 추천 받고<br>취업에 유리한 기술 추천까지 받아보세요!</div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="multiselect-container">', unsafe_allow_html=True)


tab2, tab1, tab3, tab4 = st.tabs(["대시보드", "직무/기술 추천", "위치별 공고 수", "챗봇"])

with tab1:
    
    # '직무' 텍스트 추가 (완전히 붙이기)
    st.markdown("<h4 style='font-size:40px; font-weight:bold; margin-bottom: 0px;'>기술을 선택하세요.</h4>", unsafe_allow_html=True)

    # 기술 선택 멀티셀렉트
    selected_skills_input = st.multiselect(
        " ",
        options=data['Keyword'].tolist(),
        default=selected_skills,
        help="원티드 2025/02/25까지의 데이터입니다."
    )
    # 회색 선 추가
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


    # 직무 적합도 계산
    job_fitness = calculate_job_fitness(selected_skills_input, data)
    recommended_job = max(job_fitness, key=job_fitness.get).replace('_직종', '')


    # 중앙 정렬을 위한 컬럼 배치
    job1, job2, job3 = st.columns([1, 0.02, 1])  # 중간 간격을 0.02로 축소

    with job1:
        st.markdown(
            """
            <div style='display: flex; flex-direction: column; align-items: flex-end; text-align: right; padding-right: 10px;'>
                <p style='color: gray; font-size: 14px; margin-bottom: 5px;'>기술을 기반으로 한</p>
                <p style='color: black; font-size: 28px; font-weight: bold; margin-top: 0px;'>추천 직무는?</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with job2:
        st.markdown(
            """
            <div style='width: 1px; height: 80px; background-color: #D3D3D3; margin: auto;'></div>
            """, 
            unsafe_allow_html=True
        ) 

    with job3:
        st.markdown(
            f"""
            <div style='display: flex; flex-direction: column; align-items: flex-start; text-align: left; padding-left: 10px;'>
                <p style='color: gray; font-size: 14px; margin-bottom: 5px;'>기술 기반 적합한 직무는?</p>
                <p style='color: black; font-size: 28px; font-weight: bold; margin-top: 0px;'>{recommended_job} 직무를 추천합니다</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 회색 선 추가
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 추천 기술 계산 함수
    def recommend_skills(selected_skills, recommended_job, data):
        remaining_skills = data[~data['Keyword'].isin(selected_skills)]
        sorted_skills = remaining_skills.sort_values(by=[recommended_job], ascending=False)
        top_skills = sorted_skills.head(5)[['Keyword', recommended_job]]
        return top_skills

    # 추천 기술 계산
    recommended_skills = recommend_skills(selected_skills_input, recommended_job, data)


    def plot_job_fitness(job_fitness):
        fixed_range = 100
        categories = ["AI", "Back end", "Front end", "Data"]
        values = list(job_fitness.values()) + [job_fitness["AI"]]  # 마지막에 AI 추가

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[fixed_range, fixed_range, fixed_range, fixed_range, fixed_range],  # 마지막 값 추가
            theta=categories + [categories[0]],  # categories에 첫 번째 항목을 추가하여 연결
            fill=None,
            mode='lines',
            line=dict(color="black", width=2),
            showlegend=False
        ))
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],  # categories에 첫 번째 항목을 추가하여 연결
            fill='toself',
            name='직무 적합도',
            marker=dict(color='rgba(0, 102, 204, 0.8)')
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False),
                angularaxis=dict(showline=False, tickfont=dict(size=14, color="black")),
            ),
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
        return fig


    # 직무 적합도 시각화
    col1, col2 = st.columns([1, 1])  # 직무 적합도 그래프는 왼쪽, 히트맵은 오른쪽

    with col1:
        # 직무 적합도 그래프
        st.plotly_chart(plot_job_fitness(job_fitness), use_container_width=True)

    with col2:
        # 기술 스택 적합도 히트맵
        filtered_data = data[data['Keyword'].isin(selected_skills_input)]

        # Total 컬럼이 있는 경우 제거
        if 'Total' in filtered_data.columns:
            heatmap_data = filtered_data.set_index('Keyword').drop(columns=['Total'])
        else:
            heatmap_data = filtered_data.set_index('Keyword')

        if len(heatmap_data) != 0:
            # 행과 열을 바꾸어 직무가 y축에 오도록 함
            heatmap_data = heatmap_data.transpose()

            # 직무별 기본 색상 설정 (원하는 색상과 유사하게)
            custom_colors = {
                "Back end": st.session_state.color_scheme.get("Back end", "#21DDB8"),  
                "Front end": st.session_state.color_scheme.get("Front end", "#00C7F2"),
                "AI": st.session_state.color_scheme.get("AI", "#695CFB"),
                "Data": st.session_state.color_scheme.get("Data", "#FFC246")
            }

            # 원하는 직무 순서로 재정렬 (데이터에 해당 직무가 모두 존재하는 경우)
            desired_order = ["AI", "Back end", "Front end", "Data"]
            heatmap_data = heatmap_data.reindex(desired_order)

            # 전체 데이터 중 최대값 (정규화를 위해)
            overall_max = heatmap_data.max().max()

            # 직접 셀 단위로 사각형을 그려 각 직무별 기본 색상과 값의 비율에 따른 진하기를 표현
            nrows, ncols = heatmap_data.shape
            fig, ax = plt.subplots(figsize=(5, 3))
            for i, job in enumerate(heatmap_data.index):
                for j, skill in enumerate(heatmap_data.columns):
                    value = heatmap_data.loc[job, skill]
                    norm = value / overall_max if overall_max != 0 else 0  # 0~1 범위로 정규화
                    base_color = custom_colors.get(job, "#FFFFFF")
                    base_rgb = mcolors.to_rgb(base_color)
                    # 흰색과 기본 색상 사이를 선형 보간: norm=0이면 흰색, norm=1이면 기본 색상
                    blended_rgb = tuple((1 - norm) * 1 + norm * base for base in base_rgb)
                    blended_hex = mcolors.to_hex(blended_rgb)
                    # 셀 사각형 그리기 (edgecolor로 회색 테두리 추가)
                    rect = plt.Rectangle((j, i), 1, 1, facecolor=blended_hex, edgecolor='grey')
                    ax.add_patch(rect)
                    # 셀 중앙에 값 표시 (원하는 경우 주석 해제)
                    ax.text(j + 0.5, i + 0.5, f"{value:.0f}", ha="center", va="center", fontsize=9)

            # 축 설정: x축은 기술(Keyword), y축은 직무
            ax.set_xlim(0, ncols)
            ax.set_ylim(0, nrows)
            ax.set_xticks([x + 0.5 for x in range(ncols)])
            ax.set_yticks([x + 0.5 for x in range(nrows)])
            ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right")
            ax.set_yticklabels(heatmap_data.index, rotation=0)
            ax.invert_yaxis()  # 위쪽부터 첫 번째 행이 나오도록
            ax.tick_params(axis='both', labelsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('')
            plt.tight_layout()
            st.pyplot(fig)

    # 중앙 정렬을 위해 추가 CSS 스타일
    # st.markdown("""
    #     <style>
    #         .stColumn {
    #             display: flex;
    #             justify-content: center;
    #             align-items: center;  /* 세로로 중앙 정렬 */
    #             height: 100%;  /* 높이를 100%로 설정하여 세로 정렬이 제대로 이루어지게 함 */
    #         }
    #     </style>
    # """, unsafe_allow_html=True)

    # 회색 선 추가
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 상위 5개 기술 선택 (적합도 기준 정렬)
    top_skills = recommended_skills.nlargest(5, recommended_job)
    top_skills = top_skills.sort_values(by=recommended_job, ascending=True)
    # 텍스트 설명 추가
    st.markdown("""
        <div style="padding: 10px; text-align: center; background-color: #f7f7f7; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="color: #6a1b9a; font-size: 28px;">해당 직무의 추천 기술</h3>
            <p style="color: #333; font-size: 14px;">적합도에 따라 추천된 상위 5개의 기술을 확인하세요. 각 기술의 적합도는 백분율로 표시됩니다.</p>
        </div>
    """, unsafe_allow_html=True)

    # 그래프 크기 설정
    fig, ax = plt.subplots(figsize=(8, 3))  # 적당한 크기로 너비는 8, 높이는 3으로 설정

    # 가로 막대 그래프 그리기
    bars = ax.barh(top_skills['Keyword'], top_skills[recommended_job], color='#8A5CF6', height=0.4)  # 막대 높이 설정

    # 각 막대 상단 왼쪽에 기술명 표시
    for bar, skill in zip(bars, top_skills['Keyword']):
        ax.text(bar.get_x() + 0.3, bar.get_y() + bar.get_height(), skill,  
                va='bottom', ha='left', fontsize=8, fontweight='bold', color='black')


    # 막대 내부 오른쪽에 적합도 % 표시
    for bar in bars:
        ax.text(bar.get_width() - 1, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())}%", va='center', ha='right', fontsize=7, color='white')

    # 불필요한 테두리 및 축 제거
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 눈금선 및 축 숨기기
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # tight_layout()로 내부 요소 정리
    plt.tight_layout(pad=2.0)  # 그래프 내부 요소 간의 간격을 조절

    # 그래프 출력
    st.pyplot(fig, use_container_width=False)

with tab2:
    # 레이아웃 설정
    grid = st.columns([1.5, 2.5])

    # CSV 파일 및 데이터 전처리
    # 데이터 로드
    tech_data = pd.read_csv("data/tech_keywords_combined.csv")
    seniority_data = pd.read_csv('data/경력_범위별_공고_수.csv')
    # 'DB'를 'Big Data/AI'로 변경
    seniority_data['직무'] = seniority_data['직무'].replace('DB', 'Data')

    total_seniority_data = seniority_data.iloc[:, 1:].sum(axis=1)
    file_path = "data/fintech_jobs_final.csv"
    job_monthly_trend, job_growth_rate = preprocess_data(file_path, period=1)

    # 최신 월 성장률 가져오기
    latest_month = job_growth_rate.index[-1]
    growth_rates = job_growth_rate.loc[latest_month]

    with grid[0]:
        st.markdown("""
        <div class='rounded-box'>
            <h3>현재 채용 중</h3>
            <div class='help-text'>출처 : 원티드</div>
        </div>
    """, unsafe_allow_html=True)

        # 전체 채용 건수
        total_jobs = int(total_seniority_data.sum())
        
        # 직무별 총공고수 도넛 차트
        fig_donut = px.pie(
            names=seniority_data['직무'],
            values=total_seniority_data,
            color_discrete_sequence=PLOTLY_COLORS,
            hole=0.75,
        )
        
        fig_donut.update_traces(
            textposition='outside',
            textinfo='none',
            hoverinfo='skip',
            rotation=90,
            pull=[0.01] * len(seniority_data),
            marker=dict(
                line=dict(color='white', width=3)
            )
        )
        
        # 중앙에 전체 건수 텍스트 추가
        fig_donut.add_annotation(
            text=f"{total_jobs:,}건<br><span style='font-size:12px'>전체 채용 기준</span>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=16,
                family="Arial",
            ),
            xanchor='center',
            yanchor='middle'
        )
        
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        

        st.plotly_chart(fig_donut, use_container_width=True)
        
        st.markdown("<div class='rounded-box'><h3>직무별 중요 키워드</h3></div>", unsafe_allow_html=True)
        word_job = st.selectbox("직종 선택", ["Back end", "Front end", "AI", "Data"])
        word_mapping = {"Back end": "/Users/gyun/Desktop/서울대/시각화/시각화/stream_Job/image/image_back.png", 
                        "Front end": "/Users/gyun/Desktop/서울대/시각화/시각화/stream_Job/image/image_Front.png", 
                        "AI": "/Users/gyun/Desktop/서울대/시각화/시각화/stream_Job/image/image_AI.png", 
                        "Data": "/Users/gyun/Desktop/서울대/시각화/시각화/stream_Job/image/image_Data.png"}
        selected_job = word_mapping[word_job]
        st.image(selected_job, caption=word_job)
        
        # 직종 및 기술 컬럼 설정
        job_roles = ["AI", "Back end", "Front end", "Data"]

        # Sankey 차트 데이터 구성
        sources, targets, values = [], [], []
        labels = list(tech_data["Keyword"]) + job_roles
        label_map = {label: idx for idx, label in enumerate(labels)}

        # 랜덤 색상을 생성하는 함수
        def random_color():
            return f'#{random.randint(0, 255):02X}{random.randint(0, 255):02X}{random.randint(0, 255):02X}'
        
        # 각 노드에 대해 색상을 지정
        node_colors = [st.session_state.color_scheme[label] if label in st.session_state.color_scheme else random_color() for label in labels]
    
        for job in job_roles:
            top_techs = tech_data.nlargest(5, job)[["Keyword", job]]
            for _, row in top_techs.iterrows():
                sources.append(label_map[job])
                targets.append(label_map[row["Keyword"]])
                values.append(row[job])
        # Sankey Chart 생성
        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors  # 색상 적용
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
            )
        ))
        st.markdown("<div class='rounded-box'><h3>직무별 주요 기술 연관성</h3></div>", unsafe_allow_html=True)
        st.plotly_chart(fig_sankey, use_container_width=True)


    with grid[1]:
        
        st.markdown("""
        <div class='rounded-box'>
            <h3>채용 규모 성장률</h3>
            <div class='help-text'>출처 : 핀테크 포털</div>
        </div>
    """, unsafe_allow_html=True)

        def draw_chart(title, data, growth_rate, color):
            df = pd.DataFrame({
                "채용기간_월": job_monthly_trend.index,
                "채용공고수": data
            })

            chart = (
                alt.Chart(df)
                .mark_line(
                    color=color,
                    size=2,
                    interpolate='basis'
                )
                .encode(
                    x=alt.X("채용기간_월", title="", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("채용공고수", title="", scale=alt.Scale(domain=[0, df['채용공고수'].max() + 100])),
                    tooltip=["채용기간_월", "채용공고수"]
                )
            )

            growth_class = "growth-positive" if growth_rate > 0 else "growth-negative"
            growth_icon = "📈" if growth_rate > 0 else "📉"

            growth_html = f"""
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div class='title'>{title}</div>
                    <div class='subtitle'>{latest_month} 기준</div>
                </div>
                <div class='growth-box {growth_class}'><span class='growth-icon'>{growth_icon}</span>{growth_rate:.1f}%</div>
            </div>
            """

            st.markdown(growth_html, unsafe_allow_html=True)
            st.altair_chart(chart, use_container_width=True)

        # 성장률 기준 기간 선택
        growth_period = st.selectbox("", ["1개월", "3개월", "6개월", "12개월"])
        period_mapping = {"1개월": 1, "3개월": 3, "6개월": 6, "12개월": 12}
        selected_period = period_mapping[growth_period]

        job_monthly_trend, job_growth_rate = preprocess_data(file_path, period=selected_period)
        growth_rates = job_growth_rate.loc[latest_month]
        
        m_col1, m_col2 = st.columns(2)
        m_col3, m_col4 = st.columns(2)

        with m_col1:
            draw_chart("Backend", job_monthly_trend["Back end"], growth_rates["Back end"], st.session_state.color_scheme["Back end"])

        with m_col2:
            draw_chart("Frontend", job_monthly_trend["Front end"], growth_rates["Front end"], st.session_state.color_scheme["Front end"])

        with m_col3:
            draw_chart("Big Data/AI", job_monthly_trend["Big Data/AI"], growth_rates["Big Data/AI"], color_DataAI)  # 버튼 클릭 시 페이지를 새로 고침하여 다른 선택이 반영되도록 함)

        with m_col4:
            draw_chart("ALL", job_monthly_trend["ALL"], growth_rates["ALL"], color_all)

        seniority_data['총합'] = seniority_data.iloc[:, 1:].sum(axis=1)
        seniority_data_melted = seniority_data.melt(id_vars=['직무', '총합'], var_name='연차', value_name='선호도')
        seniority_data_melted['백분율'] = (seniority_data_melted['선호도'] / seniority_data_melted['총합']) * 100

        st.markdown("""
            <div class='rounded-box' style="margin-top: 27px;">
                <h3>직무/기술 네트워크</h3>
            </div>
        """, unsafe_allow_html=True)

        # Big Data/AI 색상 = Data와 AI 색상의 중간색
        color_DataAI = N_average_colors([
            st.session_state.color_scheme["Data"], 
            st.session_state.color_scheme["AI"]
        ], [1, 1])  # 색상 비율 1:1

        # [1] CSV 파일 읽기
        csv_file_name = "data/tech_keywords_combined.csv"

        df = pd.read_csv(csv_file_name)


        columns = df.columns.tolist()
    
        # 첫 열: 기술 키워드
        keyword_col = columns[0]   # 예: 'Keyword'
        # 나머지 열: 직무 (예: ['AI', 'Back end', 'Front end', 'Data'])
        job_roles = columns[1:]

        # [2] NetworkX 그래프 생성
        G = nx.Graph()

        def get_color_by_freq(role_freq_dict):
            """모든 직무의 빈도를 반영하여 색상 혼합"""
            if not role_freq_dict:
                return "lightgray"
            
            # 빈도 0이 아닌 직무만 선택
            valid_roles = {role: freq for role, freq in role_freq_dict.items() if freq > 0}

            # 모든 관련 직무의 색상과 빈도를 반영하여 혼합
            colors = [st.session_state.color_scheme[role] for role in valid_roles.keys()]
            weights = list(valid_roles.values())  # 빈도를 가중치로 사용

            return N_average_colors(colors, weights)

        # [3] 직무 노드 추가 (group 속성 제거)
        for role in job_roles:
            dark_color = st.session_state.color_scheme.get(role, "gray")

            G.add_node(
                role,
                label=role,
                shape="dot",
                color=dark_color,
                size=53,
                font=dict(size=30),
                mass=5
            )

        # [4] 기술 노드 + 엣지 추가 (group 속성 제거)
        for idx, row in df.iterrows():
            skill = row[keyword_col]

            # 직무별 빈도 딕셔너리
            role_freq_dict = {r: row[r] for r in job_roles}
            total_freq = sum(role_freq_dict.values())

            # 기술 노드 색상: 가장 많이 언급된 직무 기준
            node_color = get_color_by_freq(role_freq_dict)

            # 툴팁 ("/"로 구분)
            tooltip_lines = []
            for (r, f) in sorted(role_freq_dict.items(), key=lambda x: x[1], reverse=True)[:2]:
                pct = (f / total_freq * 100) if total_freq > 0 else 0
                tooltip_lines.append(f"{r}: {f} ({pct:.2f}%)")
            tooltip_text = " / ".join(tooltip_lines)

            # 기술 노드 크기 계산
            base_size = 10 + total_freq * 0.5
            node_size = base_size * 0.4

            # 기술 노드 추가 (group 제거)
            G.add_node(
                skill,
                label=skill,
                shape="dot",
                color=node_color,
                size=node_size,
                font=dict(size=30),
                title=tooltip_text
            )

            # freq > 0 인 직무에만 엣지 연결
            for role in job_roles:
                freq = role_freq_dict[role]
                if freq > 0:
                    edge_color = st.session_state.color_scheme.get(role, "lightgray")
                    # 가중치가 높은 경우 더 짧은 엣지 길이
                    edge_length = max(50, 300 - freq * 10)  # 최대 300, 최소 50까지
                    # 엣지 추가 (가중치와 길이 조정)
                    G.add_edge(
                        skill,
                        role,
                        value=freq * 2,    # 가중치 강화
                        color=edge_color,
                        title=f"{skill} ↔ {role}: {freq}",
                        length=edge_length  # 엣지 길이 추가
                    )

        # [5] "차수(degree)가 1인 노드"의 엣지 길이 반으로 만들기
        for node in G.nodes():
            if G.degree(node) == 1:
                edges = list(G.edges(node, data=True))
                if len(edges) == 1:
                    u, v, data_dict = edges[0]
                    data_dict["length"] = 187

        # PyVis 설정 (간격 2.5배: node_distance=450, spring_length=375 등)
        net = Network(width="100%", height="800px")
        net.repulsion(
            node_distance=350,    # 기존 450 -> 350 (조금 더 가까워지도록)
            central_gravity=0.2,  # 기존 0.1 -> 0.2 (관련된 노드끼리 밀집)
            spring_length=300,    # 기존 375 -> 300 (연결된 노드들 더 가까이)
            spring_strength=0.15, # 기존 0.05 -> 0.15 (가중치 높은 노드끼리 밀착)
            damping=0.85
        )

        # NetworkX -> PyVis
        net.from_nx(G)

        # HTML 저장 후 Streamlit에 임베드
        net.save_graph("network.html")
        with open("network.html", "r", encoding="utf-8") as f:
            html_data = f.read()

        
        # Streamlit에 그래프 임베드 (가로로 더 넓게 설정)
        html(f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
            {html_data}
        </div>
        """, height=400, width=950, scrolling=True)


    
    st.markdown("<div class='rounded-box'><h3>직무별 연차 선호도</h3></div>", unsafe_allow_html=True)
    fig_scatter = px.scatter(
        seniority_data_melted, 
        x='연차', 
        y='선호도', 
        color='직무',
        size='선호도',
        labels={'백분율': '백분율 (%)', '직무': '직무 분야'},
        hover_data={'백분율': ':.2f'},
        color_discrete_sequence=PLOTLY_COLORS
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    

with tab3:
    # CSV 파일 경로 설정 (예시로 파일명 'map_data.csv' 사용)
    df = pd.read_csv('data/map_data.csv')

    # 데이터 전처리: 지역별 직종의 개수 계산 (공고 수 계산)
    job_count = df.groupby(['지역', '직종']).size().reset_index(name='공고 수')

    # 각 지역의 총 공고 수 계산
    total_job_count = df.groupby('지역').size().reset_index(name='총 공고 수')

    # 직종별 공고 수를 계산할 때, '공고 수'와 '총 공고 수'의 타입을 int로 변환
    job_count['공고 수'] = job_count['공고 수'].astype(int)
    total_job_count['총 공고 수'] = total_job_count['총 공고 수'].astype(int)

    # 예시: 각 지역의 위도, 경도 정보 (추가/수정 필요)
    location_coords = {
        '서울 강남구': [37.5173, 127.0473],
        '서울 서초구': [37.4834, 127.0322],
        '경기 성남시': [37.4386, 127.1377],
        '서울 마포구': [37.5663, 126.9000],
        '서울 성동구': [37.5633, 127.0425],
        '서울 영등포구': [37.5267, 126.8978],
        '서울 중구': [37.5636, 126.9970],
        '서울 송파구': [37.5146, 127.1054],
        '서울 구로구': [37.4952, 126.8817],
        '서울 관악구': [37.4780, 126.9517],
        '서울 금천구': [37.4582, 126.8984],
        '서울 종로구': [37.5707, 126.9812],
        '서울 용산구': [37.5326, 126.9903],
        '서울 강서구': [37.5482, 126.8490],
        '인천 연수구': [37.4133, 126.6500],
        '경기 안양시': [37.3910, 126.9248],
        '대전 유성구': [36.3730, 127.3660],
        '경기 과천시': [37.4445, 126.9978],
        '서울 광진구': [37.5399, 127.0827],
        '경기 용인시': [37.2415, 127.1780],
        '부산 해운대구': [35.1645, 129.1603],
        '서울 동작구': [37.5113, 126.9404],
        '경기 수원시': [37.2636, 127.0286],
        '경기 고양시': [37.6487, 126.8357],
        '부산 부산진구': [35.1591, 129.0630],
        '충남 천안시': [36.8057, 127.1390],
        '서울 동대문구': [37.5743, 127.0434],
        '서울 강동구': [37.5302, 127.1235],
        '경기 화성시': [37.2045, 127.0075],
        '경기 파주시': [37.7521, 126.7732],
        '서울 서대문구': [37.5822, 126.9368],
        '경기 군포시': [37.3597, 126.9261],
        '경기 광주시': [37.4224, 127.2554],
        '경북 포항시': [36.0223, 129.3459],
        '경기 김포시': [37.6161, 126.7066],
        '경북 안동시': [36.5661, 128.7395],
        '경기 안산시': [37.3201, 126.8302],
        '경기 구리시': [37.5980, 127.1257],
        '경기 광명시': [37.4765, 126.8666],
        '대전 서구': [36.3484, 127.3844],
        '경기 부천시': [37.5047, 126.7669],
        '충남 아산시': [36.7982, 127.0271],
        '인천 부평구': [37.4877, 126.7055],
        '대구 남구': [35.8256, 128.6030],
        '서울 노원구': [37.6548, 127.0771],
        '부산': [35.1796, 129.0757],
        '경기 하남시': [37.5281, 127.2112]
    }

    # 막대 그래프 이미지 생성 함수
    def create_bar_chart_image(job_count_for_location):
        """직종별 공고 수를 나타내는 막대 그래프를 생성하고 base64 이미지로 변환"""
        job_count_for_location = job_count_for_location.sort_values('공고 수', ascending=False)

        # 그래프 설정
        fig, ax = plt.subplots(figsize=(8, 5))

        # 직종 및 공고 수
        job_categories = job_count_for_location['직종']
        job_counts = job_count_for_location['공고 수']

        # 직종별 색상 설정
        colors = [st.session_state.color_scheme.get(job, "#000000") for job in job_categories]

        ax.bar(job_categories, job_counts, color=colors)
        ax.set_xlabel('직종')
        ax.set_ylabel('공고 수')
        ax.set_title('직종별 공고 수')
        plt.xticks(rotation=45)

        # 이미지 변환
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png', bbox_inches="tight")
        img_stream.seek(0)
        plt.close(fig)

        # base64 변환
        img_b64 = base64.b64encode(img_stream.getvalue()).decode()
        img_tag = f'<img src="data:image/png;base64,{img_b64}" width="600" height="400"/>'
        
        return img_tag


    # 직종 선택 (checkbox 버튼)
    selected_jobs = []
    selected_jobs = [job for job in st.session_state.color_scheme.keys() if st.checkbox(job, key=job)]

    # 직종 선택에 맞는 지역별 공고 수 계산
    job_count_filtered = job_count[job_count['직종'].isin(selected_jobs)]

    # '공고 수'를 int로 변환
    job_count_filtered['공고 수'] = job_count_filtered['공고 수'].astype(int)

    # 직종이 선택되지 않았으면, 지도와 순위를 표시하지 않도록 처리
    # 선택된 직종이 있을 경우 지도 표시
    if selected_jobs:
        # 세션 상태에 지도 저장하여 초기화 방지
        if 'map_object' not in st.session_state:
            st.session_state.map_object = folium.Map(location=[36.5, 128], zoom_start=7)

        m = st.session_state.map_object  # 기존 지도 객체 가져오기
        marker_cluster = MarkerCluster().add_to(m)  # 마커 클러스터 추가

        job_count_filtered = job_count[job_count['직종'].isin(selected_jobs)]

        for location_name in total_job_count['지역']:
            total_jobs_for_location = total_job_count[total_job_count['지역'] == location_name]['총 공고 수'].values[0]
            job_count_for_location = job_count_filtered[job_count_filtered['지역'] == location_name]

            if not job_count_for_location.empty and location_name in location_coords:
                coordinates = location_coords[location_name]

                # 그래프 생성
                img_tag = create_bar_chart_image(job_count_for_location)

                for job_category in selected_jobs:
                    job_data = job_count_for_location[job_count_for_location['직종'] == job_category]
                    if not job_data.empty:
                        job_count_value = job_data['공고 수'].values[0]
                        marker_size = max(5, min(np.log1p(job_count_value) * 15, 200))

                        # 색상 적용
                        color = st.session_state.color_scheme.get(job_category, "#000000")

                        folium.CircleMarker(
                            location=coordinates,
                            radius=marker_size,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.4,
                            popup=folium.Popup(
                                f"""
                                <strong>{location_name}</strong><br>
                                {job_category} 공고 수: {job_count_value}<br>
                                총 공고 수: {total_jobs_for_location}<br>
                                {img_tag}  <!-- 그래프 포함 -->
                                """,
                                max_width=800
                            )
                        ).add_to(marker_cluster)


        # 선택한 직종이 있을 경우에만 실행
        if selected_jobs:
            job_count_by_location = job_count_filtered.groupby(['지역', '직종'])['공고 수'].sum().reset_index()
            job_count_by_location = job_count_by_location.sort_values('공고 수', ascending=False)

            col1, col2 = st.columns([2, 1])  # 첫 번째 열(지도) / 두 번째 열(순위)

            with col1:
                st.session_state.map_object = m  # 변경된 지도 저장

                folium_static(m, width=1000, height=700)

            with col2:
                st.header("📌 직종별 공고가 많은 지역 순위")

                top5_overall = job_count_by_location.groupby("지역")["공고 수"].sum().reset_index()
                top5_overall = top5_overall.sort_values("공고 수", ascending=False).head(5)
                st.subheader("🔹 전체 직종 기준 Top 5 지역")
                st.dataframe(top5_overall[['지역', '공고 수']], use_container_width=True)

    else:
        st.markdown("""
        <div style="background-color: #f4f4f9; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: #e76f51;">직종을 선택하세요</h2>
            <p style="color: #2a9d8f;">원하는 직종을 선택하여 공고를 확인할 수 있습니다.</p>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='title'>Job market RAG Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>구직에 대한 모든 궁금증을 여기에!</div>", unsafe_allow_html=True)

    # Initialize or retrieve chat history from session state.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # List of tuples: (user message, bot response)

    # Display existing chat history.
    chat_container = st.container()
    if st.session_state.chat_history:
        for user_msg, bot_msg in st.session_state.chat_history:
            chat_container.markdown(f"**User:** {user_msg}")
            chat_container.markdown(f"**Bot:** {bot_msg}")

    # User input for the new message.
    user_input = st.text_input("Your message:", key="user_input")

    if st.button("Send"):
        if user_input.strip() != "":
            # Append the user message with an empty bot response (to be updated)
            st.session_state.chat_history.append((user_input, ""))
            # Create a placeholder for streaming the bot's answer.
            response_placeholder = st.empty()
            full_response = ""
            # Use previous chat history (excluding the last incomplete one) as context.
            chat_history_context = st.session_state.chat_history[:-1]
            # Stream the response.
            for partial_response in chat_function(user_input, chat_history_context):
                full_response = partial_response
                response_placeholder.markdown(full_response)
                time.sleep(0.05)  # small sleep to simulate streaming
            # Update the chat history with the final answer.
            st.session_state.chat_history[-1] = (user_input, full_response)
            # The updated session state will refresh the chat display on the next interaction.

    # Button to clear the chat history.
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []

# 회색 선 추가
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

logo_image = image_to_base64(logo_image_path)
st.markdown(f"""
        <div class="logo">
            <br><br><center><img src="data:image/png;base64,{logo_image}" width='500'>
        </div>
""", unsafe_allow_html=True)