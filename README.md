# 본 프로젝트는 Streamlit으로 만들어진 직무/기술 동향 및 추천 서비스 웹페이지 입니다.
실행 방법
해당 경로로 이동하여 streamlit run streamlit.app.py로 실행
# 사용 데이터
본 프로젝트의 데이터는 원티드, 핀테크 포털 공고의 각 직무를 크롤링 하였습니다.
1. 경력_범위별_공고_수.csv : 공고별 경력 요구 사항
2. combined_output.json : 챗봇 RAG를 위한 벡터 데이터 베이스 데이터
3. fintech_jobs_final.csv : 핀테크 포톨 공고 시계열 데이터
4. map_data.csv : 위치, 직무 데이터
5. tech_keywords_combined.csv : 네트워크 시각화 및 직무/기술 추천을 위한 기술-직무 연관성 데이터
# 챗봇, 데이터 베이스는 제외 했습니다.
