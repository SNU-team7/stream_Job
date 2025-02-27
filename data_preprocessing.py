import pandas as pd

def preprocess_data(file_path, period=1):
    # CSV 파일 불러오기
    df = pd.read_csv(file_path)

    # 직무 변환
    df["직무"] = df["직무"].astype(str).fillna("")  # NaN 값을 빈 문자열로 변환
    df["직무"] = df["직무"].str.replace("퍼블리싱/디자인\(UI/UX\)", "프론트엔드 개발", regex=True)
    df["직무"] = df["직무"].str.replace("네트워크/서버/보안", "백엔드 개발", regex=True)
    df["직무"] = df["직무"].str.replace("프론트엔드 개발", "Front end", regex=True)
    df["직무"] = df["직무"].str.replace("백엔드 개발", "Back end", regex=True)
    df["직무"] = df["직무"].str.replace("빅데이터/AI", "Big Data/AI", regex=True)

    # 새로운 데이터프레임 생성 (각 직무가 포함된 경우 개별 카운트)
    job_counts = {"채용기간_월": [], "직무": []}

    def extract_jobs(row):
        for job in ["Back end", "Front end", "Big Data/AI"]:
            if job in row["직무"]:
                job_counts["채용기간_월"].append(row["채용기간"].split("~")[-1].strip()[:7])  # 연-월 추출
                job_counts["직무"].append(job)

    # 각 행에 대해 적용
    df.apply(extract_jobs, axis=1)

    df_filtered = pd.DataFrame(job_counts)

    # 직무별 월별 채용 공고 수 집계
    job_monthly_trend = df_filtered.groupby(["채용기간_월", "직무"]).size().unstack().fillna(0)

    # 'ALL' 추가: 각 월별로 모든 직무의 합계 추가
    job_monthly_trend['ALL'] = job_monthly_trend.sum(axis=1)

    # 성장률 계산 (사용자가 선택한 기간 반영)
    job_growth_rate = job_monthly_trend.pct_change(periods=period).fillna(0) * 100  # 퍼센트 단위 변환

    return job_monthly_trend, job_growth_rate
