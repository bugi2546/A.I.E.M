# 크롤링 및 AI Hub 데이터 기반 표정 감정 분석 AI 모델 구축

## 프로젝트 개요
* **기간:** 2024.09 ~ 2024.12 
* **팀 구성:** 3인 팀 프로젝트
* **한 줄 소개:** 웹 크롤링 기반의 대용량 비정형 표정 이미지 자동 수집·정제 파이프라인 구축 및 PyTorch AlexNet 기반 4종 감정 분류 AI 모델 개발
* **담당 역할:** 웹 크롤링 및 데이터 수집 (Web Crawling & Data Collection)

## 사용 기술 및 환경 
* **Environment / OS:** Linux (Ubuntu), VS Code, Jupyter Notebook
* **Crawling & Scraping:** Selenium, BeautifulSoup4, Requests
* **Language / Framework:** Python 3.9, PyTorch, Torchvision, PIL (Pillow), OS, NumPy

## 핵심 기능 및 구조
[동적 웹 크롤링] ➔ [파일명 일괄 정규화] ➔ [CustomDataset] ➔ [AlexNet CNN 학습]

* **웹 크롤링 기반 데이터 수집 (Data Collection)**
* **비정형 파일명 일괄 정규화 스크립트 (Data Preprocessing)**
* **PyTorch 커스텀 데이터셋 구축 (`CustomDataset`)**
* **이미지 전처리 및 AlexNet CNN 모델 학습**


## 트러블슈팅 & 해결 과정
* **문제 상황:** 크롤링 및 AI Hub를 통해 수집된 원본 이미지 데이터의 파일명이 너무 길고 한글/특수문자가 포함되어 있어 PyTorch `DataLoader` 및 `Image.open` 호출 시 데이터 로딩 에러 빈번히 발생
* **원인 분석:** 원천 데이터 파일명 전체에 결합된 복잡한 메타데이터(성별_연령_장소_일자 등)가 파이토치 인덱싱 파싱 과정에서 예외 유발
* **해결 방법:** Python `os` 모듈과 `try-except` 예외 처리 기반의 자동화 스크립트를 작성하여, 감정 폴더 내부 파일들을 `s0.jpg`, `s1.jpg` 형태의 인덱스 식별자로 일괄 변환하는 사전 정제 파이프라인 구축
* **결과:** 데이터 로딩 오류로 인한 학습 중단 현상을 완벽히 해결하여 안정적인 AI 모델 학습 연동 성공

## 성과 
1. End-to-End 데이터 파이프라인 경험
2. 데이터 품질 관리(Data Quality)의 중요성 체득
3. 웹 크롤링 도구 활용 역량 확보
