# Sungsoo Project

## 🏆 수상 이력

본 프로젝트는 2024 성동구 빅데이터 펠로우십에서 우수상을 수상한 프로젝트입니다. 🎉

## 📌 프로젝트 소개

Sungsoo 프로젝트는 성수동 상권 지속 가능성을 분석하는 데이터 기반 연구 프로젝트입니다. 본 연구는 성동구 성수동 지역의 폐업 및 지속 가능 상권을 구분하고, 상권 활성화를 위한 정책적 제안을 도출하는 것을 목표로 합니다.

## 🚀 주요 기능

- 성수동 지역의 상권 데이터를 활용한 분석
- 폐업 및 지속 가능한 상권을 구분하는 머신러닝 모델 개발
- SHAP 값을 활용한 변수 중요도 분석 및 인사이트 도출
- 상권별 특성을 반영한 데이터 기반 정책 제안
- 지리적 요인, 유동 인구, 매출 데이터 기반 상권 지속 가능성 평가

## 📁 프로젝트 구조

```bash
sungsoo/
│── analysis/                # 분석 코드
│── preprocess/              # 데이터 전처리 코드
│── README.md               # 프로젝트 설명서
```

## 🔧 설치 및 실행 방법

### 1. 환경 설정

이 프로젝트는 Anaconda 환경에서 실행됩니다.

```bash
conda env create -f sungsoo.yaml
conda activate sungsoo
```

### 2. 데이터 준비

- 프로젝트에서 사용한 주요 데이터:
  - **상권별 일별 매출 데이터** (카드사 제공)
  - **유동 인구 데이터** (연령대, 성별 포함, 성동구청 제공)
  - **폐업 및 지속 가능 여부에 대한 라벨링** (지방행정 인허가 데이터 활용)
  - **대중교통 데이터** (지하철, 버스, 따릉이 정류장 승하차 정보)
  - **날씨 데이터** (공공데이터포털 제공)

### 3. 코드 실행

```bash
python analysis/analyze_korean.py  # 한식 업종에 대해서 모델 학습 및 분석
python analysis/analyze_meal.py  # 외국음식 업종에 대해서 모델 학습 및 분석
python analysis/analyze_cafe.py  # 카페 업종에 대해서 모델 학습 및 분석
python analysis/visualize_correlation_moving_sales.py #유동인구와 매출액간 상관관계 분석
python analysis/visualize_paup_ratio.py #폐업비율을 시각화
```

## 🎯 연구 목표

- 성수동 상권의 지속 가능성을 결정짓는 주요 요인 분석
- 폐업과 지속 가능한 상권 간의 차이점 도출
- SHAP 분석을 활용한 변수 중요도 평가
- 데이터 기반 정책적 제안 및 상권 활성화 방안 제시
- 유동 인구, 대중교통 접근성, 날씨 등의 요인이 상권에 미치는 영향 분석

## 📊 데이터 분석 및 모델링

1. **데이터 전처리**

   - 카드사 매출 데이터 및 상권별 유동 인구 데이터 병합
   - 격자 단위 데이터 집계 (250m x 250m 격자)
   - 폐업 및 지속 상권의 정의 및 라벨링

2. **머신러닝 모델 개발**

   - 회귀 분석을 통한 매출 예측
   - Random Forest, XGBoost, LightGBM 모델 비교
   - SHAP 분석을 통한 주요 변수 도출

3. **주요 인사이트**

   - 폐업 격자는 유동 인구가 많아도 매출이 낮은 경향
   - 지속 가능 상권은 주요 고객층(20\~40대 유동 인구)과의 높은 연관성을 보임
   - 대중교통 접근성이 폐업 및 지속 가능성에 영향을 미치는 주요 변수로 확인됨

## 📢 프로젝트 활용 및 기대 효과

데이터 기반 정책 수립: 성동구청 및 관련 기관에서 상권 지속 가능성 평가에 활용 가능

창업 및 폐업 예측 모델: 창업자들에게 유용한 상권 데이터 제공

젠트리피케이션 분석: 임대료 상승 및 유동 인구 변화를 반영한 정책 수립 지원

## 🛠️ 사용 기술

- **데이터 분석**: Pandas, NumPy, Scikit-learn
- **시각화**: Matplotlib, Seaborn, QGIS
- **머신러닝 모델**: Random Forest, XGBoost, LightGBM
- **설치 환경**: Anaconda, Python 3.8

## 👥 기여자

이도현 (skypo1000@ds.seoultech.ac.kr, [GitHub](https://github.com/DDohyeon2941))

탁승연 (sytak@ds.seoultech.ac.kr, [GitHub](https://github.com/SyngyeonTak))

이희준 (bok_h22@ds.seoultech.ac.kr, [GitHub](https://github.com/bok-h22))

✨ 이 프로젝트는 성수동 상권의 지속 가능성을 연구하고, 데이터 기반 인사이트를 제공하는 데 초점을 맞추고 있습니다. 많은 관심과 기여 부탁드립니다! 🚀


 
