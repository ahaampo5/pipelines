# 데이터 파이프라인

이 프로젝트는 다양한 데이터 소스로부터 데이터를 로드하고 전처리하는 파이프라인을 제공합니다.

## 구조

```
data/
├── data_loader/
│   ├── base.py                          # Hugging Face 기반 추상화 클래스
│   ├── pii_masking_ai4privacy/
│   │   ├── __init__.py
│   │   └── loader.py                    # AI4Privacy PII 마스킹 데이터셋 로더
│   └── __init__.py
├── data_generator/
├── data_translator/
├── test_pii_loader.py                   # 테스트 스크립트
└── README.md
```

## 주요 기능

### 1. BaseHuggingFaceDataLoader (추상화 클래스)

Hugging Face 데이터셋을 효율적으로 로드하고 전처리하는 기본 클래스입니다.

**주요 기능:**
- 📦 **캐싱 시스템**: 데이터셋을 로컬에 캐시하여 재사용 최적화
- 🔤 **토크나이제이션**: 자동 배치 토크나이제이션 지원
- 🔍 **필터링**: 텍스트 길이, 품질 기반 데이터 필터링
- 📊 **배치 처리**: 메모리 효율적인 배치 단위 처리
- ✅ **검증 기능**: 데이터 품질 자동 검사
- 📝 **다양한 포맷 지원**: JSON, CSV, Parquet 내보내기
- 🌊 **스트리밍**: 대용량 데이터셋 스트리밍 로딩
- 🎯 **샘플링**: 개발/테스트용 데이터 샘플링

### 2. AI4PrivacyPIIDataLoader

개인식별정보(PII) 마스킹을 위한 AI4Privacy 데이터셋 전용 로더입니다.

**특징:**
- 🔒 **PII 유형 지원**: 이름, 이메일, 전화번호, 주소, 신용카드 등
- 🎛️ **필터링 옵션**: PII 유형별, 개수별 필터링
- 📈 **통계 분석**: PII 분포 및 통계 정보 제공
- 🔄 **마스킹 쌍 생성**: 훈련용 입력-출력 쌍 자동 생성
- 🛡️ **백업 시스템**: 원본 데이터셋 로드 실패 시 샘플 데이터 생성

## 사용법

### 기본 사용

```python
from data_loader.pii_masking_ai4privacy import create_pii_data_loader

# 데이터 로더 생성
loader = create_pii_data_loader(
    dataset_name="ai4privacy/pii-masking-300k",
    tokenizer_name="klue/bert-base",
    filter_pii_types=["PERSON", "EMAIL", "PHONE"],
    min_pii_count=1,
    max_pii_count=5
)

# 데이터셋 로드 및 전처리
dataset = loader.get_dataset()

# 기본 정보 확인
info = loader.get_dataset_info(dataset)
stats = loader.get_pii_statistics(dataset)
```

### 원스톱 로딩

```python
from data_loader.pii_masking_ai4privacy import load_and_preprocess_pii_data

# 간단한 로딩 및 전처리
dataset = load_and_preprocess_pii_data(
    dataset_name="ai4privacy/pii-masking-300k",
    tokenizer_name="klue/bert-base",
    sample_size=1000,
    apply_tokenization=True
)
```

### 고급 사용

```python
# 토크나이제이션 적용
tokenized_dataset = loader.apply_tokenization(dataset)

# PII 마스킹 훈련 쌍 생성
training_pairs = loader.create_pii_mask_pairs(dataset)

# 다양한 포맷으로 내보내기
loader.export_to_formats(dataset, "output/", ["json", "csv", "parquet"])

# 샘플링
sample_dataset = loader.sample_dataset(dataset, 100)
```

## 테스트 실행

```bash
python test_pii_loader.py
```

## 의존성

- `datasets`: Hugging Face 데이터셋 라이브러리
- `transformers`: Hugging Face 트랜스포머 라이브러리
- `pandas`: 데이터 조작
- `re`: 정규표현식

## 설치

```bash
pip install datasets transformers pandas
```

## 데이터셋 정보

### AI4Privacy PII 마스킹 데이터셋

- **데이터셋**: `ai4privacy/pii-masking-300k`
- **크기**: 약 300,000개 샘플
- **언어**: 다국어 지원
- **PII 유형**: 12가지 주요 PII 유형 지원
  - PERSON (사람 이름)
  - EMAIL (이메일 주소)
  - PHONE (전화번호)
  - ADDRESS (주소)
  - CREDIT_CARD (신용카드 번호)
  - SSN (사회보장번호)
  - DATE (날짜)
  - ORGANIZATION (조직명)
  - LOCATION (위치)
  - URL (웹사이트 주소)
  - IP_ADDRESS (IP 주소)
  - LICENSE_PLATE (차량 번호판)

## 확장 방법

새로운 데이터셋 로더를 추가하려면:

1. `BaseHuggingFaceDataLoader`를 상속받은 클래스 생성
2. `load_dataset()` 메서드 구현
3. `preprocess_data()` 메서드 구현
4. 필요한 경우 추가 전처리 메서드 구현

```python
class CustomDataLoader(BaseHuggingFaceDataLoader):
    def load_dataset(self):
        # 데이터셋 로딩 로직
        pass
    
    def preprocess_data(self, dataset):
        # 데이터 전처리 로직
        pass
``` 