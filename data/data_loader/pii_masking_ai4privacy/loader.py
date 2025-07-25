"""
AI4Privacy PII 마스킹 데이터셋 로더
"""
import logging
from typing import Optional, Dict, Any, List, Union, cast
from datasets import Dataset, DatasetDict, load_dataset, IterableDataset, IterableDatasetDict
import re
from ..base import BaseHuggingFaceDataLoader

logger = logging.getLogger(__name__)


class AI4PrivacyPIIDataLoader(BaseHuggingFaceDataLoader):
    """
    AI4Privacy PII 마스킹 데이터셋을 로드하고 전처리하는 클래스
    
    이 데이터셋은 개인식별정보(PII) 마스킹을 위한 훈련 데이터를 제공합니다.
    - 원본 텍스트와 PII가 마스킹된 텍스트 쌍을 포함
    - 다양한 PII 유형 지원 (이름, 이메일, 전화번호, 주소 등)
    """
    
    def __init__(
        self,
        dataset_name: str = "ai4privacy/pii-masking-300k",
        subset: Optional[str] = None,
        cache_dir: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 1000,
        streaming: bool = False,
        filter_pii_types: Optional[List[str]] = None,
        min_pii_count: int = 1,
        max_pii_count: Optional[int] = None
    ):
        """
        Args:
            dataset_name: Hugging Face 데이터셋 이름
            subset: 데이터셋 서브셋 (예: 'english', 'multilingual')
            cache_dir: 캐시 디렉토리 경로
            tokenizer_name: 사용할 토크나이저 이름
            max_length: 최대 토큰 길이
            batch_size: 배치 처리 크기
            streaming: 스트리밍 모드 사용 여부
            filter_pii_types: 필터링할 PII 유형 리스트
            min_pii_count: 최소 PII 개수
            max_pii_count: 최대 PII 개수
        """
        super().__init__(
            cache_dir=cache_dir,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            batch_size=batch_size,
            streaming=streaming
        )
        
        self.dataset_name = dataset_name
        self.subset = subset
        self.filter_pii_types = filter_pii_types or []
        self.min_pii_count = min_pii_count
        self.max_pii_count = max_pii_count
        
        # PII 유형 정의
        self.pii_types = {
            "PERSON": "사람 이름",
            "EMAIL": "이메일 주소",
            "PHONE": "전화번호",
            "ADDRESS": "주소",
            "CREDIT_CARD": "신용카드 번호",
            "SSN": "사회보장번호",
            "DATE": "날짜",
            "ORGANIZATION": "조직명",
            "LOCATION": "위치",
            "URL": "웹사이트 주소",
            "IP_ADDRESS": "IP 주소",
            "LICENSE_PLATE": "차량 번호판"
        }
    
    def load_dataset(self) -> Union[Dataset, DatasetDict]:
        """
        AI4Privacy PII 마스킹 데이터셋을 로드
        """
        try:
            logger.info(f"AI4Privacy 데이터셋 로드 중: {self.dataset_name}")
            
            raw_dataset = load_dataset(
                self.dataset_name,
                name=self.subset,
                cache_dir=self.cache_dir,
                streaming=self.streaming
            )
            
            # IterableDataset인 경우 Dataset으로 변환
            if isinstance(raw_dataset, (IterableDataset, IterableDatasetDict)):
                if isinstance(raw_dataset, IterableDatasetDict):
                    dataset = DatasetDict({
                        split: Dataset.from_list(list(ds.take(10000)))  # 스트리밍에서 일부만 가져오기
                        for split, ds in raw_dataset.items()
                    })
                else:
                    dataset = Dataset.from_list(list(raw_dataset.take(10000)))
            else:
                dataset = cast(Union[Dataset, DatasetDict], raw_dataset)
            
            logger.info(f"데이터셋 로드 완료: {dataset}")
            return dataset
            
        except Exception as e:
            logger.error(f"데이터셋 로드 실패: {e}")
            # 백업으로 로컬 데이터나 다른 PII 데이터셋 사용
            logger.info("백업 데이터셋을 시도합니다...")
            return self._load_backup_dataset()
    
    def _load_backup_dataset(self) -> Union[Dataset, DatasetDict]:
        """
        백업 데이터셋 로드 (예: 더 일반적인 PII 데이터셋)
        """
        try:
            # 다른 PII 관련 데이터셋들 시도
            backup_datasets = [
                "microsoft/DialoGPT-medium",  # 대화 데이터에서 PII 추출 가능
                "squad",  # 질의응답 데이터에서 PII 패턴 학습
                "cnn_dailymail"  # 뉴스 데이터에서 PII 패턴 학습
            ]
            
            for backup_name in backup_datasets:
                try:
                    logger.info(f"백업 데이터셋 시도: {backup_name}")
                    raw_dataset = load_dataset(
                        backup_name,
                        cache_dir=self.cache_dir,
                        streaming=self.streaming
                    )
                    
                    # IterableDataset인 경우 Dataset으로 변환
                    if isinstance(raw_dataset, (IterableDataset, IterableDatasetDict)):
                        if isinstance(raw_dataset, IterableDatasetDict):
                            dataset = DatasetDict({
                                split: Dataset.from_list(list(ds.take(1000)))
                                for split, ds in raw_dataset.items()
                            })
                        else:
                            dataset = Dataset.from_list(list(raw_dataset.take(1000)))
                    else:
                        dataset = cast(Union[Dataset, DatasetDict], raw_dataset)
                    
                    logger.info(f"백업 데이터셋 로드 성공: {backup_name}")
                    return dataset
                except:
                    continue
            
            # 모든 백업이 실패하면 가상 데이터셋 생성
            return self._create_sample_dataset()
            
        except Exception as e:
            logger.error(f"백업 데이터셋 로드 실패: {e}")
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> Dataset:
        """
        샘플 PII 데이터셋 생성 (테스트/개발용)
        """
        logger.info("샘플 PII 데이터셋을 생성합니다...")
        
        sample_data = [
            {
                "original_text": "안녕하세요, 제 이름은 김철수입니다. 제 이메일은 kim.chulsoo@example.com이고 전화번호는 010-1234-5678입니다.",
                "masked_text": "안녕하세요, 제 이름은 [PERSON]입니다. 제 이메일은 [EMAIL]이고 전화번호는 [PHONE]입니다.",
                "pii_types": ["PERSON", "EMAIL", "PHONE"],
                "pii_count": 3
            },
            {
                "original_text": "이 문서는 2024년 1월 15일에 작성되었으며, 서울시 강남구 테헤란로 123번지에서 발행되었습니다.",
                "masked_text": "이 문서는 [DATE]에 작성되었으며, [ADDRESS]에서 발행되었습니다.",
                "pii_types": ["DATE", "ADDRESS"],
                "pii_count": 2
            },
            {
                "original_text": "고객님의 신용카드 정보는 1234-5678-9012-3456이며, 만료일은 12/25입니다.",
                "masked_text": "고객님의 신용카드 정보는 [CREDIT_CARD]이며, 만료일은 [DATE]입니다.",
                "pii_types": ["CREDIT_CARD", "DATE"],
                "pii_count": 2
            },
            {
                "original_text": "회사 웹사이트는 https://www.example.com이고, 본사는 (주)예시회사입니다.",
                "masked_text": "회사 웹사이트는 [URL]이고, 본사는 [ORGANIZATION]입니다.",
                "pii_types": ["URL", "ORGANIZATION"],
                "pii_count": 2
            },
            {
                "original_text": "서버 IP 주소는 192.168.1.100이고, 접속 로그는 2024-01-15 14:30:25에 기록되었습니다.",
                "masked_text": "서버 IP 주소는 [IP_ADDRESS]이고, 접속 로그는 [DATE]에 기록되었습니다.",
                "pii_types": ["IP_ADDRESS", "DATE"],
                "pii_count": 2
            }
        ]
        
        # 더 많은 샘플 데이터 생성
        extended_data = []
        for i in range(100):  # 100개 샘플 생성
            for base_sample in sample_data:
                new_sample = base_sample.copy()
                new_sample["id"] = len(extended_data)
                extended_data.append(new_sample)
        
        return Dataset.from_list(extended_data)
    
    def preprocess_data(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        """
        PII 마스킹 데이터셋 전처리
        """
        logger.info("PII 데이터셋 전처리 시작...")
        
        # DatasetDict인 경우 각 split별로 처리
        if isinstance(dataset, DatasetDict):
            processed_dataset = DatasetDict()
            for split, ds in dataset.items():
                processed_dataset[split] = self._preprocess_single_dataset(ds)
            return processed_dataset
        else:
            return self._preprocess_single_dataset(dataset)
    
    def _preprocess_single_dataset(self, dataset: Dataset) -> Dataset:
        """
        단일 데이터셋 전처리
        """
        # 컬럼명 표준화
        dataset = self._standardize_columns(dataset)
        
        # PII 유형 필터링
        if self.filter_pii_types:
            dataset = self._filter_by_pii_types(dataset)
        
        # PII 개수로 필터링
        dataset = self._filter_by_pii_count(dataset)
        
        # 텍스트 정제
        dataset = dataset.map(self._clean_text, batched=True, batch_size=self.batch_size)
        
        # PII 패턴 검증
        dataset = dataset.filter(self._validate_pii_patterns)
        
        # 길이 필터링 적용
        filtered_result = self.filter_by_length(dataset, min_length=10, max_length=1000)
        if isinstance(filtered_result, DatasetDict):
            # DatasetDict가 반환된 경우, 첫 번째 split만 사용
            filtered_dataset = list(filtered_result.values())[0]
        else:
            filtered_dataset = filtered_result
        
        logger.info(f"전처리 완료: {len(filtered_dataset)}개 샘플")
        return filtered_dataset
    
    def _standardize_columns(self, dataset: Dataset) -> Dataset:
        """
        컬럼명을 표준화
        """
        column_mapping = {}
        
        # 다양한 컬럼명 패턴 처리
        for col in dataset.column_names:
            if col.lower() in ['text', 'original', 'source', 'input']:
                column_mapping[col] = 'original_text'
            elif col.lower() in ['masked', 'target', 'output', 'masked_text', 'privacy_mask']:
                column_mapping[col] = 'masked_text'
            elif col.lower() in ['labels', 'pii_types', 'entities']:
                column_mapping[col] = 'pii_types'
        
        if column_mapping:
            dataset = dataset.rename_columns(column_mapping)
            logger.info(f"컬럼명 표준화: {column_mapping}")
        
        return dataset
    
    def _filter_by_pii_types(self, dataset: Dataset) -> Dataset:
        """
        특정 PII 유형으로 필터링
        """
        def pii_type_filter(example):
            if 'pii_types' not in example:
                return True
            
            pii_types = example['pii_types']
            if isinstance(pii_types, str):
                pii_types = [pii_types]
            
            return any(pii_type in self.filter_pii_types for pii_type in pii_types)
        
        filtered_dataset = dataset.filter(pii_type_filter)
        logger.info(f"PII 유형 필터링: {len(dataset)} -> {len(filtered_dataset)} 샘플")
        return filtered_dataset
    
    def _filter_by_pii_count(self, dataset: Dataset) -> Dataset:
        """
        PII 개수로 필터링
        """
        def pii_count_filter(example):
            pii_count = example.get('pii_count', 0)
            if isinstance(pii_count, list):
                pii_count = len(pii_count)
            
            if pii_count < self.min_pii_count:
                return False
            if self.max_pii_count and pii_count > self.max_pii_count:
                return False
            return True
        
        filtered_dataset = dataset.filter(pii_count_filter)
        logger.info(f"PII 개수 필터링: {len(dataset)} -> {len(filtered_dataset)} 샘플")
        return filtered_dataset
    
    def _clean_text(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        텍스트 정제
        """
        def clean_single_text(text):
            if not isinstance(text, str):
                return ""
            
            # HTML 태그 제거
            text = re.sub(r'<[^>]+>', '', text)
            
            # 불필요한 공백 정리
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 특수 문자 정리 (PII 마스킹 토큰은 보존)
            text = re.sub(r'[^\w\s\[\]가-힣.,!?-]', '', text)
            
            return text
        
        cleaned_examples = {}
        for key, values in examples.items():
            if key in ['original_text', 'masked_text']:
                cleaned_examples[key] = [clean_single_text(text) for text in values]
            else:
                cleaned_examples[key] = values
        
        return cleaned_examples
    
    def _validate_pii_patterns(self, example) -> bool:
        """
        PII 패턴 유효성 검사
        """
        original = example.get('original_text', '')
        masked = example.get('masked_text', '')
        
        if not original or not masked:
            return False
        
        # 마스킹 토큰 패턴 확인
        mask_pattern = r'\[([A-Z_]+)\]'
        masks = re.findall(mask_pattern, masked)
        
        # 최소한의 마스킹이 있는지 확인
        if len(masks) == 0:
            return False
        
        # 원본 텍스트와 마스킹된 텍스트의 길이 차이가 너무 크지 않은지 확인
        length_ratio = len(masked) / max(len(original), 1)
        if length_ratio < 0.3 or length_ratio > 2.0:
            return False
        
        return True
    
    def get_pii_statistics(self, dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
        """
        PII 통계 정보 반환
        """
        stats = {}
        
        if isinstance(dataset, DatasetDict):
            for split, ds in dataset.items():
                stats[split] = self._calculate_pii_stats(ds)
        else:
            stats = self._calculate_pii_stats(dataset)
        
        return stats
    
    def _calculate_pii_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """
        단일 데이터셋의 PII 통계 계산
        """
        pii_type_counts = {}
        total_pii_count = 0
        
        for example in dataset:
            pii_types = example.get('pii_types', []) if isinstance(example, dict) else []
            if isinstance(pii_types, str):
                pii_types = [pii_types]
            
            for pii_type in pii_types:
                pii_type_counts[pii_type] = pii_type_counts.get(pii_type, 0) + 1
                total_pii_count += 1
        
        return {
            'total_samples': len(dataset),
            'total_pii_instances': total_pii_count,
            'avg_pii_per_sample': total_pii_count / max(len(dataset), 1),
            'pii_type_distribution': pii_type_counts,
            'unique_pii_types': len(pii_type_counts)
        }
    
    def create_pii_mask_pairs(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        """
        PII 마스킹 훈련을 위한 입력-출력 쌍 생성
        """
        def create_pairs(examples):
            inputs = []
            targets = []
            
            for original, masked in zip(examples['original_text'], examples['masked_text']):
                # 입력: 원본 텍스트 + 마스킹 지시
                input_text = f"다음 텍스트에서 개인정보를 마스킹하세요: {original}"
                inputs.append(input_text)
                targets.append(masked)
            
            return {
                'input_text': inputs,
                'target_text': targets
            }
        
        if isinstance(dataset, DatasetDict):
            return DatasetDict({
                split: ds.map(create_pairs, batched=True, batch_size=self.batch_size)
                for split, ds in dataset.items()
            })
        else:
            return dataset.map(create_pairs, batched=True, batch_size=self.batch_size)


# 사용 예시 및 유틸리티 함수
def create_pii_data_loader(
    dataset_name: str = "ai4privacy/pii-masking-300k",
    tokenizer_name: str = "klue/bert-base",
    cache_dir: Optional[str] = None,
    **kwargs
) -> AI4PrivacyPIIDataLoader:
    """
    AI4Privacy PII 데이터 로더 생성 헬퍼 함수
    """
    return AI4PrivacyPIIDataLoader(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        cache_dir=cache_dir,
        **kwargs
    )


def load_and_preprocess_pii_data(
    dataset_name: str = "ai4privacy/pii-masking-300k",
    tokenizer_name: str = "klue/bert-base",
    sample_size: Optional[int] = None,
    apply_tokenization: bool = True,
    **kwargs
) -> Union[Dataset, DatasetDict]:
    """
    PII 데이터를 로드하고 전처리하는 원스톱 함수
    """
    # 데이터 로더 생성
    loader = create_pii_data_loader(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        **kwargs
    )
    
    # 데이터셋 로드
    dataset = loader.get_dataset()
    
    # 샘플링 (선택사항)
    if sample_size:
        dataset = loader.sample_dataset(dataset, sample_size)
    
    # 토크나이제이션 적용 (선택사항)
    if apply_tokenization:
        dataset = loader.apply_tokenization(dataset)
    
    # 유효성 검사
    if not loader.validate_dataset(dataset):
        raise ValueError("데이터셋 유효성 검사 실패")
    
    # 통계 정보 출력
    stats = loader.get_pii_statistics(dataset)
    logger.info(f"PII 데이터셋 통계: {stats}")
    
    return dataset


if __name__ == "__main__":
    # 기본 사용 예시
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 데이터 로더 생성
        loader = create_pii_data_loader(
            tokenizer_name="klue/bert-base",
            filter_pii_types=["PERSON", "EMAIL", "PHONE"],
            min_pii_count=1,
            max_pii_count=5
        )
        
        # 데이터셋 로드 및 전처리
        dataset = loader.get_dataset()
        
        # 기본 정보 출력
        info = loader.get_dataset_info(dataset)
        print(f"데이터셋 정보: {info}")
        
        # PII 통계 출력
        stats = loader.get_pii_statistics(dataset)
        print(f"PII 통계: {stats}")
        
        # 샘플 데이터 확인
        if isinstance(dataset, DatasetDict):
            sample = dataset['train'][0] if 'train' in dataset else list(dataset.values())[0][0]
        else:
            sample = dataset[0]
        
        print(f"샘플 데이터: {sample}")
        
    except Exception as e:
        logger.error(f"데이터 로딩 실패: {e}")
        print("샘플 데이터로 테스트를 진행합니다.")
        
        # 샘플 데이터로 테스트
        loader = AI4PrivacyPIIDataLoader()
        dataset = loader._create_sample_dataset()
        print(f"샘플 데이터셋 크기: {len(dataset)}")
        print(f"첫 번째 샘플: {dataset[0]}")
