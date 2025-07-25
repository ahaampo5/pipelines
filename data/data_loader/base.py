"""
Hugging Face 기반 데이터 로더 추상화 클래스
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Iterator
from pathlib import Path
import logging
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import pandas as pd
import json

logger = logging.getLogger(__name__)


class BaseHuggingFaceDataLoader(ABC):
    """
    Hugging Face 데이터셋을 로드하고 전처리하는 추상화 클래스
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 1000,
        streaming: bool = False
    ):
        """
        Args:
            cache_dir: 캐시 디렉토리 경로
            tokenizer_name: 사용할 토크나이저 이름
            max_length: 최대 토큰 길이
            batch_size: 배치 처리 크기
            streaming: 스트리밍 모드 사용 여부
        """
        self.cache_dir = cache_dir
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.streaming = streaming
        self.tokenizer = None
        self._dataset = None
        
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @abstractmethod
    def load_dataset(self) -> Union[Dataset, DatasetDict]:
        """
        데이터셋을 로드하는 추상 메서드
        하위 클래스에서 구현해야 함
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        """
        데이터 전처리를 수행하는 추상 메서드
        하위 클래스에서 구현해야 함
        """
        pass
    
    def get_dataset(self, force_reload: bool = False) -> Union[Dataset, DatasetDict]:
        """
        데이터셋을 가져오거나 로드함
        
        Args:
            force_reload: 강제로 다시 로드할지 여부
        """
        if self._dataset is None or force_reload:
            logger.info("데이터셋을 로드하는 중...")
            self._dataset = self.load_dataset()
            logger.info("데이터 전처리를 수행하는 중...")
            self._dataset = self.preprocess_data(self._dataset)
            logger.info(f"데이터셋 로드 완료: {self._dataset}")
        
        return self._dataset
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        배치 토크나이제이션 함수
        
        Args:
            examples: 토크나이즈할 예제들
        """
        if self.tokenizer is None:
            raise ValueError("토크나이저가 설정되지 않았습니다.")
        
        # 텍스트 컬럼을 찾아서 토크나이즈
        text_columns = [col for col in examples.keys() if 'text' in col.lower()]
        if not text_columns:
            raise ValueError("토크나이즈할 텍스트 컬럼을 찾을 수 없습니다.")
        
        # 첫 번째 텍스트 컬럼을 사용
        text_column = text_columns[0]
        
        return self.tokenizer(
            examples[text_column],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None
        )
    
    def apply_tokenization(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        """
        데이터셋에 토크나이제이션을 적용
        """
        if self.tokenizer is None:
            logger.warning("토크나이저가 설정되지 않아 토크나이제이션을 건너뜁니다.")
            return dataset
        
        if isinstance(dataset, DatasetDict):
            return DatasetDict({
                split: ds.map(
                    self.tokenize_function,
                    batched=True,
                    batch_size=self.batch_size,
                    remove_columns=[col for col in ds.column_names if 'text' in col.lower()]
                )
                for split, ds in dataset.items()
            })
        else:
            return dataset.map(
                self.tokenize_function,
                batched=True,
                batch_size=self.batch_size,
                remove_columns=[col for col in dataset.column_names if 'text' in col.lower()]
            )
    
    def filter_by_length(self, dataset: Union[Dataset, DatasetDict], 
                        min_length: int = 10, max_length: Optional[int] = None) -> Union[Dataset, DatasetDict]:
        """
        텍스트 길이로 데이터를 필터링
        
        Args:
            dataset: 필터링할 데이터셋
            min_length: 최소 길이
            max_length: 최대 길이
        """
        def length_filter(example):
            text_columns = [col for col in example.keys() if 'text' in col.lower()]
            if not text_columns:
                return True
            
            text = example[text_columns[0]]
            text_len = len(text.split()) if isinstance(text, str) else 0
            
            if text_len < min_length:
                return False
            if max_length and text_len > max_length:
                return False
            return True
        
        if isinstance(dataset, DatasetDict):
            return DatasetDict({
                split: ds.filter(length_filter)
                for split, ds in dataset.items()
            })
        else:
            return dataset.filter(length_filter)
    
    def sample_dataset(self, dataset: Union[Dataset, DatasetDict], 
                      sample_size: int) -> Union[Dataset, DatasetDict]:
        """
        데이터셋에서 샘플을 추출
        
        Args:
            dataset: 샘플링할 데이터셋
            sample_size: 샘플 크기
        """
        if isinstance(dataset, DatasetDict):
            return DatasetDict({
                split: ds.select(range(min(sample_size, len(ds))))
                for split, ds in dataset.items()
            })
        else:
            return dataset.select(range(min(sample_size, len(dataset))))
    
    def save_to_disk(self, dataset: Union[Dataset, DatasetDict], path: str):
        """
        데이터셋을 디스크에 저장
        
        Args:
            dataset: 저장할 데이터셋
            path: 저장 경로
        """
        dataset.save_to_disk(path)
        logger.info(f"데이터셋이 {path}에 저장되었습니다.")
    
    def load_from_disk(self, path: str) -> Union[Dataset, DatasetDict]:
        """
        디스크에서 데이터셋을 로드
        
        Args:
            path: 로드할 경로
        """
        from datasets import load_from_disk
        dataset = load_from_disk(path)
        logger.info(f"데이터셋이 {path}에서 로드되었습니다.")
        return dataset
    
    def get_dataset_info(self, dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
        """
        데이터셋 정보를 반환
        
        Args:
            dataset: 정보를 가져올 데이터셋
        """
        if isinstance(dataset, DatasetDict):
            info = {}
            for split, ds in dataset.items():
                info[split] = {
                    'num_rows': len(ds),
                    'num_columns': len(ds.column_names),
                    'column_names': ds.column_names,
                    'features': ds.features
                }
            return info
        else:
            return {
                'num_rows': len(dataset),
                'num_columns': len(dataset.column_names),
                'column_names': dataset.column_names,
                'features': dataset.features
            }
    
    def validate_dataset(self, dataset: Union[Dataset, DatasetDict]) -> bool:
        """
        데이터셋 유효성 검사
        
        Args:
            dataset: 검사할 데이터셋
        """
        try:
            if isinstance(dataset, DatasetDict):
                for split, ds in dataset.items():
                    if len(ds) == 0:
                        logger.warning(f"Split '{split}'이 비어있습니다.")
                        return False
            else:
                if len(dataset) == 0:
                    logger.warning("데이터셋이 비어있습니다.")
                    return False
            
            logger.info("데이터셋 유효성 검사 통과")
            return True
        except Exception as e:
            logger.error(f"데이터셋 유효성 검사 실패: {e}")
            return False
    
    def export_to_formats(self, dataset: Union[Dataset, DatasetDict], 
                         output_dir: str, formats: List[str] = ['json', 'csv']):
        """
        다양한 포맷으로 데이터셋 내보내기
        
        Args:
            dataset: 내보낼 데이터셋
            output_dir: 출력 디렉토리
            formats: 내보낼 포맷 리스트
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(dataset, DatasetDict):
            for split, ds in dataset.items():
                for fmt in formats:
                    if fmt == 'json':
                        ds.to_json(output_path / f"{split}.json")
                    elif fmt == 'csv':
                        ds.to_csv(output_path / f"{split}.csv")
                    elif fmt == 'parquet':
                        ds.to_parquet(output_path / f"{split}.parquet")
        else:
            for fmt in formats:
                if fmt == 'json':
                    dataset.to_json(output_path / "dataset.json")
                elif fmt == 'csv':
                    dataset.to_csv(output_path / "dataset.csv")
                elif fmt == 'parquet':
                    dataset.to_parquet(output_path / "dataset.parquet")
        
        logger.info(f"데이터셋이 {output_dir}에 {formats} 포맷으로 내보내어졌습니다.")
