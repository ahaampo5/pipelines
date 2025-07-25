#!/usr/bin/env python3
"""
PII 마스킹 데이터 로더 사용 예시
"""

import sys
import os
import logging

# 현재 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader.pii_masking_ai4privacy import create_pii_data_loader, load_and_preprocess_pii_data

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """메인 실행 함수"""
    print("=== PII 마스킹 데이터 로더 테스트 ===\n")
    
    try:
        print("1. 기본 데이터 로더 생성 및 테스트")
        print("-" * 50)
        
        # 데이터 로더 생성 (한국어 BERT 토크나이저 사용)
        loader = create_pii_data_loader(
            dataset_name="ai4privacy/pii-masking-300k",  # 실제 데이터셋이 없으면 샘플 데이터 생성
            tokenizer_name="klue/bert-base",
            filter_pii_types=["PERSON", "EMAIL", "PHONE"],
            min_pii_count=1,
            max_pii_count=5,
            streaming=False
        )
        
        print(f"✓ 데이터 로더 생성 완료: {type(loader).__name__}")
        
        # 데이터셋 로드
        print("\n2. 데이터셋 로드 및 전처리")
        print("-" * 50)
        
        dataset = loader.get_dataset()
        print(f"✓ 데이터셋 로드 완료")
        
        # 데이터셋 정보 출력
        info = loader.get_dataset_info(dataset)
        print(f"✓ 데이터셋 정보: {info}")
        
        # PII 통계 출력
        stats = loader.get_pii_statistics(dataset)
        print(f"✓ PII 통계: {stats}")
        
        # 샘플 데이터 확인
        print("\n3. 샘플 데이터 확인")
        print("-" * 50)
        
        if hasattr(dataset, '__getitem__'):
            sample = dataset[0]
            print(f"샘플 데이터:")
            if isinstance(sample, dict):
                for key, value in sample.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  샘플: {sample}")
        
        # 토크나이제이션 테스트
        print("\n4. 토크나이제이션 테스트")
        print("-" * 50)
        
        try:
            tokenized_dataset = loader.apply_tokenization(dataset)
            print(f"✓ 토크나이제이션 완료")
            print(f"✓ 토크나이즈된 데이터셋 형태: {type(tokenized_dataset)}")
            
            if hasattr(tokenized_dataset, '__getitem__'):
                tokenized_sample = tokenized_dataset[0]
                print(f"토크나이즈된 샘플:")
                if isinstance(tokenized_sample, dict):
                    for key, value in tokenized_sample.items():
                        if isinstance(value, list) and len(value) > 10:
                            print(f"  {key}: {value[:10]}... (길이: {len(value)})")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"  샘플: {tokenized_sample}")
        except Exception as e:
            print(f"⚠ 토크나이제이션 건너뜀 (토크나이저 로드 실패): {e}")
        
        # PII 마스킹 쌍 생성 테스트
        print("\n5. PII 마스킹 쌍 생성 테스트")
        print("-" * 50)
        
        try:
            pair_dataset = loader.create_pii_mask_pairs(dataset)
            print(f"✓ PII 마스킹 쌍 생성 완료")
            
            if hasattr(pair_dataset, '__getitem__'):
                pair_sample = pair_dataset[0]
                print(f"마스킹 쌍 샘플:")
                if isinstance(pair_sample, dict):
                    for key, value in pair_sample.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  샘플: {pair_sample}")
        except Exception as e:
            print(f"⚠ PII 마스킹 쌍 생성 실패: {e}")
        
        # 원스톱 함수 테스트
        print("\n6. 원스톱 로딩 함수 테스트")
        print("-" * 50)
        
        try:
            quick_dataset = load_and_preprocess_pii_data(
                dataset_name="ai4privacy/pii-masking-300k",
                tokenizer_name="klue/bert-base",
                sample_size=10,
                apply_tokenization=False  # 토크나이저 로드 문제로 비활성화
            )
            print(f"✓ 원스톱 로딩 완료: {len(quick_dataset)}개 샘플")
        except Exception as e:
            print(f"⚠ 원스톱 로딩 실패: {e}")
        
        print("\n=== 테스트 완료 ===")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
