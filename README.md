# Amazon Bedrock Nova 모델 Customization 툴킷

Amazon Bedrock Nova 모델 Customization 툴킷입니다. 모델 증류(distillation), 배포, 평가, 벤치마킹 기능을 제공합니다.

## 🚀 주요 기능

- **모델 증류**: Nova Premier에서 Nova Lite로 지식 증류를 통한 커스텀 모델 생성
- **모델 배포**: 추론을 위한 커스텀 모델 온디맨드 배포
- **모델 평가**: LLM-as-a-Judge 방법론을 활용한 모델 성능 평가
- **설정 가능한 매개변수**: 모든 스크립트가 명령행 인자를 통한 유연한 설정 지원

## 📋 사전 요구사항

이 툴킷을 사용하기 전에 다음 사항들을 준비해주세요:

- 적절한 권한을 가진 활성 AWS 계정
- 선호하는 리전에서 Amazon Bedrock 액세스 활성화
- Python 3.8+ 설치
- 적절한 자격 증명으로 구성된 AWS CLI
- Bedrock 모델 커스터마이제이션 및 배포를 위한 충분한 서비스 할당량

### 필요한 AWS 권한

AWS IAM 역할/사용자에 다음 권한이 필요합니다:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:*",
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket",
                "s3:CreateBucket",
                "iam:CreateRole",
                "iam:CreatePolicy",
                "iam:AttachRolePolicy",
                "iam:PassRole",
                "sts:GetCallerIdentity"
            ],
            "Resource": "*"
        }
    ]
}
```

## 🛠️ 설치

1. 이 저장소를 클론합니다:
```bash
git clone https://github.com/takkielshim/nova-distillation.git
cd nova-distillation
```

2. 필요한 Python 패키지를 설치합니다:
```bash
pip install boto3 pandas datasets
```

3. AWS 자격 증명을 구성합니다:
```bash
aws configure
```

## 📁 프로젝트 구조

```
├── README.md                     # 본 파일
├── run_distillation_job.py       # 모델 증류 스크립트
├── run_model_deployment.py       # 모델 배포 스크립트
├── run_evaluation.py             # 모델 평가 실행기
├── llm_judge_evaluation.py       # LLM-as-a-Judge 평가 프레임워크
├── utils.py                      # AWS 작업을 위한 유틸리티 함수
├── train.jsonl                   # 증류용 샘플 훈련 데이터
└── test.jsonl                    # 평가용 샘플 테스트 데이터
```

## 🎯 사용 예제

### 1. 모델 증류

지식 증류를 사용하여 커스텀 Nova 모델을 생성합니다:

```bash
# 기본 설정으로 증류 실행
python run_distillation_job.py --bucket-name your-s3-bucket

# 커스텀 매개변수로 고급 증류
python run_distillation_job.py \
  --bucket-name your-s3-bucket \
  --input-file custom_training_data.jsonl \
  --teacher-model us.amazon.nova-premier-v1:0 \
  --student-model amazon.nova-lite-v1:0:300k \
  --region us-east-1 \
  --max-response-length 1000
```

### 2. 모델 배포

추론을 위한 커스텀 모델을 배포합니다:

```bash
# 사용 가능한 커스텀 모델 목록 조회
python run_model_deployment.py --list-models

# 특정 모델 배포
python run_model_deployment.py \
  --model-arn arn:aws:bedrock:us-east-1:123456789012:custom-model/your-model-id

# 커스텀 이름과 설명으로 배포
python run_model_deployment.py \
  --model-arn arn:aws:bedrock:us-east-1:123456789012:custom-model/your-model-id \
  --name production-deployment \
  --description "고객 서비스용 프로덕션 배포"
```

### 3. 모델 평가

LLM-as-a-Judge 방법론을 사용하여 모델 성능을 평가합니다:

```bash
# 모든 사용 가능한 모델 평가
python run_evaluation.py --file test.jsonl

# 특정 모델 평가
python run_evaluation.py --model pro --file test.jsonl

# 커스텀 모델 평가
python run_evaluation.py \
  --model custom \
  --custom-model-id arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/abc123 \
  --file test.jsonl

# 커스텀 모델을 포함한 모든 모델 비교
python run_evaluation.py \
  --model all \
  --custom-model-id arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/abc123 \
  --region us-west-2
```


## 📊 데이터 형식

### 훈련 데이터 (JSONL)

모델 증류를 위해 다음 구조의 JSONL 형식을 사용합니다:

```json
{"prompt": "프랑스의 수도는 어디인가요?", "completion": "프랑스의 수도는 파리입니다."}
{"prompt": "광합성에 대해 설명해주세요", "completion": "광합성은 식물이 햇빛을 에너지로 변환하는 과정입니다..."}
```

### 평가 데이터 (JSONL)

모델 평가를 위해 JSONL 형식을 사용합니다:

```json
{"prompt": "재생 에너지의 장점은 무엇인가요?"}
{"prompt": "머신러닝은 어떻게 작동하나요?"}
```

## 🔧 설정 옵션

### 모델 증류 옵션

| 매개변수 | 설명 | 기본값 |
|---------|------|--------|
| `--bucket-name` | 훈련 데이터와 출력을 위한 S3 버킷 | 필수 |
| `--input-file` | 로컬 훈련 데이터 파일 | `train.jsonl` |
| `--teacher-model` | 교사 모델 식별자 | `us.amazon.nova-premier-v1:0` |
| `--student-model` | 학생 모델 식별자 | `amazon.nova-lite-v1:0:300k` |
| `--region` | AWS 리전 | `us-east-1` |
| `--max-response-length` | 최대 응답 길이 | `1000` |

### 평가 옵션

| 매개변수 | 설명 | 기본값 |
|---------|------|--------|
| `--model` | 평가할 모델 (pro/lite/custom/all) | `all` |
| `--file` | 평가 데이터 파일 | `test.jsonl` |
| `--custom-model-id` | 커스텀 모델 ARN/ID | None |
| `--region` | AWS 리전 | `us-east-1` |

## 📈 출력 파일

툴킷은 다양한 출력 파일을 생성합니다:

- **증류**: Job ARN 및 상태 정보
- **배포**: 추론을 위한 배포 ARN
- **평가**: 
  - `evaluation_results_YYYYMMDD_HHMMSS.json` - 상세 결과
  - `evaluation_results_YYYYMMDD_HHMMSS_summary.csv` - 요약 통계


**참고**: 이 툴킷은 교육 및 개발 목적으로 제작되었습니다. 프로덕션 환경에서 사용할 때는 AWS 모범 사례와 보안 가이드라인을 따르세요.
