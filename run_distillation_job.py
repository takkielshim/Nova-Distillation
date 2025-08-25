import json
import os
import boto3
import argparse
from datetime import datetime
from botocore.exceptions import ClientError
from utils import create_s3_bucket, upload_training_data_to_s3, delete_s3_bucket_and_contents, \
create_model_distillation_role_and_permissions, delete_role_and_attached_policies, delete_distillation_buckets


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run model distillation job using Amazon Bedrock')
    
    parser.add_argument('--bucket-name', required=True, 
                       help='S3 bucket name for storing training data and outputs')
    parser.add_argument('--input-file', default='train.jsonl',
                       help='Local training data file (default: train.jsonl)')
    parser.add_argument('--data-prefix', default='distill',
                       help='S3 prefix for training data (default: distill)')
    parser.add_argument('--teacher-model', default='us.amazon.nova-premier-v1:0',
                       help='Teacher model identifier (default: us.amazon.nova-premier-v1:0)')
    parser.add_argument('--student-model', default='amazon.nova-lite-v1:0:300k',
                       help='Student model identifier (default: amazon.nova-lite-v1:0:300k)')
    parser.add_argument('--region', default='us-east-1',
                       help='AWS region (default: us-east-1)')
    parser.add_argument('--max-response-length', type=int, default=1000,
                       help='Maximum response length for inference (default: 1000)')
    parser.add_argument('--job-name-prefix', default='distillation-job',
                       help='Job name prefix (default: distillation-job)')
    parser.add_argument('--model-name-prefix', default='nova-premier-lite-model',
                       help='Model name prefix (default: nova-premier-lite-model)')
    
    return parser.parse_args()


def run_distillation(args):
    """Run the model distillation process"""
    
    # Create Bedrock client
    bedrock_client = boto3.client(service_name="bedrock", region_name=args.region)
    
    # Create runtime client for inference
    bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=args.region)
    
    # Region and accountID
    session = boto3.session.Session()
    sts_client = session.client('sts', region_name=args.region)
    account_id = sts_client.get_caller_identity()['Account']
    
    # Generate unique names for the job and model
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    job_name = f"{args.job_name_prefix}-{timestamp}"
    model_name = f"{args.model_name_prefix}-{timestamp}"
    
    print(f"Starting distillation job: {job_name}")
    print(f"Teacher model: {args.teacher_model}")
    print(f"Student model: {args.student_model}")
    print(f"Training data file: {args.input_file}")
    print(f"S3 bucket: {args.bucket_name}")
    
    # Configure models and IAM role
    role_name, role_arn = create_model_distillation_role_and_permissions(
        bucket_name=args.bucket_name, 
        account_id=account_id
    )
    
    # Creating training data bucket
    create_s3_bucket(bucket_name=args.bucket_name)
    
    # Specify S3 locations
    training_data = upload_training_data_to_s3(
        args.bucket_name, 
        args.input_file, 
        prefix=args.data_prefix
    )
    output_path = f"s3://{args.bucket_name}/output/"
    
    print(f"Training data uploaded to: {training_data}")
    print(f"Output will be saved to: {output_path}")
    
    # Create model customization job
    response = bedrock_client.create_model_customization_job(
        jobName=job_name,
        customModelName=model_name,
        roleArn=role_arn,
        baseModelIdentifier=args.student_model,
        customizationType="DISTILLATION",
        trainingDataConfig={
            "s3Uri": training_data
        },
        outputDataConfig={
            "s3Uri": output_path
        },
        customizationConfig={
            "distillationConfig": {
                "teacherModelConfig": {
                    "teacherModelIdentifier": args.teacher_model,
                    "maxResponseLengthForInference": args.max_response_length 
                }
            }
        }
    )
    
    job_arn = response['jobArn']
    print(f"Distillation job created with ARN: {job_arn}")
    
    # Print job status
    job_status = bedrock_client.get_model_customization_job(jobIdentifier=job_arn)["status"]
    print(f"Job status: {job_status}")
    
    return job_arn, job_status


def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        job_arn, status = run_distillation(args)
        print(f"Distillation job started successfully!")
        print(f"Job ARN: {job_arn}")
        print(f"Current status: {status}")
        
    except Exception as e:
        print(f"Error running distillation job: {str(e)}")
        raise


if __name__ == "__main__":
    main()
