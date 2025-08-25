#!/usr/bin/env python3
"""
Bedrock Custom Model Deployment Script
Creates on-demand deployment for custom models
"""

import boto3
import json
import uuid
from datetime import datetime
import argparse
from typing import List, Dict, Optional


class BedrockModelDeployer:
    """Handles deployment of custom Bedrock models"""
    
    def __init__(self, region: str = 'us-east-1'):
        """
        Initialize the deployer
        
        Args:
            region: AWS region for Bedrock service
        """
        self.bedrock_client = boto3.client('bedrock', region_name=region)
        self.region = region
    
    def list_custom_models(self) -> List[Dict]:
        """
        List all available custom models
        
        Returns:
            List of custom model summaries
        """
        try:
            response = self.bedrock_client.list_custom_models()
            models = response['modelSummaries']
            
            print("📋 Available Custom Models:")
            print("-" * 80)
            
            if not models:
                print("No custom models found.")
                return []
            
            for i, model in enumerate(models, 1):
                print(f"{i}. Model Name: {model['modelName']}")
                print(f"   ARN: {model['modelArn']}")
                print(f"   Base Model: {model['baseModelArn']}")
                print(f"   Created: {model['creationTime']}")
                print("-" * 80)
            
            return models
            
        except Exception as e:
            print(f"❌ Error listing custom models: {e}")
            return []
    
    def create_deployment(self, model_arn: str, deployment_name: Optional[str] = None, 
                         description: Optional[str] = None) -> Optional[str]:
        """
        Create a deployment for the specified custom model
        
        Args:
            model_arn: ARN of the custom model to deploy
            deployment_name: Name for the deployment (auto-generated if not provided)
            description: Description for the deployment
            
        Returns:
            Deployment ARN if successful, None otherwise
        """
        
        # Generate deployment name if not provided
        if not deployment_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            deployment_name = f"bedrock-model-deployment-{timestamp}"
        
        # Generate description if not provided
        if not description:
            description = f"Custom model deployment created at {datetime.now().isoformat()}"
        
        try:
            print(f"🚀 Creating deployment: {deployment_name}")
            print(f"📦 Model ARN: {model_arn}")
            
            # Create the deployment
            response = self.bedrock_client.create_custom_model_deployment(
                modelDeploymentName=deployment_name,
                modelArn=model_arn,
                description=description,
                tags=[
                    {'key': 'Environment', 'value': 'Development'},
                    {'key': 'CreatedBy', 'value': 'BedrockModelDeployer'},
                    {'key': 'CreatedAt', 'value': datetime.now().isoformat()},
                    {'key': 'Region', 'value': self.region}
                ],
                clientRequestToken=str(uuid.uuid4())
            )
            
            deployment_arn = response['customModelDeploymentArn']
            
            print(f"✅ Deployment created successfully!")
            print(f"📍 Deployment ARN: {deployment_arn}")
            print(f"⏳ Status: Creating (deployment is in progress)")
            print(f"💡 Use this ARN as modelId for inference once deployment is complete")
            
            return deployment_arn
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return None
    
    def get_deployment_status(self, deployment_arn: str) -> Optional[str]:
        """
        Get the status of a deployment
        
        Args:
            deployment_arn: ARN of the deployment
            
        Returns:
            Deployment status if successful, None otherwise
        """
        try:
            response = self.bedrock_client.get_custom_model_deployment(
                customModelDeploymentArn=deployment_arn
            )
            return response['status']
        except Exception as e:
            print(f"❌ Error getting deployment status: {e}")
            return None


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Deploy custom Bedrock models for inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python run_model_deployment.py --list-models
  
  # Deploy a specific model
  python run_model_deployment.py --model-arn arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model
  
  # Deploy with custom name and description
  python run_model_deployment.py --model-arn arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model --name my-deployment --description "Production deployment"
        """
    )
    
    parser.add_argument('--model-arn', 
                       help='Custom model ARN to deploy')
    parser.add_argument('--name', 
                       help='Deployment name (auto-generated if not provided)')
    parser.add_argument('--description', 
                       help='Deployment description')
    parser.add_argument('--region', 
                       default='us-east-1',
                       help='AWS region (default: us-east-1)')
    parser.add_argument('--list-models', 
                       action='store_true', 
                       help='List available custom models and exit')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize deployer
    deployer = BedrockModelDeployer(region=args.region)
    
    # List models only
    if args.list_models:
        models = deployer.list_custom_models()
        print(f"\n📊 Found {len(models)} custom models in region {args.region}")
        return
    
    # Validate required arguments for deployment
    if not args.model_arn:
        print("❌ --model-arn is required for deployment")
        print("💡 Use --list-models to see available models")
        print("💡 Use --help for usage examples")
        return
    
    # Execute deployment
    print("🚀 Starting Custom Model Deployment")
    print("=" * 60)
    print(f"Region: {args.region}")
    print("=" * 60)
    
    deployment_arn = deployer.create_deployment(
        model_arn=args.model_arn,
        deployment_name=args.name,
        description=args.description
    )
    
    if deployment_arn:
        print("\n" + "=" * 60)
        print("✅ DEPLOYMENT INITIATED SUCCESSFULLY")
        print("=" * 60)
        print(f"Deployment ARN: {deployment_arn}")
        print(f"Region: {args.region}")
        print("\n📝 Next Steps:")
        print("1. Wait for deployment to complete (usually takes a few minutes)")
        print("2. Use the deployment ARN as modelId for inference")
        print("3. Check deployment status in AWS Console or via CLI")
        print("4. Monitor deployment logs for any issues")
    else:
        print("\n" + "=" * 60)
        print("❌ DEPLOYMENT FAILED")
        print("=" * 60)
        print("Please check the error messages above and try again")


if __name__ == "__main__":
    main()
