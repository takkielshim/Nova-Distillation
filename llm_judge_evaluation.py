#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation Script
Configurable generator and evaluator models for performance assessment
"""

import json
import boto3
import time
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import csv


class LLMJudgeEvaluator:
    """Evaluates model responses using LLM-as-a-Judge methodology"""
    
    def __init__(self, region: str = 'us-east-1', generator_model: str = 'nova-pro', 
                 evaluator_model: str = 'claude-sonnet-3.7', custom_model_id: Optional[str] = None,
                 max_tokens: int = 400, temperature: float = 0.3, top_p: float = 0.8):
        """
        Initialize the evaluator
        
        Args:
            region: AWS region for Bedrock service
            generator_model: Model to generate responses ('nova-pro', 'nova-lite', 'nova-custom')
            evaluator_model: Model to evaluate responses ('claude-sonnet-3.7')
            custom_model_id: ARN/ID for custom model (required if generator_model is 'nova-custom')
            max_tokens: Maximum tokens for response generation
            temperature: Temperature for response generation
            top_p: Top-p for response generation
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region)
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Configure generator model
        self._setup_generator_model(generator_model, custom_model_id)
        
        # Configure evaluator model
        self._setup_evaluator_model(evaluator_model)
        
    def _setup_generator_model(self, generator_model: str, custom_model_id: Optional[str]):
        """Setup generator model configuration"""
        generator_model = generator_model.lower()
        
        if generator_model == 'nova-pro':
            self.nova_model_id = 'us.amazon.nova-pro-v1:0'
            self.generator_name = 'Amazon Nova Pro'
        elif generator_model == 'nova-lite':
            self.nova_model_id = 'us.amazon.nova-lite-v1:0'
            self.generator_name = 'Amazon Nova Lite'
        elif generator_model in ['nova-custom', 'custom']:
            if not custom_model_id:
                raise ValueError("custom_model_id is required when using custom generator model")
            self.nova_model_id = custom_model_id
            self.generator_name = f'Custom Nova Model ({custom_model_id.split("/")[-1] if "/" in custom_model_id else custom_model_id})'
        else:
            raise ValueError(f"Unsupported generator model: {generator_model}. Use 'nova-pro', 'nova-lite', or 'nova-custom'")
        
        self.generator_model = generator_model
    
    def _setup_evaluator_model(self, evaluator_model: str):
        """Setup evaluator model configuration"""
        evaluator_model = evaluator_model.lower()
        
        if evaluator_model in ['claude-sonnet-3.7', 'sonnet-3.7', 'claude-sonnet']:
            self.evaluator_model_id = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'
            self.evaluator_name = 'Claude Sonnet 3.7'
        else:
            raise ValueError(f"Unsupported evaluator model: {evaluator_model}. Currently only 'claude-sonnet-3.7' is supported")
    
    def load_prompts(self, file_path: str) -> List[str]:
        """
        Load prompts from JSONL file
        
        Args:
            file_path: Path to the JSONL file containing prompts
            
        Returns:
            List of prompt strings
        """
        prompts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if 'prompt' in data:
                                prompts.append(data['prompt'])
                            else:
                                print(f"Warning: Line {line_num} missing 'prompt' field")
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON on line {line_num}: {e}")
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return []
        except Exception as e:
            print(f"Error loading prompts: {e}")
            return []
        
        return prompts
    
    def generate_response_nova(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate response using Amazon Nova models
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for response formatting
            
        Returns:
            Generated response text
        """
        try:
            # Default system prompt for Korean e-commerce responses
            if system_prompt is None:
                system_prompt = """다음 질문에 대해 간결하면서도 완전한 답변을 작성해주세요.

중요한 지침:
- 답변은 150-200글자 정도로 작성
- 반드시 완전한 문장으로 마무리
- 핵심 정보만 포함하여 간결하게 작성
- 문장이 중간에 끊기지 않도록 주의

질문: {prompt}

완성된 답변:"""
            
            # Format the prompt
            formatted_prompt = system_prompt.format(prompt=prompt)
            
            # Nova request payload
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": formatted_prompt}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                    "topP": self.top_p
                }
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=self.nova_model_id,
                body=json.dumps(payload),
                contentType='application/json'
            )
            
            result = json.loads(response['body'].read())
            generated_response = result['output']['message']['content'][0]['text'].strip()
            
            # Post-process response to ensure completeness
            generated_response = self._post_process_response(generated_response)
            
            return generated_response
            
        except Exception as e:
            print(f"Error generating with {self.generator_name}: {e}")
            return f"Error: {str(e)}"
    
    def _post_process_response(self, response: str, max_length: int = 250) -> str:
        """
        Post-process generated response to ensure completeness and appropriate length
        
        Args:
            response: Raw generated response
            max_length: Maximum allowed response length
            
        Returns:
            Post-processed response
        """
        # Check if response ends properly
        proper_endings = ['.', '!', '?', '다', '습니다', '됩니다', '있습니다']
        
        if not any(response.endswith(ending) for ending in proper_endings):
            # Try to find last complete sentence
            for delimiter in ['.', '!', '?']:
                if delimiter in response:
                    parts = response.split(delimiter)
                    if len(parts) > 1:
                        # Keep complete sentences only
                        complete_parts = parts[:-1] if len(parts[-1].strip()) < 20 else parts
                        response = delimiter.join(complete_parts)
                        if not response.endswith(delimiter):
                            response += delimiter
                        break
            
            # Check for Korean sentence endings
            korean_endings = ['다.', '습니다.', '됩니다.', '있습니다.', '없습니다.', '합니다.']
            for ending in korean_endings:
                if ending in response:
                    idx = response.rfind(ending)
                    if idx != -1:
                        response = response[:idx + len(ending)]
                        break
            else:
                # If no proper ending found, add period
                if len(response) > 50:
                    words = response.split()
                    if len(words) > 10:
                        target_length = min(200, len(response))
                        truncated = ""
                        for word in words:
                            if len(truncated + word + ' ') <= target_length - 10:
                                truncated += word + ' '
                            else:
                                break
                        response = truncated.strip() + '.'
        
        # Truncate if too long
        if len(response) > max_length:
            sentences = []
            current_length = 0
            
            for delimiter in ['.', '!', '?']:
                if delimiter in response:
                    parts = response.split(delimiter)
                    for part in parts[:-1]:  # Exclude last incomplete part
                        sentence = part + delimiter
                        if current_length + len(sentence) <= max_length - 20:
                            sentences.append(sentence)
                            current_length += len(sentence)
                        else:
                            break
                    break
            
            if sentences:
                response = ''.join(sentences)
        
        return response
    
    def evaluate_with_claude(self, prompt: str, response: str, 
                           evaluation_criteria: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Evaluate response using Claude Sonnet
        
        Args:
            prompt: Original prompt
            response: Generated response to evaluate
            evaluation_criteria: Custom evaluation criteria
            
        Returns:
            Evaluation results dictionary
        """
        # Default evaluation criteria
        if evaluation_criteria is None:
            evaluation_criteria = {
                "accuracy": "답변의 사실적 정확성과 질문 이해도",
                "completeness": "답변이 충분히 상세하고 완전한가",
                "relevance": "답변이 질문과 관련이 있는가",
                "helpfulness": "답변이 사용자에게 실질적으로 도움이 되는가",
                "language_quality": "한국어 표현이 자연스럽고 적절한가"
            }
        
        # Build evaluation prompt
        criteria_text = ""
        for i, (criterion, description) in enumerate(evaluation_criteria.items(), 1):
            criteria_text += f"{i}. **{criterion.replace('_', ' ').title()} ({criterion.title()})**: {description} (1-5점)\n"
            criteria_text += self._get_scoring_guide(criterion) + "\n\n"
        
        evaluation_prompt = f"""
다음은 한국어 쇼핑/전자상거래 관련 질문과 AI의 답변입니다. 이 답변을 다음 기준으로 평가해주세요:

**원본 질문:** {prompt}

**AI 답변:** {response}

**평가 기준:**
{criteria_text}

**응답 형식:**
```json
{{
    {', '.join([f'"{k}": <점수>' for k in evaluation_criteria.keys()])},
    "overall_score": <전체 평균 점수>,
    "feedback": "<구체적인 피드백 및 개선점>"
}}
```

JSON 형식으로만 응답해주세요.
"""
        
        try:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0.3,
                "top_p": 0.9,
                "messages": [
                    {
                        "role": "user",
                        "content": evaluation_prompt
                    }
                ]
            }
            
            response_obj = self.bedrock_client.invoke_model(
                modelId=self.evaluator_model_id,
                body=json.dumps(payload),
                contentType='application/json'
            )
            
            result = json.loads(response_obj['body'].read())
            evaluation_text = result['content'][0]['text']
            
            # Extract JSON part
            start_idx = evaluation_text.find('{')
            end_idx = evaluation_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = evaluation_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {"error": "Could not parse evaluation JSON"}
                
        except Exception as e:
            print(f"Error evaluating with {self.evaluator_name}: {e}")
            return {"error": str(e)}
    
    def _get_scoring_guide(self, criterion: str) -> str:
        """Get scoring guide for specific criterion"""
        guides = {
            "accuracy": """   - 5점: 질문을 완벽히 이해하고 사실적으로 정확한 정보 제공
   - 4점: 질문을 잘 이해하고 대부분 정확한 정보 제공, 사소한 오류 있을 수 있음
   - 3점: 질문을 어느 정도 이해하고 부분적으로 정확한 정보 제공
   - 2점: 질문 이해가 부족하거나 부정확한 정보가 많음
   - 1점: 질문을 잘못 이해했거나 명백히 잘못된 정보 제공""",
            
            "completeness": """   - 5점: 질문에 대한 모든 측면을 포괄적으로 다룸
   - 4점: 주요 내용을 잘 다루지만 일부 세부사항 누락
   - 3점: 기본적인 내용은 포함하지만 불완전함
   - 2점: 중요한 정보가 많이 누락됨
   - 1점: 매우 불완전하거나 핵심 내용 누락""",
            
            "relevance": """   - 5점: 질문과 완전히 관련된 내용만 포함
   - 4점: 대부분 관련성 있지만 일부 불필요한 내용 포함
   - 3점: 어느 정도 관련성 있지만 주제에서 벗어난 부분 있음
   - 2점: 관련성이 낮거나 주제와 맞지 않는 내용이 많음
   - 1점: 질문과 거의 관련 없는 답변""",
            
            "helpfulness": """   - 5점: 사용자가 원하는 정보를 명확하고 실용적으로 제공
   - 4점: 대체로 유용하지만 일부 개선 여지 있음
   - 3점: 어느 정도 도움이 되지만 제한적임
   - 2점: 도움이 되는 정도가 낮음
   - 1점: 거의 도움이 되지 않음""",
            
            "language_quality": """   - 5점: 완벽한 한국어 표현, 자연스럽고 이해하기 쉬움
   - 4점: 대체로 자연스럽지만 사소한 어색함 있을 수 있음
   - 3점: 이해 가능하지만 어색한 표현이나 문법 오류 있음
   - 2점: 부자연스러운 표현이 많고 이해하기 어려운 부분 있음
   - 1점: 매우 어색하거나 이해하기 어려운 한국어"""
        }
        
        return guides.get(criterion, "   - 1-5점 척도로 평가")
    
    def run_evaluation(self, eval_file_path: str, output_file: Optional[str] = None, 
                      rate_limit_delay: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Run complete evaluation pipeline
        
        Args:
            eval_file_path: Path to evaluation file
            output_file: Output file path (auto-generated if None)
            rate_limit_delay: Delay between API calls to avoid rate limits
            
        Returns:
            Evaluation results dictionary
        """
        print("🚀 Starting LLM-as-a-Judge Evaluation")
        print(f"Generator: {self.generator_name}")
        print(f"Evaluator: {self.evaluator_name}")
        print(f"Region: {self.region}")
        print("-" * 60)
        
        # Load prompts
        prompts = self.load_prompts(eval_file_path)
        if not prompts:
            print("❌ No prompts loaded. Exiting.")
            return None
        
        print(f"📝 Loaded {len(prompts)} prompts")
        
        results = []
        total_scores = {
            'accuracy': 0,
            'completeness': 0,
            'relevance': 0,
            'helpfulness': 0,
            'language_quality': 0,
            'overall_score': 0
        }
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n🔄 Processing {i}/{len(prompts)}: {prompt[:50]}...")
            
            # Generate response
            print(f"  📤 Generating response with {self.generator_name}...")
            response = self.generate_response_nova(prompt)
            
            # Rate limiting
            time.sleep(rate_limit_delay)
            
            # Evaluate response
            print(f"  📊 Evaluating with {self.evaluator_name}...")
            evaluation = self.evaluate_with_claude(prompt, response)
            
            # Rate limiting
            time.sleep(rate_limit_delay)
            
            result = {
                'prompt_id': i,
                'prompt': prompt,
                'response': response,
                'evaluation': evaluation,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            results.append(result)
            
            # Accumulate scores if evaluation was successful
            if 'error' not in evaluation:
                for key in total_scores:
                    if key in evaluation:
                        total_scores[key] += evaluation[key]
                
                print(f"  ✅ Overall Score: {evaluation.get('overall_score', 'N/A')}/5")
            else:
                print(f"  ❌ Evaluation Error: {evaluation['error']}")
        
        # Calculate averages
        valid_evaluations = len([r for r in results if 'error' not in r['evaluation']])
        if valid_evaluations > 0:
            avg_scores = {k: v / valid_evaluations for k, v in total_scores.items()}
        else:
            avg_scores = {k: 0 for k in total_scores}
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.generator_model.replace('-', '_')
            output_file = f"evaluation_results_{model_name}_{timestamp}.json"
        
        # Prepare final results
        final_results = {
            'metadata': {
                'generator_model': self.nova_model_id,
                'generator_name': self.generator_name,
                'evaluator_model': self.evaluator_model_id,
                'evaluator_name': self.evaluator_name,
                'region': self.region,
                'total_prompts': len(prompts),
                'valid_evaluations': valid_evaluations,
                'evaluation_date': datetime.utcnow().isoformat(),
                'parameters': {
                    'max_tokens': self.max_tokens,
                    'temperature': self.temperature,
                    'top_p': self.top_p,
                    'rate_limit_delay': rate_limit_delay
                }
            },
            'average_scores': avg_scores,
            'detailed_results': results
        }
        
        # Save results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            
            # Save CSV summary
            csv_file = output_file.replace('.json', '_summary.csv')
            self.save_csv_summary(results, csv_file)
            
            # Print summary
            self._print_summary(len(prompts), valid_evaluations, avg_scores, output_file, csv_file)
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return final_results
    
    def _print_summary(self, total_prompts: int, valid_evaluations: int, 
                      avg_scores: Dict[str, float], output_file: str, csv_file: str):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("📊 EVALUATION SUMMARY")
        print("="*70)
        print(f"Generator: {self.generator_name}")
        print(f"Evaluator: {self.evaluator_name}")
        print(f"Total Prompts: {total_prompts}")
        print(f"Valid Evaluations: {valid_evaluations}")
        print(f"Success Rate: {valid_evaluations/total_prompts*100:.1f}%")
        print("\n📈 Average Scores:")
        for metric, score in avg_scores.items():
            print(f"  {metric.replace('_', ' ').title()}: {score:.2f}/5.0")
        
        print(f"\n💾 Results saved to:")
        print(f"  - Detailed: {output_file}")
        print(f"  - Summary: {csv_file}")
    
    def save_csv_summary(self, results: List[Dict], csv_file: str):
        """Save evaluation summary to CSV"""
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'Prompt ID', 'Prompt', 'Response Length', 
                    'Accuracy', 'Completeness', 'Relevance', 
                    'Helpfulness', 'Language Quality', 'Overall Score',
                    'Has Error', 'Feedback'
                ])
                
                # Data rows
                for result in results:
                    evaluation = result['evaluation']
                    has_error = 'error' in evaluation
                    
                    writer.writerow([
                        result['prompt_id'],
                        result['prompt'][:100] + '...' if len(result['prompt']) > 100 else result['prompt'],
                        len(result['response']),
                        evaluation.get('accuracy', 'N/A'),
                        evaluation.get('completeness', 'N/A'),
                        evaluation.get('relevance', 'N/A'),
                        evaluation.get('helpfulness', 'N/A'),
                        evaluation.get('language_quality', 'N/A'),
                        evaluation.get('overall_score', 'N/A'),
                        has_error,
                        evaluation.get('feedback', '')[:200] + '...' if len(evaluation.get('feedback', '')) > 200 else evaluation.get('feedback', '')
                    ])
        except Exception as e:
            print(f"Warning: Could not save CSV summary: {e}")


def main():
    """Main execution function for standalone usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM-as-a-Judge Evaluation')
    parser.add_argument('--file', default='test.jsonl', help='Evaluation file path')
    parser.add_argument('--generator', default='nova-pro', choices=['nova-pro', 'nova-lite', 'nova-custom'],
                       help='Generator model')
    parser.add_argument('--custom-model-id', help='Custom model ID for nova-custom')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    # Validate custom model requirements
    if args.generator == 'nova-custom' and not args.custom_model_id:
        print("❌ --custom-model-id is required when using nova-custom generator")
        return
    
    # Initialize evaluator
    evaluator = LLMJudgeEvaluator(
        region=args.region,
        generator_model=args.generator,
        custom_model_id=args.custom_model_id
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(args.file, args.output)
    
    if results:
        print("\n🎉 Evaluation completed successfully!")
    else:
        print("\n❌ Evaluation failed!")


if __name__ == "__main__":
    main()
