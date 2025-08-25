#!/usr/bin/env python3
"""
MMLU-Pro Benchmark Test for Nova Models
Configurable benchmark testing for various Bedrock models on MMLU-Pro dataset
"""

import json
import boto3
import time
import random
import argparse
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import csv
from dataclasses import dataclass


@dataclass
class MMLUProQuestion:
    """MMLU-Pro question data structure"""
    question_id: str
    question: str
    options: List[str]
    answer: str
    category: str
    difficulty: str


class MMLUProBenchmark:
    """MMLU-Pro benchmark runner for Bedrock models"""
    
    def __init__(self, region: str = 'us-east-1', custom_model_id: Optional[str] = None):
        """
        Initialize benchmark runner
        
        Args:
            region: AWS region for Bedrock service
            custom_model_id: Optional custom model ARN/ID
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region)
        self.region = region
        
        # Base model configurations
        self.base_models = {
            'nova-pro': {
                'model_id': 'us.amazon.nova-pro-v1:0',
                'name': 'Amazon Nova Pro'
            },
            'nova-lite': {
                'model_id': 'us.amazon.nova-lite-v1:0', 
                'name': 'Amazon Nova Lite'
            }
        }
        
        # Add custom model if provided
        if custom_model_id:
            self.base_models['custom-nova'] = {
                'model_id': custom_model_id,
                'name': f'Custom Nova Model ({custom_model_id.split("/")[-1] if "/" in custom_model_id else custom_model_id})'
            }
        
        self.models = self.base_models
        
    def load_mmlu_pro_sample(self, num_questions: int = 50, categories: Optional[List[str]] = None, 
                            random_seed: int = 42) -> List[MMLUProQuestion]:
        """
        Load MMLU-Pro dataset sample
        
        Args:
            num_questions: Number of questions to sample
            categories: Specific categories to include (None for all)
            random_seed: Random seed for reproducible sampling
            
        Returns:
            List of MMLU-Pro questions
        """
        print("📥 Loading MMLU-Pro dataset from Hugging Face...")
        
        try:
            from datasets import load_dataset
        except ImportError:
            print("❌ Error: 'datasets' library not found. Install with: pip install datasets")
            return []
        
        try:
            dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return []
 
        questions = []
        print(f"📊 Dataset loaded with {len(dataset)} total questions")
        
        # Filter by categories if specified
        if categories:
            filtered_indices = []
            for i, item in enumerate(dataset):
                if item.get('category', 'general').lower() in [cat.lower() for cat in categories]:
                    filtered_indices.append(i)
            print(f"🔍 Filtered to {len(filtered_indices)} questions in categories: {categories}")
        else:
            filtered_indices = list(range(len(dataset)))
        
        # Sample questions
        random.seed(random_seed)
        sample_size = min(num_questions, len(filtered_indices))
        selected_indices = random.sample(filtered_indices, sample_size)
        
        for i, idx in enumerate(selected_indices):
            item = dataset[idx]
            
            question = MMLUProQuestion(
                question_id=f"mmlu_pro_{i+1:03d}",
                question=item['question'],
                options=item['options'],
                answer=item['answer'],
                category=item.get('category', 'general'),
                difficulty=item.get('difficulty', 'medium')
            )
            
            questions.append(question)
        
        print(f"✅ Loaded {len(questions)} questions from MMLU-Pro dataset")
        
        # Print category distribution
        categories_count = {}
        for q in questions:
            categories_count[q.category] = categories_count.get(q.category, 0) + 1
        
        print("📂 Category distribution:")
        for cat, count in sorted(categories_count.items()):
            print(f"  • {cat}: {count} questions")
        
        return questions

    def generate_response(self, model_key: str, question: MMLUProQuestion, 
                         max_tokens: int = 1, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Generate response using specified model
        
        Args:
            model_key: Model identifier
            question: MMLU-Pro question
            max_tokens: Maximum tokens for response
            temperature: Temperature for generation
            
        Returns:
            Response dictionary with prediction and metadata
        """
        model_info = self.models[model_key]
        
        # Format options (A, B, C, D, E format)
        formatted_options = []
        for i, option in enumerate(question.options):
            letter = chr(65 + i)  # A, B, C, D, E
            if option.strip().startswith(f"{letter})") or option.strip().startswith(f"{letter}."):
                formatted_options.append(option.strip())
            else:
                formatted_options.append(f"{letter}) {option.strip()}")
        
        options_text = "\n".join(formatted_options)
        
        # Optimized prompt for multiple choice
        prompt = f"""You are an expert taking a multiple-choice exam. Please read the question carefully and select the best answer.

Question: {question.question}

Options:
{options_text}

Instructions:
1. Think step by step about the question
2. Consider each option carefully
3. Choose the most accurate answer
4. Respond with ONLY the letter (A, B, C, D, or E) - no explanation needed

Answer:"""

        try:
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": 1.0,
                    "stopSequences": ["\n", " ", ")", ".", ","]
                }
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=model_info['model_id'],
                body=json.dumps(payload),
                contentType='application/json'
            )
            
            result = json.loads(response['body'].read())
            generated_text = result['output']['message']['content'][0]['text'].strip()
            
            predicted_answer = self._extract_answer(generated_text)
            
            return {
                'raw_response': generated_text,
                'predicted_answer': predicted_answer,
                'correct_answer': question.answer,
                'is_correct': predicted_answer == question.answer,
                'error': None
            }
            
        except Exception as e:
            return {
                'raw_response': '',
                'predicted_answer': None,
                'correct_answer': question.answer,
                'is_correct': False,
                'error': str(e)
            }
    
    def _extract_answer(self, response_text: str) -> Optional[str]:
        """Extract A, B, C, D, E answer from response text"""
        if not response_text:
            return None
            
        response_text = response_text.upper().strip()
        valid_answers = ['A', 'B', 'C', 'D', 'E']
        
        # Exact single character match
        if len(response_text) == 1 and response_text in valid_answers:
            return response_text
        
        # Find first valid character
        for char in response_text:
            if char in valid_answers:
                return char
        
        # Pattern matching (A), A., A: etc)
        import re
        patterns = [
            r'^([A-E])\)',  # A)
            r'^([A-E])\.',  # A.
            r'^([A-E]):',   # A:
            r'([A-E])$',    # A at end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                return match.group(1)
        
        return None
    
    def run_benchmark_single_model(self, model_key: str, questions: List[MMLUProQuestion], 
                                  rate_limit_delay: float = 0.5) -> Dict[str, Any]:
        """
        Run benchmark for a single model
        
        Args:
            model_key: Model identifier
            questions: List of questions to test
            rate_limit_delay: Delay between API calls
            
        Returns:
            Benchmark results for the model
        """
        model_info = self.models[model_key]
        print(f"\n🤖 Testing {model_info['name']}...")
        print(f"📝 Total questions: {len(questions)}")
        
        results = []
        correct_count = 0
        category_stats = {}
        
        for i, question in enumerate(questions, 1):
            print(f"  🔄 Processing {i}/{len(questions)}: {question.category}...", end=' ')
            
            response = self.generate_response(model_key, question)
            
            result = {
                'question_id': question.question_id,
                'question': question.question,
                'category': question.category,
                'difficulty': question.difficulty,
                'options': question.options,
                'correct_answer': question.answer,
                'predicted_answer': response['predicted_answer'],
                'raw_response': response['raw_response'],
                'is_correct': response['is_correct'],
                'error': response['error']
            }
            results.append(result)
            
            # Update statistics
            if response['is_correct']:
                correct_count += 1
                print("✅")
            else:
                print("❌")
            
            # Category statistics
            if question.category not in category_stats:
                category_stats[question.category] = {'correct': 0, 'total': 0}
            category_stats[question.category]['total'] += 1
            if response['is_correct']:
                category_stats[question.category]['correct'] += 1
            
            # Rate limiting
            time.sleep(rate_limit_delay)
        
        # Calculate final statistics
        accuracy = correct_count / len(questions) if questions else 0
        
        category_accuracy = {}
        for category, stats in category_stats.items():
            category_accuracy[category] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        return {
            'model_name': model_info['name'],
            'model_key': model_key,
            'total_questions': len(questions),
            'correct_answers': correct_count,
            'accuracy': accuracy,
            'category_stats': category_stats,
            'category_accuracy': category_accuracy,
            'detailed_results': results
        }
    
    def run_comparative_benchmark(self, num_questions: int = 20, models_to_test: Optional[List[str]] = None,
                                 categories: Optional[List[str]] = None, random_seed: int = 42,
                                 rate_limit_delay: float = 0.5) -> Dict[str, Any]:
        """
        Run comparative benchmark across multiple models
        
        Args:
            num_questions: Number of questions to test
            models_to_test: List of model keys to test (None for all available)
            categories: Specific categories to test (None for all)
            random_seed: Random seed for reproducible results
            rate_limit_delay: Delay between API calls
            
        Returns:
            Complete benchmark results
        """
        print("🚀 Starting MMLU-Pro Comparative Benchmark")
        print("=" * 70)
        
        # Load questions
        questions = self.load_mmlu_pro_sample(num_questions, categories, random_seed)
        if not questions:
            print("❌ No questions loaded. Exiting.")
            return {}
        
        print(f"📚 Loaded {len(questions)} MMLU-Pro questions")
        
        # Determine models to test
        if models_to_test is None:
            models_to_test = list(self.models.keys())
        
        # Validate model keys
        invalid_models = [m for m in models_to_test if m not in self.models]
        if invalid_models:
            print(f"❌ Invalid model keys: {invalid_models}")
            print(f"Available models: {list(self.models.keys())}")
            return {}
        
        print(f"🤖 Testing models: {[self.models[m]['name'] for m in models_to_test]}")
        
        # Test each model
        all_results = {}
        for model_key in models_to_test:
            model_results = self.run_benchmark_single_model(model_key, questions, rate_limit_delay)
            all_results[model_key] = model_results
        
        # Generate comparison
        comparison = self._generate_comparison_report(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"mmlu_pro_benchmark_results_{timestamp}.json"
        
        final_results = {
            'metadata': {
                'benchmark': 'MMLU-Pro',
                'total_questions': len(questions),
                'models_tested': models_to_test,
                'categories_tested': categories or 'all',
                'random_seed': random_seed,
                'test_date': datetime.utcnow().isoformat(),
                'region': self.region
            },
            'individual_results': all_results,
            'comparison': comparison
        }
        
        # Save files
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        csv_file = output_file.replace('.json', '_summary.csv')
        self._save_csv_summary(all_results, csv_file)
        
        print(f"\n💾 Results saved to:")
        print(f"  - Detailed: {output_file}")
        print(f"  - Summary: {csv_file}")
        
        return final_results
    
    def _generate_comparison_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison report from all results"""
        # Overall accuracy ranking
        accuracy_ranking = sorted(
            [(k, v['accuracy']) for k, v in all_results.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Category performance comparison
        categories = set()
        for results in all_results.values():
            categories.update(results['category_accuracy'].keys())
        
        category_comparison = {}
        for category in categories:
            category_scores = {}
            for model_key, results in all_results.items():
                category_scores[model_key] = results['category_accuracy'].get(category, 0.0)
            category_comparison[category] = category_scores
        
        return {
            'overall_ranking': accuracy_ranking,
            'category_comparison': category_comparison,
            'performance_summary': {
                model_key: {
                    'accuracy': results['accuracy'],
                    'correct_answers': results['correct_answers'],
                    'total_questions': results['total_questions']
                }
                for model_key, results in all_results.items()
            }
        }
    
    def _save_csv_summary(self, all_results: Dict[str, Any], csv_file: str):
        """Save CSV summary file"""
        # Get all categories
        all_categories = set()
        for results in all_results.values():
            all_categories.update(results['category_accuracy'].keys())
        
        sorted_categories = sorted(all_categories)
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Model', 'Total Questions', 'Correct Answers', 'Accuracy (%)']
            header.extend([cat.replace('_', ' ').title() for cat in sorted_categories])
            writer.writerow(header)
            
            # Data rows
            for model_key, results in all_results.items():
                row = [
                    results['model_name'],
                    results['total_questions'],
                    results['correct_answers'],
                    f"{results['accuracy']*100:.1f}%"
                ]
                
                # Add category accuracies
                for category in sorted_categories:
                    if category in results['category_accuracy']:
                        row.append(f"{results['category_accuracy'][category]*100:.1f}%")
                    else:
                        row.append("N/A")
                
                writer.writerow(row)
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print results summary to console"""
        print("\n" + "="*70)
        print("📊 MMLU-Pro BENCHMARK RESULTS")
        print("="*70)
        
        # Overall ranking
        ranking = results['comparison']['overall_ranking']
        print("\n🏆 Overall Ranking:")
        for i, (model_key, accuracy) in enumerate(ranking, 1):
            model_name = results['individual_results'][model_key]['model_name']
            print(f"  {i}. {model_name}: {accuracy*100:.1f}%")
        
        # Detailed performance
        print(f"\n📈 Detailed Performance:")
        for model_key, model_results in results['individual_results'].items():
            print(f"\n{model_results['model_name']}:")
            print(f"  - Accuracy: {model_results['accuracy']*100:.1f}% ({model_results['correct_answers']}/{model_results['total_questions']})")
            if model_results['category_accuracy']:
                print(f"  - Category Performance:")
                for category, accuracy in model_results['category_accuracy'].items():
                    print(f"    • {category.replace('_', ' ').title()}: {accuracy*100:.1f}%")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='MMLU-Pro Benchmark for Bedrock Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with 20 questions
  python run_mmlu_pro_benchmark.py --num-questions 20
  
  # Test specific models
  python run_mmlu_pro_benchmark.py --models nova-pro nova-lite --num-questions 50
  
  # Test with custom model
  python run_mmlu_pro_benchmark.py --custom-model-id arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/abc123
  
  # Test specific categories
  python run_mmlu_pro_benchmark.py --categories mathematics physics --num-questions 30
        """
    )
    
    parser.add_argument('--num-questions', type=int, default=20,
                       help='Number of questions to test (default: 20)')
    parser.add_argument('--models', nargs='+', choices=['nova-pro', 'nova-lite', 'custom-nova'],
                       help='Models to test (default: all available)')
    parser.add_argument('--custom-model-id',
                       help='Custom model ARN/ID for testing')
    parser.add_argument('--categories', nargs='+',
                       help='Specific categories to test (default: all)')
    parser.add_argument('--region', default='us-east-1',
                       help='AWS region (default: us-east-1)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducible results (default: 42)')
    parser.add_argument('--rate-limit-delay', type=float, default=0.5,
                       help='Delay between API calls in seconds (default: 0.5)')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Validate custom model requirements
    if args.models and 'custom-nova' in args.models and not args.custom_model_id:
        print("❌ --custom-model-id is required when testing custom-nova model")
        return
    
    # Initialize benchmark
    benchmark = MMLUProBenchmark(
        region=args.region,
        custom_model_id=args.custom_model_id
    )
    
    # Run benchmark
    results = benchmark.run_comparative_benchmark(
        num_questions=args.num_questions,
        models_to_test=args.models,
        categories=args.categories,
        random_seed=args.random_seed,
        rate_limit_delay=args.rate_limit_delay
    )
    
    if results:
        # Print results
        benchmark.print_results_summary(results)
        print("\n🎉 Benchmark completed successfully!")
    else:
        print("\n❌ Benchmark failed!")


if __name__ == "__main__":
    main()
