#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation Runner
Supports evaluation of different models with configurable parameters
"""

from llm_judge_evaluation import LLMJudgeEvaluator
import argparse
import sys
import os


def run_single_evaluation(model_name, eval_file, region='us-east-1', custom_model_id=None):
    """Run evaluation for a single model"""
    print(f"🤖 LLM-as-a-Judge Evaluation")
    
    # Determine generator model configuration
    if model_name == 'custom':
        if not custom_model_id:
            print("❌ Custom model ID is required when using 'custom' model")
            return None
        print(f"Generator: Custom Model ({custom_model_id})")
        generator_model = 'nova-custom'
    elif model_name == 'pro':
        print(f"Generator: Amazon Nova Pro")
        generator_model = 'nova-pro'
    elif model_name == 'lite':
        print(f"Generator: Amazon Nova Lite")
        generator_model = 'nova-lite'
    else:
        print(f"❌ Unknown model: {model_name}")
        return None
    
    print(f"Evaluator: Claude Sonnet 3.7")
    print(f"Region: {region}")
    print(f"Evaluation file: {eval_file}")
    print("-" * 50)
    
    # Initialize evaluator
    evaluator = LLMJudgeEvaluator(
        region=region, 
        generator_model=generator_model,
        custom_model_id=custom_model_id
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(eval_file)
    
    if results:
        avg_score = results['average_scores']['overall_score']
        print(f"\n🏆 Final Average Score: {avg_score:.2f}/5.0")
        
        # Show top and bottom performing prompts
        detailed = results['detailed_results']
        valid_results = [r for r in detailed if 'error' not in r['evaluation']]
        
        if valid_results:
            # Sort by overall score
            sorted_results = sorted(valid_results, 
                                  key=lambda x: x['evaluation']['overall_score'], 
                                  reverse=True)
            
            print(f"\n🥇 Best performing prompt (Score: {sorted_results[0]['evaluation']['overall_score']}/5):")
            print(f"   {sorted_results[0]['prompt']}")
            
            print(f"\n🥉 Lowest performing prompt (Score: {sorted_results[-1]['evaluation']['overall_score']}/5):")
            print(f"   {sorted_results[-1]['prompt']}")
    
    return results


def run_comparison(eval_file, region='us-east-1', custom_model_id=None, models_to_compare=None):
    """Run evaluation for multiple models and compare results"""
    if models_to_compare is None:
        models_to_compare = ['pro', 'lite']
        if custom_model_id:
            models_to_compare.append('custom')
    
    model_names = {
        'pro': 'Nova Pro',
        'lite': 'Nova Lite', 
        'custom': 'Custom Nova'
    }
    
    print(f"🔄 Running Comparative Evaluation: {' vs '.join([model_names[m] for m in models_to_compare])}")
    print("=" * 80)
    
    results = {}
    
    # Run evaluations for each model
    for model in models_to_compare:
        print(f"\n{'='*20} {model_names[model].upper()} {'='*20}")
        model_results = run_single_evaluation(model, eval_file, region, custom_model_id)
        if model_results:
            results[model] = model_results
        else:
            print(f"❌ Failed to evaluate {model_names[model]}")
    
    # Compare results
    if len(results) >= 2:
        print(f"\n{'='*25} COMPARISON RESULTS {'='*25}")
        
        # Prepare comparison table
        print(f"{'Metric':<20}", end='')
        for model in models_to_compare:
            if model in results:
                print(f"{model_names[model]:<15}", end='')
        print()
        print("-" * (20 + 15 * len(results)))
        
        # Print metrics
        metrics = ['accuracy', 'completeness', 'relevance', 'helpfulness', 'language_quality', 'overall_score']
        for metric in metrics:
            print(f"{metric.replace('_', ' ').title():<20}", end='')
            for model in models_to_compare:
                if model in results:
                    score = results[model]['average_scores'][metric]
                    print(f"{score:<15.2f}", end='')
            print()
        
        # Overall winner
        scores = {}
        for model in models_to_compare:
            if model in results:
                scores[model_names[model]] = results[model]['average_scores']['overall_score']
        
        if scores:
            winner = max(scores, key=scores.get)
            winner_score = scores[winner]
            
            print(f"\n🏆 Winner: {winner} (Score: {winner_score:.2f})")
            
            # Show ranking
            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            print("\n📊 Final Ranking:")
            for i, (model, score) in enumerate(sorted_models, 1):
                print(f"  {i}. {model}: {score:.2f}/5.0")
    
    return results


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LLM-as-a-Judge Evaluation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all available models (default behavior)
  python run_evaluation.py --file test.jsonl
  
  # Evaluate specific model
  python run_evaluation.py --model pro --file test.jsonl
  
  # Evaluate custom model
  python run_evaluation.py --model custom --custom-model-id arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/abc123
  
  # Compare all models including custom
  python run_evaluation.py --model all --custom-model-id arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/abc123 --region us-west-2
        """
    )
    
    parser.add_argument('--model', 
                       choices=['pro', 'lite', 'custom', 'all'], 
                       help='Model to evaluate: pro, lite, custom, or all for comparison (default: all available models)')
    parser.add_argument('--file', 
                       default='test.jsonl',
                       help='Evaluation file to use (default: test.jsonl)')
    parser.add_argument('--region', 
                       default='us-east-1',
                       help='AWS region (default: us-east-1)')
    parser.add_argument('--custom-model-id',
                       help='Custom model ARN/ID for custom model evaluation')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Validate file exists
    if not os.path.exists(args.file):
        print(f"❌ Evaluation file not found: {args.file}")
        sys.exit(1)
    
    # Validate custom model requirements
    if args.model == 'custom' and not args.custom_model_id:
        print("❌ --custom-model-id is required when evaluating custom models")
        sys.exit(1)
    
    # Determine evaluation mode
    if args.model:
        if args.model == 'all':
            # Run comparison with all available models
            models_to_compare = ['pro', 'lite']
            if args.custom_model_id:
                models_to_compare.append('custom')
            run_comparison(args.file, args.region, args.custom_model_id, models_to_compare)
        else:
            # Run single model evaluation
            run_single_evaluation(args.model, args.file, args.region, args.custom_model_id)
    else:
        # Default: run comparison with available models
        models_to_compare = ['pro', 'lite']
        if args.custom_model_id:
            models_to_compare.append('custom')
        run_comparison(args.file, args.region, args.custom_model_id, models_to_compare)


if __name__ == "__main__":
    main()
