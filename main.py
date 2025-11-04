# main.py
import argparse
from examples.synthetic_experiments import linear_scm_experiment, nonlin_scm_experiment
from examples.real_world_examples import causal_generation_example, intervention_study
from examples.benchmarks import causal_discovery_benchmark, intervention_benchmark

def main():
    parser = argparse.ArgumentParser(description="Causal Generative Models")
    parser.add_argument('--experiment', 
                       choices=['linear', 'nonlinear', 'generation', 
                               'intervention', 'discovery', 'benchmark', 'all'],
                       default='all',
                       help='Experiment to run')
    
    args = parser.parse_args()
    
    experiments = {
        'linear': linear_scm_experiment,
        'nonlinear': nonlin_scm_experiment,
        'generation': causal_generation_example,
        'intervention': intervention_study,
        'discovery': causal_discovery_benchmark,
        'benchmark': intervention_benchmark
    }
    
    if args.experiment == 'all':
        for name, experiment in experiments.items():
            print(f"\n{'='*60}")
            print(f"Running {name} experiment...")
            print('='*60)
            experiment()
    else:
        experiments[args.experiment]()

if __name__ == "__main__":
    main()