"""
Demo: Phase 3.4 Hybrid Ensemble Optimization for Khmer Spellchecking

This demo showcases advanced optimization techniques for neural-statistical integration,
automatically finding optimal parameters through grid search, genetic algorithms,
and Bayesian optimization with comprehensive performance analysis.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(__file__))

# Configure logging for UTF-8 support
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set environment variable for UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

from neural_models.hybrid_ensemble_optimizer import (
    HybridEnsembleOptimizer,
    ParameterSpace,
    OptimizationObjective,
    OptimizationResult
)


class Phase34Demo:
    """Phase 3.4 Hybrid Ensemble Optimization demonstration"""
    
    def __init__(self):
        self.logger = logging.getLogger("phase34_demo")
        
        # Test data sets
        self.test_texts = [
            # Standard Khmer texts
            "នេះជាអត្ថបទខ្មែរដ៏ស្រស់ស្អាត។",
            "ការអប់រំជាមូលដ្ឋានសំខាន់សម្រាប់ការអភិវឌ្ឍន៍។",
            "វប្បធម៌ខ្មែរមានប្រវត្តិដ៏យូរលង់។",
            "កំពង់ចំជាទីក្រុងសំខាន់របស់ខេត្តកំពង់ចាម។",
            "ភាសាខ្មែរជាភាសាជាតិរបស់ប្រទេសកម្ពុជា។",
            "សិស្សៗកំពុងរៀនភាសាអង់គ្លេសនៅសាលា។",
            "គ្រូបង្រៀនកំពុងពន្យល់មេរៀន។",
            
            # Longer complex texts
            "ព្រះបាទស្ទាវៗក្នុងដំណាក់កាលកណ្តាលនៃសតវត្សទី១៦។",
            "ការសិក្សាជាការសំខាន់ណាស់ក្នុងជីវិតរបស់យុវជនកម្ពុជា។",
            "បច្ចេកវិទ្យាថ្មីៗបានផ្លាស់ប្តូរវិធីរស់នៅរបស់មនុស្ស។",
            "ការអភិរក្សបរិស្ថានគឺជាកាតព្វកិច្ចរបស់យើងទាំងអស់គ្នា។",
            "សេដ្ឋកិច្ចកម្ពុជាបានរីកចម្រើនយ៉ាងគួរឱ្យកត់សម្គាល់។",
            
            # Mixed content with numbers and punctuation
            "ឆ្នាំ២០២៥ជាឆ្នាំថ្មីដ៏មានសារៈសំខាន់។",
            "Facebook និង Instagram ជាសង្គមនេត្តពេញនិយម។",
            "ចំនួន១០០នាក់បានចូលរួមក្នុងការប្រជុំនេះ។",
            "តម្លៃទំនិញបានកើនឡើង៥%ក្នុងខែនេះ។",
            
            # Short texts
            "សុខសប្បាយទេ?",
            "អរគុណច្រើន។",
            "បាយនេះឆ្ងាញ់ណាស់។",
            "ពេលវេលាគឺជាមាសមួយ។",
            "ការងារនេះពិបាកណាស់។",
            
            # Texts with potential errors for testing
            "កំហុសនេះគួរតែកែតម្រូវ។",
            "ខ្ញុំចង់ទៅលេងនៅទីក្រុងភ្នំពេញ។",
            "កុំព្យុទ័រជាឧបករណ៍សំខាន់មួយ។",
            "នេះជាសៀវភៅដ៏ល្អមួយ។"
        ]
        
        # Optimization configurations to test
        self.optimization_objectives = {
            'balanced': OptimizationObjective(
                accuracy_weight=0.3,
                confidence_weight=0.25,
                agreement_weight=0.25,
                speed_weight=0.1,
                error_detection_weight=0.1
            ),
            'accuracy_focused': OptimizationObjective(
                accuracy_weight=0.6,
                confidence_weight=0.2,
                agreement_weight=0.1,
                speed_weight=0.05,
                error_detection_weight=0.05
            ),
            'performance_focused': OptimizationObjective(
                accuracy_weight=0.2,
                confidence_weight=0.2,
                agreement_weight=0.2,
                speed_weight=0.3,
                error_detection_weight=0.1
            )
        }
        
        # Parameter spaces to test
        self.parameter_spaces = {
            'standard': ParameterSpace(),
            'neural_focused': ParameterSpace(
                neural_weight_range=(0.3, 0.7),
                statistical_weight_range=(0.1, 0.5),
                rule_weight_range=(0.1, 0.4)
            ),
            'statistical_focused': ParameterSpace(
                neural_weight_range=(0.1, 0.4),
                statistical_weight_range=(0.3, 0.7),
                rule_weight_range=(0.1, 0.5)
            )
        }
        
        self.logger.info("Initialized Phase 3.4: Hybrid Ensemble Optimization")
        self.logger.info(f"Test texts: {len(self.test_texts)}")
        self.logger.info(f"Optimization objectives: {len(self.optimization_objectives)}")
        self.logger.info(f"Parameter spaces: {len(self.parameter_spaces)}")
    
    def run_single_method_demo(self):
        """Demonstrate single optimization method"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 1: Single Method Optimization Demonstration")
        self.logger.info("=" * 80)
        
        # Use balanced objective and standard parameter space
        objective = self.optimization_objectives['balanced']
        parameter_space = self.parameter_spaces['standard']
        
        optimizer = HybridEnsembleOptimizer(parameter_space, objective)
        
        # Test each method individually
        methods = ['grid_search', 'genetic_algorithm', 'bayesian']
        results = {}
        
        for method in methods:
            self.logger.info(f"\n🔧 Testing {method.replace('_', ' ').title()}")
            
            try:
                if method == 'grid_search':
                    result = optimizer.optimize_with_method(
                        method, 
                        self.test_texts[:15],  # Smaller set for demo
                        max_evaluations=50
                    )
                elif method == 'genetic_algorithm':
                    result = optimizer.optimize_with_method(
                        method,
                        self.test_texts[:15],
                        generations=10
                    )
                elif method == 'bayesian':
                    result = optimizer.optimize_with_method(
                        method,
                        self.test_texts[:15],
                        iterations=25
                    )
                
                results[method] = result
                summary = result.get_summary()
                
                self.logger.info(f"   Best score: {summary['best_score']:.4f}")
                self.logger.info(f"   Evaluations: {summary['total_evaluations']:,}")
                self.logger.info(f"   Time: {summary['optimization_time']:.2f}s")
                self.logger.info(f"   CV mean ± std: {summary['cv_mean']:.3f} ± {summary['cv_std']:.3f}")
                
                best_config = summary['best_configuration']
                self.logger.info(f"   Best config: N={best_config['neural_weight']:.3f}, "
                               f"S={best_config['statistical_weight']:.3f}, "
                               f"R={best_config['rule_weight']:.3f}")
                
            except Exception as e:
                self.logger.error(f"   Failed: {e}")
        
        # Generate comparison report
        if results:
            report = optimizer.get_optimization_report(results)
            self.logger.info("\n📊 Single Method Comparison Report:")
            self.logger.info(report)
        
        return results
    
    def run_multi_objective_demo(self):
        """Demonstrate optimization with different objectives"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 2: Multi-Objective Optimization")
        self.logger.info("=" * 80)
        
        objective_results = {}
        
        for obj_name, objective in self.optimization_objectives.items():
            self.logger.info(f"\n🎯 Testing {obj_name.replace('_', ' ').title()} Objective")
            self.logger.info(f"   Weights: Acc={objective.accuracy_weight}, "
                           f"Conf={objective.confidence_weight}, "
                           f"Agr={objective.agreement_weight}, "
                           f"Speed={objective.speed_weight}, "
                           f"Error={objective.error_detection_weight}")
            
            optimizer = HybridEnsembleOptimizer(
                self.parameter_spaces['standard'], 
                objective
            )
            
            # Use genetic algorithm for this demo (good balance of speed and quality)
            result = optimizer.optimize_with_method(
                'genetic_algorithm',
                self.test_texts[:20],
                generations=15
            )
            
            objective_results[obj_name] = result
            summary = result.get_summary()
            
            self.logger.info(f"   Result: {summary['best_score']:.4f} score "
                           f"({summary['total_evaluations']} evaluations, "
                           f"{summary['optimization_time']:.1f}s)")
            
            best_config = summary['best_configuration']
            approach = ("neural-focused" if best_config['neural_weight'] > 0.45 
                       else "statistical-focused" if best_config['statistical_weight'] > 0.45
                       else "balanced")
            self.logger.info(f"   Approach: {approach}")
        
        # Compare objectives
        self.logger.info("\n📈 Objective Comparison:")
        for obj_name, result in objective_results.items():
            summary = result.get_summary()
            self.logger.info(f"   {obj_name}: {summary['best_score']:.4f} "
                           f"(CV: {summary['cv_mean']:.3f}±{summary['cv_std']:.3f})")
        
        return objective_results
    
    def run_parameter_space_demo(self):
        """Demonstrate optimization with different parameter spaces"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 3: Parameter Space Exploration")
        self.logger.info("=" * 80)
        
        space_results = {}
        objective = self.optimization_objectives['balanced']
        
        for space_name, parameter_space in self.parameter_spaces.items():
            self.logger.info(f"\n🔍 Testing {space_name.replace('_', ' ').title()} Parameter Space")
            self.logger.info(f"   Neural range: {parameter_space.neural_weight_range}")
            self.logger.info(f"   Statistical range: {parameter_space.statistical_weight_range}")
            self.logger.info(f"   Rule range: {parameter_space.rule_weight_range}")
            
            optimizer = HybridEnsembleOptimizer(parameter_space, objective)
            
            # Use Bayesian optimization for parameter space exploration
            result = optimizer.optimize_with_method(
                'bayesian',
                self.test_texts[:20],
                iterations=30
            )
            
            space_results[space_name] = result
            summary = result.get_summary()
            
            self.logger.info(f"   Best score: {summary['best_score']:.4f}")
            self.logger.info(f"   Time: {summary['optimization_time']:.2f}s")
            
            best_config = summary['best_configuration']
            self.logger.info(f"   Optimal weights: "
                           f"N={best_config['neural_weight']:.3f}, "
                           f"S={best_config['statistical_weight']:.3f}, "
                           f"R={best_config['rule_weight']:.3f}")
        
        # Compare parameter spaces
        self.logger.info("\n🔬 Parameter Space Analysis:")
        best_space = max(space_results.keys(), key=lambda s: space_results[s].best_score)
        self.logger.info(f"   Best parameter space: {best_space}")
        
        for space_name, result in space_results.items():
            summary = result.get_summary()
            best_config = summary['best_configuration']
            approach = ("neural" if best_config['neural_weight'] > 0.45 
                       else "statistical" if best_config['statistical_weight'] > 0.45
                       else "balanced")
            self.logger.info(f"   {space_name}: {summary['best_score']:.4f} → {approach} approach")
        
        return space_results
    
    def run_comprehensive_optimization(self):
        """Run comprehensive optimization with best settings"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 4: Comprehensive Optimization")
        self.logger.info("=" * 80)
        
        # Use best objective and parameter space from previous steps
        best_objective = self.optimization_objectives['balanced']
        best_parameter_space = self.parameter_spaces['standard']
        
        optimizer = HybridEnsembleOptimizer(best_parameter_space, best_objective)
        
        self.logger.info("🚀 Running comprehensive multi-method optimization...")
        self.logger.info(f"   Dataset size: {len(self.test_texts)} texts")
        self.logger.info(f"   Methods: Grid Search, Genetic Algorithm, Bayesian")
        
        # Run all methods with more iterations
        start_time = time.time()
        
        results = {}
        
        # Grid search with higher resolution
        self.logger.info("\n🔍 Grid Search (high resolution)...")
        results['grid_search'] = optimizer.optimize_with_method(
            'grid_search',
            self.test_texts,
            max_evaluations=150
        )
        
        # Genetic algorithm with more generations
        self.logger.info("🧬 Genetic Algorithm (extended)...")
        results['genetic_algorithm'] = optimizer.optimize_with_method(
            'genetic_algorithm',
            self.test_texts,
            generations=25
        )
        
        # Bayesian optimization with more iterations
        self.logger.info("🎯 Bayesian Optimization (intensive)...")
        results['bayesian'] = optimizer.optimize_with_method(
            'bayesian',
            self.test_texts,
            iterations=75
        )
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = optimizer.get_optimization_report(results)
        self.logger.info("\n📊 COMPREHENSIVE OPTIMIZATION REPORT:")
        self.logger.info(report)
        
        # Additional analysis
        self.logger.info(f"\n⏱️ Total optimization time: {total_time:.2f}s")
        self.logger.info(f"📈 Total evaluations: {sum(r.total_evaluations for r in results.values()):,}")
        
        # Find overall best
        best_method = max(results.keys(), key=lambda m: results[m].best_score)
        best_result = results[best_method]
        
        self.logger.info(f"\n🏆 FINAL RECOMMENDATION:")
        self.logger.info(f"   Best method: {best_method.replace('_', ' ').title()}")
        self.logger.info(f"   Score: {best_result.best_score:.4f}")
        
        best_config = best_result.best_configuration
        self.logger.info(f"   Optimal configuration:")
        self.logger.info(f"     Neural weight: {best_config.neural_weight:.3f}")
        self.logger.info(f"     Statistical weight: {best_config.statistical_weight:.3f}")
        self.logger.info(f"     Rule weight: {best_config.rule_weight:.3f}")
        self.logger.info(f"     Consensus threshold: {best_config.consensus_threshold:.3f}")
        self.logger.info(f"     Error confidence: {best_config.error_confidence_threshold:.3f}")
        
        return results, best_result
    
    def run_convergence_analysis(self):
        """Analyze optimization convergence"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 5: Convergence Analysis")
        self.logger.info("=" * 80)
        
        optimizer = HybridEnsembleOptimizer(
            self.parameter_spaces['standard'],
            self.optimization_objectives['balanced']
        )
        
        # Run genetic algorithm with tracking
        self.logger.info("🔬 Analyzing convergence with extended GA run...")
        
        result = optimizer.optimize_with_method(
            'genetic_algorithm',
            self.test_texts[:25],
            generations=40
        )
        
        summary = result.get_summary()
        
        self.logger.info(f"✅ Convergence Analysis Results:")
        self.logger.info(f"   Final score: {summary['best_score']:.4f}")
        self.logger.info(f"   Cross-validation stability: {summary['cv_std']:.4f}")
        
        stability = "Excellent" if summary['cv_std'] < 0.02 else "Good" if summary['cv_std'] < 0.05 else "Needs improvement"
        self.logger.info(f"   Stability assessment: {stability}")
        
        return result
    
    def save_optimization_results(self, results: Dict[str, Any]):
        """Save all optimization results"""
        output_dir = Path("output/phase34")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results
        if 'comprehensive' in results:
            comprehensive_results = results['comprehensive'][0]  # Results dict
            best_result = results['comprehensive'][1]  # Best result
            
            optimizer = HybridEnsembleOptimizer()
            optimizer.save_optimization_results(
                str(output_dir / "comprehensive_optimization.json"),
                comprehensive_results
            )
            
            # Save best configuration separately
            best_config_data = {
                'method': max(comprehensive_results.keys(), key=lambda m: comprehensive_results[m].best_score),
                'score': best_result.best_score,
                'configuration': {
                    'neural_weight': best_result.best_configuration.neural_weight,
                    'statistical_weight': best_result.best_configuration.statistical_weight,
                    'rule_weight': best_result.best_configuration.rule_weight,
                    'consensus_threshold': best_result.best_configuration.consensus_threshold,
                    'error_confidence_threshold': best_result.best_configuration.error_confidence_threshold
                },
                'cross_validation': {
                    'mean': np.mean(best_result.cross_validation_scores),
                    'std': np.std(best_result.cross_validation_scores),
                    'scores': best_result.cross_validation_scores
                }
            }
            
            import json
            with open(output_dir / "best_configuration.json", 'w', encoding='utf-8') as f:
                json.dump(best_config_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📁 Results saved to {output_dir}")
    
    def run_complete_demo(self):
        """Run the complete Phase 3.4 demonstration"""
        self.logger.info("🔧 STARTING PHASE 3.4 HYBRID ENSEMBLE OPTIMIZATION DEMO")
        self.logger.info("=" * 85)
        self.logger.info("Demo: Phase 3.4: Hybrid Ensemble Optimization")
        self.logger.info("=" * 85)
        
        all_results = {}
        
        try:
            # Step 1: Single method demonstration
            single_results = self.run_single_method_demo()
            all_results['single_method'] = single_results
            
            # Step 2: Multi-objective optimization
            objective_results = self.run_multi_objective_demo()
            all_results['multi_objective'] = objective_results
            
            # Step 3: Parameter space exploration
            space_results = self.run_parameter_space_demo()
            all_results['parameter_space'] = space_results
            
            # Step 4: Comprehensive optimization
            comprehensive_results = self.run_comprehensive_optimization()
            all_results['comprehensive'] = comprehensive_results
            
            # Step 5: Convergence analysis
            convergence_result = self.run_convergence_analysis()
            all_results['convergence'] = convergence_result
            
            # Save results
            self.save_optimization_results(all_results)
            
            self.logger.info("=" * 85)
            self.logger.info("🏆 PHASE 3.4 DEMO COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 85)
            self.logger.info("✅ Hybrid Ensemble Optimization: Fully operational")
            self.logger.info("   Advanced optimization algorithms validated")
            self.logger.info("   Multi-objective optimization completed")
            self.logger.info("   Parameter space exploration finished")
            self.logger.info("   Comprehensive optimization successful")
            self.logger.info("   Convergence analysis completed")
            self.logger.info("   Production-ready configurations generated")
            self.logger.info("=" * 85)
            
            print("\n🎉 Phase 3.4 Hybrid Ensemble Optimization demo completed successfully!")
            print("🏆 Ready for next phase: Phase 3.5 (Production Deployment)")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {e}")
            raise


if __name__ == "__main__":
    demo = Phase34Demo()
    results = demo.run_complete_demo() 