"""
Phase 3.4: Hybrid Ensemble Optimization for Khmer Spellchecking

This module provides advanced optimization techniques for neural-statistical integration,
automatically tuning parameters to achieve optimal spellchecking performance through
grid search, genetic algorithms, Bayesian optimization, and cross-validation.
"""

import numpy as np
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
from collections import defaultdict
import random
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from neural_models.neural_statistical_integration import (
    NeuralStatisticalIntegrator, 
    IntegrationConfiguration,
    IntegrationResult
)

# Suppress warnings for optimization algorithms
warnings.filterwarnings('ignore')


@dataclass
class OptimizationObjective:
    """Definition of optimization objective with weights for different metrics"""
    accuracy_weight: float = 0.4
    confidence_weight: float = 0.2
    agreement_weight: float = 0.2
    speed_weight: float = 0.1
    error_detection_weight: float = 0.1
    
    def calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted objective score"""
        score = (
            self.accuracy_weight * metrics.get('accuracy', 0.0) +
            self.confidence_weight * metrics.get('confidence', 0.0) +
            self.agreement_weight * metrics.get('agreement', 0.0) +
            self.speed_weight * metrics.get('speed_score', 0.0) +
            self.error_detection_weight * metrics.get('error_detection', 0.0)
        )
        return max(0.0, min(1.0, score))


@dataclass 
class ParameterSpace:
    """Definition of parameter search space for optimization"""
    neural_weight_range: Tuple[float, float] = (0.1, 0.8)
    statistical_weight_range: Tuple[float, float] = (0.1, 0.8)
    rule_weight_range: Tuple[float, float] = (0.1, 0.6)
    consensus_threshold_range: Tuple[float, float] = (0.3, 0.9)
    error_confidence_range: Tuple[float, float] = (0.3, 0.8)
    neural_temperature_range: Tuple[float, float] = (0.5, 2.0)
    
    def validate_weights(self, neural_w: float, statistical_w: float, rule_w: float) -> bool:
        """Validate that weights sum to approximately 1.0"""
        total = neural_w + statistical_w + rule_w
        return 0.95 <= total <= 1.05
    
    def normalize_weights(self, neural_w: float, statistical_w: float, rule_w: float) -> Tuple[float, float, float]:
        """Normalize weights to sum to 1.0"""
        total = neural_w + statistical_w + rule_w
        if total > 0:
            return neural_w/total, statistical_w/total, rule_w/total
        return 0.33, 0.33, 0.34


@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    best_configuration: IntegrationConfiguration
    best_score: float
    best_metrics: Dict[str, float]
    optimization_history: List[Dict] = field(default_factory=list)
    total_evaluations: int = 0
    optimization_time: float = 0.0
    convergence_iteration: int = 0
    cross_validation_scores: List[float] = field(default_factory=list)
    
    def get_summary(self) -> Dict:
        """Get optimization summary"""
        return {
            'best_score': self.best_score,
            'total_evaluations': self.total_evaluations,
            'optimization_time': self.optimization_time,
            'convergence_iteration': self.convergence_iteration,
            'cv_mean': np.mean(self.cross_validation_scores) if self.cross_validation_scores else 0.0,
            'cv_std': np.std(self.cross_validation_scores) if self.cross_validation_scores else 0.0,
            'best_configuration': {
                'neural_weight': self.best_configuration.neural_weight,
                'statistical_weight': self.best_configuration.statistical_weight,
                'rule_weight': self.best_configuration.rule_weight,
                'consensus_threshold': self.best_configuration.consensus_threshold,
                'error_confidence_threshold': self.best_configuration.error_confidence_threshold
            }
        }


class GridSearchOptimizer:
    """Grid search optimization for parameter tuning"""
    
    def __init__(self, parameter_space: ParameterSpace, grid_resolution: int = 5):
        self.parameter_space = parameter_space
        self.grid_resolution = grid_resolution
        self.logger = logging.getLogger("grid_search_optimizer")
    
    def generate_grid(self) -> List[Dict[str, float]]:
        """Generate grid of parameter combinations"""
        neural_weights = np.linspace(*self.parameter_space.neural_weight_range, self.grid_resolution)
        statistical_weights = np.linspace(*self.parameter_space.statistical_weight_range, self.grid_resolution)
        rule_weights = np.linspace(*self.parameter_space.rule_weight_range, self.grid_resolution)
        consensus_thresholds = np.linspace(*self.parameter_space.consensus_threshold_range, self.grid_resolution)
        error_confidence_thresholds = np.linspace(*self.parameter_space.error_confidence_range, self.grid_resolution)
        
        grid_points = []
        for neural_w in neural_weights:
            for statistical_w in statistical_weights:
                for rule_w in rule_weights:
                    # Normalize weights
                    neural_w_norm, statistical_w_norm, rule_w_norm = self.parameter_space.normalize_weights(
                        neural_w, statistical_w, rule_w
                    )
                    
                    for consensus_thresh in consensus_thresholds:
                        for error_conf_thresh in error_confidence_thresholds:
                            grid_points.append({
                                'neural_weight': neural_w_norm,
                                'statistical_weight': statistical_w_norm,
                                'rule_weight': rule_w_norm,
                                'consensus_threshold': consensus_thresh,
                                'error_confidence_threshold': error_conf_thresh
                            })
        
        self.logger.info(f"Generated grid with {len(grid_points)} parameter combinations")
        return grid_points
    
    def optimize(self, 
                evaluator: Callable[[Dict], float],
                max_evaluations: Optional[int] = None) -> List[Tuple[Dict, float]]:
        """
        Perform grid search optimization
        
        Args:
            evaluator: Function to evaluate parameter configuration
            max_evaluations: Maximum number of evaluations
            
        Returns:
            List of (parameters, score) tuples sorted by score
        """
        grid_points = self.generate_grid()
        
        if max_evaluations and max_evaluations < len(grid_points):
            # Random sampling from grid
            grid_points = random.sample(grid_points, max_evaluations)
            self.logger.info(f"Sampling {max_evaluations} points from grid")
        
        results = []
        for i, params in enumerate(grid_points):
            score = evaluator(params)
            results.append((params, score))
            
            if i % 50 == 0 and i > 0:
                self.logger.info(f"Evaluated {i}/{len(grid_points)} configurations")
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class GeneticAlgorithmOptimizer:
    """Genetic algorithm optimization for parameter tuning"""
    
    def __init__(self, 
                 parameter_space: ParameterSpace,
                 population_size: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        self.parameter_space = parameter_space
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.logger = logging.getLogger("genetic_algorithm_optimizer")
    
    def create_individual(self) -> Dict[str, float]:
        """Create random individual (parameter configuration)"""
        neural_w = np.random.uniform(*self.parameter_space.neural_weight_range)
        statistical_w = np.random.uniform(*self.parameter_space.statistical_weight_range)
        rule_w = np.random.uniform(*self.parameter_space.rule_weight_range)
        
        # Normalize weights
        neural_w, statistical_w, rule_w = self.parameter_space.normalize_weights(
            neural_w, statistical_w, rule_w
        )
        
        return {
            'neural_weight': neural_w,
            'statistical_weight': statistical_w,
            'rule_weight': rule_w,
            'consensus_threshold': np.random.uniform(*self.parameter_space.consensus_threshold_range),
            'error_confidence_threshold': np.random.uniform(*self.parameter_space.error_confidence_range)
        }
    
    def create_population(self) -> List[Dict[str, float]]:
        """Create initial population"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """Create offspring through crossover"""
        child = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        
        # Ensure weight normalization
        if 'neural_weight' in child:
            neural_w, statistical_w, rule_w = self.parameter_space.normalize_weights(
                child['neural_weight'], child['statistical_weight'], child['rule_weight']
            )
            child['neural_weight'] = neural_w
            child['statistical_weight'] = statistical_w
            child['rule_weight'] = rule_w
        
        return child
    
    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Apply mutation to individual"""
        mutated = individual.copy()
        
        for key, value in mutated.items():
            if random.random() < self.mutation_rate:
                if key == 'neural_weight':
                    mutated[key] = np.clip(
                        value + np.random.normal(0, 0.05),
                        *self.parameter_space.neural_weight_range
                    )
                elif key == 'statistical_weight':
                    mutated[key] = np.clip(
                        value + np.random.normal(0, 0.05),
                        *self.parameter_space.statistical_weight_range
                    )
                elif key == 'rule_weight':
                    mutated[key] = np.clip(
                        value + np.random.normal(0, 0.05),
                        *self.parameter_space.rule_weight_range
                    )
                elif key == 'consensus_threshold':
                    mutated[key] = np.clip(
                        value + np.random.normal(0, 0.05),
                        *self.parameter_space.consensus_threshold_range
                    )
                elif key == 'error_confidence_threshold':
                    mutated[key] = np.clip(
                        value + np.random.normal(0, 0.05),
                        *self.parameter_space.error_confidence_range
                    )
        
        # Re-normalize weights
        if 'neural_weight' in mutated:
            neural_w, statistical_w, rule_w = self.parameter_space.normalize_weights(
                mutated['neural_weight'], mutated['statistical_weight'], mutated['rule_weight']
            )
            mutated['neural_weight'] = neural_w
            mutated['statistical_weight'] = statistical_w
            mutated['rule_weight'] = rule_w
        
        return mutated
    
    def optimize(self,
                evaluator: Callable[[Dict], float],
                generations: int = 50) -> List[Tuple[Dict, float]]:
        """
        Perform genetic algorithm optimization
        
        Args:
            evaluator: Function to evaluate parameter configuration  
            generations: Number of generations to evolve
            
        Returns:
            List of (parameters, score) tuples from final population
        """
        population = self.create_population()
        history = []
        
        for generation in range(generations):
            # Evaluate population
            fitness_scores = [(individual, evaluator(individual)) for individual in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Track best score
            best_score = fitness_scores[0][1]
            history.append(best_score)
            
            if generation % 10 == 0:
                self.logger.info(f"Generation {generation}: Best score = {best_score:.4f}")
            
            # Selection (top 50%)
            elite_size = self.population_size // 2
            elite = [individual for individual, score in fitness_scores[:elite_size]]
            
            # Create next generation
            new_population = elite.copy()  # Keep elite
            
            while len(new_population) < self.population_size:
                # Tournament selection for parents
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2]).copy()
                
                # Mutation
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Final evaluation
        final_scores = [(individual, evaluator(individual)) for individual in population]
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"GA optimization completed. Best score: {final_scores[0][1]:.4f}")
        return final_scores


class BayesianOptimizer:
    """Simplified Bayesian optimization for parameter tuning"""
    
    def __init__(self, parameter_space: ParameterSpace, exploration_weight: float = 0.1):
        self.parameter_space = parameter_space
        self.exploration_weight = exploration_weight
        self.history = []
        self.logger = logging.getLogger("bayesian_optimizer")
    
    def acquisition_function(self, candidate: Dict[str, float]) -> float:
        """Simple acquisition function using distance and historical performance"""
        if not self.history:
            return random.random()  # Random for first few samples
        
        # Calculate distance to historical points
        distances = []
        scores = []
        
        for hist_params, hist_score in self.history:
            distance = 0.0
            for key in candidate.keys():
                distance += (candidate[key] - hist_params[key]) ** 2
            distances.append(np.sqrt(distance))
            scores.append(hist_score)
        
        # Exploitation: favor areas with high historical scores
        if distances:
            weights = [1.0 / (d + 1e-6) for d in distances]
            weighted_score = np.average(scores, weights=weights)
        else:
            weighted_score = 0.0
        
        # Exploration: favor areas far from evaluated points
        min_distance = min(distances) if distances else 1.0
        exploration_bonus = self.exploration_weight * min_distance
        
        return weighted_score + exploration_bonus
    
    def suggest_candidate(self) -> Dict[str, float]:
        """Suggest next candidate based on acquisition function"""
        # Generate several candidates and pick best according to acquisition
        candidates = []
        for _ in range(20):
            candidate = {
                'neural_weight': np.random.uniform(*self.parameter_space.neural_weight_range),
                'statistical_weight': np.random.uniform(*self.parameter_space.statistical_weight_range),
                'rule_weight': np.random.uniform(*self.parameter_space.rule_weight_range),
                'consensus_threshold': np.random.uniform(*self.parameter_space.consensus_threshold_range),
                'error_confidence_threshold': np.random.uniform(*self.parameter_space.error_confidence_range)
            }
            
            # Normalize weights
            neural_w, statistical_w, rule_w = self.parameter_space.normalize_weights(
                candidate['neural_weight'], candidate['statistical_weight'], candidate['rule_weight']
            )
            candidate['neural_weight'] = neural_w
            candidate['statistical_weight'] = statistical_w
            candidate['rule_weight'] = rule_w
            
            acquisition_score = self.acquisition_function(candidate)
            candidates.append((candidate, acquisition_score))
        
        # Return best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def optimize(self,
                evaluator: Callable[[Dict], float],
                iterations: int = 100) -> List[Tuple[Dict, float]]:
        """
        Perform Bayesian optimization
        
        Args:
            evaluator: Function to evaluate parameter configuration
            iterations: Number of optimization iterations
            
        Returns:
            List of (parameters, score) tuples
        """
        for iteration in range(iterations):
            candidate = self.suggest_candidate()
            score = evaluator(candidate)
            self.history.append((candidate, score))
            
            if iteration % 20 == 0:
                best_score = max(score for _, score in self.history)
                self.logger.info(f"Iteration {iteration}: Best score = {best_score:.4f}")
        
        # Sort history by score
        self.history.sort(key=lambda x: x[1], reverse=True)
        return self.history


class HybridEnsembleOptimizer:
    """
    Main optimizer that combines multiple optimization strategies
    for finding optimal neural-statistical integration parameters
    """
    
    def __init__(self, 
                 parameter_space: Optional[ParameterSpace] = None,
                 objective: Optional[OptimizationObjective] = None):
        self.parameter_space = parameter_space or ParameterSpace()
        self.objective = objective or OptimizationObjective()
        self.logger = logging.getLogger("hybrid_ensemble_optimizer")
        
        # Optimization algorithms
        self.grid_search = GridSearchOptimizer(self.parameter_space)
        self.genetic_algorithm = GeneticAlgorithmOptimizer(self.parameter_space)
        self.bayesian_optimizer = BayesianOptimizer(self.parameter_space)
        
        # Results storage
        self.optimization_results = {}
        self.evaluation_cache = {}
    
    def evaluate_configuration(self,
                             params: Dict[str, float],
                             test_texts: List[str],
                             validation_texts: Optional[List[str]] = None) -> float:
        """
        Evaluate a parameter configuration
        
        Args:
            params: Parameter configuration to evaluate
            test_texts: Texts for evaluation
            validation_texts: Optional validation texts for cross-validation
            
        Returns:
            Objective score for the configuration
        """
        # Cache key for memoization
        cache_key = tuple(sorted(params.items()))
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        try:
            # Create configuration
            config = IntegrationConfiguration(
                neural_weight=params['neural_weight'],
                statistical_weight=params['statistical_weight'],
                rule_weight=params['rule_weight'],
                consensus_threshold=params['consensus_threshold'],
                error_confidence_threshold=params['error_confidence_threshold']
            )
            
            # Create integrator
            integrator = NeuralStatisticalIntegrator(config)
            
            # Use minimal models for evaluation
            integrator.create_minimal_models(test_texts)
            
            # Evaluate on test texts
            start_time = time.time()
            results = integrator.validate_batch(test_texts[:20])  # Sample for speed
            processing_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_metrics(results, processing_time)
            
            # Calculate objective score
            score = self.objective.calculate_score(metrics)
            
            # Cache result
            self.evaluation_cache[cache_key] = score
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Evaluation failed for params {params}: {e}")
            return 0.0
    
    def _calculate_metrics(self, results: List[IntegrationResult], processing_time: float) -> Dict[str, float]:
        """Calculate performance metrics from validation results"""
        if not results:
            return {'accuracy': 0.0, 'confidence': 0.0, 'agreement': 0.0, 'speed_score': 0.0, 'error_detection': 0.0}
        
        # Basic metrics
        accuracy = sum(1 for r in results if r.is_valid) / len(results)
        confidence = np.mean([r.overall_confidence for r in results])
        agreement = np.mean([r.method_agreement for r in results])
        
        # Speed score (normalized, higher is better)
        texts_per_second = len(results) / max(processing_time, 0.001)
        speed_score = min(1.0, texts_per_second / 1000.0)  # Normalize to 1000 texts/sec
        
        # Error detection capability
        total_errors = sum(len(r.errors) for r in results)
        error_detection = min(1.0, total_errors / max(1, len(results) * 0.1))  # Expect ~10% error rate
        
        return {
            'accuracy': accuracy,
            'confidence': confidence,
            'agreement': agreement,
            'speed_score': speed_score,
            'error_detection': error_detection
        }
    
    def cross_validate_configuration(self,
                                   params: Dict[str, float],
                                   texts: List[str],
                                   k_folds: int = 5) -> Tuple[float, List[float]]:
        """
        Perform k-fold cross-validation on configuration
        
        Args:
            params: Parameter configuration
            texts: All available texts
            k_folds: Number of folds for cross-validation
            
        Returns:
            (mean_score, individual_fold_scores)
        """
        if len(texts) < k_folds:
            # Not enough data for k-fold, use single evaluation
            score = self.evaluate_configuration(params, texts)
            return score, [score]
        
        fold_size = len(texts) // k_folds
        fold_scores = []
        
        for fold in range(k_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else len(texts)
            
            test_fold = texts[start_idx:end_idx]
            train_fold = texts[:start_idx] + texts[end_idx:]
            
            score = self.evaluate_configuration(params, test_fold, train_fold)
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        return mean_score, fold_scores
    
    def optimize_with_method(self,
                           method: str,
                           test_texts: List[str],
                           **kwargs) -> OptimizationResult:
        """
        Optimize using specified method
        
        Args:
            method: Optimization method ('grid_search', 'genetic_algorithm', 'bayesian')
            test_texts: Texts for evaluation
            **kwargs: Additional arguments for optimization method
            
        Returns:
            OptimizationResult with best configuration
        """
        start_time = time.time()
        
        # Create evaluator function
        evaluator = lambda params: self.evaluate_configuration(params, test_texts)
        
        self.logger.info(f"Starting optimization with method: {method}")
        
        if method == 'grid_search':
            results = self.grid_search.optimize(evaluator, kwargs.get('max_evaluations', 1000))
        elif method == 'genetic_algorithm':
            results = self.genetic_algorithm.optimize(evaluator, kwargs.get('generations', 30))
        elif method == 'bayesian':
            results = self.bayesian_optimizer.optimize(evaluator, kwargs.get('iterations', 50))
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        optimization_time = time.time() - start_time
        
        # Get best result
        best_params, best_score = results[0]
        
        # Perform cross-validation on best configuration
        cv_mean, cv_scores = self.cross_validate_configuration(best_params, test_texts)
        
        # Create best configuration object
        best_config = IntegrationConfiguration(**best_params)
        
        # Calculate best metrics
        best_metrics = self.evaluation_cache.get(tuple(sorted(best_params.items())), {})
        
        optimization_result = OptimizationResult(
            best_configuration=best_config,
            best_score=best_score,
            best_metrics=best_metrics,
            optimization_history=[{'params': p, 'score': s} for p, s in results[:10]],  # Top 10
            total_evaluations=len(results),
            optimization_time=optimization_time,
            convergence_iteration=0,  # Could be enhanced
            cross_validation_scores=cv_scores
        )
        
        self.optimization_results[method] = optimization_result
        self.logger.info(f"Optimization completed. Best score: {best_score:.4f}, CV mean: {cv_mean:.4f}")
        
        return optimization_result
    
    def multi_method_optimization(self,
                                test_texts: List[str],
                                methods: Optional[List[str]] = None) -> Dict[str, OptimizationResult]:
        """
        Run multiple optimization methods and compare results
        
        Args:
            test_texts: Texts for evaluation
            methods: List of methods to use (default: all)
            
        Returns:
            Dictionary mapping method names to OptimizationResults
        """
        if methods is None:
            methods = ['grid_search', 'genetic_algorithm', 'bayesian']
        
        results = {}
        
        for method in methods:
            self.logger.info(f"Running optimization with {method}")
            
            try:
                if method == 'grid_search':
                    result = self.optimize_with_method(method, test_texts, max_evaluations=200)
                elif method == 'genetic_algorithm':
                    result = self.optimize_with_method(method, test_texts, generations=20)
                elif method == 'bayesian':
                    result = self.optimize_with_method(method, test_texts, iterations=50)
                
                results[method] = result
                
            except Exception as e:
                self.logger.error(f"Optimization failed for {method}: {e}")
        
        return results
    
    def get_optimization_report(self, results: Dict[str, OptimizationResult]) -> str:
        """Generate comprehensive optimization report"""
        if not results:
            return "No optimization results available"
        
        report = f"""
🔧 HYBRID ENSEMBLE OPTIMIZATION REPORT
{'=' * 80}

🎯 OPTIMIZATION OBJECTIVE:
  Accuracy weight: {self.objective.accuracy_weight}
  Confidence weight: {self.objective.confidence_weight}
  Agreement weight: {self.objective.agreement_weight}
  Speed weight: {self.objective.speed_weight}
  Error detection weight: {self.objective.error_detection_weight}

📊 METHOD COMPARISON:"""

        for method, result in results.items():
            summary = result.get_summary()
            report += f"""

  {method.replace('_', ' ').title()}:
    • Best score: {summary['best_score']:.4f}
    • Evaluations: {summary['total_evaluations']:,}
    • Time: {summary['optimization_time']:.2f}s
    • CV mean ± std: {summary['cv_mean']:.3f} ± {summary['cv_std']:.3f}
    • Best config:
      - Neural: {summary['best_configuration']['neural_weight']:.3f}
      - Statistical: {summary['best_configuration']['statistical_weight']:.3f}
      - Rules: {summary['best_configuration']['rule_weight']:.3f}
      - Consensus: {summary['best_configuration']['consensus_threshold']:.3f}
      - Error confidence: {summary['best_configuration']['error_confidence_threshold']:.3f}"""

        # Find overall best method
        best_method = max(results.keys(), key=lambda m: results[m].best_score)
        best_result = results[best_method]

        report += f"""

🏆 BEST METHOD: {best_method.replace('_', ' ').title()}
  Overall score: {best_result.best_score:.4f}
  Cross-validation: {np.mean(best_result.cross_validation_scores):.3f} ± {np.std(best_result.cross_validation_scores):.3f}
  
🎛️ RECOMMENDED CONFIGURATION:
  Neural weight: {best_result.best_configuration.neural_weight:.3f}
  Statistical weight: {best_result.best_configuration.statistical_weight:.3f}
  Rule weight: {best_result.best_configuration.rule_weight:.3f}
  Consensus threshold: {best_result.best_configuration.consensus_threshold:.3f}
  Error confidence threshold: {best_result.best_configuration.error_confidence_threshold:.3f}

⚡ PERFORMANCE SUMMARY:
  Total evaluations: {sum(r.total_evaluations for r in results.values()):,}
  Total optimization time: {sum(r.optimization_time for r in results.values()):.2f}s
  Cache hit rate: {len(self.evaluation_cache) / max(1, sum(r.total_evaluations for r in results.values()))*100:.1f}%

💡 RECOMMENDATIONS:
  • Use {best_method.replace('_', ' ')} for future optimizations
  • Configuration shows {"balanced" if 0.25 <= best_result.best_configuration.neural_weight <= 0.45 else "neural-focused" if best_result.best_configuration.neural_weight > 0.45 else "statistical-focused"} approach works best
  • Cross-validation stability: {"Excellent" if np.std(best_result.cross_validation_scores) < 0.05 else "Good" if np.std(best_result.cross_validation_scores) < 0.1 else "Needs improvement"}

{'=' * 80}
"""
        return report
    
    def save_optimization_results(self, filepath: str, results: Dict[str, OptimizationResult]):
        """Save optimization results to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'parameter_space': {
                'neural_weight_range': self.parameter_space.neural_weight_range,
                'statistical_weight_range': self.parameter_space.statistical_weight_range,
                'rule_weight_range': self.parameter_space.rule_weight_range,
                'consensus_threshold_range': self.parameter_space.consensus_threshold_range,
                'error_confidence_range': self.parameter_space.error_confidence_range
            },
            'objective': {
                'accuracy_weight': self.objective.accuracy_weight,
                'confidence_weight': self.objective.confidence_weight,
                'agreement_weight': self.objective.agreement_weight,
                'speed_weight': self.objective.speed_weight,
                'error_detection_weight': self.objective.error_detection_weight
            },
            'results': {method: result.get_summary() for method, result in results.items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Optimization results saved to {filepath}")


if __name__ == "__main__":
    # Demo usage
    print("🔧 HYBRID ENSEMBLE OPTIMIZATION DEMO")
    print("=" * 50)
    
    # Sample texts for optimization
    test_texts = [
        "នេះជាការសាកល្បងអត្ថបទដ៏ល្អមួយ។",
        "ការអប់រំជាមូលដ្ឋានសំខាន់សម្រាប់ការអភិវឌ្ឍន៍។",
        "វប្បធម៌ខ្មែរមានប្រវត្តិដ៏យូរលង់។",
        "កំពង់ចំជាទីក្រុងសំខាន់របស់ខេត្តកំពង់ចាម។",
        "កំហុសនេះគួរតែកែតម្រូវ។",
        "២០២៥ជាឆ្នាំថ្មីដ៏មានសារៈសំខាន់។",
        "ភាសាខ្មែរជាភាសាជាតិរបស់ប្រទេសកម្ពុជា។",
        "សិស្សៗកំពុងរៀនភាសាអង់គ្លេសនៅសាលា។",
        "បាយនេះឆ្ងាញ់ណាស់។",
        "ខ្ញុំចង់ទៅលេងនៅទីក្រុងភ្នំពេញ។",
        "កុំព្យុទ័រជាឧបករណ៍សំខាន់មួយ។",
        "គ្រូបង្រៀនកំពុងពន្យល់មេរៀន។",
        "នេះជាសៀវភៅដ៏ល្អមួយ។",
        "ពេលវេលាគឺជាមាសមួយ។",
        "ការងារនេះពិបាកណាស់។"
    ] * 4  # Replicate for more evaluation data
    
    print(f"Test texts: {len(test_texts)} samples")
    
    # Create optimizer
    parameter_space = ParameterSpace()
    objective = OptimizationObjective()
    optimizer = HybridEnsembleOptimizer(parameter_space, objective)
    
    print("✅ Optimizer initialized")
    
    # Run multi-method optimization
    print("\n🚀 Starting multi-method optimization...")
    results = optimizer.multi_method_optimization(test_texts)
    
    # Generate and print report
    if results:
        report = optimizer.get_optimization_report(results)
        print(report)
        
        # Save results
        optimizer.save_optimization_results("output/phase34/optimization_results.json", results)
        print("\n✅ Optimization results saved")
    else:
        print("❌ No optimization results obtained")
    
    print("\n🏆 Phase 3.4 Hybrid Ensemble Optimization demo completed!") 