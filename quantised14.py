import gymnasium as gym
import numpy as np
import neat
import pickle
import os
import time
import csv
import random
import datetime
import ale_py
from collections import OrderedDict

# Environment setup
ENV_NAME = 'ALE/Breakout-v5'
RANDOM_SEED = 42

# RAM addresses for game state
RAM = {
    'ball_x': 99,
    'ball_y': 101,
    'paddle_x': 72,
    'lives': 57,
    'score': 84
}

MAX_STEPS = 43000000
CURRENT_GENERATION = 1

# Cache configuration
ABLATION_CONFIG = {
    'USE_CACHING': True,
    'CACHE_SIZE': 100000,
    'CACHE_PERSISTENCE': True,
    'CACHE_EVICTION_PERCENTAGE': 0.1,
    'COMPUTATION_COMPLEXITY': 'complex',
    'USE_CLAMPING': True,
    'CLAMP_METHOD': 'fp16',
    'USE_ROUNDING': True,
    'PRECISION_LEVEL': 'maximum_cache',
    'USE_FAST_KEYS': True,
    'USE_CONDITIONAL_CACHING': True,
    'FEATURE_QUANTIZATION': True,
    'FEATURE_SIZE': 128,
    'ENABLE_DETAILED_TIMING': True,
    'CACHE_DEBUG': True,
    'TRACK_GENERATION_CACHE': True
}

def force_complete_cache_reset():
    """Reset all cache variables"""
    global generational_cache, cache_stats
    
    if 'generational_cache' in globals():
        del generational_cache
    if 'cache_stats' in globals():
        del cache_stats
    
    generational_cache = OrderedDict()    
    cache_stats = {
        'hits': 0, 'misses': 0, 'total_calls': 0,
        'generation_hits': 0, 'generation_misses': 0,
        'cache_size_history': [], 'hit_rate_history': []
    }
    
    print("COMPLETE CACHE RESET: All variables forcibly cleared and recreated")
    print(f"Cache size after reset: {len(generational_cache)}")
    print(f"Cache stats after reset: {cache_stats}")

# Initialize cache variables
generational_cache = OrderedDict()
cache_stats = {
    'hits': 0, 'misses': 0, 'total_calls': 0,
    'generation_hits': 0, 'generation_misses': 0,
    'cache_size_history': [], 'hit_rate_history': []
}

def process_neural_output(raw_output):
    """Process neural network output with precision control"""
    result = raw_output
    
    if ABLATION_CONFIG['USE_CLAMPING']:
        clamp_method = ABLATION_CONFIG.get('CLAMP_METHOD', 'standard')
        
        if clamp_method == 'fp16':
            result = float(np.float16(result))
            result = np.clip(result, -5.0, 5.0)  
        else:
            result = np.clip(result, -5.0, 5.0)
    
    if ABLATION_CONFIG['USE_ROUNDING']:
        precision = ABLATION_CONFIG['PRECISION_LEVEL']
        
        if precision == 'maximum_cache':
            result = round(result * 20) * 0.05      
        elif precision == 'high_cache':
            result = round(result * 100) / 100    
        elif precision == 'optimal':
            result = round(result * 10000) / 10000
        elif precision == 'minimal_cache':
            result = round(result * 100000) / 100000
    
    return float(result)

def create_cache_key(inputs_weights, bias, response):
    """Create cache key with collision resistance"""
    if len(inputs_weights) == 0:
        return None
    
    if ABLATION_CONFIG['USE_FAST_KEYS']:
        key_val = 0
        limit = min(len(inputs_weights), 16)  
        
        for i, (input_val, weight) in enumerate(inputs_weights[:limit]):
            int_input = int(input_val * 10000) & 0xFFFFFFFF
            int_weight = int(weight * 10000) & 0xFFFFFFFF
            
            combined = (int_input << 16) ^ int_weight
            key_val ^= combined
            key_val = ((key_val << 3) | (key_val >> 29)) & 0xFFFFFFFF
        
        key_val ^= (int(bias * 10000) << 8) ^ int(response * 10000)
        return key_val & 0xFFFFFFFF
    else:
        key_parts = []
        for input_val, weight in inputs_weights[:12]:
            int_input = int(input_val * 1000)   
            int_weight = int(weight * 1000)     
            key_parts.append((int_input, int_weight))
        
        int_bias = int(bias * 1000)         
        int_response = int(response * 1000) 
        return hash((tuple(key_parts), int_bias, int_response))

def should_cache_computation(inputs_weights):
    """Determine if computation is worth caching"""
    if not ABLATION_CONFIG['USE_CONDITIONAL_CACHING']:
        return True
    
    if len(inputs_weights) < 2 or len(inputs_weights) > 50:
        return False
    
    for input_val, weight in inputs_weights:
        if abs(input_val) > 2.0 or abs(weight) > 3.0:
            return False
    
    return True

def compute_neuron_direct(inputs_weights, bias, response):
    """Direct neuron computation with configurable complexity"""
    complexity = ABLATION_CONFIG['COMPUTATION_COMPLEXITY']
    
    # Phase 1: Input processing
    total = 0.0
    if complexity == 'simple':
        for input_val, weight in inputs_weights:
            total += input_val * weight
    
    elif complexity == 'medium':
        for input_val, weight in inputs_weights:
            linear = input_val * weight
            quadratic_input = 0.1 * (input_val ** 2) * weight
            quadratic_weight = 0.05 * input_val * (weight ** 2)
            total += linear + quadratic_input + quadratic_weight
    
    elif complexity == 'complex':
        for input_val, weight in inputs_weights:
            linear = input_val * weight
            quadratic_input = 0.1 * (input_val ** 2) * weight
            quadratic_weight = 0.05 * input_val * (weight ** 2)
            cubic = 0.01 * (input_val ** 3) * weight
            total += linear + quadratic_input + quadratic_weight + cubic
    
    elif complexity == 'ultra_complex':
        for input_val, weight in inputs_weights:
            linear = input_val * weight
            quad_input = 0.1 * (input_val ** 2) * weight
            quad_weight = 0.05 * input_val * (weight ** 2)
            cubic_input = 0.01 * (input_val ** 3) * weight
            cubic_weight = 0.005 * input_val * (weight ** 3)
            quartic = 0.001 * (input_val ** 4) * weight
            cross_term = 0.002 * (input_val ** 2) * (weight ** 2)
            
            sqrt_term = 0.001 * np.sqrt(abs(input_val) + 1e-8) * weight
            log_term = 0.0005 * np.log(abs(input_val) + 1) * weight
            
            total += (linear + quad_input + quad_weight + cubic_input +
                     cubic_weight + quartic + cross_term + sqrt_term + log_term)
    
    # Phase 2: Bias and response processing
    stage1 = total + bias
    stage2 = stage1 * response
    
    if complexity in ['complex', 'ultra_complex']:
        stage3 = stage2 + 0.1 * np.sin(stage1 * np.pi)
        stage4 = stage3 + 0.05 * np.exp(-abs(stage2 * 0.5))
        pre_activation = stage4
        
        if complexity == 'ultra_complex':
            stage5 = stage4 + 0.03 * np.cos(stage1 * 2 * np.pi)
            stage6 = stage5 + 0.02 * np.sqrt(abs(stage4) + 1e-8)
            stage7 = stage6 + 0.01 * np.log(abs(stage5) + 1)
            pre_activation = stage7
    else:
        pre_activation = stage2
    
    # Phase 3: Activation function
    if complexity == 'simple':
        output = np.tanh(pre_activation)
    elif complexity == 'medium':
        tanh_comp = np.tanh(pre_activation)
        sigmoid_comp = 1 / (1 + np.exp(-np.clip(pre_activation * 0.5, -10, 10)))
        output = 0.7 * tanh_comp + 0.3 * sigmoid_comp
    else:
        tanh_comp = np.tanh(pre_activation)
        sigmoid_comp = 1 / (1 + np.exp(-np.clip(pre_activation * 0.5, -10, 10)))
        swish_comp = pre_activation / (1 + np.exp(-np.clip(pre_activation, -10, 10)))
        
        if complexity == 'ultra_complex':
            relu_comp = max(0, pre_activation)
            elu_comp = pre_activation if pre_activation > 0 else (np.exp(np.clip(pre_activation, -10, 10)) - 1)
            output = (0.4 * tanh_comp + 0.2 * sigmoid_comp + 0.15 * swish_comp +
                     0.15 * relu_comp + 0.1 * elu_comp)
        else:
            output = 0.6 * tanh_comp + 0.3 * sigmoid_comp + 0.1 * swish_comp
    
    # Phase 4: Post-processing
    normalized_output = output / (1 + abs(output))
    
    if complexity in ['complex', 'ultra_complex']:
        final_output = normalized_output + 0.01 * np.cos(output * 2 * np.pi)
        if complexity == 'ultra_complex':
            final_output += 0.005 * np.sin(final_output * 3 * np.pi)
            final_output += 0.002 * np.sqrt(abs(final_output) + 1e-8)
    else:
        final_output = normalized_output
    
    return process_neural_output(final_output)

def compute_neuron_with_generational_cache(inputs_weights, bias, response):
    """Compute neuron output with LRU caching"""
    global cache_stats
    
    cache_stats['total_calls'] += 1
    
    if not ABLATION_CONFIG['USE_CACHING']:
        return compute_neuron_direct(inputs_weights, bias, response)
    
    if ABLATION_CONFIG['USE_CONDITIONAL_CACHING'] and not should_cache_computation(inputs_weights):
        return compute_neuron_direct(inputs_weights, bias, response)
    
    cache_key = create_cache_key(inputs_weights, bias, response)
    if cache_key is None:
        return compute_neuron_direct(inputs_weights, bias, response)
    
    # Check cache
    if cache_key in generational_cache:
        # Cache HIT - move to end for LRU tracking
        cache_stats['hits'] += 1
        cache_stats['generation_hits'] += 1
        
        value = generational_cache[cache_key]
        del generational_cache[cache_key]
        generational_cache[cache_key] = value
        return value
    else:
        # Cache MISS - compute and store
        cache_stats['misses'] += 1
        cache_stats['generation_misses'] += 1
        
        result = compute_neuron_direct(inputs_weights, bias, response)
        generational_cache[cache_key] = result
        return result

class GenerationalNeuralNetwork:
    """Neural network that builds cache across generations"""
    
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        
        self.input_keys = tuple(sorted(config.genome_config.input_keys))
        self.output_keys = tuple(sorted(config.genome_config.output_keys))
        
        # Build network topology
        self.node_inputs = {}
        self.node_bias = {}
        self.node_response = {}
        
        for (input_key, output_key), conn in genome.connections.items():
            if conn.enabled:
                if output_key not in self.node_inputs:
                    self.node_inputs[output_key] = []
                self.node_inputs[output_key].append((input_key, conn.weight))
        
        for node_key, node in genome.nodes.items():
            self.node_bias[node_key] = getattr(node, 'bias', 0.0)
            self.node_response[node_key] = getattr(node, 'response', 1.0)
        
        self.eval_order = self._compute_evaluation_order()
        self.node_values = {key: 0.0 for key in genome.nodes.keys()}
    
    def _compute_evaluation_order(self):
        """Compute topological evaluation order"""
        in_degree = {node_key: 0 for node_key in self.genome.nodes.keys()}
        
        for output_key in self.node_inputs:
            in_degree[output_key] = len(self.node_inputs[output_key])
        
        queue = list(self.input_keys)
        evaluation_order = []
        
        while queue:
            current_node = queue.pop(0)
            evaluation_order.append(current_node)
            
            for dependent_node in self.node_inputs:
                for input_node, _ in self.node_inputs[dependent_node]:
                    if input_node == current_node:
                        in_degree[dependent_node] -= 1
                        if in_degree[dependent_node] == 0 and dependent_node not in queue:
                            queue.append(dependent_node)
        
        return tuple(evaluation_order)
    
    def activate(self, inputs):
        """Activate network using generational cache"""
        for key in self.node_values:
            self.node_values[key] = 0.0
        
        for i, input_key in enumerate(self.input_keys):
            if i < len(inputs):
                self.node_values[input_key] = inputs[i]
        
        for node_key in self.eval_order:
            if node_key not in self.input_keys:
                inputs_list = self.node_inputs.get(node_key, [])
                if inputs_list:
                    inputs_weights = tuple((self.node_values[input_key], weight)
                                         for input_key, weight in inputs_list)
                    bias = self.node_bias.get(node_key, 0.0)
                    response = self.node_response.get(node_key, 1.0)
                    
                    self.node_values[node_key] = compute_neuron_with_generational_cache(
                        inputs_weights, bias, response
                    )
        
        return [self.node_values[key] for key in self.output_keys]

def create_cache_optimized_features(obs):
    """Create feature vector with quantization for caching"""
    try:
        px = obs[RAM['paddle_x']] / 255.0
        bx = obs[RAM['ball_x']] / 255.0
        by = obs[RAM['ball_y']] / 255.0
        lives = obs[RAM['lives']] / 5.0
        score = min(obs[RAM['score']] / 1000.0, 10.0)
        
        if ABLATION_CONFIG['FEATURE_QUANTIZATION']:
            precision = 50  
            px = round(px * precision) / precision
            bx = round(bx * precision) / precision
            by = round(by * precision) / precision
            lives = round(lives * precision) / precision
            score = round(score * precision) / precision
        
        features = [
            px, bx, by, lives, score,
            round(abs(px - bx), 3),
            round(abs(px - by), 3),
            round(px ** 2, 3), round(bx ** 2, 3), round(by ** 2, 3),
            round(px * bx, 3), round(px * by, 3), round(bx * by, 3),
            round((px - getattr(create_cache_optimized_features, 'prev_px', px)), 3),
            round((bx - getattr(create_cache_optimized_features, 'prev_bx', bx)), 3),
            round((by - getattr(create_cache_optimized_features, 'prev_by', by)), 3),
            round((px - 0.5) ** 2, 3),
            round(px / (bx + 0.01), 3),
            round(px * lives, 3),
            round(bx * score / 1000.0, 3),
            round(np.sin(px * 2 * np.pi), 3),
            round(np.cos(px * 2 * np.pi), 3),
            round(np.sin(bx * 2 * np.pi), 3),
            round(np.cos(bx * 2 * np.pi), 3),
        ]
        
        create_cache_optimized_features.prev_px = px
        create_cache_optimized_features.prev_bx = bx
        create_cache_optimized_features.prev_by = by
        
        # Pad to target size
        target_size = ABLATION_CONFIG['FEATURE_SIZE']
        while len(features) < target_size:
            hash_input = len(features) + int(px * 1000) + int(bx * 1000)
            noise_val = (hash(hash_input) % 1000) / 1000.0
            features.append(round(noise_val, 3))
        
        features = features[:target_size]
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        print(f"Error in feature generation: {e}")
        return (obs / 255.0 + np.random.uniform(-0.001, 0.001, obs.shape)).astype(np.float32)

def create_generational_network(genome, config):
    return GenerationalNeuralNetwork(genome, config)

def eval_genome(genome, config):
    """Evaluate genome with generational cache building"""
    try:
        net = create_generational_network(genome, config)
    except Exception as e:
        print(f"Network creation error: {e}")
        return -1.0
    
    env_seed = RANDOM_SEED + hash(str(genome.key)) % 1000
    env = gym.make(ENV_NAME, obs_type="ram", render_mode=None)
    env.action_space.seed(env_seed)
    obs, _ = env.reset(seed=env_seed)
    
    # Initialize tracking variables
    fitness = 0
    step = 0
    prev_px = int(obs[RAM['paddle_x']])
    prev_bx = int(obs[RAM['ball_x']])
    prev_by = int(obs[RAM['ball_y']])
    prev_lives = 5
    prev_score = 0
    hits = 0
    in_play_steps = 0
    launched = False
    paddle_positions = set()
    no_ball_counter = 0
    
    try:
        while step < MAX_STEPS:
            by = int(obs[RAM['ball_y']])
            
            if by == 0:
                action = 1  # Fire ball
            else:
                try:
                    game_inputs = create_cache_optimized_features(obs)
                    network_outputs = net.activate(game_inputs)
                    action = int(np.argmax(network_outputs) % env.action_space.n)
                except Exception as network_error:
                    print(f"Network error: {network_error}")
                    action = 1
            
            obs, reward, done, truncated, _ = env.step(action)
            
            # Extract current game state
            px = int(obs[RAM['paddle_x']])
            bx = int(obs[RAM['ball_x']])
            by = int(obs[RAM['ball_y']])
            lives = int(obs[RAM['lives']])
            score = int(obs[RAM['score']])
            
            # Fitness calculations
            fitness += reward
            fitness += 0.01 if px != prev_px else -0.002
            
            b = px // 10
            if b not in paddle_positions:
                paddle_positions.add(b)
                fitness += 0.02
            
            if bx > 0 and by > 0 and by < 200:
                dist = min(abs(px - bx), 256 - abs(px - bx))
                prev_dist = min(abs(prev_px - prev_bx), 256 - abs(prev_px - prev_bx))
                if dist < prev_dist:
                    fitness += 0.05
                if dist < 10:
                    fitness += 0.1
            
            if by > 0:
                in_play_steps += 1
                fitness += 0.002
                no_ball_counter = 0
            else:
                no_ball_counter += 1
                if launched and no_ball_counter > 200:
                    fitness -= 0.01 * (no_ball_counter / 200)
            
            if by < 40 and by > 0 and min(abs(bx - px), 256 - abs(bx - px)) < 12:
                if prev_by > by and step - hits > 15:
                    hits += 1
                    fitness += 15.0 + hits * 2.0 + (5.0 if step < 1000 else 0)
            
            if score > prev_score:
                fitness += (score - prev_score) * 0.5
            
            if lives < prev_lives:
                fitness -= 1.0
                hits = 0
            
            fitness += 0.0002
            
            if by > 0 and prev_by == 0:
                launched = True
                fitness += 5.0
            
            prev_px, prev_bx, prev_by = px, bx, by
            prev_lives, prev_score = lives, score
            step += 1
            
            if done or truncated:
                break
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        fitness = 0
    finally:
        env.close()
    
    # Final fitness adjustments
    fitness += len(paddle_positions) * 0.2
    fitness += min(in_play_steps * 0.02, 10.0)
    fitness += hits * 8.0
    
    if not launched:
        fitness -= 5.0
    
    return max(fitness, -1.0)

def get_generational_cache_stats():
    """Get comprehensive cache statistics"""
    total_calls = cache_stats['total_calls']
    
    if not ABLATION_CONFIG['USE_CACHING']:
        return {
            'cache_performance': {
                'hit_rate': 0.0, 'total_hits': 0, 'total_misses': 0,
                'hits': 0, 'misses': 0, 'generation_hits': 0,
                'generation_misses': 0, 'total_calls': total_calls,
                'cache_size': 0, 'max_cache_size': 0, 'cache_utilization': 0.0
            },
            'configuration': dict(ABLATION_CONFIG)
        }
    
    total_cache_ops = cache_stats['hits'] + cache_stats['misses']
    hit_rate = cache_stats['hits'] / total_cache_ops if total_cache_ops > 0 else 0
    
    return {
        'cache_performance': {
            'hit_rate': hit_rate,
            'total_hits': cache_stats['hits'],
            'total_misses': cache_stats['misses'],
            'hits': cache_stats['hits'],
            'misses': cache_stats['misses'],
            'generation_hits': cache_stats['generation_hits'],
            'generation_misses': cache_stats['generation_misses'],
            'total_calls': total_calls,
            'cache_size': len(generational_cache),
            'max_cache_size': ABLATION_CONFIG['CACHE_SIZE'],
            'cache_utilization': len(generational_cache) / ABLATION_CONFIG['CACHE_SIZE']
        },
        'configuration': dict(ABLATION_CONFIG)
    }

def trim_cache_between_generations():
    """Trim cache using LRU eviction"""
    global generational_cache
    
    if not ABLATION_CONFIG['USE_CACHING']:
        return
    
    target_size = ABLATION_CONFIG['CACHE_SIZE']
    current_size = len(generational_cache)
    
    print(f"\nBETWEEN-GENERATION CACHE TRIMMING")
    print(f"Current cache size: {current_size:,}")
    print(f"Target cache size: {target_size:,}")
    
    if current_size <= target_size:
        print(f"✅ Cache within limits - no trimming needed")
        return
    
    excess = current_size - target_size
    print(f"Need to remove: {excess:,} entries")
    
    trim_start = time.time()
    keys_to_remove = list(generational_cache.keys())[:excess]
    
    for key in keys_to_remove:
        if key in generational_cache:
            del generational_cache[key]
    
    trim_time = time.time() - trim_start
    final_size = len(generational_cache)
    
    print(f"Trimming completed in {trim_time:.3f}s")
    print(f"Before: {current_size:,} → After: {final_size:,}")

def clear_generational_cache():
    """Clear entire generational cache"""
    global generational_cache, cache_stats
    
    generational_cache.clear()
    cache_stats = {
        'hits': 0, 'misses': 0, 'total_calls': 0,
        'generation_hits': 0, 'generation_misses': 0,
        'cache_size_history': [], 'hit_rate_history': []
    }

def print_generational_cache_analysis():
    """Print detailed cache performance analysis"""
    if not ABLATION_CONFIG['CACHE_DEBUG']:
        return
    
    stats = get_generational_cache_stats()
    cache_perf = stats['cache_performance']
    config = stats['configuration']
    
    print("\nGENERATIONAL CACHE ANALYSIS")
    print(f"Configuration:")
    print(f"  • Caching: {'ENABLED' if config['USE_CACHING'] else 'DISABLED'}")
    print(f"  • Target cache size: {config['CACHE_SIZE']:,}")
    print(f"  • Computation complexity: {config['COMPUTATION_COMPLEXITY']}")
    print(f"  • Precision level: {config['PRECISION_LEVEL']}")
    
    print(f"\nCache Performance:")
    print(f"  • Current cache size: {cache_perf['cache_size']:,} / {cache_perf['max_cache_size']:,}")
    print(f"  • Overall hit rate: {cache_perf['hit_rate']:.1%}")
    print(f"  • Total hits: {cache_perf['total_hits']:,}")
    print(f"  • Total misses: {cache_perf['total_misses']:,}")
    print(f"  • This generation hits: {cache_perf['generation_hits']:,}")
    print(f"  • This generation misses: {cache_perf['generation_misses']:,}")
    
    if cache_perf['generation_hits'] + cache_perf['generation_misses'] > 0:
        gen_hit_rate = cache_perf['generation_hits'] / (cache_perf['generation_hits'] + cache_perf['generation_misses'])
        print(f"  • This generation hit rate: {gen_hit_rate:.1%}")
    
    cache_utilization = cache_perf['cache_size'] / cache_perf['max_cache_size'] if cache_perf['max_cache_size'] > 0 else 0
    print(f"  • Cache utilization: {cache_utilization:.1%}")

def eval_genomes(genomes, config):
    """Evaluate genomes with cache management"""
    global CURRENT_GENERATION
    
    # Trim cache before generation (except first)
    if CURRENT_GENERATION > 1:
        print(f"\n🔧 BEFORE Generation {CURRENT_GENERATION}: Trimming cache")
        trim_cache_between_generations()
    else:
        print(f"\n🔧 Generation {CURRENT_GENERATION}: First generation - no trimming needed")
    
    cache_stats['generation_hits'] = 0
    cache_stats['generation_misses'] = 0
    
    cache_start_size = len(generational_cache)
    
    print(f"\nGENERATION {CURRENT_GENERATION} - EVALUATION PHASE")
    print(f"Cache size at start: {cache_start_size:,}")
    
    generation_start_time = time.time()
    
    # Evaluate each genome
    for i, (genome_id, genome) in enumerate(genomes):
        print(f"Genome {i+1}/{len(genomes)}: {genome_id}", end=" ")
        start_time = time.time()
        genome.fitness = eval_genome(genome, config)
        eval_time = time.time() - start_time
        print(f"Fitness: {genome.fitness:.2f} ({eval_time:.3f}s)")
    
    generation_time = time.time() - generation_start_time
    cache_end_size = len(generational_cache)
    cache_growth = cache_end_size - cache_start_size
    
    print(f"\nGeneration {CURRENT_GENERATION} Summary:")
    print(f"  • Time: {generation_time:.2f}s")
    print(f"  • Cache growth: {cache_start_size:,} → {cache_end_size:,} (+{cache_growth:,})")
    print(f"  • Cache hits: {cache_stats['generation_hits']:,}")
    print(f"  • Cache misses: {cache_stats['generation_misses']:,}")
    
    if cache_end_size > ABLATION_CONFIG['CACHE_SIZE']:
        excess = cache_end_size - ABLATION_CONFIG['CACHE_SIZE']
        print(f"  • Cache now over limit by {excess:,} - will trim BEFORE next generation")
    else:
        print(f"  • Cache still within limits")
    
    print_generational_cache_analysis()

def get_research_metrics():
    """Get comprehensive metrics for research"""
    stats = get_generational_cache_stats()
    cache_perf = stats['cache_performance']
    
    cache_utilization = cache_perf['cache_size'] / cache_perf['max_cache_size'] if cache_perf['max_cache_size'] > 0 else 0.0
    
    return {
        'cache_metrics': {
            'hit_rate': cache_perf['hit_rate'],
            'total_hits': cache_perf['total_hits'],
            'total_misses': cache_perf['total_misses'],
            'cache_size': cache_perf['cache_size'],
            'max_cache_size': cache_perf['max_cache_size'],
            'cache_utilization': cache_utilization,
            'generation_hits': cache_perf['generation_hits'],
            'generation_misses': cache_perf['generation_misses']
        },
        'configuration': stats['configuration'],
        'generation': CURRENT_GENERATION,
        'cache_growth': cache_stats['cache_size_history'],
        'hit_rate_progression': cache_stats['hit_rate_history']
    }

class AblationReporter(neat.reporting.BaseReporter):
    """Reporter for ablation study experiments"""
    
    def __init__(self, config_path, experiment_name="generational_cache"):
        self.best_genomes = []
        self.config_path = config_path
        self.experiment_name = experiment_name
        self.generation_stats = []
        self.start_time = time.time()
        
        self.results_dir = f'ablation_results_{experiment_name}'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create CSV for results
        self.csv_filename = f'{self.results_dir}/ablation_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        with open(self.csv_filename, 'w', newline='') as csvfile:
            csv.writer(csvfile).writerow([
                'Generation', 'Best_Fitness', 'Avg_Fitness', 'Elapsed_Time_s',
                'Cache_Hit_Rate', 'Cache_Hits', 'Cache_Misses', 'Cache_Size',
                'Cache_Utilization', 'Generation_Hits', 'Generation_Misses',
                'Use_Caching', 'Computation_Complexity', 'Precision_Level',
                'Use_Fast_Keys', 'Use_Conditional_Caching', 'Feature_Quantization',
                'Feature_Size', 'Cache_Max_Size', 'Cache_Eviction_Percentage', 
                'Generation_Time_s', 'Timestamp'
            ])
    
    def post_evaluate(self, config, population, species, best_genome):
        """Called after each generation evaluation"""
        global CURRENT_GENERATION
        generation = len(self.best_genomes) + 1
        CURRENT_GENERATION = generation
        
        elapsed_time = time.time() - self.start_time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        fitness_values = [genome.fitness for genome in population.values()]
        avg_fitness = sum(fitness_values) / len(fitness_values)
        
        metrics = get_research_metrics()
        cache_metrics = metrics['cache_metrics']
        config_metrics = metrics['configuration']
        
        cache_utilization = cache_metrics['cache_utilization']
        
        generation_data = {
            'generation': generation,
            'best_fitness': best_genome.fitness,
            'avg_fitness': avg_fitness,
            'elapsed_time': elapsed_time,
            'timestamp': timestamp,
            **metrics
        }
        self.generation_stats.append(generation_data)
        
        # Write to CSV
        with open(self.csv_filename, 'a', newline='') as csvfile:
            csv.writer(csvfile).writerow([
                generation, best_genome.fitness, avg_fitness, elapsed_time,
                cache_metrics['hit_rate'], cache_metrics['total_hits'],
                cache_metrics['total_misses'], cache_metrics['cache_size'],
                cache_utilization,
                cache_metrics['generation_hits'], cache_metrics['generation_misses'], 
                config_metrics['USE_CACHING'], config_metrics['COMPUTATION_COMPLEXITY'], 
                config_metrics['PRECISION_LEVEL'], config_metrics['USE_FAST_KEYS'], 
                config_metrics['USE_CONDITIONAL_CACHING'], config_metrics['FEATURE_QUANTIZATION'], 
                config_metrics['FEATURE_SIZE'], config_metrics['CACHE_SIZE'], 
                config_metrics.get('CACHE_EVICTION_PERCENTAGE', 0.1),
                0, timestamp
            ])
        
        # Save best genome
        self.best_genomes.append(best_genome)
        genome_filename = f'{self.results_dir}/best_genome_gen_{generation}.pkl'
        with open(genome_filename, 'wb') as f:
            pickle.dump(best_genome, f)
        
        # Print summary
        print(f"\n=== LRU CACHE REPORT - Generation {generation} ===")
        print(f"Best fitness: {best_genome.fitness:.2f}")
        print(f"Average fitness: {avg_fitness:.2f}")
        if config_metrics['USE_CACHING']:
            print(f"Cache performance: {cache_metrics['hit_rate']:.1%} hit rate")
            print(f"Cache size: {cache_metrics['cache_size']:,} / {cache_metrics['max_cache_size']:,}")
            print(f"Cache utilization: {cache_utilization:.1%}")

def run_ablation_experiment(config_path, experiment_config, num_generations=50, experiment_name="test"):
    """Run ablation experiment with specified configuration"""
    global ABLATION_CONFIG, CURRENT_GENERATION
    
    ABLATION_CONFIG.update(experiment_config)
    CURRENT_GENERATION = 1
    
    print("="*80)
    print(f"ABLATION EXPERIMENT: {experiment_name.upper()}")
    print("="*80)
    
    print("Configuration:")
    for key, value in ABLATION_CONFIG.items():
        print(f"  • {key}: {value}")
    print()
    
    clear_generational_cache()
    
    # Set up NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(5))
    
    reporter = AblationReporter(config_path, experiment_name)
    population.add_reporter(reporter)
    
    def eval_genomes_wrapper(genomes, config):
        global CURRENT_GENERATION
        CURRENT_GENERATION = len(reporter.best_genomes) + 1
        eval_genomes(genomes, config)
    
    try:
        winner = population.run(eval_genomes_wrapper, num_generations)
        
        print(f"\nExperiment '{experiment_name}' completed!")
        print(f"Best genome fitness: {winner.fitness:.2f}")
        
        final_metrics = get_research_metrics()
        cache_final = final_metrics['cache_metrics']
        
        print(f"\nFinal Cache Statistics:")
        print(f"  • Hit rate: {cache_final['hit_rate']:.1%}")
        print(f"  • Cache size: {cache_final['cache_size']:,}")
        print(f"  • Total computations saved: {cache_final['total_hits']:,}")
        
        if len(cache_stats['hit_rate_history']) > 1:
            initial_hit_rate = cache_stats['hit_rate_history'][0] if cache_stats['hit_rate_history'] else 0
            final_hit_rate = cache_stats['hit_rate_history'][-1]
            improvement = final_hit_rate - initial_hit_rate
            print(f"  • Hit rate improvement: {initial_hit_rate:.1%} → {final_hit_rate:.1%} (+{improvement:.1%})")
        
        return winner, reporter.generation_stats
    
    except Exception as e:
        print(f"ERROR in experiment '{experiment_name}': {e}")
        import traceback
        traceback.print_exc()
        return None, reporter.generation_stats

def validate_cache_reset():
    """Validate that cache is properly reset"""
    assert len(generational_cache) == 0, f"Cache not empty: {len(generational_cache)} entries"
    assert cache_stats['hits'] == 0, f"Cache hits not zero: {cache_stats['hits']}"
    assert cache_stats['misses'] == 0, f"Cache misses not zero: {cache_stats['misses']}"
    print("✅ Cache reset validation passed")

# Main entry point
if __name__ == "__main__":
    force_complete_cache_reset()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    validate_cache_reset()
    
    print("🚀 Starting experiment...")
    run_ablation_experiment('config-feedforward-breakout', dict(ABLATION_CONFIG), 50, "neat_cache")
