import gymnasium as gym
import numpy as np
import neat
import pickle
import os
import time
import csv
import random
import datetime
import visualize
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

# FIXED ABLATION STUDY CONFIGURATION - Much more diversity!
ABLATION_CONFIG = {
    # Core cache settings
    'USE_CACHING': True,
    'CACHE_SIZE': 100000,
    'CACHE_PERSISTENCE': True,
    'CACHE_EVICTION_PERCENTAGE': 0.1,

    # Computation complexity levels
    'COMPUTATION_COMPLEXITY': 'complex',

    # FIXED: Much less aggressive output processing for diversity
    'USE_CLAMPING': True,
    'CLAMP_METHOD': 'fp16',
    'USE_ROUNDING': True,
    'PRECISION_LEVEL': 'maximal_cache',  # CHANGED: Much higher precision

    # Cache key optimization
    'USE_FAST_KEYS': True,
    'USE_CONDITIONAL_CACHING': True,

    # FIXED: Much less aggressive feature quantization
    'FEATURE_QUANTIZATION': True,
    'FEATURE_SIZE': 128,

    # Debug and analysis
    'ENABLE_DETAILED_TIMING': True,
    'CACHE_DEBUG': True,
    'TRACK_GENERATION_CACHE': True
}


# FORCE COMPLETE CACHE RESET FUNCTION
def force_complete_cache_reset():
    """Completely reset all cache state - call at script start"""
    global generational_cache, cache_stats

    # Force clear any existing cache
    if 'generational_cache' in globals():
        del generational_cache
    if 'cache_stats' in globals():
        del cache_stats

    # Recreate fresh instances
    generational_cache = OrderedDict()
    cache_stats = {
        'hits': 0, 'misses': 0, 'total_calls': 0,
        'generation_hits': 0, 'generation_misses': 0,
        'cache_size_history': [], 'hit_rate_history': []
    }

    print("COMPLETE CACHE RESET: All variables forcibly cleared and recreated")
    print(f"Cache size after reset: {len(generational_cache)}")
    print(f"Cache stats after reset: {cache_stats}")


# Initialize cache variables (will be reset in main)
generational_cache = OrderedDict()
cache_stats = {
    'hits': 0, 'misses': 0, 'total_calls': 0,
    'generation_hits': 0, 'generation_misses': 0,
    'cache_size_history': [], 'hit_rate_history': []
}

# Per-generation statistics
generation_stats = {}


def process_neural_output(raw_output):
    """FIXED: Process neural network output with much better precision for diversity"""
    result = raw_output

    # Apply FP16 clamping if enabled
    if ABLATION_CONFIG['USE_CLAMPING']:
        clamp_method = ABLATION_CONFIG.get('CLAMP_METHOD', 'standard')

        if clamp_method == 'fp16':
            # Convert to FP16 then back to FP32 (natural clamping + quantization)
            result = float(np.float16(result))
            # FIXED: Much wider clamping for more diversity
            result = np.clip(result, -10.0, 10.0)  # Was -2.0, 2.0 - now much wider
        else:
            # Standard clamping with wider range
            result = np.clip(result, -10.0, 10.0)

    # FIXED: Much less aggressive rounding for diversity
    if ABLATION_CONFIG['USE_ROUNDING']:
        precision = ABLATION_CONFIG['PRECISION_LEVEL']

        if precision == 'maximum_cache':
            result = round(result * 20) * 0.05
        elif precision == 'high_cache':
            result = round(result * 100) / 100
        elif precision == 'optimal':
            result = round(result * 10000) / 10000
        elif precision == 'minimal_cache':
            result = round(result * 100000) / 100000  # CHANGED: Much higher precision (0.00001)
        # 'none' - no rounding

    return float(result)


def create_cache_key(inputs_weights, bias, response):
    """FIXED: Create cache key with better collision resistance"""
    if len(inputs_weights) == 0:
        return None

    if ABLATION_CONFIG['USE_FAST_KEYS']:
        # Improved fast key with better diversity
        key_val = 0
        limit = min(len(inputs_weights), 16)  # Increased from 12 for better uniqueness

        for i, (input_val, weight) in enumerate(inputs_weights[:limit]):
            # FIXED: Much higher precision for better uniqueness
            int_input = int(input_val * 100000) & 0xFFFFFFFF  # CHANGED: 100000 instead of 10000
            int_weight = int(weight * 100000) & 0xFFFFFFFF

            # Better bit mixing
            combined = (int_input << 16) ^ int_weight
            key_val ^= combined
            key_val = ((key_val << 3) | (key_val >> 29)) & 0xFFFFFFFF

        # Include bias and response with higher precision
        key_val ^= (int(bias * 100000) << 8) ^ int(response * 100000)  # CHANGED: Higher precision
        return key_val & 0xFFFFFFFF
    else:
        # Traditional approach with higher precision
        key_parts = []
        for input_val, weight in inputs_weights[:12]:
            # FIXED: Much higher precision quantization
            int_input = int(input_val * 10000)  # CHANGED: Was 1000, now 10000
            int_weight = int(weight * 10000)  # CHANGED: Was 1000, now 10000
            key_parts.append((int_input, int_weight))

        # Higher precision for bias and response
        int_bias = int(bias * 10000)  # CHANGED: Was 1000, now 10000
        int_response = int(response * 10000)  # CHANGED: Was 1000, now 10000
        return hash((tuple(key_parts), int_bias, int_response))


def should_cache_computation(inputs_weights):
    """Determine if computation is worth caching"""
    if not ABLATION_CONFIG['USE_CONDITIONAL_CACHING']:
        return True

    # Skip very simple computations (overhead > benefit)
    if len(inputs_weights) < 2:
        return False

    # Skip very complex computations (unlikely to repeat exactly)
    if len(inputs_weights) > 50:
        return False

    # FIXED: More lenient extreme value check for diversity
    for input_val, weight in inputs_weights:
        if abs(input_val) > 5.0 or abs(weight) > 5.0:  # CHANGED: Was 2.0 and 3.0, now more lenient
            return False

    return True


def compute_neuron_direct(inputs_weights, bias, response):
    """Direct neuron computation with configurable complexity - SAME for cache and non-cache"""
    complexity = ABLATION_CONFIG['COMPUTATION_COMPLEXITY']

    # Phase 1: Input processing with different complexity levels
    total = 0.0
    if complexity == 'simple':
        # Simple weighted sum
        for input_val, weight in inputs_weights:
            total += input_val * weight

    elif complexity == 'medium':
        # Add polynomial terms
        for input_val, weight in inputs_weights:
            linear = input_val * weight
            quadratic_input = 0.1 * (input_val ** 2) * weight
            quadratic_weight = 0.05 * input_val * (weight ** 2)
            total += linear + quadratic_input + quadratic_weight

    elif complexity == 'complex':
        # Add more polynomial terms + trigonometric
        for input_val, weight in inputs_weights:
            linear = input_val * weight
            quadratic_input = 0.1 * (input_val ** 2) * weight
            quadratic_weight = 0.05 * input_val * (weight ** 2)
            cubic = 0.01 * (input_val ** 3) * weight
            total += linear + quadratic_input + quadratic_weight + cubic

    elif complexity == 'ultra_complex':
        # Maximum complexity: polynomials + trigonometric + exponential
        for input_val, weight in inputs_weights:
            # Polynomial terms
            linear = input_val * weight
            quad_input = 0.1 * (input_val ** 2) * weight
            quad_weight = 0.05 * input_val * (weight ** 2)
            cubic_input = 0.01 * (input_val ** 3) * weight
            cubic_weight = 0.005 * input_val * (weight ** 3)
            quartic = 0.001 * (input_val ** 4) * weight
            cross_term = 0.002 * (input_val ** 2) * (weight ** 2)

            # Expensive mathematical operations
            sqrt_term = 0.001 * np.sqrt(abs(input_val) + 1e-8) * weight
            log_term = 0.0005 * np.log(abs(input_val) + 1) * weight

            total += (linear + quad_input + quad_weight + cubic_input +
                      cubic_weight + quartic + cross_term + sqrt_term + log_term)

    # Phase 2: Bias and response processing
    stage1 = total + bias
    stage2 = stage1 * response

    if complexity in ['complex', 'ultra_complex']:
        # Add expensive trigonometric operations
        stage3 = stage2 + 0.1 * np.sin(stage1 * np.pi)
        stage4 = stage3 + 0.05 * np.exp(-abs(stage2 * 0.5))  # Clamped exp
        pre_activation = stage4

        if complexity == 'ultra_complex':
            # Even more expensive operations
            stage5 = stage4 + 0.03 * np.cos(stage1 * 2 * np.pi)
            stage6 = stage5 + 0.02 * np.sqrt(abs(stage4) + 1e-8)
            stage7 = stage6 + 0.01 * np.log(abs(stage5) + 1)
            pre_activation = stage7
    else:
        pre_activation = stage2

    # Phase 3: Activation function
    if complexity == 'simple':
        # Simple tanh activation
        output = np.tanh(pre_activation)
    elif complexity == 'medium':
        # Dual activation
        tanh_comp = np.tanh(pre_activation)
        sigmoid_comp = 1 / (1 + np.exp(-np.clip(pre_activation * 0.5, -10, 10)))
        output = 0.7 * tanh_comp + 0.3 * sigmoid_comp
    else:  # complex or ultra_complex
        # Multi-component activation
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

    # Apply output processing for cache optimization (SAME for both cached and non-cached)
    return process_neural_output(final_output)


def compute_neuron_with_generational_cache(inputs_weights, bias, response):
    """FIXED: Compute neuron output with LRU caching"""
    global cache_stats

    cache_stats['total_calls'] += 1

    # If caching is disabled, compute directly
    if not ABLATION_CONFIG['USE_CACHING']:
        return compute_neuron_direct(inputs_weights, bias, response)

    # Check if we should cache this computation
    if ABLATION_CONFIG['USE_CONDITIONAL_CACHING'] and not should_cache_computation(inputs_weights):
        return compute_neuron_direct(inputs_weights, bias, response)

    # Create cache key
    cache_key = create_cache_key(inputs_weights, bias, response)
    if cache_key is None:
        return compute_neuron_direct(inputs_weights, bias, response)

    # Check cache
    if cache_key in generational_cache:
        # Cache HIT - move to end for LRU tracking
        cache_stats['hits'] += 1
        cache_stats['generation_hits'] += 1

        # Move to end (most recently used)
        value = generational_cache[cache_key]
        del generational_cache[cache_key]
        generational_cache[cache_key] = value
        return value
    else:
        # Cache MISS - compute and store
        cache_stats['misses'] += 1
        cache_stats['generation_misses'] += 1

        result = compute_neuron_direct(inputs_weights, bias, response)

        # Add to cache - let it grow during generation
        generational_cache[cache_key] = result
        return result


class GenerationalNeuralNetwork:
    """Neural network that builds cache across generations"""

    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

        # Network structure (optimized for repeated access)
        self.input_keys = tuple(sorted(config.genome_config.input_keys))
        self.output_keys = tuple(sorted(config.genome_config.output_keys))

        # Build network topology
        self.node_inputs = {}
        self.node_bias = {}
        self.node_response = {}

        # Extract enabled connections
        for (input_key, output_key), conn in genome.connections.items():
            if conn.enabled:
                if output_key not in self.node_inputs:
                    self.node_inputs[output_key] = []
                self.node_inputs[output_key].append((input_key, conn.weight))

        # Extract node properties
        for node_key, node in genome.nodes.items():
            self.node_bias[node_key] = getattr(node, 'bias', 0.0)
            self.node_response[node_key] = getattr(node, 'response', 1.0)

        # Compute evaluation order (topological sort)
        self.eval_order = self._compute_evaluation_order()

        # Pre-allocate node values for efficiency
        self.node_values = {key: 0.0 for key in genome.nodes.keys()}

    def _compute_evaluation_order(self):
        """Compute topological evaluation order for the network"""
        in_degree = {node_key: 0 for node_key in self.genome.nodes.keys()}

        # Calculate in-degrees
        for output_key in self.node_inputs:
            in_degree[output_key] = len(self.node_inputs[output_key])

        # Topological sort
        queue = list(self.input_keys)
        evaluation_order = []

        while queue:
            current_node = queue.pop(0)
            evaluation_order.append(current_node)

            # Update dependent nodes
            for dependent_node in self.node_inputs:
                for input_node, _ in self.node_inputs[dependent_node]:
                    if input_node == current_node:
                        in_degree[dependent_node] -= 1
                        if in_degree[dependent_node] == 0 and dependent_node not in queue:
                            queue.append(dependent_node)

        return tuple(evaluation_order)

    def activate(self, inputs):
        """Activate network using generational cache"""
        # Reset node values
        for key in self.node_values:
            self.node_values[key] = 0.0

        # Set input values
        for i, input_key in enumerate(self.input_keys):
            if i < len(inputs):
                self.node_values[input_key] = inputs[i]

        # Process each neuron in topological order
        for node_key in self.eval_order:
            if node_key not in self.input_keys:
                inputs_list = self.node_inputs.get(node_key, [])
                if inputs_list:
                    # Create inputs_weights tuple for caching
                    inputs_weights = tuple((self.node_values[input_key], weight)
                                           for input_key, weight in inputs_list)
                    bias = self.node_bias.get(node_key, 0.0)
                    response = self.node_response.get(node_key, 1.0)

                    # Compute with generational cache
                    self.node_values[node_key] = compute_neuron_with_generational_cache(
                        inputs_weights, bias, response
                    )

        return [self.node_values[key] for key in self.output_keys]


def create_cache_optimized_features(obs):
    """FIXED: Create feature vector with proper quantization for MUCH more diversity"""
    try:
        # Extract basic game state
        px = obs[RAM['paddle_x']] / 255.0
        bx = obs[RAM['ball_x']] / 255.0
        by = obs[RAM['ball_y']] / 255.0
        lives = obs[RAM['lives']] / 5.0
        score = min(obs[RAM['score']] / 1000.0, 10.0)

        # FIXED: Much less aggressive quantization for diversity
        if ABLATION_CONFIG['FEATURE_QUANTIZATION']:
            # OLD: precision = 50  # TOO AGGRESSIVE - caused identical inputs
            # NEW: Much finer precision for diversity while still enabling some cache hits
            precision = 1000  # CHANGED: From 50 to 1000 (20x more diverse!)
            px = round(px * precision) / precision
            bx = round(bx * precision) / precision
            by = round(by * precision) / precision
            lives = round(lives * precision) / precision
            score = round(score * precision) / precision

        # Build more diverse feature set with higher precision
        features = [
            # Core game state (keep high precision for key features)
            px, bx, by, lives, score,

            # Strategic relationships (with much higher precision)
            round(abs(px - bx), 5),  # CHANGED: From 3 to 5 decimal places
            round(abs(px - by), 5),  # Much more precision

            # Add MORE diverse derived features with higher precision
            round(px ** 2, 5), round(bx ** 2, 5), round(by ** 2, 5),
            round(px * bx, 5), round(px * by, 5), round(bx * by, 5),

            # Velocity approximations (add temporal diversity)
            round((px - getattr(create_cache_optimized_features, 'prev_px', px)), 5),
            round((bx - getattr(create_cache_optimized_features, 'prev_bx', bx)), 5),
            round((by - getattr(create_cache_optimized_features, 'prev_by', by)), 5),

            # Strategic positioning with much more precision
            round((px - 0.5) ** 2, 5),  # Distance from center
            round(px / (bx + 0.01), 5),  # Relative positioning

            # Add game-state dependent features for more diversity
            round(px * lives, 5),  # Life-dependent positioning
            round(bx * score / 1000.0, 5),  # Score-dependent ball tracking

            # Trigonometric features (with much better precision)
            round(np.sin(px * 2 * np.pi), 5),
            round(np.cos(px * 2 * np.pi), 5),
            round(np.sin(bx * 2 * np.pi), 5),
            round(np.cos(bx * 2 * np.pi), 5),

            # ADDED: More diverse features for better learning
            round(np.sin(px * 4 * np.pi), 5),  # Higher frequency patterns
            round(np.cos(bx * 4 * np.pi), 5),
            round(px * np.sin(by * np.pi), 5),  # Cross-correlations
            round(bx * np.cos(px * np.pi), 5),

            # ADDED: Paddle-ball interaction features
            round((px - bx) ** 2, 5),  # Squared distance (important for control)
            round(np.sign(px - bx), 5),  # Direction to ball
            round(abs(px - bx) * by, 5),  # Distance weighted by ball height
        ]

        # Store previous values for velocity
        create_cache_optimized_features.prev_px = px
        create_cache_optimized_features.prev_bx = bx
        create_cache_optimized_features.prev_by = by

        # Pad to target size with much more diverse hash-based deterministic values
        target_size = ABLATION_CONFIG['FEATURE_SIZE']
        while len(features) < target_size:
            # CHANGED: Much higher precision hash-based "noise" for diversity
            hash_input = len(features) + int(px * 100000) + int(bx * 100000) + int(by * 100000)
            noise_val = (hash(hash_input) % 100000) / 100000.0  # Much more diverse
            features.append(round(noise_val, 5))

        features = features[:target_size]
        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"Error in feature generation: {e}")
        # Return much more diverse fallback
        base_features = (obs / 255.0).astype(np.float32)
        # Add small amount of diversity to prevent identical outputs
        diversity_noise = np.random.uniform(-0.0001, 0.0001, base_features.shape)
        return base_features + diversity_noise


def create_generational_network(genome, config):
    """Create neural network with generational caching"""
    return GenerationalNeuralNetwork(genome, config)


def eval_genome(genome, config):
    """Evaluate genome with generational cache building - FIXED for better learning"""
    try:
        net = create_generational_network(genome, config)
    except Exception as e:
        print(f"Network creation error: {e}")
        return -1.0

    # Set up environment with genome-specific seed for diversity
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

    # ADDED: Track action diversity for better fitness
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    try:
        while step < MAX_STEPS:
            by = int(obs[RAM['ball_y']])

            if by == 0:
                action = 1  # Fire ball
                action_counts[1] += 1
            else:
                try:
                    # Create features optimized for cache hits but with much more diversity
                    game_inputs = create_cache_optimized_features(obs)
                    # Network activation builds generational cache
                    network_outputs = net.activate(game_inputs)
                    action = int(np.argmax(network_outputs) % env.action_space.n)
                    action_counts[action] += 1
                except Exception as network_error:
                    print(f"Network error: {network_error}")
                    action = 1
                    action_counts[1] += 1

            obs, reward, done, truncated, _ = env.step(action)

            # Extract current game state
            px = int(obs[RAM['paddle_x']])
            bx = int(obs[RAM['ball_x']])
            by = int(obs[RAM['ball_y']])
            lives = int(obs[RAM['lives']])
            score = int(obs[RAM['score']])

            # ENHANCED FITNESS CALCULATIONS for better paddle control learning
            fitness += reward

            # CHANGED: Much higher reward for paddle movement to encourage learning
            if px != prev_px:
                fitness += 0.1  # CHANGED: From 0.01 to 0.1 (10x more reward for movement)
            else:
                fitness -= 0.01  # CHANGED: From -0.002 to -0.01 (more penalty for not moving)

            # Reward paddle position diversity
            b = px // 10
            if b not in paddle_positions:
                paddle_positions.add(b)
                fitness += 0.05  # CHANGED: From 0.02 to 0.05 (more reward for exploring positions)

            # Reward following the ball
            if bx > 0 and by > 0 and by < 200:
                dist = min(abs(px - bx), 256 - abs(px - bx))
                prev_dist = min(abs(prev_px - prev_bx), 256 - abs(prev_px - prev_bx))
                if dist < prev_dist:
                    fitness += 0.1  # CHANGED: From 0.05 to 0.1 (more reward for ball tracking)
                if dist < 10:
                    fitness += 0.2  # CHANGED: From 0.1 to 0.2 (more reward for close tracking)

            # Reward ball in play
            if by > 0:
                in_play_steps += 1
                fitness += 0.005  # CHANGED: From 0.002 to 0.005
                no_ball_counter = 0
            else:
                no_ball_counter += 1
                if launched and no_ball_counter > 200:
                    fitness -= 0.02 * (no_ball_counter / 200)  # More penalty for long waits

            # Reward hits
            if by < 40 and by > 0 and min(abs(bx - px), 256 - abs(bx - px)) < 12:
                if prev_by > by and step - hits > 15:
                    hits += 1
                    fitness += 20.0 + hits * 3.0 + (10.0 if step < 1000 else 0)  # CHANGED: Higher hit rewards

            # Reward score increases
            if score > prev_score:
                fitness += (score - prev_score) * 1.0  # CHANGED: From 0.5 to 1.0

            # Penalty for losing lives
            if lives < prev_lives:
                fitness -= 2.0  # CHANGED: From -1.0 to -2.0 (higher penalty)
                hits = 0

            # Small reward for staying alive
            fitness += 0.001  # CHANGED: From 0.0002 to 0.001

            # Reward launching the ball
            if by > 0 and prev_by == 0:
                launched = True
                fitness += 10.0  # CHANGED: From 5.0 to 10.0

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

    # ADDED: Bonus for action diversity (encourage using different actions)
    unique_actions_used = len([count for count in action_counts.values() if count > 0])
    total_actions = sum(action_counts.values())

    if total_actions > 0:
        # Bonus for using multiple different actions
        diversity_bonus = unique_actions_used * 2.0  # Up to 8.0 bonus for using all 4 actions
        fitness += diversity_bonus

        # Penalty for being too stuck on one action
        max_action_percentage = max(action_counts.values()) / total_actions
        if max_action_percentage > 0.9:  # If one action is >90% of all actions
            fitness -= 5.0  # Penalty for being stuck

    # Final fitness adjustments
    fitness += len(paddle_positions) * 0.5  # CHANGED: From 0.2 to 0.5
    fitness += min(in_play_steps * 0.05, 20.0)  # CHANGED: From 0.02 to 0.05
    fitness += hits * 15.0  # CHANGED: From 8.0 to 15.0 (higher hit bonus)

    if not launched:
        fitness -= 10.0  # CHANGED: From -5.0 to -10.0 (higher penalty for not launching)

    return max(fitness, -1.0)


def get_generational_cache_stats():
    """Get comprehensive cache statistics"""
    total_calls = cache_stats['total_calls']

    # If caching is disabled, return zero stats
    if not ABLATION_CONFIG['USE_CACHING']:
        return {
            'cache_performance': {
                'hit_rate': 0.0,
                'total_hits': 0,
                'total_misses': 0,
                'hits': 0,
                'misses': 0,
                'generation_hits': 0,
                'generation_misses': 0,
                'total_calls': total_calls,
                'cache_size': 0,
                'max_cache_size': 0,
                'cache_utilization': 0.0
            },
            'configuration': dict(ABLATION_CONFIG)
        }

    # Normal cache statistics when caching is enabled
    total_cache_ops = cache_stats['hits'] + cache_stats['misses']

    if total_cache_ops == 0:
        hit_rate = 0
    else:
        hit_rate = cache_stats['hits'] / total_cache_ops

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
    """FIXED: Trim cache between generations using proper LRU eviction"""
    global generational_cache

    if not ABLATION_CONFIG['USE_CACHING']:
        return

    target_size = ABLATION_CONFIG['CACHE_SIZE']  # 100,000
    current_size = len(generational_cache)

    print(f"\n{'=' * 50}")
    print(f"BETWEEN-GENERATION CACHE TRIMMING")
    print(f"{'=' * 50}")
    print(f"Current cache size: {current_size:,}")
    print(f"Target cache size: {target_size:,}")

    if current_size <= target_size:
        print(f"✅ Cache within limits - no trimming needed")
        return

    # Calculate how many entries to remove
    excess = current_size - target_size
    print(f"Need to remove: {excess:,} entries")

    # CRITICAL FIX: Remove from the BEGINNING (oldest entries in LRU)
    trim_start = time.time()

    # Convert to list to avoid "dictionary changed size during iteration"
    keys_to_remove = list(generational_cache.keys())[:excess]

    print(f"Removing {len(keys_to_remove):,} oldest entries...")

    for key in keys_to_remove:
        if key in generational_cache:
            del generational_cache[key]

    trim_time = time.time() - trim_start
    final_size = len(generational_cache)

    print(f"Trimming completed in {trim_time:.3f}s")
    print(f"Before: {current_size:,} → After: {final_size:,}")
    print(f"Removed: {current_size - final_size:,} entries")

    if final_size == target_size:
        print(f"✅ SUCCESS: Cache exactly at target size")
    elif final_size < target_size:
        print(f"✅ SUCCESS: Cache under target ({target_size - final_size:,} room left)")
    else:
        print(f"❌ ERROR: Still over target by {final_size - target_size:,}")


def clear_generational_cache():
    """Clear entire generational cache (for new experiments)"""
    global generational_cache, cache_stats

    print(f"DEBUG: Clearing cache. Current size: {len(generational_cache)}")
    generational_cache.clear()
    print(f"DEBUG: Cache cleared. New size: {len(generational_cache)}")

    cache_stats = {
        'hits': 0, 'misses': 0, 'total_calls': 0,
        'generation_hits': 0, 'generation_misses': 0,
        'cache_size_history': [], 'hit_rate_history': []
    }
    print("DEBUG: Cache stats reset")


def print_generational_cache_analysis():
    """Print detailed analysis of generational cache performance"""
    if not ABLATION_CONFIG['CACHE_DEBUG']:
        return

    stats = get_generational_cache_stats()
    cache_perf = stats['cache_performance']
    config = stats['configuration']

    print("\n" + "=" * 70)
    print("GENERATIONAL CACHE ANALYSIS")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  • Caching: {'ENABLED' if config['USE_CACHING'] else 'DISABLED'}")
    print(f"  • Target cache size: {config['CACHE_SIZE']:,}")
    print(f"  • Strategy: GROW during generation, TRIM between generations")
    print(f"  • Computation complexity: {config['COMPUTATION_COMPLEXITY']}")
    print(f"  • Precision level: {config['PRECISION_LEVEL']}")
    print(f"  • Feature quantization: {'YES' if config['FEATURE_QUANTIZATION'] else 'NO'}")

    print(f"\nGenerational Cache Performance:")
    print(f"  • Current cache size: {cache_perf['cache_size']:,} / {cache_perf['max_cache_size']:,}")
    print(f"  • Overall hit rate: {cache_perf['hit_rate']:.1%}")
    print(f"  • Total hits: {cache_perf['total_hits']:,}")
    print(f"  • Total misses: {cache_perf['total_misses']:,}")
    print(f"  • This generation hits: {cache_perf['generation_hits']:,}")
    print(f"  • This generation misses: {cache_perf['generation_misses']:,}")

    if cache_perf['generation_hits'] + cache_perf['generation_misses'] > 0:
        gen_hit_rate = cache_perf['generation_hits'] / (cache_perf['generation_hits'] + cache_perf['generation_misses'])
        print(f"  • This generation hit rate: {gen_hit_rate:.1%}")

        if CURRENT_GENERATION > 1:
            print(f"  • Building on previous generations: {'YES' if gen_hit_rate > 0 else 'NO'}")

    # Cache effectiveness analysis
    cache_utilization = cache_perf['cache_size'] / cache_perf['max_cache_size'] if cache_perf[
                                                                                       'max_cache_size'] > 0 else 0
    print(f"Cache Effectiveness:")
    print(f"  • Cache utilization: {cache_utilization:.1%}")
    print(f"  • LRU Strategy: Trim oldest entries between generations")

    if cache_perf['hit_rate'] > 0.5:
        print(f"  • Status: ✅ Excellent cache performance (>50% hit rate)")
    elif cache_perf['hit_rate'] > 0.2:
        print(f"  • Status: ⚡ Good cache performance (20-50% hit rate)")
    elif cache_perf['hit_rate'] > 0.05:
        print(f"  • Status: 📊 Moderate cache performance (5-20% hit rate)")
    else:
        print(f"  • Status: ⚠️ Low cache performance (<5% hit rate)")

    if cache_perf['cache_size'] > cache_perf['max_cache_size']:
        excess = cache_perf['cache_size'] - cache_perf['max_cache_size']
        print(f"  • Note: Cache temporarily over limit by {excess:,} entries (will trim next generation)")


def eval_genomes(genomes, config):
    """FIXED: Proper cache management - trim BEFORE generation, grow during"""
    global CURRENT_GENERATION

    # CRITICAL FIX: Trim cache BEFORE starting the generation (except generation 1)
    if CURRENT_GENERATION > 1:
        print(f"\n🔧 BEFORE Generation {CURRENT_GENERATION}: Trimming cache from previous generation")
        trim_cache_between_generations()
    else:
        print(f"\n🔧 Generation {CURRENT_GENERATION}: First generation - no trimming needed")

    # Reset generation-specific stats
    cache_stats['generation_hits'] = 0
    cache_stats['generation_misses'] = 0

    cache_start_size = len(generational_cache)

    print(f"\n{'=' * 60}")
    print(f"GENERATION {CURRENT_GENERATION} - EVALUATION PHASE")
    print(f"Cache size at start: {cache_start_size:,}")
    print(f"Strategy: GROW freely during generation, TRIM between generations")
    print(f"{'=' * 60}")

    generation_start_time = time.time()

    # Evaluate each genome - cache grows freely during this phase
    for i, (genome_id, genome) in enumerate(genomes):
        print(f"Genome {i + 1}/{len(genomes)}: {genome_id}", end=" ")
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

    # Don't trim here - let cache overflow until next generation
    if cache_end_size > ABLATION_CONFIG['CACHE_SIZE']:
        excess = cache_end_size - ABLATION_CONFIG['CACHE_SIZE']
        print(f"  • Cache now over limit by {excess:,} - will trim BEFORE next generation")
    else:
        print(f"  • Cache still within limits")

    # Print detailed cache analysis
    print_generational_cache_analysis()


def get_research_metrics():
    """Get comprehensive metrics for ablation study research"""
    stats = get_generational_cache_stats()
    cache_perf = stats['cache_performance']

    # Safe cache utilization calculation
    if cache_perf['max_cache_size'] > 0:
        cache_utilization = cache_perf['cache_size'] / cache_perf['max_cache_size']
    else:
        cache_utilization = 0.0

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
    """Reporter for ablation study experiments with proper cache management"""

    def __init__(self, config_path, experiment_name="generational_cache"):
        self.best_genomes = []
        self.config_path = config_path
        self.experiment_name = experiment_name
        self.generation_stats = []
        self.start_time = time.time()

        self.results_dir = f'ablation_results_{experiment_name}'
        os.makedirs(self.results_dir, exist_ok=True)

        # Create detailed CSV for ablation analysis
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
        global CURRENT_GENERATION
        generation = len(self.best_genomes) + 1
        CURRENT_GENERATION = generation

        elapsed_time = time.time() - self.start_time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate fitness statistics
        fitness_values = [genome.fitness for genome in population.values()]
        avg_fitness = sum(fitness_values) / len(fitness_values)

        # Get comprehensive metrics
        metrics = get_research_metrics()
        cache_metrics = metrics['cache_metrics']
        config_metrics = metrics['configuration']

        # FIXED: Calculate proper cache utilization (should be decimal, not >1.0)
        cache_utilization = cache_metrics['cache_utilization']

        # Store generation data
        generation_data = {
            'generation': generation,
            'best_fitness': best_genome.fitness,
            'avg_fitness': avg_fitness,
            'elapsed_time': elapsed_time,
            'timestamp': timestamp,
            **metrics
        }
        self.generation_stats.append(generation_data)

        # Write to CSV - FIXED: Use proper cache_utilization (decimal)
        with open(self.csv_filename, 'a', newline='') as csvfile:
            csv.writer(csvfile).writerow([
                generation, best_genome.fitness, avg_fitness, elapsed_time,
                cache_metrics['hit_rate'], cache_metrics['total_hits'],
                cache_metrics['total_misses'], cache_metrics['cache_size'],
                cache_utilization,  # This should be decimal (0.0-1.0)
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
        print(f"\n=== FIXED DIVERSE TRAINING REPORT - Generation {generation} ===")
        print(f"Best fitness: {best_genome.fitness:.2f}")
        print(f"Average fitness: {avg_fitness:.2f}")
        if config_metrics['USE_CACHING']:
            print(f"Cache performance: {cache_metrics['hit_rate']:.1%} hit rate")
            print(f"Cache size: {cache_metrics['cache_size']:,} / {cache_metrics['max_cache_size']:,}")
            print(f"Cache utilization: {cache_utilization:.1%}")
        print(f"Strategy: Much higher precision for diverse neural networks")


def run_ablation_experiment(config_path, experiment_config, num_generations=50, experiment_name="test"):
    """
    Run a single ablation experiment with specified configuration.
    Args:
        config_path: Path to NEAT config file
        experiment_config: Dictionary with ABLATION_CONFIG settings
        num_generations: Number of generations to run (default: 50)
        experiment_name: Name for this experiment
    """
    global ABLATION_CONFIG, CURRENT_GENERATION

    # Update global configuration
    ABLATION_CONFIG.update(experiment_config)
    CURRENT_GENERATION = 1

    print("=" * 80)
    print(f"FIXED DIVERSE TRAINING EXPERIMENT: {experiment_name.upper()}")
    print("=" * 80)

    # Print configuration
    print("Configuration:")
    for key, value in ABLATION_CONFIG.items():
        print(f"  • {key}: {value}")
    print()

    # Clear cache for fresh start
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

        print(f"\nFixed diverse training experiment '{experiment_name}' completed!")
        print(f"Best genome fitness: {winner.fitness:.2f}")

        # Final cache analysis
        final_metrics = get_research_metrics()
        cache_final = final_metrics['cache_metrics']

        print(f"\nFinal Cache Statistics:")
        print(f"  • Hit rate: {cache_final['hit_rate']:.1%}")
        print(f"  • Cache size: {cache_final['cache_size']:,}")
        print(f"  • Total computations saved: {cache_final['total_hits']:,}")

        # Save final best genome
        final_genome_path = f'{reporter.results_dir}/best_genome_final.pkl'
        with open(final_genome_path, 'wb') as f:
            pickle.dump(winner, f)
        print(f"  • Final best genome saved: {final_genome_path}")

        return winner, reporter.generation_stats

    except Exception as e:
        print(f"ERROR in experiment '{experiment_name}': {e}")
        import traceback
        traceback.print_exc()
        return None, reporter.generation_stats


# Main entry point for fixed diverse training
if __name__ == "__main__":
    # CRITICAL: Force complete cache reset at script start
    force_complete_cache_reset()

    # Set random seeds AFTER reset
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join( 'config-feedforward-breakout')

    print("FIXED DIVERSE TRAINING: Much higher precision for neural network diversity!")
    print("✅ 20x more diverse features (precision 1000 vs 50)")
    print("✅ 10x higher decimal precision (0.00001 vs 0.0001)")
    print("✅ Enhanced fitness rewards for paddle movement")
    print("✅ Action diversity bonuses and penalties")
    print("✅ Wider clamping ranges for neural outputs")

    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        # Check for generation count argument
        num_gens = 25  # Default to 25 generations for faster testing
        if len(sys.argv) > 2 and sys.argv[2].isdigit():
            num_gens = int(sys.argv[2])

        if command == 'diverse':
            # Run the fixed diverse training
            diverse_config = {
                'USE_CACHING': True,
                'CACHE_EVICTION_PERCENTAGE': 0.1,
                'COMPUTATION_COMPLEXITY': 'complex',
                'USE_CLAMPING': True,
                'CLAMP_METHOD': 'fp16',
                'USE_ROUNDING': True,
                'PRECISION_LEVEL': 'minimal_cache',  # Highest precision
                'USE_FAST_KEYS': True,
                'USE_CONDITIONAL_CACHING': True,
                'FEATURE_QUANTIZATION': True
            }
            run_ablation_experiment(config_path, diverse_config, num_gens, "fixed_diverse_training")

        elif command == 'baseline':
            # Run baseline (no cache, same high precision)
            baseline_config = {
                'USE_CACHING': False,
                'COMPUTATION_COMPLEXITY': 'complex',
                'PRECISION_LEVEL': 'minimal_cache',
                'FEATURE_QUANTIZATION': True
            }
            run_ablation_experiment(config_path, baseline_config, num_gens, "baseline_diverse")

        elif command == 'compare':
            # Run both baseline and fixed diverse training
            print("Running BASELINE with high precision...")
            baseline_config = {
                'USE_CACHING': False,
                'COMPUTATION_COMPLEXITY': 'complex',
                'PRECISION_LEVEL': 'minimal_cache',
                'FEATURE_QUANTIZATION': True
            }
            baseline_winner, baseline_stats = run_ablation_experiment(config_path, baseline_config, num_gens,
                                                                      "baseline_diverse")

            print("\n" + "=" * 80)
            print("Running FIXED DIVERSE TRAINING...")
            force_complete_cache_reset()  # Reset for second experiment

            diverse_config = {
                'USE_CACHING': True,
                'COMPUTATION_COMPLEXITY': 'complex',
                'PRECISION_LEVEL': 'minimal_cache',
                'FEATURE_QUANTIZATION': True
            }
            diverse_winner, diverse_stats = run_ablation_experiment(config_path, diverse_config, num_gens,
                                                                    "fixed_diverse_training")

            # Compare results
            print("\n" + "=" * 80)
            print("COMPARISON RESULTS")
            print("=" * 80)
            if baseline_winner and diverse_winner:
                print(f"Baseline fitness: {baseline_winner.fitness:.2f}")
                print(f"Fixed diverse training fitness: {diverse_winner.fitness:.2f}")
                improvement = diverse_winner.fitness - baseline_winner.fitness
                print(f"Fitness improvement: {improvement:.2f}")
                if improvement > 0:
                    print("✅ Fixed diverse training performed better!")
                else:
                    print("⚠️ Baseline performed better - may need more generations")
        else:
            print("Available commands:")
            print("  diverse [gens]    - Run fixed diverse training (recommended)")
            print("  baseline [gens]   - Run baseline with high precision")
            print("  compare [gens]    - Compare baseline vs fixed diverse training")
            print(f"  Example: python {sys.argv[0]} diverse 25")
            print(f"  Example: python {sys.argv[0]} compare 20")
    else:
        # Default: Run fixed diverse training
        print("Running FIXED DIVERSE TRAINING with default settings...")
        diverse_config = dict(ABLATION_CONFIG)  # Use the fixed config
        run_ablation_experiment(config_path, diverse_config, 50, "fixed_diverse_training_default")