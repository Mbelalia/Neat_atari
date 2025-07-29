import gymnasium as gym
import numpy as np
import neat
import pickle
import os
import time
import argparse
import glob
import ale_py
import pygame
from collections import OrderedDict

# Setup the Atari Breakout environment
ENV_NAME = 'ALE/Breakout-v5'

# Set fixed seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# RAM addresses for game state (same as training)
RAM = {
    'ball_x': 99,
    'ball_y': 101,
    'paddle_x': 72,
    'lives': 57,
    'score': 84
}

# COPIED FROM TRAINING: Feature creation function
def create_cache_optimized_features(obs):
    """Create feature vector EXACTLY like training code"""
    try:
        # Extract basic game state
        px = obs[RAM['paddle_x']] / 255.0
        bx = obs[RAM['ball_x']] / 255.0
        by = obs[RAM['ball_y']] / 255.0
        lives = obs[RAM['lives']] / 5.0
        score = min(obs[RAM['score']] / 1000.0, 10.0)

        # Apply the same quantization as training
        precision = 1000  # Same as training code
        px = round(px * precision) / precision
        bx = round(bx * precision) / precision
        by = round(by * precision) / precision
        lives = round(lives * precision) / precision
        score = round(score * precision) / precision

        # Build the same feature set as training
        features = [
            # Core game state
            px, bx, by, lives, score,

            # Strategic relationships
            round(abs(px - bx), 5),
            round(abs(px - by), 5),

            # Derived features
            round(px ** 2, 5), round(bx ** 2, 5), round(by ** 2, 5),
            round(px * bx, 5), round(px * by, 5), round(bx * by, 5),

            # Velocity approximations
            round((px - getattr(create_cache_optimized_features, 'prev_px', px)), 5),
            round((bx - getattr(create_cache_optimized_features, 'prev_bx', bx)), 5),
            round((by - getattr(create_cache_optimized_features, 'prev_by', by)), 5),

            # Strategic positioning
            round((px - 0.5) ** 2, 5),
            round(px / (bx + 0.01), 5),

            # Game-state dependent features
            round(px * lives, 5),
            round(bx * score / 1000.0, 5),

            # Trigonometric features
            round(np.sin(px * 2 * np.pi), 5),
            round(np.cos(px * 2 * np.pi), 5),
            round(np.sin(bx * 2 * np.pi), 5),
            round(np.cos(bx * 2 * np.pi), 5),

            # Additional diverse features (same as training)
            round(np.sin(px * 4 * np.pi), 5),
            round(np.cos(bx * 4 * np.pi), 5),
            round(px * np.sin(by * np.pi), 5),
            round(bx * np.cos(px * np.pi), 5),

            # Paddle-ball interaction features
            round((px - bx) ** 2, 5),
            round(np.sign(px - bx), 5),
            round(abs(px - bx) * by, 5),
        ]

        # Store previous values
        create_cache_optimized_features.prev_px = px
        create_cache_optimized_features.prev_bx = bx
        create_cache_optimized_features.prev_by = by

        # Pad to target size (128 features like training)
        target_size = 128
        while len(features) < target_size:
            hash_input = len(features) + int(px * 100000) + int(bx * 100000) + int(by * 100000)
            noise_val = (hash(hash_input) % 100000) / 100000.0
            features.append(round(noise_val, 5))

        features = features[:target_size]
        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"Error in feature generation: {e}")
        # Fallback to simple normalization
        return (obs / 255.0).astype(np.float32)[:128]  # Take first 128 if needed

class ReplayGenerationalNetwork:
    """Network that can handle genomes trained with the new training code"""

    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

        # Try to create network in multiple ways for compatibility
        self.network = None
        self.network_type = None

        # Method 1: Try the training code's custom network structure
        try:
            self.network = self._create_custom_network(genome, config)
            self.network_type = "custom"
            print("✅ Using custom generational network structure")
        except Exception as e1:
            print(f"Custom network failed: {e1}")

            # Method 2: Try standard NEAT network
            try:
                self.network = neat.nn.FeedForwardNetwork.create(genome, config)
                self.network_type = "standard"
                print("✅ Using standard NEAT network")
            except Exception as e2:
                print(f"Standard network failed: {e2}")
                raise Exception(f"Could not create network: Custom={e1}, Standard={e2}")

    def _create_custom_network(self, genome, config):
        """Try to recreate the custom network from training code"""
        # This mimics the GenerationalNeuralNetwork structure
        input_keys = tuple(sorted(config.genome_config.input_keys))
        output_keys = tuple(sorted(config.genome_config.output_keys))

        # Build network topology
        node_inputs = {}
        node_bias = {}
        node_response = {}

        # Extract enabled connections
        for (input_key, output_key), conn in genome.connections.items():
            if conn.enabled:
                if output_key not in node_inputs:
                    node_inputs[output_key] = []
                node_inputs[output_key].append((input_key, conn.weight))

        # Extract node properties
        for node_key, node in genome.nodes.items():
            node_bias[node_key] = getattr(node, 'bias', 0.0)
            node_response[node_key] = getattr(node, 'response', 1.0)

        # Create a simple network object
        class SimpleNetwork:
            def __init__(self):
                self.input_keys = input_keys
                self.output_keys = output_keys
                self.node_inputs = node_inputs
                self.node_bias = node_bias
                self.node_response = node_response
                self.node_values = {key: 0.0 for key in genome.nodes.keys()}

            def activate(self, inputs):
                # Reset node values
                for key in self.node_values:
                    self.node_values[key] = 0.0

                # Set input values
                for i, input_key in enumerate(self.input_keys):
                    if i < len(inputs):
                        self.node_values[input_key] = inputs[i]

                # Simple forward pass (without caching for replay)
                for node_key in sorted(self.node_values.keys()):
                    if node_key not in self.input_keys:
                        inputs_list = self.node_inputs.get(node_key, [])
                        if inputs_list:
                            total = 0.0
                            for input_key, weight in inputs_list:
                                total += self.node_values[input_key] * weight

                            # Add bias and response
                            total += self.node_bias.get(node_key, 0.0)
                            total *= self.node_response.get(node_key, 1.0)

                            # Apply activation (tanh)
                            self.node_values[node_key] = np.tanh(total)

                return [self.node_values[key] for key in self.output_keys]

        return SimpleNetwork()

    def activate(self, inputs):
        """Activate the network"""
        if self.network_type == "custom":
            return self.network.activate(inputs)
        else:
            return self.network.activate(inputs)

def play_genome_debug(config_path, genome_path, render_mode='human', delay=0.05, max_steps=5000, window_size=(820, 410)):
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load the genome
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)

    print(f"Loaded genome {genome.key} with fitness {genome.fitness}")

    # Create the network (compatible with new training)
    try:
        net = ReplayGenerationalNetwork(genome, config)
        print(f"✅ Network created successfully")
    except Exception as e:
        print(f"❌ Network creation failed: {e}")
        print("Trying fallback to standard NEAT network...")
        try:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            print("✅ Using standard NEAT network as fallback")
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return 0

    # Create the environment
    actual_render_mode = 'rgb_array' if render_mode == 'human' else render_mode
    env = gym.make(ENV_NAME, obs_type="ram", render_mode=actual_render_mode)

    # DEBUGGING: Print environment action meanings
    print(f"Available actions: {env.unwrapped.get_action_meanings()}")
    print(f"Action space size: {env.action_space.n}")

    # Set environment seed
    env.action_space.seed(RANDOM_SEED)
    env.observation_space.seed(RANDOM_SEED)
    observation, info = env.reset(seed=RANDOM_SEED)

    # Initialize pygame
    pygame_initialized = False
    screen = None
    current_delay = delay

    if render_mode == 'human':
        pygame.init()
        pygame_initialized = True
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(
            f"FIXED REPLAY - {os.path.basename(genome_path)} - Network: {getattr(net, 'network_type', 'unknown')}")

    # Game state
    done = False
    truncated = False
    total_reward = 0
    step_count = 0

    # DEBUG: Track actions taken
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    recent_actions = []
    recent_outputs = []

    # Feature processing tracking
    use_new_features = True
    feature_type = "NEW FEATURES (128 enhanced)"  # FIXED: Initialize feature_type

    try:
        while not (done or truncated) and step_count < max_steps:
            # Process pygame events
            if pygame_initialized:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            current_delay = max(0.01, current_delay - 0.01)
                            pygame.display.set_caption(f"FIXED REPLAY - Speed: {current_delay:.3f}s")
                        elif event.key == pygame.K_DOWN:
                            current_delay += 0.01
                            pygame.display.set_caption(f"FIXED REPLAY - Speed: {current_delay:.3f}s")
                        elif event.key == pygame.K_SPACE:
                            if current_delay > 0:
                                saved_delay = current_delay
                                current_delay = 0
                                print("Game paused - Press SPACE to resume")
                            else:
                                current_delay = saved_delay
                                print(f"Game resumed - Delay: {current_delay:.3f}s")
                        elif event.key == pygame.K_f:
                            # Toggle between new features and old raw RAM
                            use_new_features = not use_new_features
                            feature_type = "NEW FEATURES" if use_new_features else "RAW RAM"
                            print(f"Switched to {feature_type}")

            # Get ball position for fire control
            ball_y = int(observation[RAM['ball_y']])

            if ball_y == 0:  # Ball not in play
                action = 1  # Fire ball
                feature_type = "FIRE MODE (ball not in play)"  # FIXED: Set feature_type for fire mode
                print(f"Step {step_count}: FIRING BALL (ball_y = 0)")
            else:
                # Process observation for network
                if use_new_features:
                    # Use the SAME feature processing as training
                    try:
                        processed_observation = create_cache_optimized_features(observation)
                        feature_type = "NEW FEATURES (128 enhanced)"
                    except Exception as fe:
                        print(f"Feature processing failed: {fe}, falling back to raw RAM")
                        processed_observation = observation / 255.0
                        feature_type = "RAW RAM (fallback)"
                else:
                    # Use raw RAM like old training
                    processed_observation = observation / 255.0
                    feature_type = "RAW RAM"

                # Get network output
                try:
                    if hasattr(net, 'activate'):
                        network_outputs = net.activate(processed_observation)
                    else:
                        network_outputs = net.activate(processed_observation)
                except Exception as ne:
                    print(f"Network activation failed: {ne}")
                    # Fallback to random action
                    action = np.random.choice([0, 2, 3])  # Avoid getting stuck
                    network_outputs = [0, 0, 0, 0]
                    network_outputs[action] = 1.0

                # DEBUG: Print network behavior
                recent_outputs.append(
                    network_outputs.copy() if hasattr(network_outputs, 'copy') else list(network_outputs))
                if len(recent_outputs) > 10:
                    recent_outputs.pop(0)

                # Choose action
                try:
                    action = int(np.argmax(network_outputs) % env.action_space.n)
                except:
                    action = 0  # NOOP fallback

                # DEBUG: Print detailed info every 30 steps
                if step_count % 30 == 0:
                    print(f"\n--- Step {step_count} DEBUG INFO ---")
                    print(f"Ball Y: {ball_y} (in play)")
                    print(f"Feature type: {feature_type}")
                    print(f"Input shape: {np.array(processed_observation).shape}")
                    print(f"Raw network outputs: {network_outputs}")
                    print(f"Output range: {np.min(network_outputs):.4f} to {np.max(network_outputs):.4f}")
                    print(f"Output variance: {np.var(network_outputs):.6f}")
                    print(f"Chosen action: {action} ({env.unwrapped.get_action_meanings()[action]})")

                    # Check if outputs are too similar
                    output_diff = np.max(network_outputs) - np.min(network_outputs)
                    if output_diff < 0.01:
                        print("⚠️  WARNING: Network outputs are very similar! Network may be stuck.")

                    # Show recent action pattern
                    if recent_actions:
                        action_pattern = recent_actions[-10:]
                        unique_actions = len(set(action_pattern))
                        print(f"Recent actions: {action_pattern}")
                        print(f"Action diversity: {unique_actions}/10")

                        if unique_actions == 1:
                            print("⚠️  WARNING: Agent is stuck on one action!")

            # Track actions
            if action in action_counts:
                action_counts[action] += 1
            recent_actions.append(action)
            if len(recent_actions) > 20:
                recent_actions.pop(0)

            # Execute action
            observation, reward, done, truncated, info = env.step(action)

            # Print basic game state every 60 steps
            if step_count % 60 == 0:
                ball_x = int(observation[RAM['ball_x']])
                ball_y = int(observation[RAM['ball_y']])
                paddle_x = int(observation[RAM['paddle_x']])
                lives = int(observation[RAM['lives']])
                score = int(observation[RAM['score']])

                print(f"\n=== GAME STATE - Step {step_count} ===")
                print(f"Score: {total_reward} | Ball: ({ball_x}, {ball_y}) | Paddle: {paddle_x} | Lives: {lives}")
                print(f"Action distribution: {action_counts}")
                print(f"Using: {feature_type}")

                # Calculate action diversity
                total_actions = sum(action_counts.values())
                if total_actions > 0:
                    diversity = len([v for v in action_counts.values() if v > 0])
                    print(f"Total actions taken: {total_actions}, Unique actions used: {diversity}/4")

                    if diversity == 1:
                        print("🚨 PROBLEM: Only using 1 action type!")
                    elif diversity == 2:
                        print("⚠️  Limited: Only using 2 action types")
                    else:
                        print("✅ Good: Using multiple action types")

            # Render
            if pygame_initialized:
                rgb_frame = env.render()
                frame_surface = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))
                frame_surface = pygame.transform.scale(frame_surface, window_size)
                screen.blit(frame_surface, (0, 0))
                pygame.display.flip()

                if current_delay > 0:
                    time.sleep(current_delay)

            total_reward += reward
            step_count += 1

    except Exception as e:
        print(f"Error during gameplay: {e}")
        import traceback
        traceback.print_exc()

    finally:
        env.close()
        if pygame_initialized:
            pygame.quit()

    # Final analysis
    print(f"\n{'=' * 50}")
    print(f"FINAL ANALYSIS")
    print(f"{'=' * 50}")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward}")
    print(f"Final action distribution: {action_counts}")

    total_actions = sum(action_counts.values())
    if total_actions > 0:
        print(f"Action percentages:")
        for action, count in action_counts.items():
            pct = (count / total_actions) * 100
            action_name = env.unwrapped.get_action_meanings()[action] if action < len(
                env.unwrapped.get_action_meanings()) else f"Action_{action}"
            print(f"  {action_name}: {count} ({pct:.1f}%)")

    # Diagnose the problem
    unique_actions_used = len([v for v in action_counts.values() if v > 0])

    if unique_actions_used == 1:
        dominant_action = max(action_counts, key=action_counts.get)
        print(
            f"\n🚨 DIAGNOSIS: Network is stuck on action {dominant_action} ({env.unwrapped.get_action_meanings()[dominant_action]})")
        print("SOLUTION: Train with the new enhanced training code for better diversity")
    elif action_counts[2] == 0 and action_counts[3] == 0:
        print(f"\n🚨 DIAGNOSIS: Network never moves paddle (no LEFT/RIGHT actions)")
        print("SOLUTION: Use the new training code with enhanced movement rewards")
    elif action_counts[2] + action_counts[3] < total_actions * 0.1:
        print(f"\n⚠️  DIAGNOSIS: Network rarely moves paddle (<10% movement actions)")
        print("SOLUTION: The new training code will fix this with better fitness rewards")
    else:
        print(f"\n✅ DIAGNOSIS: Network uses multiple actions appropriately")

    return total_reward

def main():
    parser = argparse.ArgumentParser(description='FIXED Replay - Compatible with New Training')
    parser.add_argument('--config', type=str, default='config-feedforward-breakout',
                        help='Path to NEAT configuration file')
    parser.add_argument('--genome', type=str, default='best_genome_gen_50.pkl',
                        help='Path to specific genome file to replay')
    parser.add_argument('--delay', type=float, default=0.05,
                        help='Delay between frames (seconds)')
    parser.add_argument('--width', type=int, default=840,
                        help='Width of the game window')
    parser.add_argument('--height', type=int, default=460,
                        help='Height of the game window')

    args = parser.parse_args()

    # Find config file
    local_dir = os.path.dirname(__file__)
    if os.path.exists(args.config):
        config_path = args.config
    else:
        config_path = os.path.join(os.path.dirname(local_dir), args.config)

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    if not os.path.exists(args.genome):
        print(f"Genome file not found: {args.genome}")
        return

    window_size = (args.width, args.height)

    print(f"🔄 FIXED REPLAY - No More Errors!")
    print(f"Genome: {args.genome}")
    print(f"Config: {config_path}")
    print("Features: Can use both new enhanced features and old raw RAM")
    print("Controls: F = toggle features, SPACE = pause, UP/DOWN = speed")
    print("=" * 60)

    score = play_genome_debug(config_path, args.genome,
                              render_mode='human', delay=args.delay,
                              window_size=window_size)

    print(f"\nFinal score: {score}")

if __name__ == "__main__":
    main()