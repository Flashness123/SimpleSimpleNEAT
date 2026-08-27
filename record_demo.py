import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sys
import random
sys.path.insert(0, "src")

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf

import neat
from gym_environment_wrapper import GymEnvironmentWrapper

SEED = 0
MAX_GENERATIONS = 150
POPULATION_SIZE = 150
SOLVED_FITNESS = -160.0   # MountainCar reward is -1/step; reaching the flag early beats the -200 floor

random.seed(SEED)
np.random.seed(SEED)

config = OmegaConf.load("src/config.yaml")
config.environment.name = "MountainCar-v0"
config.environment.seeds = [SEED]
config.environment.population_size = POPULATION_SIZE
config.environment.target_fitness = SOLVED_FITNESS
config.training.number_of_generations = MAX_GENERATIONS

wrapper = GymEnvironmentWrapper(config.environment.name)
evaluator = neat.EnvironmentEvaluator(config, wrapper)

population = neat.Population(config)
population.initialize_population(wrapper.input_count, wrapper.output_count)

print(f"Training NEAT on {config.environment.name} "
      f"(pop {POPULATION_SIZE}, up to {MAX_GENERATIONS} gens)...")
winner = population.run_generations(config=config, evaluator=evaluator, tournament_size=3)
print(f"Best fitness found: {winner.fitness:.1f}")

# Replay the winner and capture frames
network = neat.NeuralNetwork(winner)
env = wrapper.create_environment(render_mode="rgb_array")
observation, _ = env.reset(seed=SEED)
frames = []
total = 0.0
while True:
    frames.append(env.render())
    action = wrapper.outputs_to_action(network.activate(wrapper.observation_to_inputs(observation)))
    observation, reward, terminated, truncated, _ = env.step(action)
    total += reward
    if terminated or truncated:
        break
env.close()
print(f"Replay episode reward: {total:.0f} over {len(frames)} steps"
      f"{' (reached the flag)' if terminated else ''}")

os.makedirs("assets", exist_ok=True)
# downscale 2x and keep every 2nd frame to keep the GIF small
small = [f[::2, ::2] for f in frames[::2]]
imageio.mimsave("assets/demo.gif", small, duration=1 / 30, loop=0)
print(f"Wrote assets/demo.gif ({len(small)} frames)")
