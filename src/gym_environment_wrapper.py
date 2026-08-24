import gymnasium as gym
import math
    
class GymEnvironmentWrapper:
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
        env = gym.make(environment_name)
        try:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.input_count = math.prod(self.observation_space.shape)
            self.output_count = self.action_space.n
        finally:
            env.close()
        
    def observation_to_inputs(self, observation):
        return [float(value) for value in observation.flatten()]
    
    def outputs_to_action(self, outputs):
        if len(outputs) != self.output_count:
            raise ValueError(f"Expected {self.output_count} network outputs, got {len(outputs)}")
        return max(range(len(outputs)), key=lambda index: outputs[index])
    
    def create_environment(self, render_mode=None):
        return gym.make(id=self.environment_name, render_mode=render_mode) #returns the index of the highest value
        