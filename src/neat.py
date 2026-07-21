from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
import os
import random
from typing import ClassVar, Literal, TypeAlias

import gymnasium as gym


NodeType: TypeAlias = Literal["input", "hidden", "output"]


class NodeGene:
    VALID_NODE_TYPES: ClassVar[set[str]] = {"input", "hidden", "output"}

    def __init__(self, node_id: int, node_type: NodeType, node_bias: Real) -> None:
        self.node_id: int = node_id
        self.node_type: NodeType = node_type
        self.node_bias: float = float(node_bias)

    def is_input(self) -> bool:
        return self.node_type == "input"

    def is_hidden(self) -> bool:
        return self.node_type == "hidden"

    def is_output(self) -> bool:
        return self.node_type == "output"

    def copy_node(self) -> NodeGene:
        return NodeGene(self.node_id, self.node_type, self.node_bias)

    def validate(self) -> bool:
        valid_id = isinstance(self.node_id, int) and not isinstance(self.node_id, bool) and self.node_id >= 0
        valid_type = self.node_type in self.VALID_NODE_TYPES
        valid_bias = isinstance(self.node_bias, Real) and not isinstance(self.node_bias, bool)
        return valid_id and valid_type and valid_bias and math.isfinite(float(self.node_bias))


class ConnectionGene:
    def __init__(self, source_id: int, destination_id: int, weight: Real, enabled: bool, innovation_number: int) -> None:
        self.source_id: int = source_id
        self.destination_id: int = destination_id
        self.weight: float = float(weight)
        self.enabled: bool = enabled
        self.innovation_number: int = innovation_number

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled

    def is_matching_innovation(self, other_connection: ConnectionGene) -> bool:
        return self.innovation_number == other_connection.innovation_number

    def connects(self, source_id: int, destination_id: int) -> bool:
        return self.source_id == source_id and self.destination_id == destination_id

    def copy_connection(self) -> ConnectionGene:
        return ConnectionGene(
            self.source_id,
            self.destination_id,
            self.weight,
            self.enabled,
            self.innovation_number,
        )

    def validate(self) -> bool:
        valid_source = (
            isinstance(self.source_id, int)
            and not isinstance(self.source_id, bool)
            and self.source_id >= 0
        )
        valid_destination = (
            isinstance(self.destination_id, int)
            and not isinstance(self.destination_id, bool)
            and self.destination_id >= 0
        )
        valid_weight = (
            isinstance(self.weight, Real)
            and not isinstance(self.weight, bool)
            and math.isfinite(float(self.weight))
        )
        valid_innovation = (
            isinstance(self.innovation_number, int)
            and not isinstance(self.innovation_number, bool)
            and self.innovation_number >= 0
        )

        return (
            valid_source
            and valid_destination
            and self.source_id != self.destination_id
            and valid_weight
            and isinstance(self.enabled, bool)
            and valid_innovation
        )


class Genome:
    def __init__(self, nodes: dict[int, NodeGene] | None = None, connections: dict[int, ConnectionGene] | None = None, fitness: float = -math.inf) -> None:
        self.nodes: dict[int, NodeGene] = {} if nodes is None else dict(nodes)
        self.connections: dict[int, ConnectionGene] = {} if connections is None else dict(connections)
        self.fitness: float = fitness

    def add_node(self, node_gene: NodeGene) -> bool:
        if not isinstance(node_gene, NodeGene):
            print("add_node failed: object is not a NodeGene")
            return False

        if not node_gene.validate():
            print("add_node failed: invalid NodeGene")
            return False

        if node_gene.node_id in self.nodes:
            print("add_node failed: node ID already exists")
            return False

        self.nodes[node_gene.node_id] = node_gene
        return True

    def add_connection(self, connection_gene: ConnectionGene) -> bool:
        if not isinstance(connection_gene, ConnectionGene):
            print("add_connection failed: object is not a ConnectionGene")
            return False

        if not connection_gene.validate():
            print("add_connection failed: invalid ConnectionGene")
            return False

        if connection_gene.source_id not in self.nodes:
            print("add_connection failed: source node missing")
            return False

        if connection_gene.destination_id not in self.nodes:
            print("add_connection failed: destination node missing")
            return False

        source_node = self.nodes[connection_gene.source_id]
        destination_node = self.nodes[connection_gene.destination_id]

        if source_node.is_output():
            print("add_connection failed: output nodes cannot be source nodes")
            return False

        if destination_node.is_input():
            print("add_connection failed: input nodes cannot be destination nodes")
            return False

        if connection_gene.innovation_number in self.connections:
            print("add_connection failed: innovation number already exists")
            return False

        if any(
            connection.connects(connection_gene.source_id, connection_gene.destination_id)
            for connection in self.connections.values()
        ):
            print("add_connection failed: connection already exists")
            return False

        self.connections[connection_gene.innovation_number] = connection_gene
        return True

    def copy_genome(self, preserve_fitness: bool = False) -> Genome:
        child_nodes = {node_id: node.copy_node() for node_id, node in self.nodes.items()}
        child_connections = {innovation: connection.copy_connection() for innovation, connection in self.connections.items()}
        child_fitness = self.fitness if preserve_fitness else -math.inf
        return Genome(child_nodes, child_connections, child_fitness)
        
class NeuralNetwork:
    def __init__(self, genome: Genome) -> None:
        if not isinstance(genome, Genome):
            raise TypeError("NeuralNetwork requires a Genome")

        self.genome: Genome = genome
        self.input_nodes: list[NodeGene] = sorted(
            (node for node in genome.nodes.values() if node.is_input()),
            key=lambda node: node.node_id,
        )
        self.output_nodes: list[NodeGene] = [
            node for node in genome.nodes.values() if node.is_output()
        ]
        self.hidden_nodes: list[NodeGene] = [
            node for node in genome.nodes.values() if node.is_hidden()
        ]

        if len(self.input_nodes) != 4:
            raise ValueError("NeuralNetwork requires exactly 4 input nodes")

        if len(self.output_nodes) != 1:
            raise ValueError("NeuralNetwork requires exactly 1 output node")

        if self.hidden_nodes:
            raise NotImplementedError("This network version cannot evaluate hidden nodes yet")

        self.output_node: NodeGene = self.output_nodes[0]

    def activate(self, observation: Sequence[Real]) -> int:
        if len(observation) != 4:
            raise ValueError("Observation must contain exactly 4 values")

        node_values: dict[int, float] = {
            node.node_id: float(value)
            for node, value in zip(self.input_nodes, observation, strict=True)
        }

        output_total = self.output_node.node_bias

        for connection in self.genome.connections.values():
            if not connection.is_enabled():
                continue

            if connection.destination_id != self.output_node.node_id:
                continue

            if connection.source_id not in node_values:
                raise ValueError(
                    f"Source node {connection.source_id} has no calculated value"
                )

            output_total += node_values[connection.source_id] * connection.weight

        return 0 if output_total < 0 else 1

class CartPoleEvaluator:
    def __init__(self, seeds, environment_name="CartPole-v1"):
        self.seeds = seeds
        self.environment_name = environment_name
    
    def evaluate_episode(self, genome, seed) -> float:
        network = NeuralNetwork(genome)
        env = gym.make(self.environment_name)#, render_mode="human")
        try: #try, except, else, finally
            observation, _ = env.reset(seed=seed)
            total_reward = 0
            while True:
                # env.render()
                action = network.activate(observation)
                # print(f"Chosen action to execute: {action}")
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            return total_reward
        finally:
            env.close()
    
    def evaluate(self, genome):
        return sum([self.evaluate_episode(genome, seed) for seed in self.seeds]) / len(self.seeds)
        

def mutate_weights(genome:Genome, mutation_probability:float, mutation_strength:float) -> Genome:
    for connection in genome.connections.values():
        rnd = random.random()
        if rnd < mutation_probability:
            connection.weight += random.gauss(0, mutation_strength)
    genome.fitness = -math.inf
    return genome

class Population:
    def __init__(self, genomes: list[Genome] = None, population_size: int = 10, best_genome: Genome = None):
        self.genomes = genomes
        self.population_size = population_size
        self.best_genome = best_genome

    def initialize_population(self):
        genomes = []
        for _ in range(self.population_size):
            genome = Genome()
            for node_id in range(4):
                genome.add_node(NodeGene(node_id, "input", 0.0))
            genome.add_node(NodeGene(4, "output", 0.0))
            for source_id in range(4):
                connection = ConnectionGene(source_id=source_id, destination_id=4, weight=random.uniform(-1, 1), enabled=True, innovation_number=source_id)
                genome.add_connection(connection)
            genomes.append(genome)
        self.genomes = genomes
        
    def evaluate_population(self, evaluator: CartPoleEvaluator):
        for genome in self.genomes:
            genome.fitness = evaluator.evaluate(genome)
    
    def find_best_genome(self) -> Genome:
        if not self.genomes:
            raise ValueError("find_best_genome, the population is empty")
        self.best_genome = max(self.genomes, key=lambda genome: genome.fitness)
        return self.best_genome
    
    def create_next_generation(self, mutation_probability:float, mutation_strength:float):
        if not self.genomes:
            raise ValueError("create_mext_generation, the population is empty")
        if any(genome.fitness == -math.inf for genome in self.genomes):
            raise ValueError("Population must be evaluated before reproduction")

        best_genome = self.find_best_genome().copy_genome()
        best_genome.fitness = -math.inf
        next_generation = [best_genome]
        
        while len(next_generation) < self.population_size:
            next_generation.append(mutate_weights(best_genome.copy_genome(), mutation_probability, mutation_strength))
        self.genomes = next_generation
        self.best_genome = None
        
    def run_generations(self, evaluator, number_of_generations, mutation_probability, mutation_strength):
        for generation in range(number_of_generations):
            self.evaluate_population(evaluator)
            best_genome = self.find_best_genome()
            avg_fitness = sum([genome.fitness for genome in self.genomes]) / self.population_size
            print(
                f"Generation {generation}: "
                f"best={best_genome.fitness:.2f}, "
                f"average={avg_fitness:.2f}")
            
            print("Weights of the best Genome:")
            for connection in best_genome.connections.values():
                print(connection.weight)
        
            if best_genome.fitness >= 500.0:
                print("CartPole was solved")
                return best_genome.copy_genome(preserve_fitness=True)
            
            if generation < number_of_generations - 1:
                self.create_next_generation(mutation_probability, mutation_strength)
        
        return self.find_best_genome().copy_genome(preserve_fitness=True)
        
        
    
def main() -> None:
    # create genome
    genome = Genome()
    # add four input nodes
    for node_id in range(4):
        if not genome.add_node(NodeGene(node_id, "input", 0.0)):
            raise RuntimeError(f"Failed to add input node {node_id}")
    # add one output node
    if not genome.add_node(NodeGene(4, "output", 0.0)):
        raise RuntimeError("Failed to add output node")
    # add four input-to-output connections
    weights: list[float] = [3.0, -2.0, 2.0, 1.0]
    for source_id, weight in enumerate(weights):
        connection = ConnectionGene(source_id=source_id, destination_id=4, weight=weight, enabled=True, innovation_number=source_id)
        if not genome.add_connection(connection):
            raise RuntimeError(f"Failed to add connection from node {source_id}")
    
    evaluator = CartPoleEvaluator(seeds=[0, 1, 2])

    population = Population(population_size=10)
    population.initialize_population()

    winner = population.run_generations(
        evaluator=evaluator,
        number_of_generations=200,
        mutation_probability=0.2,
        mutation_strength=0.2,
    )

    print(f"Winning fitness: {winner.fitness}")
    
    
    
if __name__ == "__main__":
    main()