# TODO: Instead of a node_value dict wouldnt it be better to give a new attrivute to nodes?
#       Should the weight differences just be added up and averaged? Not like a MSE?
#       Design idea - smart comments multiline adjusting position to code and text length

from __future__ import annotations
import math
from collections.abc import Sequence
from numbers import Real
import random
from typing import ClassVar, Literal, TypeAlias
from omegaconf import OmegaConf
from gym_environment_wrapper import GymEnvironmentWrapper


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
    
    def find_connection(self, source_id, destination_id):
        for connection in self.connections.values():
            if connection.source_id == source_id and connection.destination_id == destination_id:
                return connection
        return None

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
        self.input_nodes: list[NodeGene] = sorted((node for node in genome.nodes.values() if node.is_input()), key=lambda node: node.node_id)
        self.output_nodes: list[NodeGene] = sorted((node for node in genome.nodes.values() if node.is_output()), key=lambda node: node.node_id)
        self.hidden_nodes: list[NodeGene] = [node for node in genome.nodes.values() if node.is_hidden()]

        self.incoming_connections: dict[int, list[ConnectionGene]] = {node.node_id: [] for node in self.hidden_nodes + self.output_nodes}

        for connection in genome.connections.values():
            if not connection.is_enabled():
                continue

            if connection.source_id not in genome.nodes:
                raise ValueError(
                    f"Connection references missing source node "
                    f"{connection.source_id}"
                )

            if connection.destination_id not in genome.nodes:
                raise ValueError(
                    f"Connection references missing destination node "
                    f"{connection.destination_id}"
                )

            if connection.destination_id not in self.incoming_connections:
                raise ValueError(
                    f"Enabled connection cannot target node "
                    f"{connection.destination_id}"
                )

            self.incoming_connections[
                connection.destination_id
            ].append(connection)

        self.evaluation_order: list[NodeGene] = (
            self.build_evaluation_order()
        )

    def activate(self, observation: Sequence[Real]) -> list[float]:
        if len(observation) != len(self.input_nodes):
            raise ValueError(
                f"Observation must contain exactly "
                f"{len(self.input_nodes)} values"
            )

        node_values: dict[int, float] = {
            node.node_id: float(value)
            for node, value in zip(
                self.input_nodes,
                observation,
                strict=True,
            )
        }

        for node in self.evaluation_order:
            total = node.node_bias

            for connection in self.incoming_connections[node.node_id]:
                if connection.source_id not in node_values:
                    raise RuntimeError(
                        f"Source node {connection.source_id} "
                        "has not been evaluated"
                    )

                total += (node_values[connection.source_id] * connection.weight)

            if node.is_hidden():
                node_values[node.node_id] = math.tanh(total)
            else:
                node_values[node.node_id] = total
        return [node_values[output_node.node_id] for output_node in self.output_nodes]

    
    def build_evaluation_order(self) -> list[NodeGene]:
        calculated_node_ids: set[int] = {
            node.node_id for node in self.input_nodes
        }

        remaining_nodes: dict[int, NodeGene] = {
            node.node_id: node
            for node in self.hidden_nodes + self.output_nodes
        }

        evaluation_order: list[NodeGene] = []

        while remaining_nodes:
            ready_node_ids: list[int] = [
                node_id
                for node_id in remaining_nodes
                if all(
                    connection.source_id in calculated_node_ids
                    for connection
                    in self.incoming_connections[node_id]
                )
            ]

            if not ready_node_ids:
                unresolved_node_ids = sorted(remaining_nodes)

                raise ValueError(
                    "Network graph contains a cycle or unresolved "
                    f"dependency among nodes {unresolved_node_ids}"
                )

            for node_id in sorted(ready_node_ids):
                node = remaining_nodes.pop(node_id)
                evaluation_order.append(node)
                calculated_node_ids.add(node_id)

        return evaluation_order

class EnvironmentEvaluator:
    def __init__(self, config, wrapper):
        self.seeds = config.environment.seeds
        self.wrapper = wrapper
    
    def evaluate_episode(self, genome, seed) -> float:
        network = NeuralNetwork(genome)
        env = self.wrapper.create_environment()
        try: #try, except, else, finally
            observation, _ = env.reset(seed=seed)
            total_reward = 0
            while True:
                # env.render()
                inputs = self.wrapper.observation_to_inputs(observation)
                outputs = network.activate(inputs)
                action = self.wrapper.outputs_to_action(outputs)
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
            #print(f"Mutated by {mutation_strength}")
    genome.fitness = -math.inf
    return genome

def creates_cycle_check(genome: Genome, source_id, destination_id):
    queue = [destination_id]
    visited = set()
    
    while queue:
        current = queue.pop(0)
        if current == source_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        for connection in genome.connections.values():
            if not connection.is_enabled():
                continue
            if connection.source_id == current:
                queue.append(connection.destination_id)
    return False
    
def mutate_add_connection(genome: Genome, innovation_tracker: InnovationTracker):
    possible_connections = []
    for source_node in genome.nodes.values():
        if source_node.is_output():
            continue
        for destination_node in genome.nodes.values():
            if destination_node.is_input() or source_node == destination_node:
                continue
            existing_connection = genome.find_connection(source_node.node_id, destination_node.node_id)
            if existing_connection and existing_connection.is_enabled():
                continue
            
            if creates_cycle_check(genome, source_node.node_id, destination_node.node_id):
                continue
            
            possible_connections.append((source_node.node_id, destination_node.node_id, existing_connection))
            
    if not possible_connections:
        return False
    
    source_id, destination_id, existing_connection = random.choice(possible_connections)
    if existing_connection is not None:
        existing_connection.enable()
        genome.fitness = -math.inf
        return True
    
    innovation_number = innovation_tracker.get_connection_innovation(source_id, destination_id)
    new_connection = ConnectionGene(source_id=source_id, destination_id=destination_id, weight=random.uniform(-1, 1), enabled=True, innovation_number=innovation_number)
    if not genome.add_connection(connection_gene=new_connection):
        return False
    genome.fitness = -math.inf
    return True

def mutate_add_node(genome: Genome, innovation_tracker: InnovationTracker):
    enabled_connections = []
    for connection in genome.connections.values():
        if connection.is_enabled():
            enabled_connections.append(connection)
    if not enabled_connections:
        return False
    selected_connection = random.choice(enabled_connections)
    connection_split_record = innovation_tracker.get_or_create_connection_split(selected_connection)
    inserted_node_id = connection_split_record["inserted_node_id"]
    if inserted_node_id not in genome.nodes:
        genome.add_node(NodeGene(inserted_node_id, "hidden", 0))
    
    first_connection = genome.find_connection(selected_connection.source_id, inserted_node_id)

    if first_connection is None:
        first_connection = ConnectionGene(source_id=selected_connection.source_id, destination_id=inserted_node_id, weight=1.0, enabled=True, innovation_number=connection_split_record["source_to_inserted_node_innovation"])

        if not genome.add_connection(first_connection):
            raise RuntimeError("Failed to add source-to-inserted-node connection")
    else:
        first_connection.enable()

    second_connection = genome.find_connection(inserted_node_id, selected_connection.destination_id)

    if second_connection is None:
        second_connection = ConnectionGene(source_id=inserted_node_id, destination_id=selected_connection.destination_id, weight=selected_connection.weight, enabled=True, innovation_number=connection_split_record["inserted_node_to_destination_innovation"])

        if not genome.add_connection(second_connection):
            raise RuntimeError("Failed to add inserted-node-to-destination connection")
    else:
        second_connection.enable()

    selected_connection.disable()
    genome.fitness = -math.inf
    return True

def crossover(parent_a: Genome, parent_b: Genome, disabled_inheritance_probability = 0.75):
    if parent_a.fitness > parent_b.fitness:
        fitter_parent = parent_a
        weaker_parent = parent_b
    elif parent_b.fitness > parent_a.fitness:
        fitter_parent = parent_b
        weaker_parent = parent_a
    else:
        fitter_parent, weaker_parent = random.choice(((parent_a, parent_b), (parent_b, parent_a)))

    child = Genome()
    for node_id, fitter_node in fitter_parent.nodes.items():
        weaker_node = weaker_parent.nodes.get(node_id)

        if weaker_node is None:
            inherited_node = fitter_node
        else:
            inherited_node = random.choice((fitter_node, weaker_node))

        if not child.add_node(inherited_node.copy_node()):
            raise RuntimeError(f"Failed to add node {node_id} during crossover")

    for innovation, fitter_connection in fitter_parent.connections.items():
        weaker_connection = weaker_parent.connections.get(innovation)

        if weaker_connection is None:
            inherited_connection = (fitter_connection.copy_connection())
        else:
            if not weaker_connection.connects(fitter_connection.source_id, fitter_connection.destination_id):
                raise RuntimeError(
                    f"Innovation {innovation} refers to different "
                    "connections in the two parents"
                )

            inherited_connection = random.choice(
                (fitter_connection, weaker_connection)
            ).copy_connection()

            if (not fitter_connection.is_enabled() or not weaker_connection.is_enabled()):
                if (random.random() < disabled_inheritance_probability):
                    inherited_connection.disable()
                else:
                    inherited_connection.enable()

        if inherited_connection.is_enabled():
            if creates_cycle_check(child, inherited_connection.source_id, inherited_connection.destination_id):
                inherited_connection.disable()

        if not child.add_connection(inherited_connection):
            raise RuntimeError(
                f"Failed to add innovation {innovation} "
                "during crossover"
            )

    child.fitness = -math.inf
    return child

def measure_compatibility_dist(config, genome1: Genome, genome2: Genome) -> float:
    innovations1 = set(genome1.connections.keys()) # the innovation numbers of this genome
    innovations2 = set(genome2.connections.keys())

    matching = innovations1 & innovations2
    weight_diff = 0.0
    for innovation in matching:
        weight_diff += abs(genome1.connections[innovation].weight - genome2.connections[innovation].weight)
        print(f"Weight diff: {weight_diff}")

    mismatching = innovations1 ^ innovations2
    # divide disjoint into disjoint and excess
    highest_shared_innovation_num = min(max(innovations1) if innovations1 else 0, max(innovations2) if innovations2 else 0)
    disjoint_cnt = excess_cnt = 0
    for innovation in mismatching:
        if innovation <= highest_shared_innovation_num:
            disjoint_cnt += 1
        else:
            excess_cnt += 1

    max_genome_size = max(len(innovations1), len(innovations2), 1)
    return (excess_cnt * config.training.loss_weight_excess + disjoint_cnt * config.training.loss_weight_disjoint) / max_genome_size + weight_diff * config.training.loss_weight_difference
        


class InnovationTracker:
    def __init__(self, initial_genome: Genome):
        self.connection_history: dict[tuple[int, int], int] = {}
        self.innovation_history: dict[int, tuple[int, int]] = {}
        self.split_history = {}
        
        self.next_node_id = ( max(initial_genome.nodes) + 1 if initial_genome.nodes else 0 )
        self.next_innovation_number = ( max(connection.innovation_number for connection in initial_genome.connections.values()) + 1 if initial_genome.connections else 0 )
        
        for connection in initial_genome.connections.values():
            self.register_connection(connection)
        
    def register_connection(self, connection: ConnectionGene) -> None:
        if not isinstance(connection, ConnectionGene):
            raise TypeError("register_connection requires a ConnectionGene")

        if not connection.validate():
            raise ValueError("Cannot register an invalid ConnectionGene")

        connection_key = (connection.source_id, connection.destination_id)
        innovation_number = connection.innovation_number

        existing_innovation = self.connection_history.get(connection_key)

        if (existing_innovation is not None and existing_innovation != innovation_number):
            raise ValueError(
                f"Connection {connection_key} is already registered "
                f"with innovation {existing_innovation}, not "
                f"{innovation_number}"
            )

        existing_connection = self.innovation_history.get(innovation_number)

        if (existing_connection is not None and existing_connection != connection_key):
            raise ValueError(
                f"Innovation {innovation_number} is already registered "
                f"for connection {existing_connection}, not "
                f"{connection_key}"
            )

        self.connection_history[connection_key] = innovation_number
        self.innovation_history[innovation_number] = connection_key

        if innovation_number >= self.next_innovation_number:
            self.next_innovation_number = innovation_number + 1
    def _get_next_node_id(self):
        next_node_id = self.next_node_id
        self.next_node_id += 1
        return next_node_id
    def _get_next_innovation_number(self):
        next_innovation_number = self.next_innovation_number
        self.next_innovation_number += 1
        return next_innovation_number
    
    def get_connection_innovation(self, source_id, destination_id):
        existing_innovation = self.connection_history.get((source_id, destination_id))
        if existing_innovation is not None:
            registered_connection = self.innovation_history.get(existing_innovation)
            if registered_connection != (source_id, destination_id):
                raise RuntimeError("Innovation history is internally inconsistent")
            return existing_innovation 
        innovation_number = self._get_next_innovation_number()
        self.connection_history[(source_id, destination_id)] = innovation_number
        self.innovation_history[innovation_number] = (source_id, destination_id)
        return innovation_number
    
    def get_or_create_connection_split(self, connection: ConnectionGene):
        original_innovation = connection.innovation_number
        original_connection = (connection.source_id, connection.destination_id)
        registered_connection = self.innovation_history.get(original_innovation)
        if registered_connection != original_connection:
            raise ValueError(
                f"Innovation {original_innovation} is not registered "
                f"for connection {original_connection}"
            )
        existing_split = self.split_history.get(original_innovation)
        if existing_split:
            return existing_split.copy()
        
        inserted_node_id = self._get_next_node_id()
        source_to_inserted_node_innovation = (self.get_connection_innovation(connection.source_id, inserted_node_id))
        inserted_node_to_destination_innovation = (self.get_connection_innovation(inserted_node_id, connection.destination_id))

        connection_split_record = {
            "original_source_id": connection.source_id,
            "original_destination_id": (connection.destination_id),
            "inserted_node_id": inserted_node_id,
            "source_to_inserted_node_innovation": (source_to_inserted_node_innovation),
            "inserted_node_to_destination_innovation": (inserted_node_to_destination_innovation),
        }

        self.split_history[original_innovation] = connection_split_record
        return connection_split_record.copy()
    

        

class Population:
    def __init__(self, config, genomes: list[Genome] = None, population_size: int = None, best_genome: Genome = None):
        self.config = config
        self.genomes = genomes
        self.population_size = self.config.environment.population_size if population_size is None else population_size
        self.best_genome = best_genome
        self.innovation_tracker = None

    def initialize_population(self, input_count, output_count):
        genomes = []
        for _ in range(self.population_size):
            genome = Genome()

            for node_id in range(input_count):
                genome.add_node(NodeGene(node_id, "input", 0.0))
                
            for node_id in range(input_count, input_count + output_count):
                genome.add_node(NodeGene(node_id, "output", 0.0))
                
            innovation_number = 0
            
            for source_id in range(input_count):
                for destination_id in range(input_count, input_count + output_count):
                    connection = ConnectionGene(source_id=source_id, destination_id=destination_id, weight=random.uniform(-1, 1), enabled=True, innovation_number=innovation_number)
                    genome.add_connection(connection)
                    innovation_number += 1
            genomes.append(genome)
        self.genomes = genomes
        self.innovation_tracker = InnovationTracker(self.genomes[0])
        
        
    def evaluate_population(self, evaluator: EnvironmentEvaluator):
        for genome in self.genomes:
            genome.fitness = evaluator.evaluate(genome)
    
    def find_best_genome(self) -> Genome:
        if not self.genomes:
            raise ValueError("find_best_genome, the population is empty")
        self.best_genome = max(self.genomes, key=lambda genome: genome.fitness)
        return self.best_genome
    
    def _select_parent(self, tournament_size, excluded_parent = None):
        candidates = [genome for genome in self.genomes if genome is not excluded_parent]
        selected_candidates = random.sample(candidates, k = min(tournament_size, len(candidates)))
        return max(selected_candidates, key=lambda genome: genome.fitness)
        
    def create_next_generation(self, config, tournament_size = 3):
        if not self.genomes:
            raise ValueError("create_mext_generation, the population is empty")
        if any(genome.fitness == -math.inf for genome in self.genomes):
            raise ValueError("Population must be evaluated before reproduction")

        best_genome = self.find_best_genome().copy_genome()
        best_genome.fitness = -math.inf
        next_generation = [best_genome]
        
        while len(next_generation) < self.population_size:
            parent_a = self._select_parent(tournament_size=tournament_size)

            if random.random() < config.training.crossover_probability:
                parent_b = self._select_parent(tournament_size=tournament_size, excluded_parent=parent_a)
                child = crossover(parent_a, parent_b)
            else:
                child = parent_a.copy_genome()

            mutate_weights(genome=child, mutation_probability=(config.training.weight_mutation_probability), mutation_strength=config.training.weight_mutation_strength)

            if random.random() < config.training.add_node_probability: 
                mutate_add_node(genome=child, innovation_tracker=self.innovation_tracker)
            if random.random() < config.training.add_connection_probability: 
                mutate_add_connection(genome=child, innovation_tracker=self.innovation_tracker)
                    
            next_generation.append(child)

        self.genomes = next_generation
        self.best_genome = None
        
    def run_generations(self, config, evaluator, tournament_size = 3):
        for generation in range(config.training.number_of_generations):
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
        
            if best_genome.fitness >= self.config.environment.target_fitness:
                print("Environment was solved")
                return best_genome.copy_genome(preserve_fitness=True)
            
            if generation < config.training.number_of_generations - 1:
                self.create_next_generation(config, tournament_size)
        
        return self.find_best_genome().copy_genome(preserve_fitness=True)

class Species:
    def __init__(self):
        pass

def main() -> None:
    config = OmegaConf.load("src/config.yaml")
    # random.seed(config.environment.random_seed)
    # gym_wrapper = GymEnvironmentWrapper(environment_name=config.environment.name)
    # evaluator = EnvironmentEvaluator(config, gym_wrapper)

    # population = Population(config)
    # population.initialize_population(input_count=gym_wrapper.input_count, output_count=gym_wrapper.output_count)

    # winner = population.run_generations(
    #     config=config,
    #     evaluator=evaluator,
    #     tournament_size=3,
    # )

    ################################################################################################################
    genome_a = Genome()
    genome_b = Genome()

    # Same nodes in both genomes
    for node_id in range(3):
        genome_a.add_node(NodeGene(node_id, "input", 0.0))
        genome_b.add_node(NodeGene(node_id, "input", 0.0))

    genome_a.add_node(NodeGene(3, "output", 0.0))
    genome_b.add_node(NodeGene(3, "output", 0.0))

    # Matching genes: innovations 0 and 1
    genome_a.add_connection(
        ConnectionGene(
            source_id=0,
            destination_id=3,
            weight=0.5,
            enabled=True,
            innovation_number=0,
        )
    )

    genome_b.add_connection(
        ConnectionGene(
            source_id=0,
            destination_id=3,
            weight=0.8,
            enabled=True,
            innovation_number=0,
        )
    )

    genome_a.add_connection(
        ConnectionGene(
            source_id=1,
            destination_id=3,
            weight=-0.2,
            enabled=True,
            innovation_number=1,
        )
    )

    genome_b.add_connection(
        ConnectionGene(
            source_id=1,
            destination_id=3,
            weight=0.1,
            enabled=True,
            innovation_number=1,
        )
    )

    # Only genome A has innovation 2
    genome_a.add_connection(
        ConnectionGene(
            source_id=2,
            destination_id=3,
            weight=0.7,
            enabled=True,
            innovation_number=2,
        )
    )

    # Only genome B has innovation 3
    genome_b.add_connection(ConnectionGene(source_id=2,destination_id=3,weight=-0.4,enabled=True,innovation_number=3,))
    distance = measure_compatibility_dist(config, genome_a, genome_b)
    print("Compatibility distance:", distance)

if __name__ == "__main__":
    main()