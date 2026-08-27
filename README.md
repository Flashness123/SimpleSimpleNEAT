# SimpleSimpleNEAT

A from-scratch implementation of **NEAT** (NeuroEvolution of Augmenting
Topologies) — evolving neural networks to solve Gymnasium control tasks, no
neuroevolution library involved.

<!-- TODO: drop in the training GIF here once recorded -->
<!-- ![NEAT solving Acrobot](assets/demo.gif) -->
*(demo GIF coming soon — a population learning to swing up Acrobot)*

## Why I built this

Part of my reinforcement learning journey. I love evolutionary algorithms — the
idea that selection, mutation and crossover alone can produce competent behaviour,
no gradients required — and what I enjoy most is combining that with neural
networks: letting evolution grow both the weights *and* the shape of the network
instead of fixing the architecture up front. NEAT is the classic version of that
idea, so I wanted to build it myself rather than import it.

## What it does

Starts from a population of minimal networks and evolves them over generations —
mutating weights, adding nodes and connections, and recombining the best
performers — until they get good at the task. Everything (environment, population
size, mutation rates, generations) is set in `src/config.yaml`. Defaults to
`Acrobot-v1`.

Speciation is the next piece I want to finish; right now selection is plain
tournament selection over the whole population.

## Running it

Managed with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
uv run python src/neat.py      # run from the repo root
```
