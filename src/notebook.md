---
jupyter:
  jupytext:
    formats: ipynb,md
    notebook_metadata_filter: blog
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.2"
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import seaborn
import matplotlib

seaborn.set_theme(font_scale=1.25)
```

```python
import pandas
import random
import numpy


class Agent:
    def __init__(self, arms: int, steps: int, epsilon: float) -> None:
        self.epsilon = epsilon
        self.arms = arms
        self.steps = steps

        # Internal action value data
        self.action_values = numpy.zeros(arms)
        self.action_counts = numpy.zeros(arms)

        # Log of rewards at each time step
        self.action_taken = numpy.zeros(steps)
        self.reward = numpy.zeros(steps)
        self.action_difference = numpy.zeros(steps)
        self.optimal_action = numpy.zeros(steps)

    def pick_action(self) -> int:
        """Determine which action to take, update counts of actions taken."""

        # Pick action using epsilon
        if numpy.random.uniform() < epsilon:
            selected_action_idx = random.randrange(self.arms)
        else:
            selected_action_idx = numpy.argmax(self.action_values)

        # Update action counts
        self.action_counts[selected_action_idx] += 1

        return selected_action_idx

    def update(
        self, timestep: int, selected_action_idx: int, reward: float, is_optimal: bool
    ) -> None:
        """Update action values and rewards"""
        self.reward[timestep] = reward
        self.optimal_action[timestep] = is_optimal
        self.action_taken[timestep] = selected_action_idx

        # Update the action values
        change_in_action_value = (
            self.reward[timestep] - self.action_values[selected_action_idx]
        ) / self.action_counts[selected_action_idx]

    def to_data_frame(self) -> pandas.DataFrame():
        """Generate a dataframe from the results"""
        return pandas.DataFrame(
            {
                "timestep": range(0, self.steps),
                "reward": self.reward,
                "epsilon": self.epsilon,
                "optimal_action": self.optimal_action,
                "action_taken": self.action_taken,
            }
        )
```

```python
import seaborn

N_RUNS = 10
N_STEPS = 1000
K_ARMS = 100
EPSILONS = [0, 0.0125, 0.025, 0.05, 0.1, 0.2]


agents = []

for run in range(0, N_RUNS):

    # Initialise q*(a) for each run
    true_action_values = numpy.random.normal(size=K_ARMS)

    for epsilon in EPSILONS:

        agent = Agent(K_ARMS, N_STEPS, epsilon)

        for t in range(0, N_STEPS):

            selected_action_idx = agent.pick_action()

            # Determine which is the best action at this point .
            optimal_action_idx = numpy.argmax(true_action_values)

            # Calculate reward
            action_reward = numpy.random.normal(true_action_values[selected_action_idx])

            # Update the agent.
            agent.update(
                timestep=t,
                reward=action_reward,
                selected_action_idx=selected_action_idx,
                is_optimal=selected_action_idx == optimal_action_idx,
            )

        agents.append(agent)


agent_data = pandas.concat([x.to_data_frame() for x in agents])
```

```python
agent_data
```

```python
averaged_data = (
    agent_data.groupby(["epsilon", "timestep"]).mean().drop(columns=["action_taken"])
)
averaged_data
```

```python
optimal_action_plot = seaborn.lineplot(
    data=averaged_data, x="timestep", y="optimal_action", hue="epsilon"
)


_ = optimal_action_plot.set(
    xlabel="Time step.",
    ylabel="Optimal action taken.",
    title="Optimal action selection plot",
)
```

```python
average_reward_plot = seaborn.lineplot(
    data=averaged_data, x="timestep", y="reward", hue="epsilon"
)


_ = average_reward_plot.set(
    xlabel="Time step.",
    ylabel="Average reward.",
    title="Average reward plot by agent.",
)
```
