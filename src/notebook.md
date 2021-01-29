---
jupyter:
  jupytext:
    formats: ipynb,md
    notebook_metadata_filter: blog
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import seaborn
import matplotlib
import pandas

pandas.options.display.max_rows = 100

seaborn.set_theme(font_scale=1.25)
```

```python
import pandas
import random
import numpy
import abc
import dataclasses
import math

@dataclasses.dataclass()
class Agent(abc.ABC):
    """Represents a single loop of the training."""
    
    arms: int
    steps: int
        
    # Tuning parameter, usage varies by agent type
    learning_parameter: float
    
    
    def __post_init__(self) -> None:
        """Set up internal agent state."""

        # Internal action value data
        self.action_values = numpy.zeros(self.arms)
        self.action_counts = numpy.zeros(self.arms)

        # Log of actions and rewards at each time step
        self.action_taken = numpy.zeros(self.steps)
        self.reward = numpy.zeros(self.steps)
        self.is_optimal = numpy.zeros(self.steps)
        
    
    @abc.abstractmethod
    def action_difference(self, timestep: int, selected_action_idx: int) -> float:
        pass
    
    @abc.abstractmethod
    def action_selection(self, timestamp: int) -> int:
        pass

        
    def act(self, timestep: int) -> int:
        """Determine which action to take, update counts of actions taken."""
        selected_action_idx = self.action_selection(timestep)
        self.action_counts[selected_action_idx] += 1
        self.action_taken[timestep] = selected_action_idx
        return selected_action_idx

    
    def update(
        self, timestep: int, selected_action_idx: int, reward: float, is_optimal: bool
    ) -> None:
        """Update action values and rewards"""
        self.reward[timestep] = reward
        self.is_optimal[timestep] = is_optimal
        self.action_taken[timestep] = selected_action_idx        
        self.action_values[selected_action_idx] += self.action_difference(timestep, selected_action_idx)

        
    def to_data_frame(self, non_stationary: bool, agent_type: str) -> pandas.DataFrame():
        """Generate a dataframe from the results"""
        return pandas.DataFrame(
            {
                "timestep": range(0, self.steps),
                "reward": self.reward,
                "learning_parameter": self.learning_parameter,
                "is_optimal": self.is_optimal,
                "action_taken": self.action_taken,
                "non_stationary": non_stationary,
                "agent_type": agent_type
            }
        )
    
    
class SampleAverageAgent(Agent):
    """Agent implementing sample average agent with epsion greedy."""
    
    def action_difference(self, timestep: int, selected_action_idx: int) -> float:
        return (self.reward[timestep] - self.action_values[selected_action_idx]) / self.action_counts[selected_action_idx]
    
    def action_selection(self, timestamp: int) -> int:
        """Epsilon greedy search."""
        # Pick action using epsilon
        if numpy.random.uniform() < self.learning_parameter:
            return random.randrange(self.arms)
        return numpy.argmax(self.action_values)

    
class ConstantStepSizeAgent(Agent):
    """Agent implementing the constant step size algorithm."""
    
    def action_selection(self, timestamp: int) -> int:
        """Pick largest action-value."""
        return numpy.argmax(self.action_values)
    
    def action_difference(self, timestep: int, selected_action_idx: int) -> float:
        return 0.9 * (self.reward[timestep] - self.action_values[selected_action_idx])


class UpperConfidenceBound(Agent):
    """Agent implementing the upper confidence bound algorithm."""
    
    def action_selection(self, timestep: int) -> int:
        """Pick upper confidence bound action value."""
        
        # If any action has not been used yet, return that
        unused_action = self.action_counts.tolist().index(0)
        print(unused_action)
        if unused_action is not None:
            return unused_action
        
        return numpy.argmax(self.action_values + self.learning_parameter * numpy.sqrt(math.log(timestep) / self.action_counts))
    
    def action_difference(self, timestep: int, selected_action_idx: int) -> float:
        return (self.reward[timestep] - self.action_values[selected_action_idx]) / self.action_counts[selected_action_idx]
    

```

```python
import seaborn
import itertools

N_RUNS = 100 # 2000
N_STEPS = 1000
K_ARMS = 10
EPSILONS = [0, 0.025, 0.05, 0.1]
NON_STATIONARY = [False, True]
AGENT = [UpperConfidenceBound, SampleAverageAgent, ConstantStepSizeAgent]


per_run_data = []

for run, epsilon, non_stationary, AgentType in itertools.product(range(0, N_RUNS), EPSILONS, NON_STATIONARY, AGENT):

    # Initialise q*(a) for each run
    true_action_values = numpy.random.normal(size=K_ARMS)
    agent = AgentType(K_ARMS, N_STEPS, epsilon)

    for t in range(0, N_STEPS):

        selected_action_idx = agent.act(t)

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
        
        if non_stationary:
            # If non stationary, update true action values
            true_action_values += numpy.random.normal(size=K_ARMS, scale=0.01)
        
    per_run_data.append(agent.to_data_frame(non_stationary, AgentType.__name__))

agent_data = pandas.concat(per_run_data)
```

```python
averaged_data = (
    agent_data.groupby(["learning_parameter", "timestep", "non_stationary", "agent_type"]).mean().drop(columns=["action_taken"]).reset_index()
)
averaged_data
```

```python
grid = seaborn.FacetGrid(averaged_data, col="non_stationary", row="agent_type")
grid.map_dataframe(seaborn.lineplot, x="timestep", y="is_optimal", hue="learning_parameter")


_ = grid.set(
    xlabel="Time step.",
    ylabel="Optimal action taken.",
    xscale="log",
)

grid.fig.set_size_inches(14,7)
```

```python
average_reward_plot = seaborn.lineplot(
    data=averaged_data, x="timestep", y="reward", hue="epsilon"
)


_ = average_reward_plot.set(
    xlabel="Time step.",
    ylabel="Average reward.",
    title="Average reward plot by agent.",
    xscale="log"
)
```
