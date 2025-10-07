"""
Comprehensive user guide documentation module for plume_nav_sim providing accessible, research-focused
documentation for scientists, researchers, and students using the plume navigation reinforcement learning
environment. Serves as the primary entry point for users with step-by-step tutorials, practical examples,
configuration guidance, and research workflow integration patterns optimized for scientific applications
and educational use.

This module implements a complete documentation system including UserGuideGenerator for structured content
creation, TutorialManager for progressive learning experiences, and ConfigurationGuideGenerator for
parameter documentation. The implementation provides comprehensive coverage of environment usage,
visualization techniques, reproducibility practices, and research integration patterns.

Key Features:
- Comprehensive user guide generation with tutorials, examples, and research integration
- Progressive tutorial system with hands-on exercises and learning assessment
- Configuration documentation with parameter analysis and research presets
- Visualization tutorials covering dual-mode rendering and analysis techniques
- Reproducibility guides with seeding workflows and scientific methodology
- Research integration patterns for RL frameworks and scientific Python ecosystem
- Interactive examples with Jupyter notebook integration and workflow automation
- Troubleshooting guidance and FAQ sections with practical solutions

Educational Design:
- Accessible content for users new to plume navigation and reinforcement learning
- Progressive complexity with clear learning objectives and skill development
- Practical examples with executable code and expected outcomes
- Research workflow integration with academic and industrial application patterns
- Scientific reproducibility emphasis with experimental consistency practices
- Performance optimization guidance with system requirements and tuning

Architecture Integration:
- Integrates with plume_nav_sim environment for comprehensive usage examples
- Supports demonstration functions from examples/ for practical tutorials
- Coordinates with seeding utilities for reproducibility documentation
- Utilizes rendering components for visualization tutorials
- Provides research workflow templates for scientific applications
"""

import inspect  # >=3.10 - Code introspection utilities for automatic user documentation generation and interactive tutorial features
import pathlib  # >=3.10 - Object-oriented filesystem paths for cross-platform path handling and file operations
import textwrap  # >=3.10 - Text formatting utilities for generating well-formatted user documentation strings and tutorial content
from typing import (  # >=3.10 - Type annotations for clear user documentation and tutorial parameter specifications with educational clarity
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# External imports with version requirements
import gymnasium as gym  # >=0.29.0 - Reinforcement learning environment framework for user tutorials demonstrating standard RL workflows and gym.make() usage patterns
import matplotlib.pyplot as plt  # >=3.9.0 - Visualization framework for user rendering tutorials, plot generation examples, and research data visualization workflows
import numpy as np  # >=2.1.0 - Array operations and mathematical functions for user data analysis examples, observation processing tutorials, and research workflow integration

from examples.basic_usage import (  # Basic demonstration functions for user tutorial integration and practical examples
    demonstrate_basic_episode,
    demonstrate_reproducibility,
)
from examples.visualization_demo import (  # Rendering demonstration function for user visualization tutorials and practical guidance
    demonstrate_rendering_modes,
)
from plume_nav_sim.core.constants import (  # Default configuration constants for user configuration examples and tutorials
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_SOURCE_LOCATION,
)
from plume_nav_sim.core.enums import (  # Action enumeration for user-friendly action space documentation and tutorial examples
    Action,
)
from plume_nav_sim.envs.plume_search_env import (  # Main environment class and factory function for comprehensive user tutorials and practical usage examples
    PlumeSearchEnv,
    create_plume_search_env,
)

# Internal imports - Environment registration and core functionality
from plume_nav_sim.registration.register import (  # Environment registration function and identifier for Gymnasium compatibility and user tutorial examples
    ENV_ID,
    register_env,
)
from plume_nav_sim.utils.seeding import (  # Seed management utilities for user reproducibility tutorials and scientific research examples
    SeedManager,
    validate_seed,
)

# Global constants for user guide configuration
USER_GUIDE_VERSION = "1.0.0"  # Version identifier for user guide documentation and compatibility tracking
TUTORIAL_SEED = 42  # Default seed value for reproducible tutorial examples and consistent user experience
QUICK_START_STEPS = (
    10  # Number of steps in quick start tutorial for immediate user productivity
)
EXAMPLE_GRID_SIZE = (
    32,
    32,
)  # Smaller grid size for tutorial examples and faster demonstration execution
VISUALIZATION_DELAY = (
    0.1  # Delay between visualization updates for smooth tutorial demonstrations
)
RESEARCH_WORKFLOW_EXAMPLES = (
    {}
)  # Dictionary storing research workflow examples and templates for scientific applications
CONFIGURATION_TEMPLATES = (
    {}
)  # Dictionary storing configuration templates and presets for common research scenarios
TROUBLESHOOTING_FAQ = (
    {}
)  # Dictionary storing troubleshooting information and frequently asked questions

# Module exports for comprehensive user guide functionality
__all__ = [
    "generate_user_guide",  # Main function for generating comprehensive user guide with tutorials and research integration
    "create_quick_start_tutorial",  # Creates accessible quick start tutorial for immediate user productivity
    "create_configuration_guide",  # Creates comprehensive configuration guide with parameter documentation
    "create_visualization_tutorial",  # Creates visualization tutorial covering dual-mode rendering and research workflows
    "create_reproducibility_guide",  # Creates scientific reproducibility guide with seeding workflows
    "create_research_integration_guide",  # Creates research integration guide with framework examples and workflow automation
    "UserGuideGenerator",  # Comprehensive user guide generator for structured documentation creation
    "TutorialManager",  # Specialized tutorial management for progressive learning experiences
    "ConfigurationGuideGenerator",  # Specialized configuration guide generator for parameter documentation
]


def generate_user_guide(
    output_format: Optional[str] = "markdown",
    include_jupyter_examples: bool = True,
    include_research_workflows: bool = True,
    output_directory: Optional["pathlib.Path"] = None,
    guide_options: dict = None,
) -> str:
    """
    Main function for generating comprehensive user guide documentation including quick start tutorials,
    configuration guidance, visualization examples, reproducibility workflows, and research integration
    patterns optimized for scientists and researchers using plume_nav_sim.

    This function orchestrates the creation of a complete user guide covering all aspects of the
    plume navigation environment from basic usage to advanced research applications. It integrates
    multiple tutorial components, configuration documentation, and research workflow examples.

    Args:
        output_format (Optional[str]): Output format for user guide ('markdown', 'html', 'jupyter', 'pdf')
        include_jupyter_examples (bool): Include Jupyter notebook examples with interactive research patterns
        include_research_workflows (bool): Include advanced research workflows with experimental design patterns
        output_directory (Optional[pathlib.Path]): Directory for output files, defaults to current directory
        guide_options (dict): Additional guide configuration options and customization settings

    Returns:
        str: Complete user guide formatted for target audience with tutorials, examples, and comprehensive
        research integration documentation ready for distribution and educational use
    """
    # Initialize user guide structure with version information and accessibility-focused organization
    guide_sections = {
        "header": f"# Plume Navigation Simulation User Guide v{USER_GUIDE_VERSION}",
        "table_of_contents": "",
        "introduction": "",
        "quick_start": "",
        "installation": "",
        "configuration": "",
        "visualization": "",
        "reproducibility": "",
        "research_integration": "",
        "jupyter_examples": "",
        "advanced_workflows": "",
        "troubleshooting": "",
        "faq": "",
        "appendices": "",
    }

    # Apply guide options and customization settings with default values for comprehensive coverage
    if guide_options is None:
        guide_options = {}

    tutorial_complexity = guide_options.get("tutorial_complexity", "intermediate")
    include_performance_tips = guide_options.get("include_performance_tips", True)
    include_troubleshooting = guide_options.get("include_troubleshooting", True)
    target_audience = guide_options.get("target_audience", "researchers")

    try:
        # Generate comprehensive introduction section explaining plume navigation research context and environment benefits
        guide_sections[
            "introduction"
        ] = f"""
## Introduction

Welcome to the Plume Navigation Simulation (plume_nav_sim) user guide! This comprehensive resource is designed
for researchers, scientists, and students working with reinforcement learning and plume navigation problems.

### What is Plume Navigation?

Plume navigation refers to the challenge of autonomous agents navigating chemical or odor plumes to locate
their sources. This problem is fundamental in:

- **Robotics**: Autonomous search and rescue, environmental monitoring
- **Biology**: Understanding animal foraging and navigation behaviors
- **Environmental Science**: Pollution source detection and monitoring
- **Machine Learning**: Developing robust navigation algorithms

### About plume_nav_sim

The plume_nav_sim environment provides a Gymnasium-compatible reinforcement learning environment for studying
plume navigation algorithms. Key features include:

- **Standard RL Interface**: Full Gymnasium API compliance for easy integration
- **Static Gaussian Plume Model**: Mathematical concentration field for consistent experiments
- **Dual Rendering Modes**: Both programmatic (`rgb_array`) and interactive (`human`) visualization
- **Scientific Reproducibility**: Comprehensive seeding system for deterministic experiments
- **Educational Focus**: Progressive tutorials and clear documentation for learning

### Who This Guide Is For

This guide serves multiple audiences with different needs and experience levels:

- **Researchers**: Comprehensive documentation for scientific applications and experimental design
- **Students**: Progressive tutorials and clear explanations for learning plume navigation concepts
- **Practitioners**: Practical examples and workflow integration patterns for real-world applications
- **Educators**: Teaching resources and structured learning materials for classroom use

### How to Use This Guide

The guide is organized progressively from basic concepts to advanced research applications:

1. **Quick Start**: Get up and running in minutes with immediate practical examples
2. **Configuration**: Understand environment parameters and customization options
3. **Visualization**: Master both rendering modes and analysis techniques
4. **Reproducibility**: Learn scientific best practices for consistent experiments
5. **Research Integration**: Integrate with RL frameworks and scientific workflows
6. **Advanced Topics**: Jupyter examples and research workflow automation

Each section includes executable code examples, expected outputs, and practical tips for success.
        """

        # Create quick start tutorial using create_quick_start_tutorial for immediate user productivity
        guide_sections["quick_start"] = create_quick_start_tutorial(
            include_installation=True,
            include_basic_visualization=True,
            tutorial_steps=QUICK_START_STEPS,
        )

        # Generate installation and setup guide with step-by-step instructions for research environments
        guide_sections[
            "installation"
        ] = f"""
## Installation and Setup

### System Requirements

Before installing plume_nav_sim, ensure your system meets these requirements:

**Python Version**: Python 3.10 or higher
- The environment requires modern Python for optimal performance and compatibility
- Verify your version: `python --version`

**Operating System Support**:
- **Linux**: Full support with all features
- **macOS**: Full support with all features
- **Windows**: Limited support (PRs accepted but not officially supported)

**Memory Requirements**:
- Minimum: 4GB RAM for basic usage
- Recommended: 8GB+ RAM for research applications
- Large grids (>256x256) may require additional memory

### Installation Steps

#### Step 1: Create Virtual Environment

We strongly recommend using a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv plume_nav_env

# Activate environment (Linux/macOS)
source plume_nav_env/bin/activate

# Activate environment (Windows)
plume_nav_env\\Scripts\\activate
```

#### Step 2: Install plume_nav_sim

Currently, plume_nav_sim supports local installation for development:

```bash
# Clone or navigate to plume_nav_sim directory
cd path/to/plume_nav_sim

# Install in development mode
pip install -e .

# Install with development dependencies (optional)
pip install -e .[dev]
```

#### Step 3: Verify Installation

Test your installation with this simple verification script:

```python
import gymnasium as gym
from plume_nav_sim.registration import register_env, ENV_ID

# Register the environment
register_env()

# Create environment instance
env = gym.make(ENV_ID)

# Test basic functionality
obs, info = env.reset(seed=42)
print(f"Initial observation: {{obs}}")
print(f"Observation shape: {{obs.shape}}")
print(f"Action space: {{env.action_space}}")
print(f"Observation space: {{env.observation_space}}")

# Clean up
env.close()
print("Installation verified successfully!")
```

**Expected Output**:
```
Initial observation: [0.xxxx]
Observation shape: (1,)
Action space: Discrete(4)
Observation space: Box(0.0, 1.0, (1,), float32)
Installation verified successfully!
```

### Troubleshooting Installation

**Common Issues and Solutions**:

1. **Python Version Error**: Upgrade to Python 3.10+
2. **Gymnasium Import Error**: Update gymnasium: `pip install --upgrade gymnasium`
3. **NumPy Compatibility**: Update NumPy: `pip install --upgrade numpy`
4. **Matplotlib Display Issues**: Install GUI backend: `pip install PyQt5` or `pip install tkinter`

### Development Setup

For contributors and advanced users developing with plume_nav_sim:

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests to verify setup
pytest tests/

# Generate documentation (if available)
# Additional setup instructions for development
```
        """

        # Create configuration guide using create_configuration_guide with parameter explanations and research examples
        guide_sections["configuration"] = create_configuration_guide(
            include_parameter_effects=True,
            include_research_presets=include_research_workflows,
            include_validation_examples=True,
        )

        # Generate visualization tutorial using create_visualization_tutorial with dual-mode rendering guidance
        guide_sections["visualization"] = create_visualization_tutorial(
            include_matplotlib_customization=True,
            include_data_analysis=True,
            include_jupyter_integration=include_jupyter_examples,
        )

        # Create reproducibility guide using create_reproducibility_guide for scientific research requirements
        guide_sections["reproducibility"] = create_reproducibility_guide(
            include_validation_procedures=True,
            include_research_methodology=include_research_workflows,
            include_statistical_analysis=include_research_workflows,
        )

        # Generate research integration guide using create_research_integration_guide with practical workflow examples
        if include_research_workflows:
            guide_sections["research_integration"] = create_research_integration_guide(
                target_frameworks=["stable_baselines3", "ray_rllib", "tianshou"],
                include_experimental_design=True,
                include_data_collection=True,
            )

        # Include Jupyter notebook examples if include_jupyter_examples enabled with interactive research patterns
        if include_jupyter_examples:
            guide_sections[
                "jupyter_examples"
            ] = f"""
## Jupyter Notebook Integration

Jupyter notebooks provide an excellent environment for interactive exploration and research with plume_nav_sim.
This section demonstrates how to effectively use the environment in notebook-based workflows.

### Setting Up Jupyter Environment

```bash
# Install Jupyter in your plume_nav_sim environment
pip install jupyter jupyterlab

# Start Jupyter Lab
jupyter lab
```

### Basic Notebook Setup

```python
# Notebook cell 1: Imports and setup
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from plume_nav_sim.registration import register_env, ENV_ID
from plume_nav_sim.utils.seeding import validate_seed

# Configure matplotlib for inline plotting
%matplotlib inline
plt.style.use('seaborn-v0_8')

# Register environment
register_env()
print("Environment registered successfully!")
```

### Interactive Environment Exploration

```python
# Notebook cell 2: Create and explore environment
env = gym.make(ENV_ID)

# Reset with seed for reproducibility
obs, info = env.reset(seed=42)

print("Environment Information:")
print(f"Action Space: {{env.action_space}}")
print(f"Observation Space: {{env.observation_space}}")
print(f"Initial Observation: {{obs}}")
print(f"Info Dictionary: {{info}}")

# Explore action meanings
from plume_nav_sim.core.enums import Action
print("\\nAction Mappings:")
for action in Action:
    print(f"{{action.value}}: {{action.name}}")
```

### Visualization in Notebooks

```python
# Notebook cell 3: Visualization setup
# Enable interactive plotting
%matplotlib widget

# Create figure for real-time updates
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Function for updating visualization
def update_visualization(env, ax1, ax2, step_count):
    # Get RGB array
    rgb_array = env.render(mode='rgb_array')

    # Display environment state
    ax1.clear()
    ax1.imshow(rgb_array)
    ax1.set_title(f'Environment State (Step {{step_count}})')
    ax1.axis('off')

    # Display concentration profile (if available)
    # This would require access to internal plume model
    # ax2.clear()
    # concentration = env.get_concentration_field()  # hypothetical method
    # ax2.imshow(concentration, cmap='viridis')
    # ax2.set_title('Concentration Field')

    plt.tight_layout()
    return fig

# Initial visualization
fig = update_visualization(env, ax1, ax2, 0)
plt.show()
```

### Interactive Episode Execution

```python
# Notebook cell 4: Interactive episode with widgets
from IPython.widgets import interact, IntSlider, Button
from IPython.display import display, clear_output

class InteractiveEpisode:
    def __init__(self, env):
        self.env = env
        self.obs, self.info = env.reset(seed=42)
        self.step_count = 0
        self.episode_history = []

    def take_action(self, action):
        if self.step_count < 100:  # Limit steps for demonstration
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.episode_history.append({
                'step': self.step_count,
                'action': action,
                'observation': obs.copy(),
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                'info': info.copy()
            })

            self.obs = obs
            self.info = info
            self.step_count += 1

            # Update visualization
            clear_output(wait=True)
            rgb_array = self.env.render(mode='rgb_array')
            plt.figure(figsize=(8, 6))
            plt.imshow(rgb_array)
            plt.title(f'Step {{self.step_count}}: Action {{action}}, Reward {{reward:.3f}}')
            plt.axis('off')
            plt.show()

            # Display step information
            print(f"Step {{self.step_count}}: Action {{action}} -> Reward: {{reward:.3f}}")
            print(f"Observation: {{obs[0]:.4f}}")
            print(f"Terminated: {{terminated}}, Truncated: {{truncated}}")

            if terminated:
                print("🎉 Goal reached!")
            elif truncated:
                print("Episode truncated due to step limit")

        else:
            print("Episode limit reached. Reset environment to continue.")

# Create interactive episode
interactive_ep = InteractiveEpisode(env)

# Interactive controls
def step_with_action(action):
    interactive_ep.take_action(action)

# Create buttons for actions
from IPython.widgets import Button, HBox
buttons = []
action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
for i, name in enumerate(action_names):
    button = Button(description=f'{{name}} ({{i}})')
    button.on_click(lambda b, action=i: step_with_action(action))
    buttons.append(button)

display(HBox(buttons))
```

### Data Analysis and Visualization

```python
# Notebook cell 5: Episode analysis
import pandas as pd

def analyze_episode(history):
    if not history:
        print("No episode data available.")
        return

    # Convert to DataFrame for analysis
    df = pd.DataFrame(history)

    # Basic statistics
    print("Episode Statistics:")
    print(f"Total Steps: {{len(df)}}")
    print(f"Total Reward: {{df['reward'].sum():.3f}}")
    print(f"Average Observation: {{df['observation'].apply(lambda x: x[0]).mean():.4f}}")
    print(f"Goal Reached: {{df['terminated'].any()}}")

    # Plot trajectory data
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Observation over time
    observations = df['observation'].apply(lambda x: x[0])
    ax1.plot(observations)
    ax1.set_title('Concentration Observations')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Concentration')
    ax1.grid(True)

    # Rewards over time
    ax2.plot(df['reward'], 'ro-', markersize=3)
    ax2.set_title('Rewards Over Time')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.grid(True)

    # Action distribution
    action_counts = df['action'].value_counts().sort_index()
    action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    ax3.bar(range(len(action_counts)), action_counts.values)
    ax3.set_title('Action Distribution')
    ax3.set_xlabel('Action')
    ax3.set_ylabel('Count')
    ax3.set_xticks(range(len(action_names)))
    ax3.set_xticklabels(action_names)
    ax3.grid(True, axis='y')

    # Cumulative reward
    ax4.plot(df['reward'].cumsum())
    ax4.set_title('Cumulative Reward')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Cumulative Reward')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

# Analyze the interactive episode
analyze_episode(interactive_ep.episode_history)
```

### Research Workflow Template

```python
# Notebook cell 6: Research workflow template
def run_experiment_series(num_episodes=10, max_steps=100, seeds=None):
    \"\"\"
    Template for running systematic experiments in Jupyter notebooks.
    \"\"\"
    if seeds is None:
        seeds = range(num_episodes)

    results = []

    for i, seed in enumerate(seeds):
        print(f"Running episode {{i+1}}/{{num_episodes}} with seed {{seed}}")

        # Reset environment with seed
        env = gym.make(ENV_ID)
        obs, info = env.reset(seed=seed)

        episode_data = {
            'seed': seed,
            'steps': 0,
            'total_reward': 0,
            'final_observation': 0,
            'goal_reached': False,
            'trajectory': []
        }

        # Run episode
        for step in range(max_steps):
            # Random policy for demonstration
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            episode_data['steps'] += 1
            episode_data['total_reward'] += reward
            episode_data['trajectory'].append({
                'step': step,
                'action': action,
                'observation': obs[0],
                'reward': reward
            })

            if terminated:
                episode_data['goal_reached'] = True
                break
            elif truncated:
                break

        episode_data['final_observation'] = obs[0]
        results.append(episode_data)
        env.close()

    # Analyze results
    success_rate = sum(1 for r in results if r['goal_reached']) / len(results)
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['total_reward'] for r in results])

    print(f"\\nExperiment Results:")
    print(f"Success Rate: {{success_rate:.1%}}")
    print(f"Average Steps: {{avg_steps:.1f}}")
    print(f"Average Reward: {{avg_reward:.3f}}")

    return results

# Run experiment series
experiment_results = run_experiment_series(num_episodes=5, max_steps=50)
```

### Saving and Sharing Results

```python
# Notebook cell 7: Save results for sharing
import json
from datetime import datetime

def save_experiment_results(results, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plume_nav_experiment_{{timestamp}}.json"

    # Prepare data for JSON serialization
    serializable_results = []
    for result in results:
        serializable_result = result.copy()
        # Convert numpy arrays to lists if present
        for key, value in serializable_result.items():
            if hasattr(value, 'tolist'):
                serializable_result[key] = value.tolist()
        serializable_results.append(serializable_result)

    # Save to file
    with open(filename, 'w') as f:
        json.dump({{
            'metadata': {{
                'timestamp': datetime.now().isoformat(),
                'environment_id': ENV_ID,
                'user_guide_version': USER_GUIDE_VERSION
            }},
            'results': serializable_results
        }}, f, indent=2)

    print(f"Results saved to {{filename}}")
    return filename

# Save results
save_experiment_results(experiment_results)
```

### Integration with External Tools

The notebook environment integrates well with external research tools:

- **Weights & Biases**: Log experiments and visualize results
- **TensorBoard**: Monitor training progress and metrics
- **MLflow**: Track experiments and manage models
- **Hydra**: Manage configuration for complex experiments

Example W&B integration:

```python
# Optional: Weights & Biases integration
# pip install wandb

import wandb

# Initialize W&B project
wandb.init(project="plume-navigation", name="exploration-experiment")

# Log experiment results
for i, result in enumerate(experiment_results):
    wandb.log({{
        "episode": i,
        "steps": result['steps'],
        "total_reward": result['total_reward'],
        "goal_reached": int(result['goal_reached']),
        "final_observation": result['final_observation']
    }})

wandb.finish()
```
            """

        # Add advanced research workflows if include_research_workflows enabled with experimental design patterns
        if include_research_workflows:
            guide_sections[
                "advanced_workflows"
            ] = f"""
## Advanced Research Workflows

This section covers sophisticated research workflows and experimental design patterns using plume_nav_sim
for academic and industrial research applications.

### Systematic Parameter Studies

```python
# Parameter sweep example for scientific research
import itertools
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def parameter_sweep_study():
    # Define parameter space
    grid_sizes = [(32, 32), (64, 64), (128, 128)]
    goal_radii = [0, 1, 2, 5]
    source_locations = [(16, 16), (32, 32), (48, 48)]

    # Generate all parameter combinations
    param_combinations = list(itertools.product(grid_sizes, goal_radii, source_locations))

    print(f"Total parameter combinations: {{len(param_combinations)}}")

    results = []
    for i, (grid_size, goal_radius, source_location) in enumerate(param_combinations):
        print(f"Testing configuration {{i+1}}/{{len(param_combinations)}}")

        # Create environment with specific parameters
        env = create_plume_search_env(
            grid_size=grid_size,
            source_location=source_location,
            goal_radius=goal_radius,
            max_steps=200
        )

        # Run multiple episodes for statistical significance
        episode_results = []
        for seed in range(10):  # 10 repetitions per configuration
            obs, info = env.reset(seed=seed)

            episode_data = {{
                'seed': seed,
                'grid_size': grid_size,
                'goal_radius': goal_radius,
                'source_location': source_location,
                'success': False,
                'steps': 0,
                'final_distance': float('inf')
            }}

            # Simple random policy for baseline
            for step in range(200):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                episode_data['steps'] = step + 1

                if terminated:
                    episode_data['success'] = True
                    episode_data['final_distance'] = 0
                    break
                elif truncated:
                    # Calculate final distance to source
                    agent_pos = info.get('agent_position', (0, 0))
                    distance = np.sqrt((agent_pos[0] - source_location[0])**2 +
                                     (agent_pos[1] - source_location[1])**2)
                    episode_data['final_distance'] = distance
                    break

            episode_results.append(episode_data)

        # Aggregate results for this configuration
        config_results = {{
            'grid_size': str(grid_size),
            'goal_radius': goal_radius,
            'source_location': str(source_location),
            'success_rate': np.mean([r['success'] for r in episode_results]),
            'avg_steps': np.mean([r['steps'] for r in episode_results]),
            'avg_final_distance': np.mean([r['final_distance'] for r in episode_results
                                         if r['final_distance'] != float('inf')])
        }}

        results.append(config_results)
        env.close()

    return pd.DataFrame(results)

# Run parameter study
param_study_results = parameter_sweep_study()
print(param_study_results.describe())

# Visualize results
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Success rate by goal radius
sns.boxplot(data=param_study_results, x='goal_radius', y='success_rate', ax=axes[0,0])
axes[0,0].set_title('Success Rate vs Goal Radius')

# Average steps by grid size
sns.boxplot(data=param_study_results, x='grid_size', y='avg_steps', ax=axes[0,1])
axes[0,1].set_title('Average Steps vs Grid Size')
axes[0,1].tick_params(axis='x', rotation=45)

# Success rate by source location
sns.boxplot(data=param_study_results, x='source_location', y='success_rate', ax=axes[1,0])
axes[1,0].set_title('Success Rate vs Source Location')
axes[1,0].tick_params(axis='x', rotation=45)

# Heatmap of success rates
pivot_data = param_study_results.pivot_table(
    values='success_rate', index='goal_radius', columns='grid_size', aggfunc='mean'
)
sns.heatmap(pivot_data, annot=True, cmap='viridis', ax=axes[1,1])
axes[1,1].set_title('Success Rate Heatmap')

plt.tight_layout()
plt.show()
```

### Comparative Algorithm Study

```python
# Compare different navigation strategies
class NavigationStrategy:
    def __init__(self, name):
        self.name = name

    def select_action(self, obs, action_space, info=None):
        raise NotImplementedError

class RandomStrategy(NavigationStrategy):
    def __init__(self):
        super().__init__("Random")

    def select_action(self, obs, action_space, info=None):
        return action_space.sample()

class GradientStrategy(NavigationStrategy):
    def __init__(self):
        super().__init__("Gradient Following")
        self.prev_obs = None

    def select_action(self, obs, action_space, info=None):
        if self.prev_obs is None:
            self.prev_obs = obs[0]
            return action_space.sample()

        # Simple gradient following heuristic
        if obs[0] > self.prev_obs:
            # Concentration increased, continue in same direction
            self.prev_obs = obs[0]
            return self.last_action if hasattr(self, 'last_action') else action_space.sample()
        else:
            # Concentration decreased, try a different direction
            self.prev_obs = obs[0]
            action = action_space.sample()
            self.last_action = action
            return action

def compare_strategies(strategies, num_episodes=20, max_steps=100):
    results = {{strategy.name: [] for strategy in strategies}}

    for strategy in strategies:
        print(f"Testing {{strategy.name}} strategy...")

        for episode in range(num_episodes):
            env = gym.make(ENV_ID)
            obs, info = env.reset(seed=episode)

            episode_data = {{
                'strategy': strategy.name,
                'episode': episode,
                'success': False,
                'steps': 0,
                'total_reward': 0,
                'observations': [obs[0]]
            }}

            for step in range(max_steps):
                action = strategy.select_action(obs, env.action_space, info)
                obs, reward, terminated, truncated, info = env.step(action)

                episode_data['steps'] = step + 1
                episode_data['total_reward'] += reward
                episode_data['observations'].append(obs[0])

                if terminated:
                    episode_data['success'] = True
                    break
                elif truncated:
                    break

            results[strategy.name].append(episode_data)
            env.close()

    return results

# Run strategy comparison
strategies = [RandomStrategy(), GradientStrategy()]
strategy_results = compare_strategies(strategies)

# Analyze and visualize results
def analyze_strategy_comparison(results):
    analysis = {{}}

    for strategy_name, episodes in results.items():
        success_rate = np.mean([ep['success'] for ep in episodes])
        avg_steps = np.mean([ep['steps'] for ep in episodes])
        avg_reward = np.mean([ep['total_reward'] for ep in episodes])

        analysis[strategy_name] = {{
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward,
            'episodes': episodes
        }}

        print(f"{{strategy_name}}:")
        print(f"  Success Rate: {{success_rate:.1%}}")
        print(f"  Average Steps: {{avg_steps:.1f}}")
        print(f"  Average Reward: {{avg_reward:.3f}}")
        print()

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Success rates
    strategies = list(analysis.keys())
    success_rates = [analysis[s]['success_rate'] for s in strategies]
    axes[0,0].bar(strategies, success_rates)
    axes[0,0].set_title('Success Rates by Strategy')
    axes[0,0].set_ylabel('Success Rate')

    # Average steps
    avg_steps = [analysis[s]['avg_steps'] for s in strategies]
    axes[0,1].bar(strategies, avg_steps)
    axes[0,1].set_title('Average Steps by Strategy')
    axes[0,1].set_ylabel('Steps')

    # Learning curves (observation trajectories)
    for strategy_name, episodes in results.items():
        avg_obs_trajectory = np.mean([ep['observations'] for ep in episodes[:5]], axis=0)
        axes[1,0].plot(avg_obs_trajectory, label=strategy_name)
    axes[1,0].set_title('Average Observation Trajectories')
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('Concentration')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # Reward distributions
    for strategy_name, strategy_data in analysis.items():
        rewards = [ep['total_reward'] for ep in strategy_data['episodes']]
        axes[1,1].hist(rewards, alpha=0.7, label=strategy_name, bins=10)
    axes[1,1].set_title('Reward Distributions')
    axes[1,1].set_xlabel('Total Reward')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()

    return analysis

strategy_analysis = analyze_strategy_comparison(strategy_results)
```

### Statistical Analysis and Hypothesis Testing

```python
# Statistical analysis for research publications
from scipy import stats
import scipy.stats as stats

def statistical_analysis(results_dict):
    \"\"\"
    Perform statistical analysis suitable for research publications.
    \"\"\"
    print("Statistical Analysis Report")
    print("=" * 50)

    # Extract data for statistical tests
    strategy_names = list(results_dict.keys())

    if len(strategy_names) >= 2:
        # Compare two strategies
        strategy1_data = [ep['total_reward'] for ep in results_dict[strategy_names[0]]]
        strategy2_data = [ep['total_reward'] for ep in results_dict[strategy_names[1]]]

        # Descriptive statistics
        print(f"\\n{{strategy_names[0]}} Strategy:")
        print(f"  Mean Reward: {{np.mean(strategy1_data):.4f}} ± {{np.std(strategy1_data):.4f}}")
        print(f"  Median Reward: {{np.median(strategy1_data):.4f}}")
        print(f"  Min/Max: {{np.min(strategy1_data):.4f}} / {{np.max(strategy1_data):.4f}}")

        print(f"\\n{{strategy_names[1]}} Strategy:")
        print(f"  Mean Reward: {{np.mean(strategy2_data):.4f}} ± {{np.std(strategy2_data):.4f}}")
        print(f"  Median Reward: {{np.median(strategy2_data):.4f}}")
        print(f"  Min/Max: {{np.min(strategy2_data):.4f}} / {{np.max(strategy2_data):.4f}}")

        # Normality tests
        print("\\nNormality Tests (Shapiro-Wilk):")
        stat1, p1 = stats.shapiro(strategy1_data)
        stat2, p2 = stats.shapiro(strategy2_data)
        print(f"  {{strategy_names[0]}}: p-value = {{p1:.4f}}")
        print(f"  {{strategy_names[1]}}: p-value = {{p2:.4f}}")

        # Choose appropriate statistical test
        if p1 > 0.05 and p2 > 0.05:
            # Both normal - use t-test
            statistic, p_value = stats.ttest_ind(strategy1_data, strategy2_data)
            test_name = "Independent t-test"
        else:
            # Non-normal - use Mann-Whitney U
            statistic, p_value = stats.mannwhitneyu(strategy1_data, strategy2_data,
                                                   alternative='two-sided')
            test_name = "Mann-Whitney U test"

        print(f"\\n{{test_name}} Results:")
        print(f"  Statistic: {{statistic:.4f}}")
        print(f"  p-value: {{p_value:.4f}}")

        alpha = 0.05
        if p_value < alpha:
            print(f"  Result: Significant difference (p < {{alpha}})")
        else:
            print(f"  Result: No significant difference (p ≥ {{alpha}})")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(strategy1_data) - 1) * np.var(strategy1_data, ddof=1) +
                             (len(strategy2_data) - 1) * np.var(strategy2_data, ddof=1)) /
                            (len(strategy1_data) + len(strategy2_data) - 2))
        cohens_d = (np.mean(strategy1_data) - np.mean(strategy2_data)) / pooled_std
        print(f"  Effect size (Cohen's d): {{cohens_d:.4f}}")

        # Interpretation
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        print(f"  Effect size interpretation: {{effect_interpretation}}")

# Run statistical analysis
statistical_analysis(strategy_results)
```

### Automated Experiment Pipeline

```python
# Automated experiment management system
import yaml
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
import time

@dataclass
class ExperimentConfig:
    name: str
    description: str
    parameters: Dict[str, Any]
    num_repetitions: int
    max_steps: int
    seeds: List[int]

class ExperimentPipeline:
    def __init__(self, config_file: str = None):
        self.experiments = []
        self.results = []
        self.setup_logging()

        if config_file:
            self.load_config(config_file)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('experiment_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def add_experiment(self, config: ExperimentConfig):
        self.experiments.append(config)
        self.logger.info(f"Added experiment: {{config.name}}")

    def run_all_experiments(self):
        self.logger.info(f"Starting pipeline with {{len(self.experiments)}} experiments")
        start_time = time.time()

        for i, experiment in enumerate(self.experiments):
            self.logger.info(f"Running experiment {{i+1}}/{{len(self.experiments)}}: {{experiment.name}}")

            experiment_results = self.run_single_experiment(experiment)
            self.results.append(experiment_results)

        total_time = time.time() - start_time
        self.logger.info(f"Pipeline completed in {{total_time:.1f}} seconds")

        return self.results

    def run_single_experiment(self, config: ExperimentConfig):
        results = {{
            'config': config,
            'episodes': [],
            'summary': {{}},
            'timestamp': time.time()
        }}

        for rep in range(config.num_repetitions):
            seed = config.seeds[rep % len(config.seeds)]

            # Create environment with experiment parameters
            env = create_plume_search_env(**config.parameters)
            obs, info = env.reset(seed=seed)

            episode_data = {{
                'repetition': rep,
                'seed': seed,
                'success': False,
                'steps': 0,
                'total_reward': 0,
                'trajectory': []
            }}

            # Run episode
            for step in range(config.max_steps):
                # Use random policy for baseline
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                episode_data['steps'] = step + 1
                episode_data['total_reward'] += reward
                episode_data['trajectory'].append({{
                    'step': step,
                    'observation': obs[0],
                    'action': action,
                    'reward': reward
                }})

                if terminated:
                    episode_data['success'] = True
                    break
                elif truncated:
                    break

            results['episodes'].append(episode_data)
            env.close()

        # Calculate summary statistics
        episodes = results['episodes']
        results['summary'] = {{
            'success_rate': np.mean([ep['success'] for ep in episodes]),
            'avg_steps': np.mean([ep['steps'] for ep in episodes]),
            'std_steps': np.std([ep['steps'] for ep in episodes]),
            'avg_reward': np.mean([ep['total_reward'] for ep in episodes]),
            'std_reward': np.std([ep['total_reward'] for ep in episodes])
        }}

        self.logger.info(f"Experiment {{config.name}} completed: "
                        f"{{results['summary']['success_rate']:.1%}} success rate")

        return results

# Example usage
pipeline = ExperimentPipeline()

# Define experiments
small_grid_exp = ExperimentConfig(
    name="Small Grid Baseline",
    description="Baseline performance on small 32x32 grid",
    parameters={{'grid_size': (32, 32), 'goal_radius': 1}},
    num_repetitions=10,
    max_steps=100,
    seeds=list(range(10))
)

large_grid_exp = ExperimentConfig(
    name="Large Grid Challenge",
    description="Performance comparison on large 128x128 grid",
    parameters={{'grid_size': (128, 128), 'goal_radius': 2}},
    num_repetitions=10,
    max_steps=200,
    seeds=list(range(10))
)

pipeline.add_experiment(small_grid_exp)
pipeline.add_experiment(large_grid_exp)

# Run experiments
all_results = pipeline.run_all_experiments()

# Generate summary report
def generate_summary_report(results):
    print("\\nExperiment Pipeline Summary Report")
    print("=" * 50)

    for result in results:
        config = result['config']
        summary = result['summary']

        print(f"\\nExperiment: {{config.name}}")
        print(f"Description: {{config.description}}")
        print(f"Parameters: {{config.parameters}}")
        print(f"Success Rate: {{summary['success_rate']:.1%}}")
        print(f"Average Steps: {{summary['avg_steps']:.1f}} ± {{summary['std_steps']:.1f}}")
        print(f"Average Reward: {{summary['avg_reward']:.3f}} ± {{summary['std_reward']:.3f}}")

generate_summary_report(all_results)
```
            """

        # Create troubleshooting section with common issues, solutions, and user support resources
        if include_troubleshooting:
            guide_sections["troubleshooting"] = create_troubleshooting_section(
                include_diagnostic_procedures=True,
                include_platform_specific=True,
                include_performance_issues=include_performance_tips,
            )

            # Generate frequently asked questions section with research-focused Q&A and practical guidance
            guide_sections["faq"] = create_faq_section(
                include_research_focused=include_research_workflows,
                include_technical_details=True,
                include_comparison_questions=True,
            )

        # Generate table of contents based on included sections
        toc_entries = []
        for section_name, content in guide_sections.items():
            if (
                content
                and section_name != "header"
                and section_name != "table_of_contents"
            ):
                # Extract main headings from content
                lines = content.strip().split("\n")
                for line in lines:
                    if line.startswith("##") and not line.startswith("###"):
                        heading = line.replace("##", "").strip()
                        anchor = (
                            heading.lower()
                            .replace(" ", "-")
                            .replace("&", "")
                            .replace("(", "")
                            .replace(")", "")
                        )
                        toc_entries.append(f"- [{heading}](#{anchor})")

        guide_sections[
            "table_of_contents"
        ] = f"""
## Table of Contents

{chr(10).join(toc_entries)}
        """

        # Create comprehensive appendices with reference materials and additional resources
        guide_sections[
            "appendices"
        ] = f"""
## Appendices

### Appendix A: API Reference Summary

#### Environment Parameters
- `grid_size`: Tuple of (width, height) for environment dimensions
- `source_location`: Tuple of (x, y) for plume source position
- `max_steps`: Maximum steps per episode before truncation
- `goal_radius`: Distance threshold for goal achievement
- `render_mode`: Visualization mode ('rgb_array' or 'human')

#### Action Space
- 0: Move UP (decrease y coordinate)
- 1: Move RIGHT (increase x coordinate)
- 2: Move DOWN (increase y coordinate)
- 3: Move LEFT (decrease x coordinate)

#### Observation Space
- Box(0.0, 1.0, (1,), float32): Concentration value at agent position

### Appendix B: Mathematical Formulation

#### Gaussian Plume Model
The concentration field follows a static Gaussian distribution:

```
C(x, y) = exp(-((x - sx)² + (y - sy)²) / (2σ²))
```

Where:
- `C(x, y)` is the concentration at position (x, y)
- `(sx, sy)` is the source location
- `σ` is the dispersion parameter (default: 12.0)

#### Reward Function
The environment uses sparse binary rewards:

```
reward = 1.0 if distance_to_source ≤ goal_radius else 0.0
```

### Appendix C: Performance Benchmarks

#### Target Performance Metrics
- Environment step execution: <1ms average latency
- Episode reset operations: <10ms initialization time
- RGB array rendering: <5ms frame generation
- Memory usage: <50MB for default 128×128 grid

#### Optimization Tips
- Use smaller grid sizes for faster execution
- Enable performance mode in renderers
- Batch episode execution when possible
- Monitor memory usage for large-scale experiments

### Appendix D: Research Resources

#### Related Publications
- Plume navigation research papers and surveys
- Reinforcement learning environment design principles
- Gymnasium API documentation and best practices

#### Community Resources
- GitHub repository: [plume_nav_sim](https://github.com/example/plume_nav_sim)
- Discussion forums and user community
- Issue tracker for bug reports and feature requests

#### Contributing Guidelines
- Development setup and contribution process
- Code style guidelines and testing requirements
- Documentation standards and example templates

### Appendix E: Version History

#### Version {USER_GUIDE_VERSION}
- Initial release with comprehensive user guide
- Quick start tutorial and configuration documentation
- Visualization tutorials and reproducibility guides
- Research integration patterns and Jupyter examples
- Advanced workflow templates and statistical analysis tools

### Appendix F: License and Citation

#### Software License
plume_nav_sim is released under [appropriate license - to be determined]

#### Citation Format
If you use plume_nav_sim in your research, please cite:

```bibtex
@software{{plume_nav_sim,
  title={{Plume Navigation Simulation Environment}},
  author={{[Authors]}},
  year={{2024}},
  version={{{USER_GUIDE_VERSION}}},
  url={{https://github.com/example/plume_nav_sim}}
}}
```

#### Acknowledgments
- Gymnasium framework developers
- NumPy and Matplotlib communities
- Research contributors and beta testers
- Educational institutions and industrial partners
        """

        # Format complete user guide with consistent styling, clear navigation, and educational progression
        complete_guide = []

        # Add header and table of contents
        complete_guide.append(guide_sections["header"])
        complete_guide.append(guide_sections["table_of_contents"])

        # Add main content sections in logical order
        section_order = [
            "introduction",
            "installation",
            "quick_start",
            "configuration",
            "visualization",
            "reproducibility",
            "research_integration",
            "jupyter_examples",
            "advanced_workflows",
            "troubleshooting",
            "faq",
            "appendices",
        ]

        for section_name in section_order:
            if section_name in guide_sections and guide_sections[section_name]:
                complete_guide.append(guide_sections[section_name])

        # Join all sections with appropriate spacing
        formatted_guide = "\n\n".join(complete_guide)

        # Apply output format-specific formatting
        if output_format == "html":
            # Convert markdown to HTML (simplified)
            formatted_guide = formatted_guide.replace("##", "<h2>").replace("#", "<h1>")
            # Note: Full HTML conversion would require markdown library
        elif output_format == "jupyter":
            # Add notebook metadata and cell structure
            # Note: Full Jupyter conversion would require nbformat library
            pass
        elif output_format == "pdf":
            # Add PDF-specific formatting
            # Note: PDF generation would require additional libraries
            pass

        # Return comprehensive user guide optimized for researchers and scientific applications
        return formatted_guide

    except Exception as e:
        # Handle any errors in guide generation with fallback content
        error_guide = f"""
# Plume Navigation Simulation User Guide v{USER_GUIDE_VERSION}

## Error in Guide Generation

An error occurred while generating the comprehensive user guide: {str(e)}

### Quick Start (Fallback)

```python
# Basic usage example
import gymnasium as gym
from plume_nav_sim.registration import register_env, ENV_ID

# Register and create environment
register_env()
env = gym.make(ENV_ID)

# Run basic episode
obs, info = env.reset(seed=42)
for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {{step}}: Observation={{obs[0]:.4f}}, Reward={{reward}}")
    if terminated or truncated:
        break

env.close()
```

Please check the system configuration and try again. For support, refer to the troubleshooting section or contact the development team.
        """

        return error_guide


def create_quick_start_tutorial(
    include_installation: bool = True,
    include_basic_visualization: bool = True,
    tutorial_steps: int = 10,
) -> str:
    """
    Creates accessible quick start tutorial for new users providing immediate hands-on experience
    with environment creation, basic episode execution, and essential functionality demonstration
    for rapid user onboarding and productive usage.

    This function generates a step-by-step tutorial designed to get users up and running with
    plume_nav_sim as quickly as possible while ensuring they understand the core concepts and
    can execute basic workflows successfully.

    Args:
        include_installation (bool): Include installation instructions and verification steps
        include_basic_visualization (bool): Include RGB array mode demonstration and basic rendering
        tutorial_steps (int): Number of steps to include in the hands-on episode demonstration

    Returns:
        str: Complete quick start tutorial with step-by-step instructions, code examples,
        and immediate user productivity focus for successful environment usage
    """
    tutorial_sections = []

    # Create tutorial introduction explaining plume navigation environment and immediate benefits for users
    tutorial_sections.append(
        f"""
## Quick Start Tutorial

Welcome to the plume_nav_sim quick start tutorial! In just a few minutes, you'll be running your first
plume navigation experiment and understanding the key concepts needed for productive research.

### What You'll Learn

By the end of this tutorial, you will:

1. **Understand the Problem**: Know what plume navigation is and why it matters
2. **Create Environments**: Successfully instantiate and configure the simulation
3. **Run Episodes**: Execute basic agent-environment interactions
4. **Interpret Results**: Understand observations, rewards, and termination conditions
5. **Visualize Behavior**: Generate and interpret visual representations
6. **Ensure Reproducibility**: Use seeding for consistent experimental results

### Time Required

- **Complete tutorial**: 10-15 minutes
- **Basic functionality**: 3-5 minutes for experienced RL users
- **With visualization**: Additional 5 minutes for rendering setup

### Prerequisites

- Python 3.10+ installed and accessible
- Basic familiarity with Python programming
- Optional: Understanding of reinforcement learning concepts (helpful but not required)
    """
    )

    # Generate installation section if include_installation enabled with pip install instructions and verification
    if include_installation:
        tutorial_sections.append(
            f"""
### Step 1: Installation Verification

First, let's ensure your plume_nav_sim installation is working correctly.

#### Quick Installation Check

```python
# Test 1: Import core modules
try:
    import gymnasium as gym
    print("✓ Gymnasium available")
except ImportError as e:
    print("✗ Gymnasium missing - install with: pip install gymnasium>=0.29.0")

try:
    import numpy as np
    print("✓ NumPy available")
except ImportError as e:
    print("✗ NumPy missing - install with: pip install numpy>=2.1.0")

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib available")
except ImportError as e:
    print("✗ Matplotlib missing - install with: pip install matplotlib>=3.9.0")

print("\\nCore dependencies check complete!")
```

#### Test Environment Import

```python
# Test 2: Import plume_nav_sim components
try:
    from plume_nav_sim.registration import register_env, ENV_ID
    print("✓ Environment registration available")

    from plume_nav_sim.envs.plume_search_env import PlumeSearchEnv
    print("✓ Main environment class available")

    from plume_nav_sim.core.enums import Action
    print("✓ Core types available")

    print("\\n🎉 plume_nav_sim successfully installed!")

except ImportError as e:
    print(f"✗ plume_nav_sim import failed: {{e}}")
    print("Solution: Ensure you've installed with 'pip install -e .' from the project directory")
```

**Expected Output**: All checks should show ✓ marks. If you see any ✗ marks, follow the installation instructions in the previous section.
        """
        )

    # Create environment registration tutorial with register_env() and gym.make() usage examples
    tutorial_sections.append(
        f"""
### Step 2: Environment Registration and Creation

Now let's register the environment and create your first plume navigation simulation.

#### Register the Environment

```python
# Import registration components
from plume_nav_sim.registration import register_env, ENV_ID

# Register the environment with Gymnasium
register_env()

print(f"Environment registered with ID: {{ENV_ID}}")
print("Registration successful! ✓")
```

#### Create Your First Environment

```python
import gymnasium as gym

# Create environment using gym.make()
env = gym.make(ENV_ID)

print("Environment created successfully!")
print(f"Action space: {{env.action_space}}")
print(f"Observation space: {{env.observation_space}}")

# Display environment information
print(f"\\nEnvironment Details:")
print(f"- Actions: 4 discrete movements (UP, RIGHT, DOWN, LEFT)")
print(f"- Observations: Concentration values between 0.0 and 1.0")
print(f"- Goal: Navigate to the plume source for reward")
```

**Expected Output**:
```
Environment registered with ID: PlumeNav-StaticGaussian-v0
Registration successful! ✓
Environment created successfully!
Action space: Discrete(4)
Observation space: Box(0.0, 1.0, (1,), float32)
```

#### Understanding the Environment

The plume navigation environment simulates an agent moving in a 2D grid world:

- **Grid World**: Default 128×128 cells with configurable dimensions
- **Agent**: Moves in 4 cardinal directions (up, right, down, left)
- **Plume Source**: Fixed location emitting a chemical/odor plume
- **Concentration Field**: Gaussian distribution with highest values at source
- **Goal**: Agent must navigate to source location for maximum reward
    """
    )

    # Generate first episode tutorial with reset(), step(), and basic action selection for immediate success
    tutorial_sections.append(
        f"""
### Step 3: Your First Episode

Let's run a complete episode and understand the agent-environment interaction cycle.

#### Initialize Episode

```python
# Reset environment to start new episode
obs, info = env.reset(seed={TUTORIAL_SEED})  # Using seed for reproducibility

print("Episode started!")
print(f"Initial observation: {{obs}}")
print(f"Observation shape: {{obs.shape}}")
print(f"Initial concentration: {{obs[0]:.4f}}")
print(f"Info dictionary: {{info}}")
```

#### Understanding Observations

The observation tells us the chemical concentration at the agent's current position:

- **Value range**: 0.0 (no chemical) to 1.0 (maximum concentration)
- **Higher values**: Closer to the plume source
- **Navigation goal**: Follow increasing concentration to find the source

#### Take Actions and Explore

```python
from plume_nav_sim.core.enums import Action
import numpy as np

# Display action meanings
print("Action meanings:")
print(f"{{Action.UP.value}}: {{Action.UP.name}} (decrease y)")
print(f"{{Action.RIGHT.value}}: {{Action.RIGHT.name}} (increase x)")
print(f"{{Action.DOWN.value}}: {{Action.DOWN.name}} (increase y)")
print(f"{{Action.LEFT.value}}: {{Action.LEFT.name}} (decrease x)")

# Run episode for specified number of steps
episode_history = []
max_concentration = obs[0]
best_action = None

print(f"\\nRunning {{tutorial_steps}}-step episode:")
print("-" * 50)

for step in range(tutorial_steps):
    # Choose action (random for demonstration)
    action = env.action_space.sample()

    # Take step in environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Track best concentration found
    if obs[0] > max_concentration:
        max_concentration = obs[0]
        best_action = action

    # Record step information
    step_info = {{
        'step': step + 1,
        'action': action,
        'action_name': Action(action).name,
        'observation': obs[0],
        'reward': reward,
        'terminated': terminated,
        'truncated': truncated
    }}
    episode_history.append(step_info)

    # Display step results
    print(f"Step {{step + 1:2d}}: Action {{action}} ({{Action(action).name:5s}}) "
          f"-> Obs: {{obs[0]:.4f}}, Reward: {{reward:.1f}}")

    # Check if episode finished
    if terminated:
        print("🎉 Goal reached! Episode terminated successfully.")
        break
    elif truncated:
        print("⏰ Episode truncated due to step limit.")
        break

print("-" * 50)
print(f"Episode completed after {{len(episode_history)}} steps")
print(f"Maximum concentration found: {{max_concentration:.4f}}")
if best_action is not None:
    print(f"Best action was: {{best_action}} ({{Action(best_action).name}})")
```

#### Episode Analysis

```python
# Analyze episode performance
total_reward = sum(step['reward'] for step in episode_history)
avg_concentration = np.mean([step['observation'] for step in episode_history])
goal_reached = any(step['terminated'] for step in episode_history)

print(f"\\nEpisode Summary:")
print(f"- Total steps: {{len(episode_history)}}")
print(f"- Total reward: {{total_reward:.1f}}")
print(f"- Average concentration: {{avg_concentration:.4f}}")
print(f"- Goal reached: {{'Yes' if goal_reached else 'No'}}")

# Show concentration trajectory
concentrations = [step['observation'] for step in episode_history]
print(f"- Concentration trajectory: {{[f'{{c:.3f}}' for c in concentrations[:5]]}}...")
if len(concentrations) > 5:
    print(f"  Final values: {{[f'{{c:.3f}}' for c in concentrations[-3:]]}}")
```

**Understanding Results**:

- **Reward = 1.0**: Agent reached the goal (source location)
- **Reward = 0.0**: Agent has not yet reached the goal
- **Higher concentrations**: Agent is closer to the source
- **Terminated = True**: Episode ended successfully (goal reached)
- **Truncated = True**: Episode ended due to step limit
    """
    )

    # Create observation and action space explanation with practical examples and user-friendly descriptions
    tutorial_sections.append(
        f"""
### Step 4: Understanding Spaces and Mechanics

Let's explore the environment's action and observation spaces in detail.

#### Action Space Deep Dive

```python
# Examine action space properties
print("Action Space Analysis:")
print(f"Type: {{type(env.action_space).__name__}}")
print(f"Number of actions: {{env.action_space.n}}")
print(f"Valid actions: {{list(range(env.action_space.n))}}")

# Test action sampling
print("\\nRandom action sampling:")
for i in range(5):
    sample_action = env.action_space.sample()
    print(f"Sample {{i+1}}: {{sample_action}} ({{Action(sample_action).name}})")

# Test action validation
print("\\nAction validation:")
for test_action in [0, 1, 2, 3, 4, -1]:
    valid = test_action in range(env.action_space.n)
    status = "✓ Valid" if valid else "✗ Invalid"
    print(f"Action {{test_action}}: {{status}}")
```

#### Observation Space Details

```python
# Examine observation space properties
print("\\nObservation Space Analysis:")
print(f"Type: {{type(env.observation_space).__name__}}")
print(f"Shape: {{env.observation_space.shape}}")
print(f"Data type: {{env.observation_space.dtype}}")
print(f"Low bound: {{env.observation_space.low}}")
print(f"High bound: {{env.observation_space.high}}")

# Sample observations
print("\\nObservation sampling:")
for i in range(3):
    sample_obs = env.observation_space.sample()
    print(f"Sample {{i+1}}: {{sample_obs}} (concentration: {{sample_obs[0]:.4f}})")

# Test observation validation
print("\\nObservation interpretation:")
test_concentrations = [0.0, 0.5, 1.0, 0.123]
for conc in test_concentrations:
    if conc == 0.0:
        meaning = "No chemical detected (far from source)"
    elif conc < 0.3:
        meaning = "Low concentration (distant from source)"
    elif conc < 0.7:
        meaning = "Moderate concentration (approaching source)"
    else:
        meaning = "High concentration (near source)"

    print(f"Concentration {{conc:.3f}}: {{meaning}}")
```

#### Movement Mechanics

```python
# Demonstrate movement mechanics with a controlled example
print("\\nMovement Mechanics Demonstration:")

# Reset to known position
obs, info = env.reset(seed={TUTORIAL_SEED})
print(f"Starting position - Concentration: {{obs[0]:.4f}}")

# Test each action systematically
actions_to_test = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
movement_results = []

for action in actions_to_test:
    # Reset to same starting position
    env.reset(seed={TUTORIAL_SEED})

    # Take one step in each direction
    obs, reward, terminated, truncated, info = env.step(action.value)

    result = {{
        'action': action.name,
        'concentration': obs[0],
        'reward': reward,
        'terminated': terminated
    }}
    movement_results.append(result)

    print(f"{{action.name:5s}}: Concentration {{obs[0]:.4f}}, Reward {{reward:.1f}}")

# Find best initial direction
best_move = max(movement_results, key=lambda x: x['concentration'])
print(f"\\nBest initial move: {{best_move['action']}} "
      f"(concentration: {{best_move['concentration']:.4f}})")
```
    """
    )

    # Add basic rendering tutorial if include_basic_visualization enabled with rgb_array mode demonstration
    if include_basic_visualization:
        tutorial_sections.append(
            f"""
### Step 5: Basic Visualization

Visualization is crucial for understanding agent behavior and environment dynamics. Let's explore the RGB array mode.

#### Enable RGB Array Rendering

```python
# Create environment with rgb_array mode
env_visual = gym.make(ENV_ID, render_mode='rgb_array')
obs, info = env_visual.reset(seed={TUTORIAL_SEED})

print("Environment created with rgb_array rendering")
print(f"Render mode: {{env_visual.render_mode}}")
```

#### Generate Visual Frames

```python
# Render initial state
rgb_frame = env_visual.render()

print(f"Frame information:")
print(f"- Shape: {{rgb_frame.shape}}")
print(f"- Data type: {{rgb_frame.dtype}}")
print(f"- Value range: {{rgb_frame.min()}} to {{rgb_frame.max()}}")

# Frame interpretation
height, width, channels = rgb_frame.shape
print(f"\\nFrame details:")
print(f"- Dimensions: {{width}} x {{height}} pixels")
print(f"- Channels: {{channels}} (RGB)")
print(f"- Total pixels: {{width * height:,}}")
```

#### Display with Matplotlib

```python
import matplotlib.pyplot as plt

# Set up visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display initial frame
axes[0].imshow(rgb_frame)
axes[0].set_title('Initial State')
axes[0].axis('off')

# Take a few steps and show progression
for i, ax in enumerate(axes[1:], 1):
    # Take random action
    action = env_visual.action_space.sample()
    obs, reward, terminated, truncated, info = env_visual.step(action)

    # Render new state
    rgb_frame = env_visual.render()

    # Display frame
    ax.imshow(rgb_frame)
    ax.set_title(f'After Step {{i}} (Action: {{Action(action).name}})')
    ax.axis('off')

    print(f"Step {{i}}: Action {{action}} ({{Action(action).name}}) "
          f"-> Concentration: {{obs[0]:.4f}}")

plt.tight_layout()
plt.show()

print("\\nVisualization complete!")
print("In the images above:")
print("- Gray areas represent plume concentration (darker = higher)")
print("- Red square shows agent position")
print("- White cross marks plume source location")
```

#### Understanding Visual Elements

The RGB array visualization includes several key elements:

1. **Background (Grayscale)**: Plume concentration field
   - Black/Dark: Low or no concentration
   - White/Light: High concentration near source

2. **Agent (Red Square)**: Current agent position
   - 3×3 pixel red marker for visibility
   - Updates as agent moves through environment

3. **Source (White Cross)**: Plume source location
   - 5×5 pixel white cross marker
   - Fixed position, represents navigation goal

4. **Grid Structure**: Underlying coordinate system
   - Environment coordinates map to pixel positions
   - Agent movement affects marker position

#### Programmatic Analysis

```python
# Analyze RGB array for automated processing
def analyze_rgb_frame(frame):
    \"\"\"Extract information from RGB frame for analysis.\"\"\"

    # Convert to grayscale for concentration analysis
    grayscale = np.mean(frame, axis=2)

    # Find agent position (look for red pixels)
    red_pixels = frame[:, :, 0] > frame[:, :, 1] + frame[:, :, 2]
    agent_positions = np.where(red_pixels)

    if len(agent_positions[0]) > 0:
        agent_y = int(np.mean(agent_positions[0]))
        agent_x = int(np.mean(agent_positions[1]))
    else:
        agent_x, agent_y = None, None

    # Find source position (look for white pixels)
    white_pixels = np.all(frame > 200, axis=2)  # High values in all channels
    source_positions = np.where(white_pixels)

    if len(source_positions[0]) > 0:
        source_y = int(np.mean(source_positions[0]))
        source_x = int(np.mean(source_positions[1]))
    else:
        source_x, source_y = None, None

    return {{
        'agent_position': (agent_x, agent_y) if agent_x is not None else None,
        'source_position': (source_x, source_y) if source_x is not None else None,
        'concentration_field': grayscale,
        'frame_stats': {{
            'mean_brightness': np.mean(grayscale),
            'max_brightness': np.max(grayscale),
            'min_brightness': np.min(grayscale)
        }}
    }}

# Analyze current frame
analysis = analyze_rgb_frame(rgb_frame)
print(f"\\nFrame Analysis:")
if analysis['agent_position']:
    print(f"Agent position: {{analysis['agent_position']}}")
if analysis['source_position']:
    print(f"Source position: {{analysis['source_position']}}")
print(f"Brightness stats: {{analysis['frame_stats']}}")
```

#### Cleanup

```python
# Clean up visualization environment
env_visual.close()
print("Visualization environment closed.")
```
        """
        )

    # Generate episode completion tutorial with termination conditions and reward understanding
    tutorial_sections.append(
        f"""
### Step 6: Episode Completion and Rewards

Understanding when and how episodes end is crucial for effective reinforcement learning. Let's explore termination and truncation conditions.

#### Reward Structure

The plume navigation environment uses a sparse binary reward system:

```python
print("Reward Structure:")
print("- Reward = 1.0: Agent reaches the goal (source location)")
print("- Reward = 0.0: Agent has not yet reached the goal")
print("- Goal achieved when: distance_to_source ≤ goal_radius")
print("- Default goal_radius = 0 (must reach exact source location)")
```

#### Termination vs Truncation

```python
# Demonstrate different episode ending conditions
def explore_episode_endings():
    results = {{'terminated': [], 'truncated': [], 'ongoing': []}}

    # Test multiple episodes with different outcomes
    for episode in range(5):
        env.reset(seed={TUTORIAL_SEED} + episode)

        episode_info = {{
            'episode': episode + 1,
            'steps': 0,
            'ended_by': None,
            'final_reward': 0,
            'max_concentration': 0
        }}

        # Run episode with step limit
        for step in range(20):  # Short episodes for demonstration
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            episode_info['steps'] = step + 1
            episode_info['final_reward'] += reward
            episode_info['max_concentration'] = max(episode_info['max_concentration'], obs[0])

            if terminated:
                episode_info['ended_by'] = 'terminated'
                results['terminated'].append(episode_info)
                break
            elif truncated:
                episode_info['ended_by'] = 'truncated'
                results['truncated'].append(episode_info)
                break

        if episode_info['ended_by'] is None:
            episode_info['ended_by'] = 'ongoing'
            results['ongoing'].append(episode_info)

        print(f"Episode {{episode + 1}}: {{episode_info['ended_by']}} after {{episode_info['steps']}} steps "
              f"(reward: {{episode_info['final_reward']:.1f}}, max_conc: {{episode_info['max_concentration']:.3f}})")

    return results

# Run episode analysis
print("Episode Ending Analysis:")
print("-" * 40)
ending_results = explore_episode_endings()

print(f"\\nSummary:")
print(f"- Terminated episodes: {{len(ending_results['terminated'])}}")
print(f"- Truncated episodes: {{len(ending_results['truncated'])}}")
print(f"- Ongoing episodes: {{len(ending_results['ongoing'])}}")
```

#### Goal Achievement Strategy

```python
# Demonstrate a simple gradient-following strategy
def simple_gradient_strategy(max_steps=50):
    \"\"\"
    Basic strategy: follow increasing concentration gradients.
    \"\"\"
    env.reset(seed={TUTORIAL_SEED})

    print("Testing gradient-following strategy:")
    print("-" * 40)

    prev_concentration = 0.0
    best_action = 0
    steps_without_improvement = 0

    for step in range(max_steps):
        # Try each action and see which gives highest concentration
        current_obs = env.reset(seed={TUTORIAL_SEED})  # Reset to test actions

        # Skip ahead to current step
        for _ in range(step):
            env.step(best_action)

        # Test each possible action
        action_concentrations = []
        for test_action in range(4):
            # Save current state
            temp_obs, temp_info = env.reset(seed={TUTORIAL_SEED})

            # Skip to current step
            for _ in range(step):
                env.step(best_action)

            # Test this action
            obs, reward, terminated, truncated, info = env.step(test_action)
            action_concentrations.append((test_action, obs[0], reward, terminated))

        # Choose action with highest concentration
        best_choice = max(action_concentrations, key=lambda x: x[1])
        chosen_action, concentration, reward, terminated = best_choice

        # Execute chosen action in main episode
        env.reset(seed={TUTORIAL_SEED})
        for _ in range(step):
            env.step(best_action)
        obs, reward, terminated, truncated, info = env.step(chosen_action)

        best_action = chosen_action

        print(f"Step {{step + 1:2d}}: Action {{chosen_action}} ({{Action(chosen_action).name:5s}}) "
              f"-> Concentration: {{concentration:.4f}}, Reward: {{reward:.1f}}")

        if terminated:
            print("🎉 Goal achieved using gradient-following strategy!")
            return step + 1, True
        elif truncated:
            print("⏰ Episode truncated before reaching goal.")
            return step + 1, False

        # Track improvement
        if concentration > prev_concentration:
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        # Early termination if stuck
        if steps_without_improvement > 5:
            print("Strategy appears stuck - ending early.")
            return step + 1, False

        prev_concentration = concentration

    print("Maximum steps reached without finding goal.")
    return max_steps, False

# Test the strategy
final_steps, success = simple_gradient_strategy()
print(f"\\nStrategy Results:")
print(f"- Steps taken: {{final_steps}}")
print(f"- Goal achieved: {{'Yes' if success else 'No'}}")
print(f"- Success rate: {{'100%' if success else '0%'}} (single trial)")
```
    """
    )

    # Create seed usage tutorial with deterministic episode reproduction for scientific consistency
    tutorial_sections.append(
        f"""
### Step 7: Reproducibility with Seeding

Scientific reproducibility is essential for reliable research. Let's learn how to use seeding for consistent results.

#### Basic Seeding

```python
from plume_nav_sim.utils.seeding import validate_seed

# Test seed validation
test_seeds = [{TUTORIAL_SEED}, 12345, 0, -1, "invalid"]

print("Seed Validation Tests:")
for seed in test_seeds:
    try:
        is_valid = validate_seed(seed) if isinstance(seed, int) else False
        print(f"Seed {{seed}}: {{'Valid' if is_valid else 'Invalid'}}")
    except Exception as e:
        print(f"Seed {{seed}}: Invalid ({{e}})")
```

#### Reproducing Identical Episodes

```python
def demonstrate_reproducibility(seed={TUTORIAL_SEED}, steps=10):
    \"\"\"Show that identical seeds produce identical episodes.\"\"\"

    episodes = []

    # Run same episode twice with identical seed
    for run in range(2):
        print(f"\\nRun {{run + 1}} with seed {{seed}}:")

        obs, info = env.reset(seed=seed)
        episode_data = [('reset', obs[0])]

        for step in range(steps):
            # Use deterministic action sequence for demo
            action = (step % 4)  # Cycle through all actions
            obs, reward, terminated, truncated, info = env.step(action)

            episode_data.append((f'step_{{step + 1}}', obs[0], reward))
            print(f"  Step {{step + 1}}: Action {{action}} -> Obs: {{obs[0]:.6f}}, Reward: {{reward}}")

            if terminated or truncated:
                break

        episodes.append(episode_data)

    # Compare episodes
    print(f"\\nReproducibility Check:")
    identical = episodes[0] == episodes[1]
    print(f"Episodes identical: {{'✓ Yes' if identical else '✗ No'}}")

    if identical:
        print("Perfect reproducibility achieved!")
    else:
        print("Checking differences...")
        for i, (data1, data2) in enumerate(zip(episodes[0], episodes[1])):
            if data1 != data2:
                print(f"  Difference at step {{i}}: {{data1}} vs {{data2}}")

    return episodes

# Demonstrate reproducibility
episodes = demonstrate_reproducibility()
```

#### Seed Management Best Practices

```python
from plume_nav_sim.utils.seeding import SeedManager

# Create seed manager for systematic seed handling
seed_manager = SeedManager(default_seed={TUTORIAL_SEED}, enable_validation=True)

print("Seed Management Best Practices:")

# Generate reproducible seed sequences
experiment_seeds = [seed_manager.generate_episode_seed(i) for i in range(5)]
print(f"\\nGenerated experiment seeds: {{experiment_seeds}}")

# Test seed validation
print(f"\\nSeed validation:")
for seed in experiment_seeds:
    is_valid = seed_manager.validate_seed(seed)
    print(f"  Seed {{seed}}: {{'Valid' if is_valid else 'Invalid'}}")

# Demonstrate seed independence
print(f"\\nSeed Independence Test:")
results = []

for i, seed in enumerate(experiment_seeds[:3]):
    env.reset(seed=seed)

    # Take 3 random actions
    observations = []
    for _ in range(3):
        action = env.action_space.sample()  # This should be deterministic with proper seeding
        obs, _, _, _, _ = env.step(action)
        observations.append(obs[0])

    results.append(observations)
    print(f"  Seed {{seed}}: {{[f'{{obs:.4f}}' for obs in observations]}}")

# Check that different seeds give different results
all_same = all(result == results[0] for result in results[1:])
print(f"\\nAll results identical: {{'No (good)' if not all_same else 'Yes (concerning)'}}")
```

#### Research-Grade Reproducibility

```python
def research_reproducibility_demo():
    \"\"\"Demonstrate reproducibility practices for research.\"\"\"

    # Experimental configuration
    config = {{
        'base_seed': {TUTORIAL_SEED},
        'num_repetitions': 3,
        'episode_length': 15,
        'environment_params': {{}},  # Default parameters
    }}

    print("Research-Grade Reproducibility Demonstration")
    print("=" * 50)
    print(f"Configuration: {{config}}")

    # Run systematic experiment
    all_results = []

    for rep in range(config['num_repetitions']):
        rep_seed = config['base_seed'] + rep * 1000  # Ensure seed separation
        print(f"\\nRepetition {{rep + 1}}/{{config['num_repetitions']}} (seed: {{rep_seed}}):")

        obs, info = env.reset(seed=rep_seed)
        rep_results = {{
            'repetition': rep + 1,
            'seed': rep_seed,
            'observations': [obs[0]],
            'actions': [],
            'rewards': [],
            'final_concentration': obs[0]
        }}

        # Execute deterministic episode
        for step in range(config['episode_length']):
            # Use deterministic policy for reproducibility
            action = (step + rep) % 4  # Vary by repetition
            obs, reward, terminated, truncated, info = env.step(action)

            rep_results['observations'].append(obs[0])
            rep_results['actions'].append(action)
            rep_results['rewards'].append(reward)
            rep_results['final_concentration'] = obs[0]

            if step < 5 or step >= config['episode_length'] - 3:  # Show first and last few
                print(f"  Step {{step + 1:2d}}: A={{action}} -> O={{obs[0]:.4f}}, R={{reward}}")
            elif step == 5:
                print("  ... (intermediate steps)")

            if terminated or truncated:
                break

        all_results.append(rep_results)

    # Analyze reproducibility across repetitions
    print(f"\\nReproducibility Analysis:")
    print("-" * 30)

    # Check final concentrations
    final_concentrations = [r['final_concentration'] for r in all_results]
    print(f"Final concentrations: {{[f'{{c:.4f}}' for c in final_concentrations]}}")

    # Statistical summary
    mean_final = np.mean(final_concentrations)
    std_final = np.std(final_concentrations)
    print(f"Mean ± std: {{mean_final:.4f}} ± {{std_final:.4f}}")

    # Check total rewards
    total_rewards = [sum(r['rewards']) for r in all_results]
    print(f"Total rewards: {{total_rewards}}")

    return all_results, config

# Run research demo
research_results, research_config = research_reproducibility_demo()

print(f"\\n✅ Reproducibility demonstration complete!")
print(f"Key takeaways:")
print(f"- Always use explicit seeds for reproducible experiments")
print(f"- Validate seeds before use")
print(f"- Use separate seed ranges for different experimental conditions")
print(f"- Document seed values in research publications")
```
    """
    )

    # Add troubleshooting guidance and common issue resolution for section topics
    tutorial_sections.append(
        f"""
### Step 8: Troubleshooting and Next Steps

Let's address common issues and provide guidance for continuing your plume navigation research.

#### Common Issues and Solutions

```python
# Issue 1: Environment creation fails
try:
    test_env = gym.make(ENV_ID)
    print("✓ Environment creation successful")
    test_env.close()
except Exception as e:
    print(f"✗ Environment creation failed: {{e}}")
    print("Solution: Ensure register_env() was called before gym.make()")

# Issue 2: Import errors
try:
    from plume_nav_sim.core.enums import Action
    print("✓ Core types import successful")
except ImportError as e:
    print(f"✗ Import failed: {{e}}")
    print("Solution: Check plume_nav_sim installation with 'pip install -e .'")

# Issue 3: Rendering issues
try:
    test_env = gym.make(ENV_ID, render_mode='rgb_array')
    test_obs, _ = test_env.reset()
    test_frame = test_env.render()
    print(f"✓ Rendering successful: {{test_frame.shape}}")
    test_env.close()
except Exception as e:
    print(f"✗ Rendering failed: {{e}}")
    print("Solution: Check matplotlib installation and display settings")

# Issue 4: Seeding problems
try:
    test_env = gym.make(ENV_ID)
    obs1, _ = test_env.reset(seed=42)
    obs2, _ = test_env.reset(seed=42)
    identical = np.allclose(obs1, obs2)
    print(f"✓ Seeding working: {{'identical' if identical else 'different'}} results")
    test_env.close()
except Exception as e:
    print(f"✗ Seeding failed: {{e}}")
    print("Solution: Ensure seed parameter is integer and environment supports seeding")
```

#### Performance Optimization Tips

```python
# Performance monitoring
import time

def measure_performance():
    \"\"\"Measure environment performance for optimization.\"\"\"

    env = gym.make(ENV_ID)

    # Measure reset performance
    reset_times = []
    for i in range(10):
        start_time = time.perf_counter()
        obs, info = env.reset(seed=i)
        reset_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        reset_times.append(reset_time)

    # Measure step performance
    obs, info = env.reset(seed=42)
    step_times = []
    for i in range(100):
        action = env.action_space.sample()
        start_time = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        step_times.append(step_time)

        if terminated or truncated:
            obs, info = env.reset(seed=42 + i)

    # Measure rendering performance (if enabled)
    env_render = gym.make(ENV_ID, render_mode='rgb_array')
    obs, info = env_render.reset(seed=42)

    render_times = []
    for i in range(10):
        start_time = time.perf_counter()
        frame = env_render.render()
        render_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        render_times.append(render_time)

    # Display results
    print("Performance Measurements:")
    print(f"- Reset time: {{np.mean(reset_times):.2f}} ± {{np.std(reset_times):.2f}} ms")
    print(f"- Step time: {{np.mean(step_times):.3f}} ± {{np.std(step_times):.3f}} ms")
    print(f"- Render time: {{np.mean(render_times):.2f}} ± {{np.std(render_times):.2f}} ms")

    # Performance targets
    print(f"\\nPerformance Targets:")
    print(f"- Reset: <10ms (current: {{np.mean(reset_times):.1f}}ms)")
    print(f"- Step: <1ms (current: {{np.mean(step_times):.2f}}ms)")
    print(f"- Render: <5ms (current: {{np.mean(render_times):.1f}}ms)")

    env.close()
    env_render.close()

# Run performance measurement
measure_performance()
```

#### Next Steps and Advanced Topics

After completing this tutorial, you're ready to explore more advanced topics:

```python
print("Congratulations! You've completed the quick start tutorial. 🎉")
print("\\nNext steps for your plume navigation research:")
print()

print("1. **Configuration and Customization**")
print("   - Learn to customize grid sizes, source locations, and goal radii")
print("   - Understand parameter effects on navigation difficulty")
print("   - Create custom environment configurations for your research")
print()

print("2. **Advanced Visualization**")
print("   - Master human mode rendering for interactive visualization")
print("   - Create publication-quality figures and animations")
print("   - Analyze agent trajectories and behavior patterns")
print()

print("3. **Reproducible Research Practices**")
print("   - Implement systematic experimental designs")
print("   - Use proper statistical analysis and hypothesis testing")
print("   - Document experiments for publication and replication")
print()

print("4. **Research Integration**")
print("   - Integrate with RL training frameworks (Stable-Baselines3, Ray RLLib)")
print("   - Use Jupyter notebooks for interactive research workflows")
print("   - Export data for analysis with pandas, scikit-learn, etc.")
print()

print("5. **Algorithm Development**")
print("   - Implement and test custom navigation strategies")
print("   - Compare algorithm performance systematically")
print("   - Optimize hyperparameters and training procedures")
print()

print("Continue with the full user guide for detailed coverage of these topics!")
```

#### Tutorial Completion

```python
# Final cleanup
env.close()

# Summary of what was learned
tutorial_summary = {{
    'environment_creation': True,
    'basic_episodes': True,
    'action_observation_spaces': True,
    'visualization_basics': True,
    'reproducibility': True,
    'troubleshooting': True
}}

completed_topics = sum(tutorial_summary.values())
total_topics = len(tutorial_summary)

print(f"\\n🎓 Tutorial Complete!")
print(f"Topics mastered: {{completed_topics}}/{{total_topics}}")
print(f"You're now ready to begin productive research with plume_nav_sim!")

# Provide quick reference
print(f"\\n📋 Quick Reference:")
print(f"- Environment ID: {{ENV_ID}}")
print(f"- Registration: register_env()")
print(f"- Creation: gym.make(ENV_ID)")
print(f"- Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT")
print(f"- Observations: [0.0, 1.0] concentration values")
print(f"- Goal: Navigate to source for reward=1.0")
print(f"- Reproducibility: Always use seeds for research!")
```
    """
    )

    # Combine all sections into complete tutorial
    complete_tutorial = "\n\n".join(tutorial_sections)

    return complete_tutorial


def create_configuration_guide(
    include_parameter_effects: bool = True,
    include_research_presets: bool = True,
    include_validation_examples: bool = True,
) -> str:
    """
    Creates comprehensive configuration guide documenting all environment parameters, default values,
    research-relevant customizations, and practical configuration patterns for scientific applications
    and experimental design.

    This function generates detailed documentation of all configurable environment parameters,
    their effects on navigation difficulty and behavior, and provides research-focused presets
    for common experimental scenarios.

    Args:
        include_parameter_effects (bool): Include detailed analysis of parameter effects on behavior and performance
        include_research_presets (bool): Include predefined configurations for common research scenarios
        include_validation_examples (bool): Include parameter validation examples with error handling guidance

    Returns:
        str: Detailed configuration guide with parameter documentation, research examples,
        and practical customization patterns for scientific applications
    """
    guide_sections = []

    # Generate configuration overview explaining environment customization capabilities and research benefits
    guide_sections.append(
        f"""
## Configuration Guide

The plume navigation environment provides extensive configuration options to support diverse research
applications and experimental designs. This guide covers all parameters, their effects, and provides
practical examples for scientific use.

### Configuration Philosophy

plume_nav_sim follows a **configuration-first** approach where:

1. **Default Values**: Provide reasonable starting points for immediate use
2. **Explicit Parameters**: All configuration options are clearly documented and validated
3. **Research Focus**: Parameters designed to support systematic experimental variation
4. **Reproducibility**: Configuration settings directly impact experimental reproducibility
5. **Performance Optimization**: Parameter choices affect computational performance

### Overview of Configurable Parameters

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| `grid_size` | Tuple[int, int] | (128, 128) | (16, 16) to (512, 512) | Environment dimensions |
| `source_location` | Tuple[int, int] | (64, 64) | Within grid bounds | Plume source position |
| `max_steps` | int | 1000 | 10 to 10000 | Episode length limit |
| `goal_radius` | float | 0.0 | 0.0 to grid diagonal | Goal detection threshold |
| `render_mode` | str | 'rgb_array' | 'rgb_array', 'human' | Visualization mode |

### Basic Configuration Example

```python
from plume_nav_sim.envs.plume_search_env import create_plume_search_env
import gymnasium as gym

# Method 1: Using environment factory function
env = create_plume_search_env(
    grid_size=(64, 64),          # Smaller grid for faster episodes
    source_location=(32, 32),    # Center position
    max_steps=500,               # Shorter episodes
    goal_radius=2.0,             # Allow nearby goal detection
    render_mode='human'          # Interactive visualization
)

# Method 2: Using gym.make() with registration
from plume_nav_sim.registration import register_env, ENV_ID

register_env()
env = gym.make(ENV_ID)  # Uses default parameters

# Method 3: Direct class instantiation
from plume_nav_sim.envs.plume_search_env import PlumeSearchEnv

env = PlumeSearchEnv(
    grid_size=(32, 32),
    source_location=(16, 16),
    max_steps=200,
    goal_radius=1.0
)
```
    """
    )

    # Document core parameters including grid_size, source_location, max_steps, and goal_radius with research context
    guide_sections.append(
        f"""
### Core Parameters Documentation

#### Grid Size Configuration

The `grid_size` parameter defines the environment's spatial dimensions and significantly impacts navigation difficulty.

```python
# Grid size examples and implications
configurations = [
    {{'size': (32, 32), 'cells': 1024, 'difficulty': 'Easy', 'use_case': 'Rapid prototyping, algorithm testing'}},
    {{'size': (64, 64), 'cells': 4096, 'difficulty': 'Medium', 'use_case': 'Standard experiments, educational demos'}},
    {{'size': (128, 128), 'cells': 16384, 'difficulty': 'Hard', 'use_case': 'Research benchmarks, publication results'}},
    {{'size': (256, 256), 'cells': 65536, 'difficulty': 'Very Hard', 'use_case': 'Scalability testing, advanced algorithms'}}
]

print("Grid Size Analysis:")
for config in configurations:
    print(f"{{config['size']}}: {{config['cells']:,}} cells, {{config['difficulty']}} - {{config['use_case']}}")
```

**Considerations for Grid Size Selection**:
- **Computational Cost**: Scales quadratically with dimensions
- **Navigation Difficulty**: Larger grids require more sophisticated strategies
- **Memory Usage**: Approximately 4 bytes per cell for concentration storage
- **Visualization Quality**: Larger grids provide finer detail but slower rendering

#### Source Location Configuration

The `source_location` parameter determines where the plume originates and affects navigation complexity.

```python
# Source location strategies for research
def analyze_source_locations(grid_size=(64, 64)):
    \"\"\"Analyze different source location strategies.\"\"\"

    width, height = grid_size

    locations = {{
        'center': (width // 2, height // 2),
        'corner': (width // 4, height // 4),
        'edge': (0, height // 2),
        'off_center': (width // 3, 2 * height // 3),
        'near_corner': (width - 5, height - 5)
    }}

    print(f"Source Location Analysis for {{grid_size}} grid:")

    for name, (x, y) in locations.items():
        # Calculate maximum possible distance
        max_dist_from_corner = np.sqrt((width - x)**2 + (height - y)**2)
        max_dist_from_center = np.sqrt((width//2 - x)**2 + (height//2 - y)**2)

        print(f"\\n{{name.capitalize()}} ({{x}}, {{y}}):")
        print(f"  Max distance from corner: {{max_dist_from_corner:.1f}}")
        print(f"  Distance from center: {{max_dist_from_center:.1f}}")
        print(f"  Navigation difficulty: {{'Low' if max_dist_from_corner < 30 else 'Medium' if max_dist_from_corner < 60 else 'High'}}")

# Run analysis
analyze_source_locations()

# Demonstrate configuration impact
def test_source_location_impact():
    \"\"\"Test how source location affects initial observations.\"\"\"

    locations = [(16, 16), (32, 32), (48, 48)]  # For 64x64 grid

    for loc in locations:
        env = create_plume_search_env(
            grid_size=(64, 64),
            source_location=loc,
            max_steps=100
        )

        # Test multiple random starting positions
        concentrations = []
        for seed in range(10):
            obs, info = env.reset(seed=seed)
            concentrations.append(obs[0])

        avg_initial_conc = np.mean(concentrations)
        std_initial_conc = np.std(concentrations)

        print(f"Source at {{loc}}: avg_initial={{avg_initial_conc:.4f}} ± {{std_initial_conc:.4f}}")
        env.close()

test_source_location_impact()
```

#### Episode Length Configuration

The `max_steps` parameter controls episode duration and affects learning dynamics.

```python
# Episode length analysis for different research objectives
episode_configs = {{
    'quick_test': {{'max_steps': 50, 'purpose': 'Rapid algorithm testing'}},
    'standard': {{'max_steps': 200, 'purpose': 'Standard RL training'}},
    'exploration': {{'max_steps': 500, 'purpose': 'Exploration-heavy algorithms'}},
    'thorough': {{'max_steps': 1000, 'purpose': 'Publication benchmarks'}},
    'extended': {{'max_steps': 2000, 'purpose': 'Long-horizon planning'}}
}}

print("Episode Length Configuration Guide:")
for name, config in episode_configs.items():
    steps = config['max_steps']
    purpose = config['purpose']

    # Estimate computational cost
    est_time_seconds = steps * 0.001  # Assuming 1ms per step

    print(f"\\n{{name.capitalize()}}: {{steps}} steps")
    print(f"  Purpose: {{purpose}}")
    print(f"  Estimated time: {{est_time_seconds:.1f}}s per episode")
    print(f"  Recommended for: {{'Quick validation' if steps < 100 else 'Standard research' if steps < 500 else 'Comprehensive evaluation'}}")

# Demonstrate episode length effects
def episode_length_impact_study():
    \"\"\"Study how episode length affects success probability.\"\"\"

    max_steps_values = [50, 100, 200, 500]
    results = []

    for max_steps in max_steps_values:
        env = create_plume_search_env(max_steps=max_steps, grid_size=(32, 32))

        successes = 0
        avg_steps = 0

        for seed in range(20):  # 20 trials per configuration
            obs, info = env.reset(seed=seed)

            for step in range(max_steps):
                action = env.action_space.sample()  # Random policy
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated:
                    successes += 1
                    avg_steps += step + 1
                    break
                elif truncated:
                    avg_steps += step + 1
                    break

        success_rate = successes / 20
        avg_episode_length = avg_steps / 20

        results.append({{
            'max_steps': max_steps,
            'success_rate': success_rate,
            'avg_episode_length': avg_episode_length
        }})

        print(f"Max steps {{max_steps}}: {{success_rate:.1%}} success, avg length {{avg_episode_length:.1f}}")
        env.close()

    return results

print("\\nEpisode Length Impact Study (Random Policy):")
length_results = episode_length_impact_study()
```

#### Goal Radius Configuration

The `goal_radius` parameter defines goal achievement tolerance and affects task difficulty.

```python
# Goal radius analysis for research applications
def analyze_goal_radius_effects():
    \"\"\"Analyze how goal radius affects navigation task difficulty.\"\"\"

    radius_configs = [
        {{'radius': 0.0, 'description': 'Exact location required'}},
        {{'radius': 1.0, 'description': 'Adjacent cells accepted'}},
        {{'radius': 2.0, 'description': 'Nearby region accepted'}},
        {{'radius': 5.0, 'description': 'Large target area'}},
        {{'radius': 10.0, 'description': 'Very forgiving goal'}}
    ]

    print("Goal Radius Configuration Analysis:")

    for config in radius_configs:
        radius = config['radius']
        description = config['description']

        # Calculate target area
        if radius == 0:
            target_cells = 1
        else:
            target_cells = int(np.pi * radius**2)

        # Estimate difficulty based on target size
        difficulty_score = 1.0 / (1.0 + target_cells)
        difficulty = 'Very Hard' if difficulty_score > 0.5 else 'Hard' if difficulty_score > 0.1 else 'Medium' if difficulty_score > 0.02 else 'Easy'

        print(f"\\nRadius {{radius}}: {{description}}")
        print(f"  Target area: {{target_cells}} cells")
        print(f"  Difficulty: {{difficulty}}")
        print(f"  Recommended for: {{'Precise navigation' if radius < 1 else 'Balanced learning' if radius < 3 else 'Exploration focus'}}")

analyze_goal_radius_effects()

# Practical goal radius testing
def test_goal_radius_impact():
    \"\"\"Test goal radius impact on success rates.\"\"\"

    radii = [0.0, 1.0, 2.0, 5.0]

    print("\\nGoal Radius Impact Test:")

    for radius in radii:
        env = create_plume_search_env(
            grid_size=(32, 32),
            goal_radius=radius,
            max_steps=100
        )

        successes = 0
        for seed in range(50):  # More trials for statistical significance
            obs, info = env.reset(seed=seed)

            # Simple gradient-following heuristic
            for step in range(100):
                # Choose action that maximizes concentration (simplified)
                action = env.action_space.sample()  # Random for demo
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated:
                    successes += 1
                    break
                elif truncated:
                    break

        success_rate = successes / 50
        print(f"Radius {{radius:3.1f}}: {{success_rate:5.1%}} success rate")
        env.close()

test_goal_radius_impact()
```
    """
    )

    # Create parameter effects section if include_parameter_effects enabled with performance and behavior implications
    if include_parameter_effects:
        guide_sections.append(
            f"""
### Parameter Effects and Interactions

Understanding how parameters interact is crucial for designing effective experiments and choosing appropriate configurations.

#### Grid Size vs Performance Trade-offs

```python
import time

def performance_vs_grid_size_analysis():
    \"\"\"Analyze computational performance across different grid sizes.\"\"\"

    grid_sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
    performance_data = []

    print("Grid Size Performance Analysis:")
    print("-" * 60)

    for grid_size in grid_sizes:
        print(f"Testing {{grid_size}}...")

        # Create environment
        start_time = time.perf_counter()
        env = create_plume_search_env(grid_size=grid_size)
        creation_time = (time.perf_counter() - start_time) * 1000

        # Measure reset performance
        reset_times = []
        for i in range(10):
            start_time = time.perf_counter()
            obs, info = env.reset(seed=i)
            reset_times.append((time.perf_counter() - start_time) * 1000)

        # Measure step performance
        obs, info = env.reset(seed=0)
        step_times = []
        for i in range(100):
            action = env.action_space.sample()
            start_time = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_times.append((time.perf_counter() - start_time) * 1000)

            if terminated or truncated:
                obs, info = env.reset(seed=i)

        # Calculate memory estimate
        width, height = grid_size
        estimated_memory_mb = (width * height * 4) / (1024 * 1024)  # 4 bytes per cell

        perf_data = {{
            'grid_size': grid_size,
            'total_cells': width * height,
            'creation_time_ms': creation_time,
            'avg_reset_time_ms': np.mean(reset_times),
            'avg_step_time_ms': np.mean(step_times),
            'estimated_memory_mb': estimated_memory_mb
        }}

        performance_data.append(perf_data)

        print(f"  Cells: {{width * height:,}}")
        print(f"  Creation: {{creation_time:.2f}}ms")
        print(f"  Reset: {{np.mean(reset_times):.3f}}ms")
        print(f"  Step: {{np.mean(step_times):.3f}}ms")
        print(f"  Memory: {{estimated_memory_mb:.1f}}MB")
        print()

        env.close()

    # Performance recommendations
    print("Performance Recommendations:")
    for data in performance_data:
        grid = data['grid_size']
        step_time = data['avg_step_time_ms']
        memory = data['estimated_memory_mb']

        if step_time < 0.5 and memory < 10:
            recommendation = "Excellent for rapid prototyping and testing"
        elif step_time < 1.0 and memory < 50:
            recommendation = "Good for standard research applications"
        elif step_time < 2.0 and memory < 200:
            recommendation = "Suitable for publication benchmarks"
        else:
            recommendation = "Consider for scalability studies only"

        print(f"{{grid}}: {{recommendation}}")

    return performance_data

perf_data = performance_vs_grid_size_analysis()
```

#### Parameter Interaction Effects

```python
def parameter_interaction_study():
    \"\"\"Study how different parameters interact to affect task difficulty.\"\"\"

    # Define parameter combinations to test
    test_configs = [
        # Easy configurations
        {{'grid_size': (32, 32), 'goal_radius': 2.0, 'max_steps': 200, 'difficulty': 'Easy'}},
        {{'grid_size': (32, 32), 'goal_radius': 1.0, 'max_steps': 100, 'difficulty': 'Easy-Medium'}},

        # Medium configurations
        {{'grid_size': (64, 64), 'goal_radius': 1.0, 'max_steps': 300, 'difficulty': 'Medium'}},
        {{'grid_size': (64, 64), 'goal_radius': 0.0, 'max_steps': 500, 'difficulty': 'Medium-Hard'}},

        # Hard configurations
        {{'grid_size': (128, 128), 'goal_radius': 1.0, 'max_steps': 500, 'difficulty': 'Hard'}},
        {{'grid_size': (128, 128), 'goal_radius': 0.0, 'max_steps': 1000, 'difficulty': 'Very Hard'}}
    ]

    print("Parameter Interaction Study:")
    print("=" * 50)

    results = []

    for i, config in enumerate(test_configs):
        print(f"\\nConfiguration {{i+1}}: {{config['difficulty']}}")
        print(f"  Grid: {{config['grid_size']}}")
        print(f"  Goal radius: {{config['goal_radius']}}")
        print(f"  Max steps: {{config['max_steps']}}")

        # Create environment with configuration
        env = create_plume_search_env(
            grid_size=config['grid_size'],
            goal_radius=config['goal_radius'],
            max_steps=config['max_steps']
        )

        # Test with random policy
        success_count = 0
        step_counts = []

        num_trials = 25
        for trial in range(num_trials):
            obs, info = env.reset(seed=trial)

            for step in range(config['max_steps']):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated:
                    success_count += 1
                    step_counts.append(step + 1)
                    break
                elif truncated:
                    step_counts.append(step + 1)
                    break

        success_rate = success_count / num_trials
        avg_steps = np.mean(step_counts) if step_counts else config['max_steps']

        result = {{
            'config': config,
            'success_rate': success_rate,
            'avg_steps_to_completion': avg_steps,
            'efficiency_score': success_rate / (avg_steps / config['max_steps'])
        }}

        results.append(result)

        print(f"  Success rate: {{success_rate:.1%}}")
        print(f"  Avg steps: {{avg_steps:.1f}}")
        print(f"  Efficiency: {{result['efficiency_score']:.3f}}")

        env.close()

    # Analyze results
    print(f"\\nInteraction Analysis Summary:")
    print("-" * 30)

    # Sort by difficulty (success rate)
    sorted_results = sorted(results, key=lambda x: x['success_rate'], reverse=True)

    for i, result in enumerate(sorted_results):
        config = result['config']
        print(f"{{i+1}}. {{config['difficulty']}}: {{result['success_rate']:.1%}} success")
        print(f"   Grid {{config['grid_size']}}, radius {{config['goal_radius']}}, "
              f"steps {{config['max_steps']}}")

    return results

interaction_results = parameter_interaction_study()
```

#### Visualization Parameter Effects

```python
def visualization_parameter_effects():
    \"\"\"Demonstrate how parameters affect visualization quality and performance.\"\"\"

    print("Visualization Parameter Effects:")
    print("=" * 40)

    # Test different configurations for visualization
    vis_configs = [
        {{'grid_size': (32, 32), 'render_mode': 'rgb_array', 'description': 'Small, fast rendering'}},
        {{'grid_size': (128, 128), 'render_mode': 'rgb_array', 'description': 'Standard quality'}},
        {{'grid_size': (256, 256), 'render_mode': 'rgb_array', 'description': 'High resolution'}},
    ]

    for config in vis_configs:
        print(f"\\nTesting: {{config['description']}}")
        print(f"Grid size: {{config['grid_size']}}")

        env = create_plume_search_env(
            grid_size=config['grid_size'],
            render_mode=config['render_mode']
        )

        obs, info = env.reset(seed=42)

        # Measure rendering performance
        render_times = []
        for i in range(10):
            start_time = time.perf_counter()
            frame = env.render()
            render_time = (time.perf_counter() - start_time) * 1000
            render_times.append(render_time)

        avg_render_time = np.mean(render_times)
        frame_size_mb = (frame.nbytes) / (1024 * 1024)

        print(f"  Frame shape: {{frame.shape}}")
        print(f"  Frame size: {{frame_size_mb:.2f}}MB")
        print(f"  Render time: {{avg_render_time:.1f}}ms")

        # Quality assessment
        if avg_render_time < 5:
            quality_assessment = "Excellent for real-time use"
        elif avg_render_time < 20:
            quality_assessment = "Good for interactive use"
        else:
            quality_assessment = "Better for batch processing"

        print(f"  Assessment: {{quality_assessment}}")

        env.close()

visualization_parameter_effects()
```
        """
        )

    # Add research preset configurations if include_research_presets enabled with common experimental designs
    if include_research_presets:
        guide_sections.append(
            f"""
### Research Configuration Presets

Pre-defined configurations for common research scenarios and experimental designs.

#### Educational and Teaching Presets

```python
# Educational configurations for teaching and learning
EDUCATIONAL_PRESETS = {{
    'beginner_demo': {{
        'grid_size': (16, 16),
        'source_location': (8, 8),
        'max_steps': 50,
        'goal_radius': 2.0,
        'render_mode': 'human',
        'description': 'Simple demo for first-time users',
        'expected_success_rate': 0.8,
        'typical_episode_length': 15
    }},

    'classroom_exercise': {{
        'grid_size': (32, 32),
        'source_location': (16, 16),
        'max_steps': 100,
        'goal_radius': 1.0,
        'render_mode': 'rgb_array',
        'description': 'Standard configuration for classroom exercises',
        'expected_success_rate': 0.4,
        'typical_episode_length': 35
    }},

    'student_project': {{
        'grid_size': (64, 64),
        'source_location': (32, 32),
        'max_steps': 200,
        'goal_radius': 0.0,
        'render_mode': 'rgb_array',
        'description': 'Challenging config for student research projects',
        'expected_success_rate': 0.1,
        'typical_episode_length': 80
    }}
}}

def create_educational_environment(preset_name):
    \"\"\"Create environment using educational preset.\"\"\"

    if preset_name not in EDUCATIONAL_PRESETS:
        raise ValueError(f"Unknown preset: {{preset_name}}. Available: {{list(EDUCATIONAL_PRESETS.keys())}}")

    config = EDUCATIONAL_PRESETS[preset_name]

    print(f"Creating '{{preset_name}}' environment:")
    print(f"  Description: {{config['description']}}")
    print(f"  Expected success rate: {{config['expected_success_rate']:.1%}}")
    print(f"  Typical episode length: {{config['typical_episode_length']}} steps")

    env = create_plume_search_env(
        grid_size=config['grid_size'],
        source_location=config['source_location'],
        max_steps=config['max_steps'],
        goal_radius=config['goal_radius'],
        render_mode=config['render_mode']
    )

    return env

# Demonstrate educational presets
for preset_name in EDUCATIONAL_PRESETS.keys():
    env = create_educational_environment(preset_name)

    # Quick validation
    obs, info = env.reset(seed=42)
    print(f"  Initial observation: {{obs[0]:.4f}}")

    env.close()
    print()
```
        """
        )

    return "\n\n".join(guide_sections)
