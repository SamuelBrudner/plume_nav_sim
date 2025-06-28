# Gymnasium Migration Guide

**From OpenAI Gym 0.26 to Gymnasium 0.29.x**

This comprehensive guide provides step-by-step instructions for migrating from the legacy OpenAI Gym 0.26 to the modern Gymnasium 0.29.x API in the plume_nav_sim library (v0.3.0).

## Table of Contents

1. [Overview](#overview)
2. [Quick Start Migration](#quick-start-migration)
3. [API Differences](#api-differences)
4. [Step-by-Step Migration](#step-by-step-migration)
5. [Compatibility Shim Usage](#compatibility-shim-usage)
6. [Environment Registration](#environment-registration)
7. [Code Examples](#code-examples)
8. [Performance Considerations](#performance-considerations)
9. [Troubleshooting](#troubleshooting)
10. [Deprecation Timeline](#deprecation-timeline)

---

## Overview

The plume_nav_sim library v0.3.0 introduces full support for Gymnasium 0.29.x while maintaining backward compatibility with legacy OpenAI Gym patterns. This migration enables:

- **Modern API compliance** with 5-tuple step returns and improved reset handling
- **Enhanced type safety** with better type annotations and validation
- **Performance optimizations** with frame caching and sub-10ms step execution
- **Extensibility hooks** for custom observations, rewards, and episode handling
- **Automatic compatibility** through the built-in compatibility shim

### Key Benefits

✅ **Zero Breaking Changes**: Existing code continues to work unchanged  
✅ **Automatic Detection**: Smart API detection based on imports and usage patterns  
✅ **Performance Gains**: Optimized execution with <10ms step times  
✅ **Future-Proof**: Built on Gymnasium's actively maintained codebase  

---

## Quick Start Migration

### For New Projects (Recommended)

```python
# Modern Gymnasium approach
import gymnasium as gym
from plume_nav_sim.environments import GymnasiumEnv

# Create environment with new ID
env = gym.make("PlumeNavSim-v0", video_path="data/plume_movie.mp4")

# Modern reset (returns tuple)
obs, info = env.reset(seed=42)

# Modern step (returns 5-tuple)
obs, reward, terminated, truncated, info = env.step(action)

# Handle episode completion
if terminated or truncated:
    obs, info = env.reset()
```

### For Existing Projects (Compatibility Mode)

```python
# Legacy gym approach - continues to work
from plume_nav_sim.shims import gym_make

# Uses compatibility shim with deprecation warning
env = gym_make("PlumeNavSim-v0", video_path="data/plume_movie.mp4")

# Legacy reset (returns observation only)
obs = env.reset()

# Legacy step (returns 4-tuple)
obs, reward, done, info = env.step(action)

# Handle episode completion
if done:
    obs = env.reset()
```

---

## API Differences

### Key Changes Summary

| Component | Legacy Gym 0.26 | Modern Gymnasium 0.29.x |
|-----------|------------------|--------------------------|
| **Import** | `import gym` | `import gymnasium` |
| **Environment ID** | `OdorPlumeNavigation-v1` | `PlumeNavSim-v0` |
| **Reset Return** | `obs` | `obs, info` |
| **Step Return** | `obs, reward, done, info` | `obs, reward, terminated, truncated, info` |
| **Seed Parameter** | `env.seed(seed)` | `env.reset(seed=seed)` |

### Detailed API Comparison

#### 1. Environment Reset

**Legacy Gym:**
```python
obs = env.reset()
env.seed(42)  # Separate seed call
```

**Modern Gymnasium:**
```python
obs, info = env.reset(seed=42)  # Seed integrated into reset
```

#### 2. Environment Step

**Legacy Gym (4-tuple):**
```python
obs, reward, done, info = env.step(action)
if done:
    # Episode finished (any reason)
    obs = env.reset()
```

**Modern Gymnasium (5-tuple):**
```python
obs, reward, terminated, truncated, info = env.step(action)
if terminated or truncated:
    # terminated: episode ended naturally (success/failure)
    # truncated: episode ended due to time/step limits
    obs, info = env.reset()
```

#### 3. Done Flag Logic

**Legacy (single flag):**
```python
done = env_terminated_naturally or time_limit_reached
```

**Modern (separate flags):**
```python
terminated = env_terminated_naturally  # Success/failure
truncated = time_limit_reached         # Time/step limit
```

---

## Step-by-Step Migration

### Phase 1: Install Dependencies

```bash
# Update to Gymnasium
pip install "gymnasium>=0.29.0"

# Optional: Remove legacy gym to avoid confusion
pip uninstall gym

# Update plume_nav_sim
pip install "plume_nav_sim>=0.3.0"
```

### Phase 2: Update Imports

**Before:**
```python
import gym
from stable_baselines3 import PPO
```

**After:**
```python
import gymnasium as gym  # Note: Import as 'gym' for minimal changes
from stable_baselines3 import PPO
```

### Phase 3: Update Environment Creation

**Before:**
```python
env = gym.make("OdorPlumeNavigation-v1", video_path="data/plume.mp4")
```

**After:**
```python
env = gym.make("PlumeNavSim-v0", video_path="data/plume.mp4")
```

### Phase 4: Update Reset Logic

**Before:**
```python
def reset_environment(env, seed=None):
    if seed is not None:
        env.seed(seed)
    obs = env.reset()
    return obs
```

**After:**
```python
def reset_environment(env, seed=None):
    obs, info = env.reset(seed=seed)
    return obs  # Return only obs for backward compatibility
    # Or return obs, info for full modern API
```

### Phase 5: Update Step Logic

**Before:**
```python
def run_episode(env, policy):
    obs = env.reset()
    total_reward = 0
    
    while True:
        action = policy.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    return total_reward
```

**After:**
```python
def run_episode(env, policy, seed=None):
    obs, info = env.reset(seed=seed)
    total_reward = 0
    
    while True:
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return total_reward
```

### Phase 6: Update Training Loops

**Before:**
```python
for episode in range(num_episodes):
    obs = env.reset()
    episode_reward = 0
    
    while True:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        
        agent.learn(obs, action, reward, next_obs, done)
        
        obs = next_obs
        episode_reward += reward
        
        if done:
            break
```

**After:**
```python
for episode in range(num_episodes):
    obs, info = env.reset(seed=episode)  # Optional: seed for reproducibility
    episode_reward = 0
    
    while True:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Pass both flags to agent (or combine as needed)
        done = terminated or truncated
        agent.learn(obs, action, reward, next_obs, done)
        
        obs = next_obs
        episode_reward += reward
        
        if terminated or truncated:
            break
```

---

## Compatibility Shim Usage

The plume_nav_sim library provides a compatibility shim for seamless migration without code changes.

### Automatic API Detection

The shim automatically detects your usage pattern:

```python
# Detected as legacy usage (emits deprecation warning)
import gym
env = gym.make("OdorPlumeNavigation-v1")

# Detected as modern usage (no warning)
import gymnasium
env = gymnasium.make("PlumeNavSim-v0")
```

### Manual Compatibility Mode

For explicit control over API mode:

```python
from plume_nav_sim.shims import gym_make

# Force legacy mode (4-tuple returns)
env = gym_make("PlumeNavSim-v0", 
               video_path="data/plume.mp4",
               _force_legacy_api=True)

# Explicit modern mode
from plume_nav_sim.environments import GymnasiumEnv
env = GymnasiumEnv(video_path="data/plume.mp4")
```

### Working with Existing Codebases

For large codebases where gradual migration is preferred:

```python
# Step 1: Use shim to maintain existing behavior
from plume_nav_sim.shims import gym_make

def create_env():
    # This returns legacy 4-tuple format automatically
    return gym_make("PlumeNavSim-v0", video_path="data/plume.mp4")

# Step 2: Gradually migrate individual functions
def modern_create_env():
    import gymnasium as gym
    env = gym.make("PlumeNavSim-v0", video_path="data/plume.mp4")
    return env  # Returns modern 5-tuple format
```

---

## Environment Registration

### New Environment IDs

| Purpose | Legacy ID | Modern ID | Status |
|---------|-----------|-----------|---------|
| Primary | `OdorPlumeNavigation-v1` | `PlumeNavSim-v0` | ✅ Recommended |
| Legacy Support | `OdorPlumeNavigation-v0` | `PlumeNavSim-v0` | ⚠️ Deprecated |

### Registration Details

**Modern Gymnasium Registration:**
```python
import gymnasium as gym
from plume_nav_sim.environments import register_environments

# Register all environments
register_environments()

# Available environments
envs = gym.envs.registry.env_specs
print("PlumeNavSim-v0" in envs)  # True
```

**Custom Environment Configuration:**
```python
import gymnasium as gym

# With custom configuration
env = gym.make(
    "PlumeNavSim-v0",
    video_path="data/custom_plume.mp4",
    max_episode_steps=500,
    initial_position=(320, 240),
    max_speed=2.0,
    include_multi_sensor=True,
    num_sensors=3
)
```

---

## Code Examples

### Example 1: Basic Environment Usage

```python
"""Basic environment usage with modern Gymnasium API."""
import gymnasium as gym
import numpy as np

# Create environment
env = gym.make("PlumeNavSim-v0", 
               video_path="data/plume_movie.mp4",
               max_episode_steps=1000)

# Reset with seed for reproducibility
obs, info = env.reset(seed=42)
print(f"Initial observation keys: {list(obs.keys())}")
print(f"Initial info: {info}")

# Run a few steps
for step in range(5):
    # Sample random action
    action = env.action_space.sample()
    
    # Execute step
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step + 1}:")
    print(f"  Reward: {reward:.3f}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")
    print(f"  Agent position: {obs['agent_position']}")
    print(f"  Odor concentration: {obs['odor_concentration']:.3f}")
    
    if terminated or truncated:
        print(f"Episode ended after {step + 1} steps")
        obs, info = env.reset()
        break

env.close()
```

### Example 2: Training with Stable-Baselines3

```python
"""Training example with stable-baselines3 and Gymnasium."""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Create vectorized environments
def make_env():
    return gym.make("PlumeNavSim-v0", 
                   video_path="data/plume_movie.mp4",
                   max_episode_steps=500)

# Create training and evaluation environments
train_env = make_vec_env(make_env, n_envs=4)
eval_env = make_vec_env(make_env, n_envs=1)

# Create evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/eval",
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Create and train model
model = PPO(
    "MultiInputPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./logs/tensorboard"
)

# Train the model
model.learn(
    total_timesteps=100000,
    callback=eval_callback
)

# Save final model
model.save("ppo_plume_navigation")

# Test trained model
obs = eval_env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    if done.any():
        obs = eval_env.reset()
```

### Example 3: Custom Environment Extensions

```python
"""Example showing extensibility hooks for custom processing."""
import gymnasium as gym
import numpy as np
from plume_nav_sim.environments import GymnasiumEnv

class CustomPlumeEnv(GymnasiumEnv):
    """Extended environment with custom observations and rewards."""
    
    def compute_additional_obs(self, base_obs):
        """Add custom observations."""
        # Add distance to estimated source
        agent_pos = base_obs["agent_position"]
        source_estimate = getattr(self, "_source_estimate", [320, 240])
        distance = np.linalg.norm(agent_pos - source_estimate)
        
        return {
            "distance_to_source": np.array([distance], dtype=np.float32),
            "exploration_progress": np.array([self._get_exploration_progress()], dtype=np.float32)
        }
    
    def compute_extra_reward(self, base_reward, info):
        """Add reward shaping."""
        # Bonus for reducing distance to source
        if hasattr(self, "_previous_distance"):
            current_distance = info.get("distance_to_source", 0)
            distance_change = self._previous_distance - current_distance
            distance_bonus = distance_change * 0.1
            self._previous_distance = current_distance
            return distance_bonus
        else:
            self._previous_distance = info.get("distance_to_source", 0)
            return 0.0
    
    def on_episode_end(self, final_info):
        """Custom episode-end processing."""
        episode_length = final_info.get("step", 0)
        final_reward = final_info.get("total_reward", 0)
        
        print(f"Episode completed:")
        print(f"  Length: {episode_length} steps")
        print(f"  Total reward: {final_reward:.2f}")
        print(f"  Final odor: {final_info.get('odor_concentration', 0):.3f}")
    
    def _get_exploration_progress(self):
        """Calculate exploration progress."""
        return float(np.sum(self._exploration_grid) / self._exploration_grid.size)

# Use custom environment
env = CustomPlumeEnv(video_path="data/plume_movie.mp4")
obs, info = env.reset(seed=42)

# Custom observations are automatically included
print(f"Available observations: {list(obs.keys())}")
```

### Example 4: Performance Monitoring

```python
"""Example showing performance monitoring and optimization."""
import gymnasium as gym
import time
import numpy as np
from plume_nav_sim.utils.frame_cache import FrameCache

# Create frame cache for performance
cache = FrameCache(mode="lru", max_size_mb=512)

# Create environment with cache
env = gym.make("PlumeNavSim-v0",
               video_path="data/plume_movie.mp4",
               frame_cache=cache,
               performance_monitoring=True)

# Performance benchmark
obs, info = env.reset(seed=42)
step_times = []

print("Running performance benchmark...")
for i in range(1000):
    start_time = time.perf_counter()
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    step_time = time.perf_counter() - start_time
    step_times.append(step_time * 1000)  # Convert to milliseconds
    
    if terminated or truncated:
        obs, info = env.reset()
    
    # Print progress every 100 steps
    if (i + 1) % 100 == 0:
        avg_time = np.mean(step_times[-100:])
        cache_stats = env.get_cache_stats()
        print(f"Steps {i+1-99}-{i+1}: {avg_time:.2f}ms avg, "
              f"cache hit rate: {cache_stats['hit_rate']:.1%}")

# Final performance summary
avg_step_time = np.mean(step_times)
p95_step_time = np.percentile(step_times, 95)
cache_stats = env.get_cache_stats()

print(f"\nPerformance Summary:")
print(f"  Average step time: {avg_step_time:.2f}ms")
print(f"  95th percentile: {p95_step_time:.2f}ms")
print(f"  Target compliance: {avg_step_time <= 10}")
print(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")
print(f"  Memory usage: {cache_stats['memory_usage_mb']:.1f}MB")

env.close()
```

---

## Performance Considerations

### Frame Caching Optimization

The new version includes enhanced frame caching for optimal performance:

```python
from plume_nav_sim.utils.frame_cache import FrameCache

# Configure cache based on available memory
cache_configs = {
    "high_memory": FrameCache(mode="all", max_size_mb=2048),     # Preload all frames
    "balanced": FrameCache(mode="lru", max_size_mb=1024),       # LRU cache
    "low_memory": None                                          # No cache
}

# Choose appropriate config
cache = cache_configs["balanced"]
env = gym.make("PlumeNavSim-v0", 
               video_path="data/plume_movie.mp4",
               frame_cache=cache)
```

### Performance Targets

| Metric | Target | Notes |
|--------|--------|--------|
| Step time | <10ms average | Critical for real-time training |
| Cache hit rate | >90% | With LRU cache enabled |
| Memory usage | <2GB per process | With frame cache limits |
| Reset time | <20ms | Including cache warmup |

### Monitoring Performance

```python
# Check performance metrics
performance_info = info.get("perf_stats", {})
if performance_info:
    print(f"Step time: {performance_info['step_time_ms']:.2f}ms")
    print(f"Cache hit rate: {performance_info['cache_hit_rate']:.1%}")
    
    # Warn if performance targets not met
    if performance_info['step_time_ms'] > 10:
        print("⚠️ Step time exceeds 10ms target")
```

---

## Troubleshooting

### Common Migration Issues

#### 1. Import Errors

**Problem:**
```python
ImportError: No module named 'gym'
```

**Solution:**
```python
# Install gymnasium instead of legacy gym
pip install "gymnasium>=0.29.0"

# Update imports
import gymnasium as gym  # Instead of: import gym
```

#### 2. Tuple Length Mismatch

**Problem:**
```python
ValueError: too many values to unpack (expected 4)
obs, reward, done, info = env.step(action)  # Fails with 5-tuple
```

**Solution:**
```python
# Option 1: Update to modern API
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated

# Option 2: Use compatibility shim
from plume_nav_sim.shims import gym_make
env = gym_make("PlumeNavSim-v0")  # Returns 4-tuple automatically
```

#### 3. Environment Not Found

**Problem:**
```python
gym.error.UnregisteredEnv: No registered env with id: PlumeNavSim-v0
```

**Solution:**
```python
# Ensure environment is registered
from plume_nav_sim.environments import register_environments
register_environments()

# Then create environment
env = gym.make("PlumeNavSim-v0")
```

#### 4. Reset Return Format

**Problem:**
```python
TypeError: 'tuple' object has no attribute 'shape'
obs = env.reset()  # Returns (obs, info) in modern API
```

**Solution:**
```python
# Option 1: Update to modern format
obs, info = env.reset(seed=42)

# Option 2: Extract observation only
reset_result = env.reset()
if isinstance(reset_result, tuple):
    obs, info = reset_result
else:
    obs = reset_result
```

#### 5. Seed Handling

**Problem:**
```python
AttributeError: 'GymnasiumEnv' object has no attribute 'seed'
env.seed(42)  # Legacy seed method not available
```

**Solution:**
```python
# Use seed parameter in reset
obs, info = env.reset(seed=42)

# Or use action_space/observation_space seeding
env.action_space.seed(42)
env.observation_space.seed(42)
```

### Performance Issues

#### 1. Slow Step Times

**Symptoms:**
- Step times >10ms consistently
- Training slower than expected

**Solutions:**
```python
# Enable frame caching
from plume_nav_sim.utils.frame_cache import FrameCache
cache = FrameCache(mode="lru", max_size_mb=1024)
env = gym.make("PlumeNavSim-v0", frame_cache=cache)

# Disable performance monitoring in production
env = gym.make("PlumeNavSim-v0", performance_monitoring=False)

# Use vectorized environments
from stable_baselines3.common.env_util import make_vec_env
env = make_vec_env(lambda: gym.make("PlumeNavSim-v0"), n_envs=4)
```

#### 2. Memory Issues

**Symptoms:**
- Out of memory errors
- Gradually increasing memory usage

**Solutions:**
```python
# Use memory-limited cache
cache = FrameCache(mode="lru", max_size_mb=512)  # Smaller cache

# Or disable caching entirely
env = gym.make("PlumeNavSim-v0", frame_cache=None)

# Clear cache periodically
env.clear_cache()
```

### Compatibility Issues

#### 1. Mixed API Usage

**Problem:**
```python
# Code mixes legacy and modern patterns
obs = env.reset()  # Legacy
obs, reward, terminated, truncated, info = env.step(action)  # Modern
```

**Solution:**
```python
# Be consistent with API usage
obs, info = env.reset(seed=42)  # Modern
obs, reward, terminated, truncated, info = env.step(action)  # Modern

# Or use compatibility mode throughout
from plume_nav_sim.shims import gym_make
env = gym_make("PlumeNavSim-v0")
obs = env.reset()  # Legacy
obs, reward, done, info = env.step(action)  # Legacy
```

#### 2. Framework Compatibility

**Problem:**
```python
# Framework expects legacy format
some_rl_library.train(env)  # Expects 4-tuple
```

**Solution:**
```python
# Use compatibility wrapper
from plume_nav_sim.environments.compat import wrap_environment
wrapped_env = wrap_environment(env)
some_rl_library.train(wrapped_env)

# Or check framework documentation for Gymnasium support
```

### Debugging Tools

#### 1. API Compatibility Check

```python
from plume_nav_sim.environments.compat import validate_compatibility

# Check environment compatibility
results = validate_compatibility(env, test_episodes=3)
print(f"Status: {results['overall_status']}")
print(f"Recommendations: {results['recommendations']}")
```

#### 2. Performance Profiling

```python
# Enable detailed logging
import logging
logging.getLogger("plume_nav_sim").setLevel(logging.DEBUG)

# Use performance monitoring
env = gym.make("PlumeNavSim-v0", performance_monitoring=True)
obs, info = env.reset()

# Check step info for timing details
obs, reward, terminated, truncated, info = env.step(action)
perf_stats = info.get("perf_stats", {})
print(f"Step timing: {perf_stats}")
```

#### 3. Environment Diagnostics

```python
from plume_nav_sim.environments import diagnose_environment_setup

# Get system diagnostic information
diagnostics = diagnose_environment_setup()
print(f"System status: {diagnostics}")
```

---

## Deprecation Timeline

### Current Status (v0.3.0)

✅ **Fully Supported:**
- Modern Gymnasium 0.29.x API
- New `PlumeNavSim-v0` environment ID
- Enhanced performance with frame caching
- Extensibility hooks for custom processing

⚠️ **Deprecated but Functional:**
- Legacy Gym 0.26 API (emits warnings)
- Legacy environment IDs (`OdorPlumeNavigation-v*`)
- 4-tuple step returns (legacy mode)
- Separate seed() method calls

### Timeline

| Version | Status | Changes |
|---------|--------|---------|
| **v0.3.0** (Current) | ✅ Full compatibility | - Gymnasium support added<br>- Compatibility shim introduced<br>- Deprecation warnings for legacy usage |
| **v0.4.0** (Q2 2024) | ⚠️ Legacy warnings increased | - More prominent deprecation warnings<br>- Performance optimizations for modern API<br>- Enhanced migration tools |
| **v0.5.0** (Q3 2024) | ⚠️ Legacy support optional | - Legacy support requires explicit flag<br>- Default to modern API only<br>- Comprehensive migration documentation |
| **v1.0.0** (Q4 2024) | ❌ Legacy support removed | - Modern Gymnasium API only<br>- Legacy compatibility shim removed<br>- Performance optimized for modern usage |

### Migration Urgency

| Usage Pattern | Urgency | Action Required |
|---------------|---------|-----------------|
| New projects | **Low** | Use modern API from start |
| Active development | **Medium** | Migrate during next sprint |
| Production systems | **Medium** | Plan migration by v0.5.0 |
| Legacy codebases | **High** | Begin migration immediately |

### Preparing for v1.0.0

To ensure smooth transition to v1.0.0:

1. **Update dependencies:** Install Gymnasium ≥0.29.0
2. **Test compatibility:** Run validation tools on your codebase
3. **Migrate gradually:** Use compatibility shim during transition
4. **Update CI/CD:** Test with both legacy and modern APIs
5. **Train team:** Ensure developers understand new API patterns

### Getting Help

- **Documentation:** [Gymnasium Documentation](https://gymnasium.farama.org/)
- **Migration Tools:** Built-in compatibility validation and diagnostics
- **Community Support:** GitHub issues and discussions
- **Professional Support:** Available through support channels

---

## Summary

The migration from OpenAI Gym 0.26 to Gymnasium 0.29.x in plume_nav_sim v0.3.0 provides significant benefits in terms of performance, maintainability, and future compatibility. The built-in compatibility shim ensures that existing code continues to work without modification, while new projects can take advantage of the modern API's enhanced capabilities.

Key takeaways:

- **Existing code works unchanged** with automatic compatibility detection
- **New projects should use modern Gymnasium API** for best performance
- **Migration can be gradual** using the compatibility shim
- **Performance improvements** are significant with frame caching
- **Extensibility hooks** enable advanced customization without core changes

The deprecation timeline provides ample time for migration, with full legacy support maintained through v0.5.0 and complete removal planned for v1.0.0 in Q4 2024.

For the most up-to-date information and additional resources, consult the project documentation and community resources.