"""Example: Record a plume navigation episode as a GIF.

Works as a standalone script or copy-paste into a Jupyter notebook.
Requires: pip install plume-nav-sim[media]
"""

import plume_nav_sim as pns
from plume_nav_sim.runner import runner
from plume_nav_sim.utils.video import save_video

# Create environment with rgb_array rendering
env = pns.make_env(
    grid_size=(64, 64),
    max_steps=50,
    render_mode="rgb_array",
    action_type="oriented",
    observation_type="concentration",
    reward_type="step_penalty",
)

# Collect frames from a short episode
frames = []
for ev in runner.stream(env, lambda obs: env.action_space.sample(), seed=42, render=True):
    if ev.frame is not None:
        frames.append(ev.frame)
    if ev.terminated or ev.truncated:
        break

# Save as GIF
save_video(frames, "demo.gif", fps=10)
print(f"Saved {len(frames)} frames to demo.gif")

env.close()
