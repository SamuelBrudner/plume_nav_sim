"""Quickstart example for `plume_nav_sim.make_env()`.

Produces a visible artifact by default:
- Tries to write a GIF to ``quickstart.gif`` (requires optional ``media`` extras)
- Falls back to a PNG ``quickstart.png`` if GIF dependencies are unavailable
"""

from pathlib import Path

import plume_nav_sim as pns


def main() -> None:
    """Run a short episode and save a visible artifact."""

    # Configure env to return RGB frames for saving
    env = pns.make_env(render_mode="rgb_array")

    try:
        obs, info = env.reset(seed=42)
        print(f"Starting position: {info['agent_xy']}")

        # Simple random policy (callable is supported by the runner)
        def random_policy(_obs):  # noqa: ANN001
            return env.action_space.sample()

        # Stream one episode with frames attached
        from plume_nav_sim.runner import runner as r

        events = list(r.stream(env, random_policy, seed=42, render=True))

        # Attempt GIF first (requires imageio via optional [media] extras)
        try:
            from plume_nav_sim.utils.video import save_video_events

            out_gif = Path("quickstart.gif")
            save_video_events(events, out_gif, fps=12)
            print(f"Wrote GIF: {out_gif.resolve()}")
        except ImportError:
            # Fallback: save the last frame as a PNG via matplotlib (base dep)
            png_path = Path("quickstart.png")
            last = next(
                (ev.frame for ev in reversed(events) if ev.frame is not None), None
            )
            if last is not None:
                import matplotlib.pyplot as plt

                plt.imsave(png_path, last)
                print(f"Wrote PNG (media extras not installed): {png_path.resolve()}")
            else:
                print("No frames available to save a fallback image.")
    finally:
        env.close()


if __name__ == "__main__":  # pragma: no cover
    main()
