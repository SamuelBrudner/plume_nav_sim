# Semantic Model

This file describes the active concepts used by `plume_nav_sim`. It is a plain-language model of the current system, not a promise that every internal detail is fixed.

## Environment

An environment instance represents one plume-navigation task with a grid, an agent state, a plume field, an observation model, an action model, and a reward function.

The supported lifecycle is:

`created -> reset -> stepping -> terminated/truncated -> reset -> ... -> closed`

Important semantics:

- the environment cannot be stepped before reset
- closing ends the usable lifecycle
- seeding should make repeated runs reproducible for the same configuration

## Agent State

The agent has a position on the grid and accumulates episode progress over time. The exact public info payload depends on the environment implementation, but the model is consistent:

- position is always interpreted in grid coordinates
- step count advances monotonically during an episode
- total reward is the cumulative sum of step rewards

## Plume Sources

Two plume source styles are active:

- static Gaussian plumes
- video-backed plumes loaded from datasets or ingested media

Video plumes carry their own dataset metadata and can override grid size based on the dataset. Runtime behavior depends on the selected plume field rather than on a single global plume model.

The authoritative dataset details for video plumes live in `src/backend/docs/contracts/video_plume_dataset.md`.

## Actions, Observations, and Rewards

These are component-driven, not globally fixed.

Actions:

- discrete grid actions
- oriented actions
- run/tumble actions

Observations:

- concentration
- antennae array
- wind vector
- wrapper-augmented observations in the component/spec path

Rewards:

- sparse goal reward
- step-penalty reward

The semantic rule is that the environment surface is defined by the configured components. Avoid documenting one observation shape or one action space as globally universal unless the code enforces it.

## Episodes and Replay

An episode is one reset-to-termination/truncation run. Replay artifacts persist that run in a schema-validated form:

- run metadata
- per-step records
- per-episode summaries

The replay loader validates schema version, record shape, and run-id consistency before returning parsed artifacts.

## Data Zoo and Movie Metadata

The Data Zoo registry maps dataset ids to download, ingest, cache, and citation metadata. At runtime, dataset entries describe where the plume data comes from and how it should be interpreted.

For raw movie inputs, sidecar metadata is used during ingest to produce dataset attrs consumed at runtime. After ingest, the dataset attrs are the runtime source of truth. Sidecar and dataset behavior should stay aligned, but this document intentionally leaves detailed field-by-field rules to the specialized video plume docs.
