# Candidate Open-Access Plume Datasets for Data Zoo Registry

**Curated:** 2026-02-24
**Purpose:** Identify open-access datasets suitable for plume navigation RL research
**Criteria:** Verified open license, downloadable, spatial concentration data, useful for RL

---

## Already in Registry (for reference)

| ID | Source | Repository | License |
|----|--------|-----------|---------|
| colorado_jet_v1 | Crimaldi PLIF wind tunnel | Zenodo 4971113 / Dryad g27mq71 | CC-BY-4.0 |
| rigolli_dns_nose_v1 | DNS turbulent channel (nose) | Zenodo 15469831 | CC-BY-4.0 |
| rigolli_dns_ground_v1 | DNS turbulent channel (ground) | Zenodo 15469831 | CC-BY-4.0 |
| emonet_smoke_v1 | Smoke plume video | Dryad 4j0zpc87z | CC0-1.0 |
| emonet_smoke_trimmed_v1 | Smoke plume video (trimmed) | Dryad 4j0zpc87z | CC0-1.0 |

---

## TIER 1 -- High Priority Candidates

These datasets provide spatial concentration fields directly usable as RL environments.

### 1. Boie et al. PLIF Odor Plumes (Information-Theoretic Analysis)

- **Title:** Data from: Information-theoretic analysis of realistic odor plumes: what cues are useful for determining location?
- **Authors:** Boie, S.D.; Connor, E.G.; McHugh, M.; Nagel, K.I.; Ermentrout, G.B.; Crimaldi, J.P.; Victor, J.D.
- **Citation:** Boie et al. (2018). PLOS Computational Biology, 14(7), e1006275.
- **Repository:** Dryad https://doi.org/10.5061/dryad.04t19 (also mirrored at Zenodo 5035406)
- **License:** CC0-1.0 (public domain)
- **Format:** HDF5 (.hdf5), 34 files
- **Size:** ~11.8 GB total
- **Description:** Three PLIF experiments from Crimaldi's wind tunnel (same apparatus as colorado_jet_v1) at different flow conditions: fast flow (10 cm/s freestream), slow flow (5 cm/s freestream), and boundary flow (10 cm/s near-bed). Includes time series from multiple spatial sensor configurations (narrow/wide grids, single/binaural sampling) plus snapshot concentration fields.
- **Spatial:** 2D concentration fields at 15 Hz; wind tunnel test section 1m x 0.3m x 0.3m
- **Suitability:** EXCELLENT. Same lab as colorado_jet_v1 but with three different flow regimes. The boundary flow condition is already ingested as colorado_jet_v1. The freestream conditions at two speeds would provide complementary plume environments for RL agent training. Already in HDF5 format -- similar ingest pipeline to existing CrimaldiFluorescenceIngest.
- **Priority:** HIGH -- natural extension of existing colorado_jet_v1

### 2. Rigolli et al. DNS Velocity Fields (Complement to Existing Concentration Data)

- **Title:** Alternation emerges as a multi-modal strategy for turbulent odor navigation - Dataset (velocity components)
- **Authors:** Rigolli, N.; Reddy, G.; Seminara, A.; Vergassola, M.
- **Citation:** Rigolli et al. (2022). eLife, 11, e76989.
- **Repository:** Zenodo https://doi.org/10.5281/zenodo.15469831
- **License:** CC-BY-4.0
- **Format:** MATLAB v7.3 (.mat, HDF5-compatible)
- **Size:** crosswind_v.mat (5.5 GB), downwind_v.mat (3.8 GB), vertical_v.mat (5.4 GB) -- total ~14.7 GB for velocity
- **Description:** The same DNS simulation that already provides nose_data.mat and ground_data.mat also includes 3-component velocity fields. These are 2D slices of the turbulent channel flow velocity, co-registered with the existing concentration data.
- **Spatial:** Same grid as rigolli_dns_nose_v1 / rigolli_dns_ground_v1
- **Suitability:** EXCELLENT. Velocity fields enable wind-aware navigation agents. An RL agent could sense both local concentration AND local wind velocity, closely mimicking insect navigation with mechanosensory + chemosensory inputs. Uses existing RigolliDNSIngest pipeline.
- **Priority:** HIGH -- complement to already-ingested data from same DOI

### 3. PPMLES -- LES Urban Dispersion Ensemble

- **Title:** PPMLES -- Perturbed-Parameter ensemble of MUST Large-Eddy Simulations
- **Authors:** Lumet, E.; Jaravel, T.; Rochoux, M.C.
- **Citation:** Lumet et al. (2025). Data in Brief, 58, 111285.
- **Repository:** Zenodo https://doi.org/10.5281/zenodo.11394347
- **License:** CC-BY-4.0
- **Format:** HDF5 (.h5) + CSV
- **Size:** ~36.5 GB total (ave_fields.h5: 17.1 GB, uncertainty_ave_fields.h5: 15.9 GB, time_series.h5: 3.1 GB, mesh.h5: 387 MB)
- **Description:** 200 large-eddy simulations of pollutant dispersion in an idealized urban environment (Mock Urban Setting Test -- MUST). Includes 3D concentration and velocity fields at sub-metric resolution, time series at 93 probe locations, and uncertainty quantification. Meteorological forcing parameters are varied across the ensemble.
- **Spatial:** 3D fields with sub-metric resolution over urban obstacle array; also time series at 93 point probes
- **Suitability:** VERY GOOD. The 3D concentration fields could be sliced into 2D planes for RL navigation in complex geometry (obstacle avoidance + plume tracking). The 200-sample ensemble provides diverse plume realizations for robust policy training. Requires new ingest pipeline for unstructured LES mesh data.
- **Priority:** HIGH -- novel obstacle-rich environment, large ensemble

---

## TIER 2 -- Good Candidates (Require More Investigation or New Ingest)

### 4. COSMOS Field Trial Plume Data (Desert + Forest)

- **Title:** Cosmos: A data-driven probabilistic time series simulator for chemical plumes across spatial scales
- **Authors:** Nag, A.; van Breugel, F.
- **Citation:** Nag & van Breugel (2025). PLOS Computational Biology (forthcoming).
- **Repository:** Dryad https://doi.org/10.5061/dryad.j3tx95xss
- **License:** CC0-1.0 (public domain)
- **Format:** Pandas HDF5 (.h5) and NumPy (.npz) inside data.zip
- **Size:** 1.22 GB
- **Description:** Real-world odor field measurements from three environments: desert high-wind-speed, desert low-wind-speed, and forest. Collected with mobile PID sensor at 200 Hz including odor concentration, wind velocity, and GPS coordinates. Also includes Rigolli CFD validation data and trained generative models.
- **Spatial:** Point measurements along trajectories (NOT gridded 2D fields). However, the data includes GPS coordinates, wind vectors, and concentration at each point.
- **Suitability:** MODERATE for direct RL use (point measurements, not spatial fields). However, this is valuable as ground-truth for validating plume simulators used in RL training. The COSMOS generative model could produce synthetic spatial fields. Contains rare real-world outdoor plume data.
- **Priority:** MEDIUM -- not spatial fields, but valuable field-trial ground truth

### 5. Kadakia et al. Odor Motion Sensing Plume Data

- **Title:** Data from: Odor motion sensing enhances navigation of complex plumes
- **Authors:** Kadakia, N.; Demir, M.; Michaelis, B.; DeAngelis, B.; Reidenbach, M.; Clark, D.; Emonet, T.
- **Citation:** Kadakia et al. (2022). Nature, 611(7937), 754-761.
- **Repository:** Dryad https://doi.org/10.5061/dryad.1ns1rn8xd
- **License:** CC0-1.0 (public domain)
- **Format:** ZIP archive containing mixed formats
- **Size:** 2.36 GB
- **Description:** Combines simulated plume data with experimental fly tracking data. The study uses spatiotemporally complex odor plumes created by stochastically perturbing an odor ribbon with lateral air jets. Contains both 2D plume simulation outputs and behavioral data.
- **Spatial:** 2D plume concentration fields from simulations; experimental VR plume stimuli
- **Suitability:** GOOD. The simulated 2D plume fields could serve as RL environments. From the same Yale/Emonet group as the existing emonet_smoke entries. Would need examination of exact data structure inside the ZIP.
- **Priority:** MEDIUM -- needs further investigation of internal data structure

### 6. Rigolli/Seminara DNS Features Dataset (OSF)

- **Title:** Learning to predict target location with turbulent odor plumes -- processed features
- **Authors:** Rigolli, N.; Magnoli, N.; Rosasco, L.; Seminara, A.
- **Citation:** Rigolli et al. (2022). eLife, 11, e72196.
- **Repository:** OSF https://osf.io/ja9xr/
- **License:** Likely CC-BY (OSF default; needs verification)
- **Format:** Unknown (OSF repository -- could not be fetched directly)
- **Size:** Unknown
- **Description:** Processed time series and computed features (mean concentration, slope, blank duration, whiff duration, intermittency factor) extracted from DNS of turbulent channel flow with odor source. DNS performed with Nek5000. Domain: L=40, W=8, H=4 (non-dimensional); grid cell size ~0.6 cm in air.
- **Spatial:** Derived from 3D DNS; features extracted at various locations
- **Suitability:** MODERATE. Processed features rather than raw concentration fields. May be more useful for supervised learning benchmarks than as RL environments. However, if raw DNS snapshots are included, those would be very valuable.
- **Priority:** MEDIUM -- needs verification of what files are actually available on OSF

---

## TIER 3 -- Lower Priority / Niche Use Cases

### 7. Wind Tunnel Concentration in Urban Canyons with Trees

- **Title:** Wind tunnel measurements of concentration and velocity in urban geometries with trees
- **Authors:** Fellini, S.; Del Ponte, A.V.; Marro, M.; Salizzoni, P.; Ridolfi, L.
- **Citation:** Del Ponte et al. (2024). Boundary-Layer Meteorology, 190(2), 6.
- **Repository:** Zenodo https://doi.org/10.5281/zenodo.15633150
- **License:** CC-BY (assumed from Zenodo default; needs verification)
- **Format:** ZIP archive
- **Size:** 12.9 MB (very small)
- **Description:** Concentration and velocity measurements in a wind tunnel simulating an urban street canyon with varying tree configurations. Three tree spacing variants tested under different wind directions.
- **Spatial:** Point measurements at discrete probe locations within street canyon (NOT full 2D/3D fields)
- **Suitability:** LOW for RL. Point measurements rather than spatial fields. Small dataset. However, the urban canyon geometry is interesting for navigation in cluttered environments.
- **Priority:** LOW

### 8. Passive Gas Plume Database for Metrics Comparison

- **Title:** Passive gas plume database for metrics comparison
- **Authors:** Vanderbecken, P.J.
- **Citation:** Vanderbecken et al. (2022). Atmospheric Measurement Techniques (preprint).
- **Repository:** Zenodo https://doi.org/10.5281/zenodo.6958047
- **License:** CC-BY-4.0
- **Format:** NetCDF (.nc)
- **Size:** 192.1 MB (Analytical: 164.7 MB, Synthetic: 27.4 MB)
- **Description:** Synthetic CO2 plumes from chemical transport model simulations plus analytical Gaussian plume references. Designed for evaluating plume comparison metrics.
- **Spatial:** 2D plume images in NetCDF
- **Suitability:** LOW-MODERATE. The 2D plume images could potentially serve as simple RL environments, but these are atmospheric-scale (km) CO2 plumes, not laboratory-scale odor plumes. The Gaussian plume model may be too smooth for training robust RL agents.
- **Priority:** LOW -- atmospheric scale, smooth plumes

### 9. Fonollosa/Vergara Gas Sensor Array in Turbulent Wind Tunnel

- **Title:** Gas sensor array exposed to turbulent gas mixtures
- **Authors:** Fonollosa, J.; Rodriguez-Lujan, I.; Trincavelli, M.; Huerta, R.
- **Citation:** Fonollosa et al. (2014). UCI ML Repository, DOI: 10.24432/C5JS5P
- **Repository:** UCI ML Repository https://archive.ics.uci.edu/dataset/309
- **License:** CC-BY-4.0
- **Format:** 180 text files (CSV-like) with time series
- **Size:** ~180 files, 300s each at 20ms sampling
- **Description:** 8 MOX gas sensors in a 2.5m x 1.2m x 0.4m wind tunnel with two independent gas sources generating turbulent binary mixtures of ethylene with methane or CO. Each measurement is 300s with 60s baseline, 180s gas release, 60s recovery.
- **Spatial:** Single point (sensor array at one location). NOT spatial concentration fields.
- **Suitability:** LOW for RL navigation (point sensor, no spatial fields). However, useful for gas identification/classification tasks that could complement navigation.
- **Priority:** LOW -- no spatial data

---

## REJECTED / NOT SUITABLE

| Dataset | Reason for Rejection |
|---------|---------------------|
| ROMEO plume campaign (Zenodo 6553092) | Field transect measurements in Excel, no spatial concentration fields |
| Plume spreading test case (Zenodo 4389353) | Ocean/estuarine plume spreading, wrong physical regime |
| JHTDB Channel Flow datasets | No passive scalar/concentration field included in available datasets |
| Alvarez-Salvado behavioral data (Dryad g27mq71) | Already ingested as colorado_jet_v1 (same underlying PLIF file) |
| CMIP7 Simple Plumes (Zenodo 15283189) | Anthropogenic aerosol climate forcing, wrong domain |
| Methane plumes from airborne surveys (Zenodo 5606120) | Remote sensing imagery, not suitable for agent-scale navigation |
| STARCOP methane segmentation (Zenodo 7863343) | Satellite hyperspectral, wrong spatial scale |

---

## Summary and Recommendations

### Immediate Additions (can reuse existing ingest pipelines)

1. **Boie et al. PLIF** -- Extends colorado_jet_v1 with two additional flow conditions from the same Crimaldi wind tunnel. CrimaldiFluorescenceIngest pipeline can be adapted.
2. **Rigolli velocity fields** -- Complements existing rigolli_dns entries with co-registered wind velocity data. RigolliDNSIngest pipeline applies directly.

### Next Wave (require new ingest pipelines)

3. **PPMLES** -- Richest new dataset candidate. 200 LES realizations with 3D concentration + velocity in obstacle-rich urban geometry. Needs HDF5-based LES mesh ingest.
4. **COSMOS field data** -- Real-world outdoor plume measurements. Not gridded, but valuable for validation.
5. **Kadakia plume data** -- Needs investigation of internal data structure.

### Key Gaps Not Filled

- **Field-trial spatial concentration fields** (Prairie Grass, MUST raw data, Moffett Field) -- these classic datasets appear not to be available as open-access spatial concentration data. The original MUST data is not openly downloadable; PPMLES reproduces it via simulation.
- **Aquatic plume data** -- No suitable open-access datasets found for underwater plume navigation.
- **3D volumetric plume data** -- The Rigolli DNS is 3D but only 2D slices are provided. Full 3D concentration fields for RL are rare in open repositories.
