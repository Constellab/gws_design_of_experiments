<p align="center">
  <img src="https://constellab.space/assets/fl-logo/constellab-logo-text-white.svg" alt="Constellab Logo" width="80%">
</p>

<br/>

# 👋 Welcome to GWS DESIGN OF EXPERIMENTS

```gws_design_of_experiments``` is a [Constellab](https://constellab.io) library (called bricks) developped by [Gencovery](https://gencovery.com/). GWS stands for Gencovery Web Services.

## 🚀 What is Constellab?


✨ [Gencovery](https://gencovery.com/) is a software company that offers [Constellab](https://constellab.io)., the leading open and secure digital infrastructure designed to consolidate data and unlock its full potential in the life sciences industry. Gencovery's mission is to provide universal access to data to enhance people's health and well-being.

🌍 With our Fair Open Access offer, you can use Constellab for free. [Sign up here](https://constellab.space/). Find more information about the Open Access offer here (link to be defined).


## ✅ Features

Gencovery brick to build, contextualize and simulate genome-scale metabolic models (GEMs) in Constellab, as a thin wrapper around the [cobrapy](https://github.com/opencobra/cobrapy) toolbox.
- Import a genome-scale metabolic model from a standard COBRA JSON file, or download one directly from the [BiGG Models](http://bigg.ucsd.edu) database
- Contextualize a model with reaction bound constraints (growth medium, gene/reaction knockouts, measured flux ranges) and soft flux targets
- Simulate a model with flux balance analysis (FBA), with an optional relaxed (quadratic) solver for partially curated models where a strict steady-state constraint would make the simulation infeasible
- Reconstruct a draft model from an organism's NCBI taxonomy id, using curated reactions from the `gws_biota` database

## 📄Documentation

📄 For `gws_design_of_experiments` brick documentation, click [here](https://constellab.community/bricks/gws_design_of_experiments/latest/doc/getting-started)

💫 For Constellab application documentation, click [here](https://constellab.community/bricks/gws_academy/latest/doc/getting-started/b38e4929-2e4f-469c-b47b-f9921a3d4c74)

## 🛠️ Installation

The `gws_design_of_experiments` brick requires the `gws_core` brick.

### 🔥 Recommended Method

The best way to install a brick is through the Constellab platform. With our Fair Open Access offer, you get a free cloud data lab where you can install bricks directly. [Sign up here](https://constellab.space/)

Learn about the data lab here : [Overview](https://constellab.community/bricks/gws_academy/latest/doc/digital-lab/overview/294e86b4-ce9a-4c56-b34e-61c9a9a8260d) and [Data lab management](https://constellab.community/bricks/gws_academy/latest/doc/digital-lab/on-cloud-digital-lab-management/4ab03b1f-a96d-4d7a-a733-ad1edf4fb53c)

### 🔧 Manual installation

This section is for users who want to install the brick manually. It can also be used to install the brick manually in the Constellab Codelab.

We recommend installing using Ubuntu 22.04 with python 3.10.

#### Usage


▶️ To start the server :

```bash
gws server run
```

🕵️ To run a given unit test

```bash
gws server test [TEST_FILE_NAME]
```

Replace `[TEST_FILE_NAME]` with the name of the test file (without `.py`) in the tests folder. Execute this command in the folder of the brick.

🕵️ To run the whole test suite, use the following command:

```bash
gws server test all
```

📌 VSCode users can use the predefined run configuration in `.vscode/launch.json`.

## 🤗 Community

🌍 Join the Constellab community [here](https://constellab.community/) to share and explore stories, code snippets and bricks with other users.

🚩 Feel free to open an issue if you have any question or suggestion.

☎️ If you have any questions or suggestions, please feel free to contact us through our website: [Constellab](https://constellab.io/).

## 🌎 License

```gws_design_of_experiments``` is completely free and open-source and licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

<br/>


This brick is maintained with ❤️ by [Gencovery](https://gencovery.com/).

<p align="center">
  <img src="https://framerusercontent.com/images/Z4C5QHyqu5dmwnH32UEV2DoAEEo.png?scale-down-to=512" alt="Gencovery Logo"  width="30%">
</p>