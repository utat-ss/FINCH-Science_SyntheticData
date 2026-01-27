# FINCH-Science_SyntheticData
Code to synthetically generate new data, for the use of further training of hyperspectral unmixing algorithms.

## Description
This repository is used for the development of synthetic data to be used by Science. Science needs to generate synthetic data (to be used in atmospheric modelling) in order to create data interpolations before training the unmixing algorithms.

## Synthesizer Models 
To perform this, there are a couple of models that have been explored so far. These are as follows:
- AutoEncoder-based:
    - Conditional AutoEncoders:
        - MLP based
        - CNN based
    - Conditional Variational AutoEncoders:
        - MLP based
        - CNN based
- Gaussian Diffusion-based: 
    - Denoising Diffusion Probabilistic Model:
        - MLP based epsilon network
        - Conditional Conformer based U-Net epsilon network
    - Denoising Diffusion Implicit Model:
        - Conditional Conformer based U-Net epsilon network

## Test Metric
We cannot use widely available testing algorithms and methods that are widely available in generative AI community, the reason being that our generated data are conditioned on a continous space and the data are inherintly 1D. Therefore, we have two indigenously developed testing metrics.
- Unmix/critic:
    - Trains an unmixing algorithm on data synthesized by a synthesizer model
    - Directly measures the practical effect of synthesizing data
- Near neighbor:
    - Calculates the first nearest neighbor of a dataset mixed in synthetic and real data
    - Directly measures the distributions learned by the synthesizer model

## Installation & Usage
### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/utat-space/FINCH-Science_SyntheticData.git](https://github.com/utat-space/FINCH-Science_SyntheticData.git)
   cd FINCH-Science_SyntheticData
2. Create a virtual environment (recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
3. Install requirements:
    ```bash
    pip install -r requirements.txt
### Usage
1. Open your terminal
2. Move to the folder you have this repo in
3. Activate the virtual environment you have created for this repo
4. For the process that you want to run (one of training synthesizer, synthesizing data, testing synthesized data), find the relevant script for the process:
    - Training -> defs/[model_type]/script.py
    - Synthesis -> defs/synthesis/script.py
    - Testing -> defs/testing/script.py
5. Take the example configuration file (in the folder that the script you want to run is in), and modify the entries as you want to, put the modified configuration file in one of the folders:
    - Training -> training/[new_folder]/
    - Synthesis -> synthesis/[new_folder]/
    - Testing -> testing/[new_folder]/
6. Login to wandb by doing:
    ```bash
    wandb login <your_api_key>
7. Run the script using the config:
    ```bash
    python -m <relative_path_to_script> <relative_path_run_config>


## Citation
If you use this code or dataset in your research or works, please cite our upcoming ISPRS 2026 paper:

## License
The code is licensed under MIT License.

## Authors
- Synthesizer models:
    - AutoEncoder-based: Shuo Chen, Kyaw Thiha
    - Gaussian Diffusion: Ege Artan, Andrew Peng
- Synthesis pipeline: Ege Artan
- Testing pipeline:
    - Critic models: Ege Artan, Sammuel Aldrich Karya
    - Nearest neighbor: Ege Artan

## Contact & Support
### Questions?
For questions regarding the codebase, the paper, or reproduction of results, please reach out to **Ege Artan**, the Science Lead of FINCH at the time of development for this code:
* **LinkedIn:** [Ege Artan](https://www.linkedin.com/in/ege-artan)
* **GitHub:** [@ege-artan](https://github.com/ege-artan)

### Found a Bug?
If you encounter any issues with the code or have feature requests, please check if the issue has already been reported. If not, feel free to open a new issue on our repository:
* [Report a Bug or Issue](https://github.com/utat-space/FINCH-Science_SyntheticData/issues)

## Project Structure
```text
.
├── data                                   
├── defs/
│   ├── diffusion/
│   │   ├── data/
│   │   │   ├── data_augmentation.py
│   │   │   └── data_preperation.py
│   │   ├── epsilon/
│   │   │   ├── auxiliary/
│   │   │   ├── mlp.py
│   │   │   └── unet.py
│   │   ├── noise/
│   │   │   ├── noise_sampling.py
│   │   │   └── noise_scheduling.py
│   │   ├── auxiliary.py
│   │   ├── cfg_run_example.yaml
│   │   ├── diffusion.py
│   │   ├── example_cmd.txt
│   │   ├── loaders.py
│   │   ├── loss.py
│   │   ├── plotting.py
│   │   ├── script.py
│   │   └── train.py
│   ├── synthesis/
│   │   ├── lean/
│   │   │   └── lean_synth.py
│   │   ├── abundance_sampler.py
│   │   ├── cfg_run_example.py
│   │   ├── data.py
│   │   ├── loaders.py
│   │   ├── script.py
│   │   ├── synth_func.py
│   │   └── synthesize.py
│   ├── testing/
│   │   ├── master/
│   │   │   ├── example_cfg.yaml
│   │   │   └── script.py
│   │   ├── nearest_neighbor/
│   │   │   ├── auxiliary/
│   │   │   ├── data/
│   │   │   ├── near_neigh/
│   │   │   └── script.py
│   │   └── unmix/
│   │       ├── auxiliary/
│   │       ├── critic_train/
│   │       ├── critics/
│   │       ├── data/
│   │       ├── loaders/
│   │       └── script.py
│   ├── tools/
│   │   ├── count_params.py
│   │   └── search_configs.py
│   └── vae/
│       ├── data
│       ├── noise
│       ├── tests
│       ├── ae.py
│       ├── auixiliary.py
│       ├── cfg_run_example.yaml
│       ├── loaders.py
│       ├── loss.py
│       ├── plotting.py
│       ├── script.py
│       ├── train.py
│       └── training_defs.py
├── synthesis
├── testing
├── training
├── .gitignore
├── LICENSE
└── README.md