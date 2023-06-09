# LunarLanderProject
This repository contains code and submission documents for the final project of ASEN 5264: Decision Making Under Uncertainty. This project aims to train the lunar lander in the OpenAI Gym Lunar Lander environment to land between two flags using reinforcement learning methods.

- [Gymnasium GitHub Repo](https://github.com/Farama-Foundation/Gymnasium): Contains instructions for installation and example API usage
- [Lunar Lander Environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/): Contains details of the lunar lander environment
- [TensorFlow Installation](https://www.tensorflow.org/install/pip#windows-wsl2): Contains instructions for installation and example API usage for WSL2
- [TensorBoard](https://www.tensorflow.org/tensorboard/get_started): Just some info

## Setup instructions
1. Create `lunar_lander` virtual environment: `$ python3.9 -m venv lunar_lander`
2. Activate the `lunar_lander` virtual environment: `$ source lunar_lander/bin/activate`
3. Install all required libraries: `$ pip install -r requirements.txt`
4. Check that things are working correctly
    1. `$ cd src`
    2. `$ python main.py`
5. Your environment is set up correctly if you see the Lunar Lander trying to land

## Workflow
Perform the following steps every time you start a new terminal session:
1. Deactivate whatever conda environment you might be in: `$ conda deactivate`
2. Activate the conda environment: `$ conda activate tf`
3. Activate the `lunar_lander` virtual environment: `$ source lunar_lander/bin/activate`
