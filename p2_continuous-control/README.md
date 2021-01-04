# Project 2: Continuous Control

### Project Details

For this project, I worked with the the Unity [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. Specifically, I trained an agent to steer a double-jointed arm to regulate its end-effector to be within a target region that moves over time.

* A reward of +0.1 is provided for each step that the agent's hand is in the target region.
* The goal is to maintain the arm's end-effector position within the target region for as many time steps as possible, within 1001 fixed timesteps.
* The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.
* The action space is continuous, 4-dimensional, bounded within -1 and +1, and correspond to directional torque components applied to the two arm joints.
* The task is considered solved when the agent gets an average score of +30 over 100 consecutive episodes (during training).

Note that I worked with the first version of the Unity environment, contianing a single agent.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file.

3. Follow [these instructions](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install conda Python 3.6 env and create `drlnd` venv.

4. Navigate terminal to (unzipped) project directory, activate `drlnd` conda venv, and run `pip install --user -r requirements.txt` to install Python dependencies. If it complaints about incompatible dependencies, you can ignore them, as long as the packages have been installed.

    For Windows installations, you may need to do these additional workarounds:

    * before `pip install .`, edit `deep-reinforcement-learning/python/requirements.txt`, and comment out the line `#torch==0.4.0`
    * install a version of numpy that is compatible with Windows 10: `pip install --user numpy==1.19.3`
    * install an up-to-date version of pytorch: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`

### Instructions

Run all cells in `Continuous_Control.ipynb` sequentially to both train and evaluate my Soft Actor-Critic (SAC) agent.

# Report

See [Report.md](Report.md).