# Project 3: Collaboration and Competition

### Project Details

For this project, I worked with the the a custom Unity-based [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis environment. Specifically, I trained an agent that controls both rackets and self-plays to bounce a ball back-and-forth over a net.

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


* A reward of +0.1 is given to an agent when it hits the ball over the net, and a reward of -0.01 is given to an agent when it lets a ball hit the ground or hits the ball out of bounds.
* It appears that the episode ends when the ball lands on the ground, or is hit out of bounds (although I can't be sure since the animations play extremely quickly).
* The goal of each agent is to keep the ball in play.
* The observation space contains 8 variables corresponding to the position and velocity of the ball and racket, and is stacked 3 times, resulting to a 24-dimensional state representation. Each agent receives its own, local observation.
* The agent-specific action space is continuous, 2-dimensional, bounded within -1 and +1, and correspond to movement toward (or away from) the net, and jumping.
* The task is considered solved when the score (a.k.a. the cumulative reward of the winning agent) in each episode is over +0.5, when averaged over 100 consecutive episodes (during training).

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

3. Follow [these instructions](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install conda Python 3.6 env and create `drlnd` venv.

4. Navigate terminal to (unzipped) project directory, activate `drlnd` conda venv, and run `pip install --user -r requirements.txt` to install Python dependencies. If it complaints about incompatible dependencies, you can ignore them, as long as the packages have been installed.

    For Windows installations, you may need to do these additional workarounds:

    * before `pip install .`, edit `deep-reinforcement-learning/python/requirements.txt`, and comment out the line `#torch==0.4.0`
    * install a version of numpy that is compatible with Windows 10: `pip install --user numpy==1.19.3`
    * install an up-to-date version of pytorch: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`

### Instructions

Run all cells in `Tennis.ipynb` sequentially to both train and evaluate my Soft Actor-Critic (SAC) agent.

# Report

See [Report.md](Report.md).