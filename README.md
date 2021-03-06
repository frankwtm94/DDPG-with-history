# DDPG with history

A deep deterministic policy gradient algorithm based on gym inverted pendulum environment, unknown external disturbance is included, and a certain period of history observations and actions are taken into consideration in order for disturbance rejection.

## Insruction

Before running the code, you need to register the newly built gym environment:

#### Step 1:
copy pendulum_disturbance.py to gym/gym/envs/clasic_control

#### Step 2:
open \_\_init\_\_.py in this folder, add a line:

    from gym.envs.classic_control.pendulum_disturbance import PendulumDisturbanceEnv

#### Step 3:
open \_\_init\_\_.py in gym/gym/envs, add:

    register(
        id='PendulumDisturbance-v0',
        entry_point='gym.envs.classic_control:PendulumDisturbanceEnv',
        max_episode_steps=200,
    )

#### Step 4:
rebuild gym: 

    cd gym
    pip install -e .
