OpenAI Gym
**********

This is a custom version of the original gym which contains support for 
goal oriented Atari games.

Currently the MsPacman is supported. The name of the environment: 

- GoalbasedMsPacman-v0
- GoalbasedMsPacman-ram-v0.

Basic usage:

.. code:: shell

    import gym
    env = make('GoalbasedMsPacman-v0')
    # (start_x, start_y, goal_x, goal_y)
    env.setup((65, 98, 76, 98))

Installation
============

After creating and activating a virtual environment:

.. code:: shell

    git clone https://github.com/adamtiger/gym.git
    cd gym
    git checkout openai-goal-based-atari
    pip install -e .

Original source
===============

`documentation <https://gym.openai.com/docs>`_
