# -*- coding: utf-8 -*-
"""

Quick test script for showing a dummy case for animal AI

Early priorities:
    -load in custom environment
    -get gt map of environment
    -get visual input out
    -get depth camera or get something from raycast
later priorities:
    -create random custom environment using generate configs
    -get agent to run around according to deterministic policy while collecting inputs
    to the other models.
    -tile place cells and get place cell activations for a trajectory following model
    from Sorcher (Ganguli) paper.


@author: ronwd
"""

#%% Imports

import sys
import random
import os
from matplotlib import pyplot as plt

from animalai.envs.environment import AnimalAIEnvironment

import generate_configs
import numpy as np

#%% Constant defaults

#need to determine spatial resolution to use
spatial_res = .01
xinds = np.arange(0,40,spatial_res)
yinds = np.arange(0,40,spatial_res)
arena_template = np.zeros([np.size(xinds),np.size(yinds)]) #arena is 40x40 with 0,0 as origin (left corner convention)

#unity uses x z y convention where middle term sets the heigh location of the object
#(i.e. what unity calls y is what we would call z)



#%% Create arena (config file) using functions from generate config
#update think will have to use config writer so have positions and sizes of objects stored
#otherwise need to figure out a way to read them out of the yaml file
#then can push them into template as ones where objects exist
#note: it looks like position is the center of object and size scales out proportionally

#%% get observation info from agent (visual, velo, absolute pose)



env_path = "../env/AnimalAI"
port = 5005 + random.randint(
    0, 1000
) 

competition_folder = "../configs/Custom/"
configuration_files = os.listdir(competition_folder)
configuration_ind = 3 #random.randint(0, len(configuration_files))
configuration_file = competition_folder + configuration_files[configuration_ind]
print(F"Using configuration file {configuration_file}")




#For first test just use the Bratenberg behavior from their lowlevelapi example.#
from animalai.envs.braitenberg import Braitenberg

totalRays = 9
env = AnimalAIEnvironment(
        file_name=env_path,
        arenas_configurations=configuration_file,
        seed = 0,
        play= False,
        useCamera=True, #The Braitenberg agent works with raycasts, but want to keep visual info
        useRayCasts=True,
        raysPerSide=int((totalRays-1)/2),
        rayMaxDegrees = 30,
    )


#set list to store info after run
scenes =[]
vel = []
pos = []


braitenbergAgent = Braitenberg(totalRays) #A simple BraitenBerg Agent that heads towards food items.
behavior = list(env.behavior_specs.keys())[0] # by default should be AnimalAI?team=0

firststep = True
for _episode in range(1): #Run episodes with the Braitenberg-style agent
        if firststep:
            env.step() # Need to make a first step in order to get an observation.
            firstep = False
        dec, term = env.get_steps(behavior)
        done = False
        episodeReward = 0
        while not done:
            raycasts = env.get_obs_dict(dec.obs)["rays"] # Get the raycast data
            temp = env.get_obs_dict(dec.obs)["camera"]
            temp2 = env.get_obs_dict(dec.obs)["velocity"]
            temp3 = env.get_obs_dict(dec.obs)["position"]
            
            scenes.append(temp)
            vel.append(temp2)
            pos.append(temp3)
            
            # print(braitenbergAgent.prettyPrint(raycasts)) #print raycasts in more readable format
            action = braitenbergAgent.get_action(raycasts)
            # print(action)
            env.set_actions(behavior, action.action_tuple)
            env.step()      
            dec, term = env.get_steps(behavior)
            if len(dec.reward) > 0:
                episodeReward += dec.reward
            if len(term) > 0: #Episode is over
                episodeReward += term.reward
                print(F"Episode Reward: {episodeReward}")
                done = True
                firststep = True

env.close()

dur = len(scenes)
vel = np.asarray(vel)
pos = np.asarray(pos)
#pull out x and y

frames = [0, 10, 20, 22, 24, 26, 30, dur-1]
plt.close('all')
for t in  frames:
    
    plt.figure()
    plt.imshow(scenes[t][:,:,2])
    plt.title('Frame #' + str(t))
    
# plt.figure()
# plt.plot(pos)

#%% old code
# # Run the environment until signal to it is lost
# try:
#     while env._process:
#         behavior = list(env.behavior_specs.keys())[0]
#         dec, term = env.get_steps(behavior)
#         temp = env.get_obs_dict(dec.obs)["camera"]
#         temp2 = env.get_obs_dict(dec.obs)["velocity"]
#         temp3 = env.get_obs_dict(dec.obs)["position"]
        
#         continue

        
# except KeyboardInterrupt:
#     pass
# finally:
#     env.close()
# plt.imshow(temp[:,:,0])

# behavior = list(env.behavior_specs.keys())[0]
# dec, term = env.get_steps(behavior)
# temp = env.get_obs_dict(dec.obs)["camera"]
# plt.imshow(temp[:,:,0])



#%% From play script: load environment and have manual control of agent.



def load_config_and_play(configuration_file: str) -> None:
    """
    Loads a configuration file for a single arena and lets you play manually
    :param configuration_file: str path to the yaml configuration
    :return: None
    """
    env_path = "../env/AnimalAI"
    port = 5005 + random.randint(
        0, 1000
    )  # use a random port to avoid problems if a previous version exits slowly

    print("initializaing AAI environment")
    environment = AnimalAIEnvironment(
        file_name=env_path,
        base_port=port,
        arenas_configurations=configuration_file,
        play=True,
    )

    # Run the environment until signal to it is lost
    try:
        while environment._process:
            continue
            
    except KeyboardInterrupt:
        pass
    finally:
        environment.close()

#%% Run file in manual mode to show arena.

#update: not clear if possible to get observations printed out while running
# the agent manually

competition_folder = "../configs/Custom/"
configuration_files = os.listdir(competition_folder)
configuration_random = 3 #random.randint(0, len(configuration_files))
configuration_file = competition_folder + configuration_files[configuration_random]
print(F"Using configuration file {configuration_file}")
        
       
    
load_config_and_play(configuration_file=configuration_file)