from gym.envs.registration import register

register(
    id=f'halfcheetah-heavy-v0',
    entry_point='alternate_envs.halfcheetah:get_cheetah_env',
    max_episode_steps=1000,
    kwargs={
        'xml_file': 'hc_heavy.xml',
    }
)

register(
    id=f'walker2dnoterm-v0',
    entry_point='alternate_envs.walker2d:get_walker_env',
    max_episode_steps=1000,
    # kwargs={
    #     'xml_file': 'hc_heavy.xml',
    # }
)

register(
    id=f'hoppernoterm-v0',
    entry_point='alternate_envs.hopper:get_hopper_env',
    max_episode_steps=1000,
)

register(
    id=f'HumanoidTruncatedObs-v2',
    entry_point='alternate_envs.humanoid_truncobs:get_humanoidtruncobs_env',
    max_episode_steps=1000,
)

register(
    id=f'HumanoidTruncatedObsMBPOReward-v2',
    entry_point='alternate_envs.humanoid_truncobs:get_humanoidtruncobsmbporeward_env',
    max_episode_steps=1000,
)

register(
    id=f'AntTruncatedObs-v2',
    entry_point='alternate_envs.ant_truncobs:get_anttruncobs_env',
    max_episode_steps=1000,
)

# register(
#     id=f'HumanoidTruncatedObs-v2',
#     entry_point='alternate_envs.humanoid_truncobs.get_humanoidtruncobs_env',
#     max_episode_steps=1000,
# )
