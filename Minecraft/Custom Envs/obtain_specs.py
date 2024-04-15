import gym

from minerl.env import _fake, _singleagent
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
#from minerl.herobraine.hero import handlers
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.hero.mc import ALL_ITEMS
from typing import List

TIMEOUT = 99999999
DIAMOND_ITEMS = [
    [["acacia_log", "birch_log", "dark_oak_log", "jungle_log", "oak_log", "spruce_log"], 10],
    #[["acacia_planks", "birch_planks", "dark_oak_planks", "jungle_planks", "oak_planks", "spruce_planks"], 2],
    [["stick"], 2],
    #[["crafting_table"], 4],
    #[["wooden_pickaxe"], 8],
    #[["cobblestone"], 16],
    #[["furnace"], 32],
    #[["stone_pickaxe"], 32],
    #[["iron_ore"], 64],
    #[["iron_ingot"], 128],
    #[["iron_pickaxe"], 256],
    #[["diamond"], 1024],
    #[["diamond_shovel"], 2048]
]


class ObtainDiamondShovelWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.vertical_angle = 0
        self.rewarded_items = DIAMOND_ITEMS
        self.seen = [0] * len(self.rewarded_items)      # keep track of what items of interest we have seen thus far (for unique rewards)
        self.have = [0] * len(self.rewarded_items)      # keep track of how may items of interest we have
        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0
        self.episode_over = False

    def step(self, action: dict):
        if self.episode_over:
            raise RuntimeError("Expected `reset` after episode terminated, not `step`.")
        observation, reward, done, info = super().step(action)
        for i, [item_list, rew] in enumerate(self.rewarded_items):
            if not self.seen[i]:
                for item in item_list:
                    # if observation["inventory"][item] != 0:
                    #     print("-" + item)
                    if observation["inventory"][item] > self.have[i]:   # if we see that we have picked up a new item of interest
                        #if i == len(self.rewarded_items) - 1:  # achieved last item in rewarded item list
                            #done = True
                        #print(item + " " + str(self.num_steps) + " Total: " + str(observation["inventory"][item]))
                        reward += rew
                        # self.seen[i] = 1                      # for unique item finds
                        self.have[i] = self.have[i] + 1         # increment the number of known items
                        break
        self.num_steps += 1
        # if self.num_steps >= self.timeout:
        #     done = True
        self.episode_over = done
        #print(self.num_steps) if (self.num_steps % 100) == 0 else None
        return observation, reward, done, info

    def reset(self):
        self.vertical_angle = 0
        self.num_steps = 0
        self.seen = [0] * len(self.rewarded_items)
        self.have = [0] * len(self.rewarded_items)  # reset how may items of interest we have
        self.episode_over = False
        obs = super().reset()
        return obs


def _obtain_diamond_shovel_gym_entrypoint(env_spec, fake=False):
    """Used as entrypoint for `gym.make`."""
    if fake:
        env = _fake._FakeSingleAgentEnv(env_spec=env_spec)
    else:
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)

    env = ObtainDiamondShovelWrapper(env)
    return env

OBTAIN_DIAMOND_SHOVEL_ENTRY_POINT = "minerl.herobraine.env_specs.obtain_specs:_obtain_diamond_shovel_gym_entrypoint"
TREECHOP_WORLD_GENERATOR_OPTIONS = """{"coordinateScale":684.412,"heightScale":684.412,"lowerLimitScale":512.0,"upperLimitScale":512.0,"depthNoiseScaleX":200.0,"depthNoiseScaleZ":200.0,"depthNoiseScaleExponent":0.5,"mainNoiseScaleX":80.0,"mainNoiseScaleY":160.0,"mainNoiseScaleZ":80.0,"baseSize":8.5,"stretchY":12.0,"biomeDepthWeight":1.0,"biomeDepthOffset":0.0,"biomeScaleWeight":1.0,"biomeScaleOffset":0.0,"seaLevel":1,"useCaves":false,"useDungeons":false,"dungeonChance":8,"useStrongholds":false,"useVillages":false,"useMineShafts":false,"useTemples":false,"useMonuments":false,"useMansions":false,"useRavines":false,"useWaterLakes":false,"waterLakeChance":4,"useLavaLakes":false,"lavaLakeChance":80,"useLavaOceans":false,"fixedBiome":4,"biomeSize":4,"riverSize":1,"dirtSize":33,"dirtCount":10,"dirtMinHeight":0,"dirtMaxHeight":256,"gravelSize":33,"gravelCount":8,"gravelMinHeight":0,"gravelMaxHeight":256,"graniteSize":33,"graniteCount":10,"graniteMinHeight":0,"graniteMaxHeight":80,"dioriteSize":33,"dioriteCount":10,"dioriteMinHeight":0,"dioriteMaxHeight":80,"andesiteSize":33,"andesiteCount":10,"andesiteMinHeight":0,"andesiteMaxHeight":80,"coalSize":17,"coalCount":20,"coalMinHeight":0,"coalMaxHeight":128,"ironSize":9,"ironCount":20,"ironMinHeight":0,"ironMaxHeight":64,"goldSize":9,"goldCount":2,"goldMinHeight":0,"goldMaxHeight":32,"redstoneSize":8,"redstoneCount":8,"redstoneMinHeight":0,"redstoneMaxHeight":16,"diamondSize":8,"diamondCount":1,"diamondMinHeight":0,"diamondMaxHeight":16,"lapisSize":7,"lapisCount":1,"lapisCenterHeight":16,"lapisSpread":16}"""

class ObtainDiamondShovelEnvSpec(HumanSurvival):
    r"""
In this environment the agent is required to obtain a diamond shovel.
The agent begins in a random starting location on a random survival map
without any items, matching the normal starting conditions for human players in Minecraft.

During an episode the agent is rewarded according to the requisite item
hierarchy needed to obtain a diamond shovel. The rewards for each item are
given here::

    <Item reward="1" type="log" />
    <Item reward="2" type="planks" />
    <Item reward="4" type="stick" />
    <Item reward="4" type="crafting_table" />
    <Item reward="8" type="wooden_pickaxe" />
    <Item reward="16" type="cobblestone" />
    <Item reward="32" type="furnace" />
    <Item reward="32" type="stone_pickaxe" />
    <Item reward="64" type="iron_ore" />
    <Item reward="128" type="iron_ingot" />
    <Item reward="256" type="iron_pickaxe" />
    <Item reward="1024" type="diamond" />
    <Item reward="2048" type="diamond_shovel" />
"""
    def __init__(self):
        super().__init__(
            name="MineRLObtainDiamondShovel-v0",
            max_episode_steps=TIMEOUT,
            # Hardcoded variables to match the pretrained models
            fov_range=[70, 70],
            resolution=[84, 84],
            gamma_range=[2, 2],
            guiscale_range=[0, 0],
            cursor_size_range=[16.0, 16.0]
        )
    def create_agent_start(self) -> List[Handler]:
        return super().create_agent_start() + [                            #super().create_agent_start() +
            handlers.AgentStartBreakSpeedMultiplier(10000),
            handlers.SimpleInventoryAgentStart([
                dict(type="diamond_axe", quantity=1)
            ])
        ]
    def _entry_point(self, fake: bool) -> str:
        return OBTAIN_DIAMOND_SHOVEL_ENTRY_POINT

    def create_observables(self) -> List[Handler]:
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(ALL_ITEMS)
        ]

    def create_monitors(self) -> List[TranslationHandler]:
        return []

    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.DefaultWorldGenerator(force_reset="true",
                                           generator_options=TREECHOP_WORLD_GENERATOR_OPTIONS
                                           )
        ]

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False
            ),
            handlers.SpawningInitialCondition(
                allow_spawning=True
            )
        ]