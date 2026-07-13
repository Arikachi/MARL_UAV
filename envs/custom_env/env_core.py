import itertools
import random

import numpy as np
np.set_printoptions(suppress=True)
from gymnasium import spaces

"""
# Understanding 
# init = base infos
# reset = starting state of each episode
# step = what happen every update
# Target is consider part of the environment

# Mechanisism
# Define the environment / reward to feed into training
# Training return policy to update the step function
# reset function is called to initialized at start of every episode

# Compare to uav_demo
# __init__:
# agents's variable (number of agent, velocity, action / observation space, state, ...)
# obstacles information (number,size, ...)
# step counter

# reset:
# initialize step counter = 0, past states of agents = []
# randomize agents positions

# step:

# rendering:

"""

class EnvCore:
    def __init__(self, length=4 , radius=0.5, num_obstacle=3, num_agents=3):
        # environment
        self.length = length
        self.radius = radius
        self.num_obstacle = num_obstacle
        self.time_step = 0.5
        self.v_max = 0.08
        self.a_max = 0.04
        self.obstacles = [Obstacle(length=length) for _ in range(self.num_obstacle)]

        # agents
        self.num_agents = num_agents  # n agents
        self.obs_dim = 14  
        """
        1, 2: global velocity x, y
        3, 4: relative goal position
        5, 6: relative target position
        7, 8, 9, 10: relative position of the closest 2 other agents
        11, 12, 13, 14: relative position of the closest 2 obstacle
        """
        self.action_dim = 2 # v_x, v_y
        self.history_positions = [[] for _ in range(num_agents)]

        # target
        self.target_pos = np.zeros(2)
        self.target_velocity = 0.05
        self.target_goal = []

        # space define
        self.agents = ["agent_0", "agent_1", "agent_2"]
        self.action_space = {
            "agent_0": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            "agent_1": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            "agent_2": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
        }
        self.observation_space = {
            "agent_0": spaces.Box(low=-np.inf, high=np.inf, shape=(14,)),
            "agent_1": spaces.Box(low=-np.inf, high=np.inf, shape=(14,)),
            "agent_2": spaces.Box(low=-np.inf, high=np.inf, shape=(14,)),
        }

        # step counters
        self.success_streak = 0
        self.step_cnt = 0
        self.MAX_STEPS = 100

    def reset(self):
        """
        # When self.num_agents is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        self.step_cnt = 0
        self.current_target_radius = np.random.uniform(0.3, 0.8)
        self.target_direction = np.random.choice([-1, 1])
        self.target_anchor_x = np.random.uniform(1.5, 2.5)
        self.target_anchor_y = np.random.uniform(1.5, 2.5)
        self.target_pos = np.array([self.target_anchor_x + self.target_direction * self.current_target_radius, self.target_anchor_y])
        self.target_goal = self.generate_surround_positions(self.target_pos, self.num_agents , self.radius)
        random.seed(random.randint(1, 1000))
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]

        for i in range(self.num_agents):
            self.multi_current_pos.append(np.random.uniform(low=0.1, high=0.4, size=(2,)))
            self.multi_current_vel.append(np.zeros(2))
        return self.get_multi_obs()

    def step(self, actions):
        """
        # When self.num_agents is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        # target path
        angular_speed = 0.05  # controls how fast it completes the circle
        
        self.target_pos[0] = self.target_anchor_x + self.target_direction * self.current_target_radius * np.cos(angular_speed * self.step_cnt)
        self.target_pos[1] = self.target_anchor_y + self.target_direction * self.current_target_radius * np.sin(angular_speed * self.step_cnt)
        self.target_goal = self.generate_surround_positions(self.target_pos, self.num_agents, self.radius)

        # agents step
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            vel_magnitude = np.linalg.norm(self.multi_current_vel[i])
            if vel_magnitude >= self.v_max:
                self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        # obstacles step
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            for dim in (0, 1):
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1

        # collision
        is_collided = self.isCollied_wrapper()

        # return values
        multi_next_obs = self.get_multi_obs()
        rewards, dones = self.cal_rewards_done(is_collided)
        infos = {agent: {} for agent in self.agents}

        if self.step_cnt >= self.MAX_STEPS:
            dones = [True] * self.num_agents
        self.step_cnt += 1

        formatted_rewards = [[r] for r in rewards]
        return [multi_next_obs, formatted_rewards, dones, infos]
    
    def get_multi_obs(self):
        total_obs = []
        
        for i in range(self.num_agents):
            # velocity global
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_vel = [vel[0] / self.v_max, vel[1] / self.v_max]
            
            # goal relative
            goal = self.target_goal[i]
            rel_goal = goal - pos
            S_goal = [rel_goal[0] / self.length, rel_goal[1] / self.length]

            # target relative
            rel_target = self.target_pos - pos
            S_target = [rel_target[0] / self.length, rel_target[1] / self.length]

            # nearby agents
            nearby_agent = []
            for j in range(self.num_agents):
                if j != i:
                    rel_pos = self.multi_current_pos[j] - pos
                    rel_distance = np.linalg.norm(rel_pos)
                    if rel_distance > self.radius:
                        continue
                    else:
                        nearby_agent.append({
                            "distance": rel_distance,
                            "coord": [rel_pos[0] / self.radius, rel_pos[1] / self.radius]
                        })
            nearby_agent.sort(key=lambda x: x["distance"])
            if len(nearby_agent) >= 2:
                nearby_agent = nearby_agent[:2]
            nearby_agent_coord = [[1.0, 1.0], [1.0, 1.0]]
            for t in range(len(nearby_agent)):
                nearby_agent_coord[t] = nearby_agent[t]["coord"]

            S_nearby_agent = [coord for pair in nearby_agent_coord for coord in pair]

            # nearby obstacles
            nearby_obstacle = []
            for j in range(self.num_obstacle):
                rel_pos = self.obstacles[j].position - pos
                rel_distance = np.linalg.norm(rel_pos)
                if rel_distance > self.radius:
                    continue
                else:
                    nearby_obstacle.append({
                        "distance": rel_distance,
                        "coord": [rel_pos[0] / self.radius, rel_pos[1] / self.radius]
                    })
            nearby_obstacle.sort(key=lambda x: x["distance"])
            if len(nearby_obstacle) >= 2:
                nearby_obstacle = nearby_obstacle[:2]
            nearby_obstacle_coord = [[1.0, 1.0], [1.0, 1.0]]
            for t in range(len(nearby_obstacle)):
                nearby_obstacle_coord[t] = nearby_obstacle[t]["coord"]
            S_nearby_obstacle = [coord for pair in nearby_obstacle_coord for coord in pair]

            single_obs = [S_vel, S_goal, S_target, S_nearby_agent, S_nearby_obstacle]
            flat_obs = list(itertools.chain(*single_obs))
            
            total_obs.append(flat_obs)
        return total_obs
    
    def cal_rewards_done(self, IsCollied):
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)

        all_agents_in_formation = True
        formation_tolerance = 0.10

        mu1 = 1.0

        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            goal = self.target_goal[i]
            dist_to_goal = np.linalg.norm(pos - goal)
            rewards[i] -= dist_to_goal * mu1

            # distance reward
            if dist_to_goal > formation_tolerance:
                all_agents_in_formation = False

            # collision reward
            if IsCollied[i]:
                rewards[i] -= 10

        # formation reward
        if all_agents_in_formation:
            self.success_streak += 1
        else:
            self.success_streak = 0
            
        if self.success_streak >= 6:
            dones = [True] * self.num_agents
            for i in range(self.num_agents):
                rewards[i] += 50.0

        return rewards, dones
    
    def isCollied_wrapper(self):
        dones = []
        agent_buffer = 0.05
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            done = False
            for obs in self.obstacles:
                dist = np.linalg.norm(pos - obs.position)
                min_dist = obs.radius + agent_buffer
                if dist < min_dist:
                    done = True
                    push_direction = (pos - obs.position) / (dist + 1e-6)
                    self.multi_current_pos[i] = obs.position + push_direction * min_dist

            for dim in (0, 1):
                if pos[dim] < agent_buffer:
                    self.multi_current_pos[i][dim] = agent_buffer
                    done = True
                elif pos[dim] > self.length - agent_buffer:
                    self.multi_current_pos[i][dim] = self.length - agent_buffer
                    done = True

            for j in range(self.num_agents):
                if j == i:
                    continue
                else:
                    pos_other = self.multi_current_pos[j]
                    dist_other = np.linalg.norm(pos - pos_other)
                    if dist_other < agent_buffer:
                        done = True
                        push_direction_other = (pos - pos_other) / (dist_other + 1e-6)
                        self.multi_current_pos[i] = pos_other + push_direction_other * agent_buffer


            if done:
                self.multi_current_vel[i] = np.zeros(2)
            dones.append(done)
        return dones

    # Render
    _HUNTER_COLORS = ("#1f77b4", "#2ca02c", "#9467bd")
    _TARGET_COLOR = "#d62728"
    _BG_COLOR = "#f7f7f9"
    _OBSTACLE_COLOR = "#5a5f66"

    # Helper
    def generate_surround_positions(self, target_pos, num_agents, radius):
        positions = []

        for i in range(num_agents):
            angle = 2 * np.pi * i / num_agents

            x = target_pos[0] + radius * np.cos(angle)
            y = target_pos[1] + radius * np.sin(angle)

            positions.append(np.array([x, y]))

        return positions

class Obstacle:
    def __init__(self, length=2):
        self.position = np.random.uniform(low=0.45, high=length-0.55, size=(2, ))
        angle = np.random.uniform(0, 2*np.pi)
        speed = 0.0
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        self.radius = np.random.uniform(0.1, 0.15)

# ==========================================
# TEST EXECUTION BLOCK (Runs only when executed directly)
# ==========================================
# if __name__ == "__main__":
#     # 1. Instantiate your custom environment
#     env = EnvCore()
    
#     print("--- Testing Reset ---")
#     # 2. Run reset to initialize your variables
#     initial_obs = env.reset()
    
#     # 3. Print your target position to verify it generated correctly
#     print(f"Target Position after reset: {env.target_pos}")
#     print(f"Target Goals after reset: {env.target_goal}")
#     print(f"Agent 0 Position after reset: {env.multi_current_pos[0]}")
    
#     print("\n--- Testing One Frame Step ---")
#     # 4. Create a dummy actions dictionary matching what MAPPO would send
#     # Agent 0, 1, and 2 trying to move with velocity [0.05, -0.02]
#     mock_actions = [
#         np.array([0.05, -0.02]),  # Agent 0
#         np.array([0.01, 0.04]),   # Agent 1
#         np.array([-0.03, 0.01]),  # Agent 2
#     ]
    
#     # 5. Run a single step to see if your designed pattern updates the target
#     obs, rewards, dones, infos = env.step(mock_actions)

#     # 6. Print the target position again to see if it moved!
#     print(f"Target Position after 1 step: {env.target_pos}")
#     print(f"Step Counter is now: {env.step_cnt}")

#     # 5. Run a single step to see if your designed pattern updates the target
#     obs, rewards, dones, infos = env.step(mock_actions)

#     # 6. Print the target position again to see if it moved!
#     print(f"Target Position after 1 step: {env.target_pos}")
#     print(f"Step Counter is now: {env.step_cnt}")