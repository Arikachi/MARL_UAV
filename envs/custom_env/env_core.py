import itertools
import random

import numpy as np
np.set_printoptions(suppress=True)
from gymnasium import spaces

import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import copy

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
    def __init__(self, length=4 , radius=0.5, num_obstacle=0, num_agents=3, all_args=None):
        # environment
        self.length = length
        self.radius = radius
        self.goal_radius = 0.8
        self.num_obstacle = num_obstacle
        self.time_step = 0.5
        self.v_max = 0.12
        self.a_max = 0.08
        self.obstacles = [Obstacle(length=length) for _ in range(self.num_obstacle)]

        # agents
        self.num_agents = num_agents  # n agents
        self.agent_num = num_agents # for env_continuous
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
        self.rewards = np.zeros(num_agents)

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
        if all_args and hasattr(all_args, "episode_length"):
            self.MAX_STEPS = all_args.episode_length
        else:
            self.MAX_STEPS = 100

    def reset(self):
        """
        # When self.num_agents is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        self.step_cnt = 0
        self.rewards = np.zeros(self.num_agents)

        # Target
        self.current_target_radius = np.random.uniform(0.3, 0.8)
        self.target_direction = np.random.choice([-1, 1])
        self.target_anchor_x = np.random.uniform(1.5, 2.5)
        self.target_anchor_y = np.random.uniform(1.5, 2.5)
        self.target_pos = np.array([self.target_anchor_x + self.target_direction * self.current_target_radius, self.target_anchor_y])
        self.target_goal = self.generate_surround_positions(self.target_pos, self.num_agents , self.goal_radius)
        self.prev_target_goal = self.target_goal
        random.seed(random.randint(1, 1000))
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]

        for i in range(self.num_agents):
            self.multi_current_pos.append(np.random.uniform(low=0.1, high=2.5, size=(2,)))
            self.multi_current_vel.append(np.zeros(2))
        return self.get_multi_obs()

    def step(self, actions):
        """
        # When self.num_agents is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        # target path
        angular_speed = 0.05 #TODO  # controls how fast it completes the circle

        self.prev_target_goal = self.target_goal
        self.target_pos[0] = self.target_anchor_x + self.target_direction * self.current_target_radius * np.cos(angular_speed * self.step_cnt)
        self.target_pos[1] = self.target_anchor_y + self.target_direction * self.current_target_radius * np.sin(angular_speed * self.step_cnt)
        self.target_goal = self.generate_surround_positions(self.target_pos, self.num_agents, self.goal_radius)

        # agents step
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            acce = actions[i] * self.time_step
            for j in [0, 1]:
                if acce[j] >= self.a_max:
                    acce[j] = self.a_max
                elif acce[j] <= -self.a_max:
                    acce[j] = -self.a_max
            # Damping
            self.multi_current_vel[i] *= 0.95
            # Accelerate
            self.multi_current_vel[i][0] += acce[0]
            self.multi_current_vel[i][1] += acce[1]
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
        formation_tolerance = 0.15 #TODO

        mu1, mul2, mul3 = 1.2, 0.8, 1.0         # distance, collision, speed

        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            goal = self.target_goal[i]
            dist_to_goal = np.linalg.norm(pos - goal)

            goal_vel = (self.target_goal[i] - self.prev_target_goal[i]) / self.time_step
            rel_speed = np.linalg.norm(vel - goal_vel)

            # distance reward
            rewards[i] -= dist_to_goal * mu1
            if dist_to_goal > formation_tolerance:
                all_agents_in_formation = False

            # collision reward
            if IsCollied[i]:
                rewards[i] -= 10 * mul2

            # smooth control
            if dist_to_goal < formation_tolerance:
                rewards[i] += 3.0
                rewards[i] -= rel_speed * mul3
            else:
                rewards[i] -= rel_speed * mul3 * 0.05

        # formation reward
        if all_agents_in_formation:
            self.success_streak += 1
        else:
            self.success_streak = 0

        if self.success_streak >= 3 and self.success_streak < 6:
            for i in range(self.num_agents):
                rewards[i] += 5.0
        elif self.success_streak >= 6:
            for i in range(self.num_agents):
                rewards[i] += 10.0

        if self.success_streak >= 15:
            for i in range(self.num_agents):
                rewards[i] += 50.0
            dones = [True] * self.num_agents

        self.rewards = rewards
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
    BG_COLOR = "#f7f7f9"
    OBSTACLE_COLOR = "#5a5f66"
    GOAL_COLOR = "#d62728"
    AGENT_COLORS = ("#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#e377c2")

    def render(self, mode="human"):
        if not hasattr(self, "history_positions"):
            self.history_positions = [[] for _ in range(self.num_agents)]

        fig = plt.gcf()
        fig.set_size_inches(6, 6)
        fig.set_facecolor(self.BG_COLOR)
        plt.clf()
        
        ax = plt.gca()
        ax.set_facecolor(self.BG_COLOR)
        ax.set_aspect("equal", adjustable="box")
        
        # Self.length dictates the arena grid size boundary dynamically
        ax.set_xlim(-0.1, self.length + 0.1)
        ax.set_ylim(-0.1, self.length + 0.1)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Draw subtle arena boundaries
        ax.add_patch(plt.Rectangle(
            (0, 0), self.length, self.length,
            fill=False, edgecolor="#cdd0d4", linewidth=0.7,
            linestyle=(0, (3, 3)),
        ))

        # 1. Draw Static/Dynamic Obstacles (Assuming self.obstacles contains obstacle instances)
        if hasattr(self, "obstacles"):
            for obstacle in self.obstacles:
                ax.add_patch(plt.Circle(
                    obstacle.position, obstacle.radius,
                    facecolor=self.OBSTACLE_COLOR, edgecolor="#2c3036",
                    alpha=0.55, linewidth=1.0, zorder=2,
                ))

        # 2. Draw Target Formations / Target Slots (self.target_goal)
        if hasattr(self, "target_goal"):
            if hasattr(self, "target_pos"):
                ax.scatter(
                    self.target_pos[0], self.target_pos[1], c="gold",
                    marker="X", s=250, edgecolors="black", linewidths=1.5,
                    zorder=5, alpha=0.9, label="Main Target"
                )

            for i in range(self.num_agents):
                goal_pos = self.target_goal[i]
                ax.scatter(
                    goal_pos[0], goal_pos[1], c=self.GOAL_COLOR,
                    marker="*", s=160, edgecolors="white", linewidths=1.0,
                    zorder=4, alpha=0.7, label="Slot Target" if i == 0 else ""
                )

        # 3. Draw Agents and their Fading Trajectories
        for i in range(self.num_agents):
            color = self.AGENT_COLORS[i % len(self.AGENT_COLORS)]
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            
            # Save position history for trails
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])

            # Draw fading line trajectories
            if len(trajectory) >= 2:
                segs = np.stack([trajectory[:-1], trajectory[1:]], axis=1)
                n_segs = len(segs)
                for k, seg in enumerate(segs):
                    alpha = 0.15 + 0.6 * (k / max(n_segs - 1, 1))
                    ax.plot(seg[:, 0], seg[:, 1], color=color, alpha=alpha, linewidth=1.8, zorder=3)

            # 3.5 DRAW LINE TO AGENT'S OWN GOAL SLOT
            if hasattr(self, "target_goal"):
                goal_pos = self.target_goal[i]
                ax.plot(
                    [pos[0], goal_pos[0]], [pos[1], goal_pos[1]],
                    color=color, linestyle=":", linewidth=1.2, alpha=0.45,
                    zorder=2, label="To Target" if i == 0 else ""
                )

            # Draw Agent Node
            ax.scatter(pos[0], pos[1], c=color, s=140, edgecolors="white",
                       linewidths=1.5, zorder=5, label=f"Agent {i}")
            
            # Draw Velocity Vectors 
            speed = np.linalg.norm(vel)
            if speed > 1e-3:
                angle = np.arctan2(vel[1], vel[0])
                arrow_len = 0.08
                ax.annotate(
                    "", xy=(pos[0] + arrow_len * np.cos(angle), pos[1] + arrow_len * np.sin(angle)),
                    xytext=(pos[0], pos[1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8, shrinkA=4, shrinkB=0),
                    zorder=6,
                )

        # 4. Display Step Counter (Assumes self.step_cnt exists, otherwise defaults to 0)
        step_val = getattr(self, "step_cnt", 0)
        max_steps_val = getattr(self, "MAX_STEPS", 100)
        streak_val = getattr(self, "success_streak", 0)
        rewards_list = getattr(self, "rewards", [0.0] * self.num_agents)

        display_text = f"step {step_val:3d}/{max_steps_val}\nstreak: {streak_val}"

        for idx in range(self.num_agents):
            rew = rewards_list[idx]
            display_text += f"\nAgent {idx} Rew: {rew:.2f}"

        ax.text(
            0.02, 0.98, display_text,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="#2c3036",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="#d0d3d8", alpha=0.85),
        )

        # Layout Arrangement and Legends Placement
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.01),
            ncol=4, frameon=False, fontsize=9,
            handletextpad=0.4, columnspacing=1.0,
        )

        fig.tight_layout(pad=0.5)
        
        # Render out to numpy RGB array for compatibility with env_runner.py
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        
        if mode == "rgb_array":
            return np.asarray(buf)
        else:
            plt.pause(0.001)
            return np.asarray(buf)

    def close(self):
        """Cleanup window objects."""
        import matplotlib.pyplot as plt
        plt.close()

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