import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
from dqn_replay_buffer import ReplayBuffer
from utils import ReplayMemory, Transition
import datetime
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, h_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, h_dim)
        self.layer2 = nn.Linear(h_dim, h_dim)
        self.layer3 = nn.Linear(h_dim, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQN_C(nn.Module):

    def __init__(self, n_observations, n_actions, h_dim):
        super(DQN_C, self).__init__()
        self.layer1 = nn.Linear(n_observations, h_dim)
        self.layer2 = nn.Linear(h_dim, h_dim)
        self.layer3 = nn.Linear(h_dim, h_dim)
        self.layer4 = nn.Linear(h_dim, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer3(F.relu(self.layer2(x))) + x)
        return self.layer4(x)


class DuelingDQN(nn.Module):

    def __init__(self, n_observations, n_actions, h_dim):
        super(DuelingDQN, self).__init__()
        self.input_dim = n_observations
        self.output_dim = n_actions
        
        self.feature_layer = nn.Sequential(
            nn.Linear(self.input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.output_dim)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals


class DQNAgent():

    def __init__(self, params):
        super(DQNAgent, self).__init__()
        self.params = params
        self.device = self.params['device']
        self.env = gym.make(self.params['game_type'])
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display
        plt.ion()
        self.n_actions = self.env.action_space.n
        state, info = self.env.reset()
        self.n_observations = len(state)
        self.policy_net = globals()[self.params['net_type']](self.n_observations, self.n_actions,
                                                             self.params['hidden_dim']).to(self.device)
        self.target_net = globals()[self.params['net_type']](self.n_observations, self.n_actions,
                                                             self.params['hidden_dim']).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = globals()[self.params['buffer_type']](100000, self.params['ALPHA'])
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.params['LR'], amsgrad=True)
        self.steps_done = 0
        self.episode_durations = []
        self.save_model_name = "./checkpoint/MC_{}_d{}_{}_{}.pkl"\
            .format(self.params['net_type'], self.params['hidden_dim'], self.params['ALPHA'], self.params['BETA'])
        self.min_t = 199
        self.log_file = self.params['log_file']

    def reinitialize(self):
        self.__init__(self.params)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.params['EPS_END'] + (self.params['EPS_START'] - self.params['EPS_END']) * \
            math.exp(-1. * self.steps_done / self.params['EPS_DECAY'])
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        # reward_t = torch.tensor(enhanced_reward, dtype=torch.float)

        plt.xlabel('Episode')
        plt.ylabel('Duration')
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
            plt.plot(durations_t.numpy())
        # plt.plot(reward_t.numpy())
        # Take 100 episode averages and plot them too
        cumulative_mean = torch.cumsum(durations_t, dim=0) / torch.arange(1, len(durations_t) + 1)
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((cumulative_mean[:99], means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory) < self.params['BATCH_SIZE']:
            return
        transitions = self.memory.sample(self.params['BATCH_SIZE'], self.params['BETA'])
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.params['BATCH_SIZE'], device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.params['GAMMA']) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    # def reset_training(self):
    #     self.steps_done = 0
    #     print('reset training')
    #     self.train()

    def train(self):
        for step_seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
            # set step_seed
            print('step_seed = ', step_seed)
            torch.manual_seed(step_seed)
            random.seed(step_seed)
            self.reinitialize()
            for i_episode in range(self.params['num_episodes']):
                # Initialize the environment and get it's state
                state, info = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                for t in count():
                    action = self.select_action(state)
                    observation, reward, terminated, truncated, _ = self.env.step(action.item())

                    reward = torch.tensor([reward], device=self.device)
                    done = terminated or truncated

                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                    # Store the transition in memory
                    self.memory.push(state, action, next_state, reward)

                    # Move to the next state
                    state = next_state

                    # Perform one step of the optimization (on the policy network)
                    self.optimize_model()

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.params['TAU'] + \
                                                     target_net_state_dict[key]*(1-self.params['TAU'])
                    self.target_net.load_state_dict(target_net_state_dict)

                    if done:
                        self.episode_durations.append(t + 1)
                        if t < self.min_t:
                            self.min_t = t
                            best_name = self.save_model_name.replace('.pkl', '_{}_e{}.pkl'
                                                                     .format(self.min_t, i_episode))
                            torch.save(self.target_net.state_dict(), best_name)

                        # enhanced_reward.append(reward_sum)
                        # if i_episode%10 == 0:
                            # self.plot_durations()
                        break
                
                last_mean = sum(self.episode_durations[-100:]) / 100
                if len(self.episode_durations) > 100 and last_mean < 125:
                    result_name = self.save_model_name.replace('.pkl', '_{}_e{}_result.pkl'
                                                               .format(last_mean, i_episode))
                    torch.save(self.target_net.state_dict(), result_name)
                    print('training completed')
                    break
            
            self.save_curve(step_seed)

    def show_curve(self):
        self.plot_durations(show_result=True)
        plt.ioff()
        plt.show()

    def render_result(self, load_model):
        self.target_net.load_state_dict(torch.load(load_model))
        shown_env = gym.make(self.params['game_type'], render_mode="human")

        state, info = shown_env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        for t in count():
            shown_env.render()
            action = self.target_net(state).max(1)[1].view(1, 1)
            observation, _, terminated, truncated, _ = shown_env.step(action.item())

            done = terminated or truncated

            # Move to the next state
            if done:
                self.env.close()
                break
            else:
                state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

    def save_curve(self, seed):
        with open(self.log_file, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([self.params['net_type'], self.params['hidden_dim'], self.params['BATCH_SIZE'],
                                self.params['ALPHA'], self.params['BETA'], self.params['buffer_type'], seed,
                                datetime.datetime.now()])
            durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
            cumulative_mean = torch.cumsum(durations_t, dim=0) / torch.arange(1, len(durations_t) + 1)
            if len(durations_t) >= 100:
                means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((cumulative_mean[:99], means))
            else:
                means = cumulative_mean
            csvwriter.writerow(means.numpy())







