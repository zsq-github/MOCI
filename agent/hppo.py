import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from .memory import HPPOBuffer
from .model import Actor, Critic

# A Deep Deterministic Policy Gradient (PPO) Approach to Distributed Multi-User Implementation
class HPPO:
    def __init__(self, num_users, num_states, num_channels, lr_a, lr_c, pmax, gamma, lam, repeat_time, batch_size,
                 eps_clip, w_entropy):
        self.actors = [Actor(num_states, 6, num_channels, pmax).cuda() for _ in range(num_users)]
        # A Deep Deterministic Policy Gradient (PPO) Approach to Distributed Multi-User Implementation
        self.critic = Critic(num_states).cuda()
        # A Critic instance is created to evaluate the value function of the state
        print(num_states)

        self.optimizer_a = torch.optim.Adam([{'params': actor.parameters(), 'lr': lr_a} for actor in self.actors])
        # An optimizer optimizer_a has been created to update the parameters for all actor instances
        self.optimizer_c = torch.optim.Adam(self.critic.parameters(), lr_c)
        # An optimizer optimizer_a has been created to update the parameters for all actor instances

        self.buffer = HPPOBuffer(num_users)

        self.pmax = pmax
        self.gamma = gamma
        self.lam = lam
        self.repeat_time = repeat_time
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.w_entropy = w_entropy

    # Use an actor network to select actions based on a given state
    def select_action(self, state, test=False):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).cuda()
            actions = []  # Stores the selected action for each user device.
            for i, actor in enumerate(self.actors):
                prob_points, prob_channels, (power_mu, power_sigma) = actor(state)
                #print( "1",prob_points, "2",prob_channels, "3",(power_mu, power_sigma))
                # point
                dist_point = Categorical(prob_points)
                point = dist_point.sample()
                # channel
                dist_channel = Categorical(prob_channels)
                channel = dist_channel.sample()
                # power
                dist_power = Normal(power_mu, power_sigma)
                power = dist_power.sample() # Sampling a Power Value from a Normal Power Distribution

                if not test:
                    self.buffer.actor_buffer[i].actions_point.append(point)
                    self.buffer.actor_buffer[i].actions_channel.append(channel)
                    self.buffer.actor_buffer[i].actions_power.append(power)

                    self.buffer.actor_buffer[i].logprobs_point.append(dist_point.log_prob(point))
                    self.buffer.actor_buffer[i].logprobs_channel.append(dist_channel.log_prob(channel))
                    self.buffer.actor_buffer[i].logprobs_power.append(dist_power.log_prob(power))

                actions.append((point.item(), channel.item(), power.clamp(1e-10, self.pmax).item()))
                # A clamp operation is performed on the power to ensure that the power is within a reasonable range and to prevent over- or under-values

                self.buffer.actor_buffer[i].info['prob_points'] = prob_points.tolist()
                self.buffer.actor_buffer[i].info['prob_channels'] = prob_channels.tolist()
                self.buffer.actor_buffer[i].info['power_mu'] = power_mu.item()
                self.buffer.actor_buffer[i].info['power_sigma'] = power_sigma.item()

        return actions

    # Updating the strategy network (actor) and value network (critic)
    def update(self):
        self.buffer.build_tensor()
        # for generate gae
        with torch.no_grad():
            pred_values_buffer = self.critic(self.buffer.states_tensor).squeeze()
        target_values_buffer, advantages_buffer = self.get_gae(pred_values_buffer)
        #+++++++++
        old_actors_params = [actor.state_dict() for actor in self.actors]

        # Generate a Generalized Advantage Estimate (GAE) and return the target and advantage values
        for _ in range(int(self.repeat_time * (len(self.buffer) / self.batch_size))):
            indices = torch.randint(len(self.buffer), size=(self.batch_size,), requires_grad=False).cuda()
            state = self.buffer.states_tensor[indices]
            target_values = target_values_buffer[indices]
            advantages = advantages_buffer[indices]
            #Get status, target value and dominance value from the buffer

            loss_point = []
            loss_channel = []
            loss_power = []
            c_wass=0.1
            for i, actor in enumerate(self.actors):
                # Calculate new probabilities of action, entropy, and loss
                logprobs_point = self.buffer.actor_buffer[i].logprobs_point_tensor[indices]
                logprobs_channel = self.buffer.actor_buffer[i].logprobs_channel_tensor[indices]
                logprobs_power = self.buffer.actor_buffer[i].logprobs_power_tensor[indices]

                new_logprobs_point, new_logprobs_channel, new_logprobs_power, \
                entropy_point, entropy_channel, entropy_power = self.eval(i, state, indices)
                self.buffer.actor_buffer[i].info['entropy_point'] = entropy_point.mean().item()
                self.buffer.actor_buffer[i].info['entropy_channel'] = entropy_channel.mean().item()
                self.buffer.actor_buffer[i].info['entropy_power'] = entropy_power.mean().item()

                # point
                ratio_point = (new_logprobs_point - logprobs_point).exp()
                # The probability that the new strategy will take the same action in a given state relative to the old strategy is greater than
                surr1_point = advantages * ratio_point
                surr2_point = advantages * torch.clamp(ratio_point, 1 - self.eps_clip, 1 + self.eps_clip)
                loss_point.append((-torch.min(surr1_point, surr2_point) - self.w_entropy * entropy_point).mean())
                # channel
                ratio_channel = (new_logprobs_channel - logprobs_channel).exp()
                surr1_channel = advantages * ratio_channel
                surr2_channel = advantages * torch.clamp(ratio_channel, 1 - self.eps_clip, 1 + self.eps_clip)
                loss_channel.append(
                    (-torch.min(surr1_channel, surr2_channel) - self.w_entropy * entropy_channel).mean())
                # power
                ratio_power = (new_logprobs_power - logprobs_power).exp()
                surr1_power = advantages * ratio_power
                surr2_power = advantages * torch.clamp(ratio_power, 1 - self.eps_clip, 1 + self.eps_clip)
                loss_power.append((-torch.min(surr1_power, surr2_power) - self.w_entropy * entropy_power).mean())
            loss_a0 = torch.stack(loss_point + loss_channel + loss_power).mean()
            # Entropy of strategies to encourage exploratory and diverse strategies

            wasserstein_loss = self.calculate_wasserstein_distance(old_actors_params,[actor.state_dict() for actor in self.actors])
            loss_a = loss_a0 + c_wass * wasserstein_loss
            #loss_a = loss_a0 - c_wass * wasserstein_loss

            self.optimizer_a.zero_grad()
            loss_a.backward()
            self.optimizer_a.step()

            pred_values = self.critic(state).squeeze()
            self.optimizer_c.zero_grad()
            loss_c = F.mse_loss(pred_values, target_values)  # Loss calculation in the value network
            # loss_c = F.smooth_l1_loss(pred_values, target_values)
            loss_c.backward()
            self.optimizer_c.step()
            self.buffer.info['loss_value'] = loss_c.item()

        self.buffer.init()

    # Log probability and entropy values used to evaluate each action
    def eval(self, idx, state, indices):
        actions_point = self.buffer.actor_buffer[idx].actions_point_tensor[indices] # Get the action point at the specified index

        actions_channel = self.buffer.actor_buffer[idx].actions_channel_tensor[indices]
        actions_power = self.buffer.actor_buffer[idx].actions_power_tensor[indices]

        prob_points, prob_channels, (power_mu, power_sigma) = self.actors[idx](state)

        dist_point = Categorical(prob_points)
        dist_channel = Categorical(prob_channels)
        dist_power = Normal(power_mu, power_sigma)

        logprobs_point = dist_point.log_prob(actions_point)
        logprobs_channel = dist_channel.log_prob(actions_channel)
        logprobs_power = dist_power.log_prob(actions_power)

        entropy_point = dist_point.entropy()
        entropy_channel = dist_channel.entropy()
        entropy_power = dist_power.entropy()

        return logprobs_point, logprobs_channel, logprobs_power, entropy_point, entropy_channel, entropy_power

    # Used to calculate Generalized Advantage Estimates (GAE)
    # Generate the target and dominance values used to update the strategy network (actor) during the training process
    def get_gae(self, pred_values_buffer):
        with torch.no_grad():
            target_values_buffer = torch.empty(len(self.buffer), dtype=torch.float32).cuda()
            advantages_buffer = torch.empty(len(self.buffer), dtype=torch.float32).cuda()
            # Used to store target and advantage values

            next_value = 0
            next_advantage = 0
            for i in reversed(range(len(self.buffer))):
                reward = self.buffer.rewards_tensor[i]
                mask = 1 - self.buffer.is_terminals_tensor[i]
                # The target value indicates the expected cumulative reward that can be obtained after performing the action in the current state
                target_values_buffer[i] = reward + mask * self.gamma * next_value
                delta = reward + mask * self.gamma * next_value - pred_values_buffer[i]
                advantages_buffer[i] = delta + mask * self.lam * next_advantage

                next_value = pred_values_buffer[i]
                next_advantage = advantages_buffer[i]
            advantages_buffer = (advantages_buffer - advantages_buffer.mean()) / (advantages_buffer.std() + 1e-5) # Normalization Process

        return target_values_buffer, advantages_buffer

    def calculate_wasserstein_distance(self, old_params, new_params):
        distance = 0
        for old_param, new_param in zip(old_params, new_params):
            for old_p, new_p in zip(old_param.values(), new_param.values()):
                distance += torch.sum(torch.abs(old_p - new_p))
        return distance

    def save_model(self, filename, args, info=None):
        dic = {'actor': [actor.state_dict() for actor in self.actors],
               'critic': self.critic.state_dict(),
               'args': args,
               'info': info}
        torch.save(dic, filename)

    def load_model(self, actor_dicts, critic_dict):
        for actor, dict in zip(self.actors, actor_dicts):
            actor.load_state_dict(dict)
        self.critic.load_state_dict(critic_dict)
