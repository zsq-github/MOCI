import numpy as np

from .channel import SubChannel
from .user import UserEquipment

normalize_factor = {'vgg11': (0.087236, 1204224),
                    'resnet18': (0.048952, 1204224),
                    'mobilenetv2': (0.060152, 1204224)}

#Indicates a mobile edge computing system
class MECsystem(object):
    def __init__(self, slot_time, num_users, num_channels, user_params, channel_params, alpha=0.5, beta=0.5):
        self.num_users = num_users
        self.UEs = [UserEquipment(**user_params) for _ in range(num_users)]
        self.slot_time = slot_time  #Length of the time slot that simulates the time interval in the system.
        self.channels = [SubChannel(**channel_params) for _ in range(num_channels)]  #Number of channels available on the system
        self.beta = beta
        self.alpha = alpha

    # Obtain system status information
    def get_state(self):
        state = []
        for u in self.UEs:
            max_time, max_data = normalize_factor[u.net]
            state.append(u.left_task_num / u.total_task)  # Degree of completion of the current task on the user's device
            state.append(u.time_left / max_time)  # Relative degree of user device reasoning progress
            state.append(u.data_left / max_data)  # Relative degree of data processing by user equipment
            state.append((u.distance - u.dmin) / (u.dmax - u.dmin))  # Location of the user device relative to the base station
        #print("get_state",state)
        return state

    # Calculate the reward value of the system
    def get_reward(self):
        energy = np.mean([u.energy_used for u in self.UEs])
        finished = np.mean([u.finished_num for u in self.UEs])
        avg_e = energy / max(finished, 0.8)
        avg_t = self.slot_time / max(finished, 0.8)
        reward = -self.alpha * avg_t - self.beta * avg_e
        return reward

    # Reset the state of the mobile edge computing system
    def reset(self):
        self.time = 0
        for u in self.UEs:
            u.reset()
            u.receive_tasks()  # Simulating the user device to receive a new task
            if u.left_task_num != 0:
                u.start_task()
        for c in self.channels:
            c.reset()
        return self.get_state()

    #MDP
    def step(self, action):
        # init
        done = False
        time_in_slot = 0  #
        # Assign Actions to User Devices in the System
        self.assign_action(action)

        for u in self.UEs:
            u.statistic_init()

        # state step
        next_time = self.stationary_time()
        while time_in_slot + next_time < self.slot_time:
            self.slot_step(next_time)
            time_in_slot += next_time
            next_time = self.stationary_time()

        if self.slot_time - time_in_slot > 0:
            self.slot_step(self.slot_time - time_in_slot)

        self.time += self.slot_time
        if self.is_done():
            done = True

        # state & reward
        state = self.get_state()
        reward = self.get_reward()

        # info
        total_time_used = self.slot_time * self.num_users
        total_energy_used = sum([u.energy_used for u in self.UEs])
        total_finished = sum([u.finished_num for u in self.UEs])
        info = {'total_time_used': total_time_used,
                'total_energy_used': total_energy_used,
                'total_finished': total_finished}

        return state, reward, done, info
        #Returns the status, reward value, end flag, and statistics for the current timestep.

    def slot_step(self, time):
        for u in self.UEs:
            if u.is_inferring:
                u.time_used += time
                u.energy_used += u.inference_power * time
                if (u.time_left - time) < 1e-10:
                    u.time_left = 0
                    # -> inferring or free
                    if u.in_local_mode():
                        u.finish_task()
                    # -> offloading
                    elif u.in_mec_mode():
                        u.offloading()
                    else:
                        raise RuntimeError('enter local inference in cloud mode')
                elif u.time_left > time:
                    u.time_left -= time
                else:
                    raise RuntimeError(f'left inference time {u.time_left}s < step time {time}s')

            elif u.is_offloading:
                u.time_used += time
                u.energy_used += u.power * time
                if (u.data_left / u.uplink_rate - time) < 1e-10:
                    # -> inferring, offloading, or free
                    u.data_left = 0
                    u.finish_task()
                elif u.data_left / u.uplink_rate > time:
                    u.data_left -= u.uplink_rate * time
                else:
                    raise RuntimeError(f'left offloading time {u.data_left / u.uplink_rate}s < step time {time}s')
            elif u.is_free:
                pass
            else:
                raise RuntimeError('unknown user state')

    #Calculate the time for the next user device to initiate a task
    def stationary_time(self):
        self.update_uplink_rate()
        min_time = self.slot_time
        for u in self.UEs:
            if u.is_inferring:
                time = u.time_left
            elif u.is_offloading:
                time = u.data_left / u.uplink_rate  # todo: divide by zero?
                # Calculate the time required for the user device to upload the remaining data
            elif u.is_free:
                time = self.slot_time
            else:
                raise RuntimeError('unknown user state')
            if time < min_time:
                min_time = time
        return min_time

    # Update the uplink rate for each channel in the system
    def update_uplink_rate(self):
        for channel in self.channels:
            channel.reset()
        for u in self.UEs:
            if u.is_offloading:
                channel_index = u.channel
                self.channels[channel_index].new_occupation(u)

        for channel in self.channels:
            channel.update_uplink_rate()

    # Assign actions to any user device in the system
    def assign_action(self, action):
        for u, a in zip(self.UEs, action):
            point = a[0]
            channel = a[1]
            power = a[2]
            u.next_point = point
            u.next_channel = channel
            u.power = power

    def is_done(self):
        for u in self.UEs:
            if not u.is_free:
                return False
        return True
