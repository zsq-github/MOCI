import numpy as np

# Behavior of simulated subchannels
class SubChannel:
    # Stores the users currently occupying the subchannel
    def __init__(self, path_loss_exponent, width, noise):
        self.path_loss_exponent = path_loss_exponent
        # Path loss index, which indicates how much signal strength decreases with distance
        self.width = width
        self.noise = noise
        self.occupying_users = []

    # Reset Subchannels
    def reset(self):
        self.occupying_users.clear()

    def power_in_channel(self):
        return sum([u.power / (u.distance ** self.path_loss_exponent) for u in self.occupying_users])

    # Adding a new busy user to a subchannel
    def new_occupation(self, user):
        self.occupying_users.append(user)

    # Calculates the uplink rate for the specified user on the subchannel
    def compute_uplink_rate(self, user):
        # user should be added in occupying_users
        user_power = user.power / (user.distance ** self.path_loss_exponent)
        interference = self.power_in_channel() - user_power
        # Calculate the interference in the channel
        total_noise = interference + self.noise
        # Calculate the total noise in the channel
        return self.width * np.log2(1 + (user_power / total_noise))

    def update_uplink_rate(self):
        for user in self.occupying_users:
            user.uplink_rate = self.compute_uplink_rate(user)
