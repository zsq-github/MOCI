'''
each list contains parameters of performing inference at a special point
    0-intermediate data size: bits
    1-compression rate
    2-front time: s
    3-front power: w
    4-front energy: j

    Assuming system power consumption of 5W at 100% CPU utilization
'''

data = [[[3 * 224 * 224 * 8, 1, 0, 0, 0],  # vgg11
         [64 * 112 * 112 * 32, 128, 0.01839, 2.704, 0.065752],
         [128 * 56 * 56 * 32, 128, 0.045, 2.6344, 0.323688],
         [256 * 28 * 28 * 32, 128, 0.115632, 2.8618, 0.287883],
         [512 * 14 * 14 * 32, 128, 0.160037, 3.0338, 0.415075],
         [0, 1, 0.087236, 2.5795, 0.488299]],
        [[3 * 224 * 224 * 8, 1, 0, 0, 0],  # resnet18
         [64 * 56 * 56 * 32, 64, 0.004359, 2.2885, 0.009976],
         [128 * 28 * 28 * 32, 64, 0.01647, 2.6328, 0.043362],
         [256 * 14 * 14 * 32, 64, 0.025477, 2.8815, 0.073412],
         [512 * 7 * 7 * 32, 32, 0.03678, 3.013, 0.110818],
         [0, 1, 0.048952, 2.1454, 0.098446]],
        [[3 * 224 * 224 * 8, 1, 0, 0, 0],  # mobilenetv2
         [16 * 112 * 112 * 32, 32, 0.005392, 1.8811, 0.010143],
         [24 * 56 * 56 * 32, 48, 0.009008, 2.4805, 0.022344],
         [32 * 28 * 28 * 32, 32, 0.015418, 2.5565, 0.039416],
         [64 * 14 * 14 * 32, 32, 0.024955, 2.2014, 0.054936],
         [0, 1, 0.060152, 1.4797, 0.077945]]]


'''
data = [[[3 * 572 * 572 * 8, 1, 0, 0, 0],  # U-Net
         [64 * 572 * 572 * 32, 64, 0.012345, 1.9876, 0.02468],
         [128 * 286 * 286 * 32, 64, 0.03210, 2.2456, 0.07204],
         [256 * 143 * 143 * 32, 64, 0.05897, 2.5123, 0.14825],
         [512 * 71 * 71 * 32, 64, 0.09382, 2.7654, 0.25817],
         [1024 * 35 * 35 * 32, 64, 0.14852, 3.0123, 0.41325],
         [512 * 71 * 71 * 32, 64, 0.09382, 2.7654, 0.25817],
         [256 * 143 * 143 * 32, 64, 0.05897, 2.5123, 0.14825],
         [128 * 286 * 286 * 32, 64, 0.03210, 2.2456, 0.07204],
         [64 * 572 * 572 * 32, 64, 0.012345, 1.9876, 0.02468],
         [0, 1, 0.01823, 1.8421, 0.03345]]]

'''
import matplotlib.pyplot as plt

def get_data(model, point):
    global data
    if model == 'vgg11':
        d = data[0]
    elif model == 'resnet18':
        d = data[1]
    elif model == 'mobilenetv2':
        d = data[2]
    else:
        raise NotImplementedError
    params = d[point]
    data_size = round(params[0] / params[1])
    latency = params[2]
    power = params[3]
    energy = params[4]
    return data_size, latency, energy, power

if __name__ == '__main__':
    import numpy as np

    def compute_uplink_rate(power):
        # user should be added in occupying_users（channel.py）
        user_power = power / (50 ** 3)
        total_noise = 1e-9
        return 1e6 * np.log2(1 + (user_power / total_noise))

    #+++++++++++++
    latency_data = []
    energy_data = []
    #++++++++++++++

    power = 5
    for p in range(6):
        data_size, latency, energy,_ = get_data('vgg11', p)
        #print(data_size, latency)
        time_offloading = data_size / compute_uplink_rate(power)
        energy_offloading = time_offloading * power
        latency += time_offloading
        energy += energy_offloading
        print(f'point: {p}, time: {latency:.6f}s (offloading: {time_offloading:.6f}s), energy: {energy:.6f}j (offloading: {energy_offloading:.6f}j)')
        #Returns delay and power information for each partition point.
        #+++++++++++++
        latency_data.append(latency)
        energy_data.append(energy)
    # +++++++++++++
    plt.plot(range(6), latency_data, label='Latency')
    plt.plot(range(6), energy_data, label='Energy')
    plt.xlabel('Point')
    plt.ylabel('Value')
    plt.title('Latency and Energy Consumption vs. Point')
    plt.legend()
    plt.grid(True)
    plt.savefig('latency_energy_plot.png')
    plt.show()


'''
resnet18
point: 0, time: 0.078771s (offloading: 0.078771s), energy: 0.393853j (offloading: 0.393853j)
point: 1, time: 0.010923s (offloading: 0.006564s), energy: 0.042797j (offloading: 0.032821j)
point: 2, time: 0.019752s (offloading: 0.003282s), energy: 0.059773j (offloading: 0.016411j)
point: 3, time: 0.027118s (offloading: 0.001641s), energy: 0.081617j (offloading: 0.008205j)
point: 4, time: 0.038421s (offloading: 0.001641s), energy: 0.119023j (offloading: 0.008205j)
point: 5, time: 0.045887s (offloading: 0.000000s), energy: 0.098446j (offloading: 0.000000j)


vgg11
point: 0, time: 0.078771s (offloading: 0.078771s), energy: 0.393853j (offloading: 0.393853j)
point: 1, time: 0.028717s (offloading: 0.013128s), energy: 0.100000j (offloading: 0.065642j)
point: 2, time: 0.051564s (offloading: 0.006564s), energy: 0.151369j (offloading: 0.032821j)
point: 3, time: 0.103877s (offloading: 0.003282s), energy: 0.304294j (offloading: 0.016411j)
point: 4, time: 0.138458s (offloading: 0.001641s), energy: 0.423280j (offloading: 0.008205j)
point: 5, time: 0.189300s (offloading: 0.000000s), energy: 0.488299j (offloading: 0.000000j)

'''