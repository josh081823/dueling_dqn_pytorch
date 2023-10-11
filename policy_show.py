import torch
from DQN_model import DQNAgent

if __name__ == '__main__':

    parameters = {}
    parameters['net_type'] = 'DuelingDQN'
    parameters['hidden_dim'] = 256
    parameters['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameters['buffer_type'] = 'ReplayBuffer'
    parameters['ALPHA'] = 0.5
    parameters['LR'] = 1e-4
    parameters['BATCH_SIZE'] = 64
    parameters['GAMMA'] = 0.99
    parameters['EPS_START'] = 0.9
    parameters['EPS_END'] = 0.05
    parameters['EPS_DECAY'] = 1000
    parameters['TAU'] = 0.005
    parameters['BETA'] = 0.4
    parameters['game_type'] = 'MountainCar-v0'
    parameters['num_episodes'] = 1500
    parameters['log_file'] = './curve.csv'

    agent = DQNAgent(parameters)
    agent.train()
    # agent.show_curve()
    '''
    load_model_name = "./checkpoint/MC_DQNc_d64_0.5_0.4_89_e488.pkl"
    agent.render_result(load_model_name)
    '''