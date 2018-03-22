import glob
import os
import json
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import pickle
import numpy as np


class QFunction(chainer.Chain):
    def __init__(self, n_actions=2):
        super().__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(6, 64, ksize=(4, 1), stride=(2, 1))
            self.c1 = L.Convolution2D(64, 128, ksize=(4, 1), stride=(2, 1))
            self.c2 = L.Convolution2D(128, 256, ksize=(4, 1), stride=(2, 1))
            self.c3 = L.Convolution2D(256, 256, ksize=(3, 1), stride=(1, 1))
            self.c4 = L.Convolution2D(256, 256, ksize=(3, 1), stride=(1, 1))
            self.fc1 = L.Linear(512 + 384, 512)
            self.fc2 = L.Linear(512, n_actions)

    def __call__(self, x):
        batchsize = x.shape[0]
        x, y = F.split_axis(x, [1], axis=3)
        y = F.reshape(y, (batchsize, -1))
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        h = F.relu(self.c4(h))
        h = F.reshape(h, (batchsize, -1))
        h = F.concat((h, y), axis=1)
        h = F.relu(self.fc1(h))
        return chainerrl.action_value.DiscreteActionValue(self.fc2(h))

 
def random_sample():
    return np.random.randint(2) 


def data_generator(data, period, base_date=64):
    start_date = np.random.randint(base_date, len(data) - period - 1)
    x = data[start_date - base_date:start_date + period + 1]
    
    d0 = np.mean(x[:,0:5], axis=1)
    d0 = np.stack([d0] * 5, axis=1)
    d1 = x[:,5:]
    d = np.concatenate((d0, d1), axis=1)

    mean = d.mean(axis=0)
    std = np.sqrt(np.mean((d - mean[np.newaxis,:]) ** 2, axis=0))

    return x, mean, std


class Environment():
    
    base_date = 64
    initial_asset = 1000.0
    
    def __init__(self, data):
        self.data = data
    
    @property
    def state(self):
        return self.data[self.date]

    @property
    def budget(self):
        return self.money
   
    @property
    def price(self):
        return self.state[0]

    @property
    def asset(self):
        return self.money + self.stock * self.price

    @property
    def revenue(self):
        return self.asset - self.initial_asset

    def step(self):
        self.date += 1
    
    def do(self, a):
        price = self.price
        done = False
        obs = self.observation()

        reward = 0
        if a == 1:
            if self.stock == 0:
                nb_stock = int(self.budget / price)
                self.stock += nb_stock
                value = nb_stock * price
                self.money -= value
            else:
                value = self.stock * price
                self.money += value
                self.stock = 0   
            reward = -value / 100
         
        # done
        done = False
        reward += (self.asset - self.prev_asset)
        
        self.prev_asset = self.asset
            
        return obs, reward, done
    
   
    def reset(self):
        self.date = self.base_date
        self.money = self.initial_asset
        self.stock = 0
        self.prev_asset = self.asset
        return self.observation()
    
    def observation(self):
        return self.data[self.date-self.base_date:self.date]


def load_data(filename, start, end):
    with open(filename, 'rb') as f:
        df = pickle.load(f)

    data = df.to_dict('records')

    d = [[d['Open'], d['Close'], d['High'], d['Low'], d['Adj Close'], d['Volume']] for d in data]
    x = np.array(d, dtype=np.float32)
    x = x[start:end]

    return x


def state_to_input(obs, mean, std):
    x = np.asarray(obs)
    x = (x - mean[np.newaxis,:]) / std[np.newaxis,:]
    x = np.transpose(x, (1, 0))
    x = x[:,:,np.newaxis]
    return x 


def to_state(obs, action, stock):
    def to_dict(o):
        return {
            'Open': o[0], 
            'Close': o[1], 
            'High': o[2], 
            'Low': o[3], 
            'Adj Close': o[4], 
            'Volume': o[5], 
        }
    ret = to_dict(obs)
    ret['Action'] = action
    ret['Stock'] = stock
    return ret

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--episode', '-e', type=int, default=30000, help='Number of episode')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    train_dataset = []
    filelist = glob.glob('data/*.pickle')
    for f in filelist:
        data = load_data(f, 0, -400)
        train_dataset.append(data)

    filelist = glob.glob('data/SNE.pickle')
    for f in filelist:
        pre_data = load_data(f, -720, -360)
        data = load_data(f, -360, -180)
        
        _, mean, std = data_generator(pre_data, len(pre_data) - 2, base_date=0)
        test_dataset = (data, mean, std)
        break
   
    q_func = QFunction(2)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        q_func.to_gpu()

    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)

    # Set the discount factor that discounts future rewards.
    gamma = 0.99

    # Use epsilon-greedy for exploration
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.2, random_action_func=random_sample)

    # DQN uses Experience Replay.
    # Specify a replay buffer and its capacity.
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    # Now create an agent that will interact with the environment.
    agent = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_interval=1,
        minibatch_size=128, target_update_interval=100)
    
    max_episode_len = 50

    os.makedirs('output', exist_ok=True)
    result = []

    for i in range(1, args.episode + 1):
        train_data = train_dataset[np.random.randint(len(train_dataset))]
        train_data, train_mean, train_std = data_generator(train_data, max_episode_len)
        train_env = Environment(train_data)

        obs = train_env.reset()
        reward = 0
        done = False
        t = 0
        R = 0
        log = []

        while not done and t < max_episode_len:
            train_env.step()
            x = state_to_input(obs, mean, std)
            s = train_env.stock > 0
            y = 2 * (np.ones_like(x) * s).astype(np.float32) - 1
            z = np.concatenate((x, y), axis=2)
 
            action = agent.act_and_train(z, reward)
            obs, reward, done = train_env.do(action)
            log.append(to_state(obs[-1], action, s))

            t += 1
            R += reward
        
        x = state_to_input(obs, train_mean, train_std)
        y = 2 * (np.ones_like(x) * train_env.stock > 0).astype(np.float32) - 1
        z = np.concatenate((x, y), axis=2)
        agent.stop_episode_and_train(z, reward, done)
        
        _result = {'episode': i, 'train/R': R, 'train/asset': train_env.asset}
        _result.update(agent.get_statistics())

        if i % 10 == 0:
            print(_result)
            result.append(_result)
           
            if i % 100 == 0:
                log_name = 'output/train_{0:08d}.pickle'.format(i)
                with open(log_name, 'wb') as f:
                    pickle.dump(log, f)

            # test
            test_data, test_mean, test_std = test_dataset
            test_episode_len = len(test_data) - Environment.base_date - 1
            test_env = Environment(test_data)
            
            obs = test_env.reset()
            reward = 0
            done = False
            t = 0
            R = 0
            log = []

            while not done and t < test_episode_len:
                test_env.step()
                x = state_to_input(obs, mean, std)
                s = test_env.stock > 0
                y = 2 * (np.ones_like(x) * s).astype(np.float32) - 1
                z = np.concatenate((x, y), axis=2)
 
                action = agent.act(z)
                obs, reward, done = test_env.do(action)
                log.append(to_state(obs[-1], action, s))

                t += 1
                R += reward
            
            x = state_to_input(obs, test_mean, test_std)
            y = 2 * (np.ones_like(x) * test_env.stock > 0).astype(np.float32) - 1
            z = np.concatenate((x, y), axis=2)
            agent.stop_episode()
           
            if i % 100 == 0:
                log_name = 'output/test_{0:08d}.pickle'.format(i)
                with open(log_name, 'wb') as f:
                    pickle.dump(log, f)
           
            _result = {'episode': i, 'validation/R': R, 'validation/asset': test_env.asset}
            _result.update(agent.get_statistics())
            print(_result) 
            result.append(_result)
           
            if i % 100 == 0:
                with open('output/log.json', 'w') as f:
                    json.dump(result, f, indent=4)


    print('Finished.')
    
    
if __name__ == '__main__':
    main()
