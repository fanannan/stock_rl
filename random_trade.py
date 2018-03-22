import glob
import os
import json
import argparse
import pickle
import numpy as np

 
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
    filelist = glob.glob('data/SNE.pickle')
    for f in filelist:
        pre_data = load_data(f, -720, -360)
        data = load_data(f, -360, -180)
        
        _, mean, std = data_generator(pre_data, len(pre_data) - 2, base_date=0)
        test_dataset = (data, mean, std)
        break
   
    max_episode_len = 50

    os.makedirs('output_random', exist_ok=True)
    result = []

    for i in range(1000):
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
 
            action = random_sample()
            obs, reward, done = test_env.do(action)
            log.append(to_state(obs[-1], action, s))

            t += 1
            R += reward
        
        x = state_to_input(obs, test_mean, test_std)
        y = 2 * (np.ones_like(x) * test_env.stock > 0).astype(np.float32) - 1
        z = np.concatenate((x, y), axis=2)
    
        if i % 100 == 0:
            log_name = 'output_random/test_{0:08d}.pickle'.format(i)
            with open(log_name, 'wb') as f:
                pickle.dump(log, f)
        
        _result = {'episode': i, 'validation/R': R, 'validation/asset': test_env.asset}
        print(_result) 
        result.append(_result)
        
    with open('output_random/log.json', 'w') as f:
        json.dump(result, f, indent=4)


    print('Finished.')
    
    
if __name__ == '__main__':
    main()
