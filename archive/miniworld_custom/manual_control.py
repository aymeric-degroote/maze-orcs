import math

import pyglet
from pyglet.window import key
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle


class ManualControl:
    def __init__(self, env, no_time_limit, domain_rand):
        self.env = env

        if no_time_limit:
            self.env.max_episode_steps = math.inf
        if domain_rand:
            self.env.domain_rand = True

    def run(self, env_name):
        print("============")
        print("Instructions")
        print("============")
        print("move: arrow keys\npickup: P\ndrop: D\ndone: ENTER\nquit: ESC")
        print("============")

        obs = self.env.reset(seed=0)[0]
        initial_obs = self.env.reset(seed = 0)[0]

        # Create the display window
        self.env.render()

        env = self.env

        class Policy(object):

            def __init__(self, obssize, actsize, lr):
                """
                obssize: size of the states
                actsize: size of the actions
                """
                # TODO DEFINE THE MODEL
                self.model = torch.nn.Sequential(
                            torch.nn.Linear(obssize, 100),
                            torch.nn.ReLU(),
                            torch.nn.Linear(100, 100),
                            torch.nn.ReLU(),
                            torch.nn.Linear(100,actsize)
                        )

                # DEFINE THE OPTIMIZER
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

                # RECORD HYPER-PARAMS
                self.obssize = obssize
                self.actsize = actsize

                # TEST
                self.compute_prob(np.random.randn(obssize).reshape(1, -1))

            def compute_prob(self, states):
                """
                compute prob distribution over all actions given state: pi(s)
                states: numpy array of size [numsamples, obssize]
                return: numpy array of size [numsamples, actsize]
                """
                states = torch.FloatTensor(states)
                prob = torch.nn.functional.softmax(self.model(states), dim=-1)
                return prob.cpu().data.numpy()

            def _to_one_hot(self, y, num_classes):
                """
                convert an integer vector y into one-hot representation
                """
                scatter_dim = len(y.size())
                y_tensor = y.view(*y.size(), -1)
                zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
                return zeros.scatter(scatter_dim, y_tensor, 1)

            def train(self, states, actions, Qs):
                """
                states: numpy array (states)
                actions: numpy array (actions)
                Qs: numpy array (Q values)
                """
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                Qs = torch.FloatTensor(Qs)

                # COMPUTE probability vector pi(s) for all s in states
                logits = self.model(states)
                prob = torch.nn.functional.softmax(logits, dim=-1)

                # Compute probaility pi(s,a) for all s,a
                action_onehot = self._to_one_hot(actions, actsize)
                prob_selected = torch.sum(prob * action_onehot, axis=-1)

                # FOR ROBUSTNESS
                prob_selected += 1e-8

                # TODO define loss function as described in the text above
                loss = -1*torch.mean(torch.mul(Qs,torch.log(prob_selected)))

                # BACKWARD PASS
                self.optimizer.zero_grad()
                loss.backward()

                # UPDATE
                self.optimizer.step()

                return loss.detach().cpu().data.numpy()

        class ValueFunction(object):

            def __init__(self, obssize, lr):
                """
                obssize: size of states
                """
                # TODO DEFINE THE MODEL
                self.model = torch.nn.Sequential(
                            #TODO
                            torch.nn.Linear(obssize, 50),
                            torch.nn.Sigmoid(),
                            torch.nn.Linear(50,1)
                        )

                # DEFINE THE OPTIMIZER
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

                # RECORD HYPER-PARAMS
                self.obssize = obssize
                self.actsize = actsize

                # TEST
                self.compute_values(np.random.randn(obssize).reshape(1, -1))

            def compute_values(self, states):
                """
                compute value function for given states
                states: numpy array of size [numsamples, obssize]
                return: numpy array of size [numsamples]
                """
                states = torch.FloatTensor(states)
                return self.model(states).cpu().data.numpy()

            def train(self, states, targets):
                """
                states: numpy array
                targets: numpy array
                """
                states = torch.FloatTensor(states)
                targets = torch.FloatTensor(targets)

                # COMPUTE Value PREDICTIONS for states
                v_preds = self.model(states)

                # LOSS
                # TODO: set LOSS as square error of predicted values compared to targets
                loss = torch.mean((v_preds - targets)**2)

                # BACKWARD PASS
                self.optimizer.zero_grad()
                loss.backward()

                # UPDATE
                self.optimizer.step()

                return loss.detach().cpu().data.numpy()

        def discounted_rewards(r, gamma):
            """ take 1D float array of rewards and compute discounted reward """
            discounted_r = np.zeros_like(r)
            running_sum = 0
            for i in reversed(range(0,len(r))):
                discounted_r[i] = running_sum * gamma + r[i]
                running_sum = discounted_r[i]
            return list(discounted_r)
        

        # parameter initializations (you can change any of these)
        alpha = 1e-3  # learning rate for PG
        beta = 1e-3  # learning rate for baseline
        numtrajs = 5  # num of trajecories from the current policy to collect in each iteration
        iterations = 100  # total num of iterations
        gamma = .99  # discount
        total_runs = 0

        # initialize environment
        obssize = 14400
        actsize = 3

        # initialize networks
        actor = Policy(obssize, actsize, alpha)  # policy initialization: IMPORTANT: this is the policy you will be scored on
        baseline = ValueFunction(obssize, beta)  # baseline initialization

        #To record training reward for logging and plotting purposes
        rrecord = []
        movingAverage=[]

        # main iteration
        for ite in range(iterations):

            # To record traectories generated from current policy
            OBS = []  # observations
            ACTS = []  # actions
            ADS = []  # advantages (to compute policy gradient)
            VAL = []  # Monte carlo value predictions (to compute baseline, and policy gradient)

            for num in range(numtrajs):
                # To keep a record of states actions and reward for each episode
                obss = []  # states
                acts = []   # actions
                rews = []  # instant rewards

                obs = self.env.reset(seed=0)[0]
                done = False
                truncation = False

                # TODO: run one episode using the current policy "actor"
                while not done and not truncation:
                    total_runs = total_runs+1
                    prob = actor.compute_prob(np.expand_dims(obs.flatten(),0))
                    prob /= np.sum(prob) #normalizing again to account for numerical errors
                    print (prob)
                    print(total_runs)
                    explore_probability=((1-(ite/iterations))*1)
                    epsilon=np.random.binomial(n=1, p= explore_probability)

                    if epsilon==1:
                        action = np.random.randint(0,3)
                        print ("random")
                    else:
                        action = np.random.choice(actsize, p=prob.flatten(), size=1).item() #choose according distribution prob
                        print ("chosen")
                    
                    newobs, reward, done, truncation = self.step(action)
                    # i=i+1

                    # TODO: record all observations (states, actions, rewards) from the epsiode in  obss, acts, rews
                    obss.append(obs.flatten())
                    acts.append(action)
                    if not done:
                            rews.append(reward)
                    else:
                            rews.append(100.0)
                    obs = newobs

                #Below is for logging training performance
                rrecord.append(np.sum(rews))

                # TODO:  Use discounted_rewards function to compute \hat{V}s/\hat{Q}s  from instant rewards in rews
                Vhat = discounted_rewards(rews, gamma)
                # TODO: record the computed \hat{V}s in VAL, states obss in OBS, and actions acts in ACTS, for batch update
                OBS += obss
                ACTS += acts
                VAL += Vhat

            # AFTER collecting numtrajs trajectories:
            VAL = np.array(VAL)
            OBS = np.array(OBS)
            ACTS = np.array(ACTS)

            #1. TODO: train baseline
            """
                Use the batch (OBS, VAL) of states and value predictions as targets to train baseline.
                Use baseline.train : note that this takes as input numpy array, so you may have to convert
                lists into numpy array using np.array()
            """
            baseline.train(OBS, VAL)

            # 2.TODO: Update the policy
            """
                Compute baselines: use basline.compute_values for states in the batch OBS
                Compute advantages ADS using VAL and computed baselines
                Update policy using actor.train using OBS, ACTS and ADS
            """
            BAS = baseline.compute_values(OBS)  # compute baseline for variance reduction
            ADS = VAL - np.squeeze(BAS,1)
            actor.train(OBS, ACTS, ADS)


            fixedWindow=10
            if len(rrecord) >= fixedWindow:
                movingAverage.append(np.mean(rrecord[len(rrecord)-fixedWindow:len(rrecord)-1]))

            plt.plot(movingAverage)
            plt.savefig(f"movingAverage_{env_name}.png")

    self.env.close()
    
    with open('rrecord_1.txt', 'w') as file:
        for item in rrecord:
            file.write(str(item) + '\n')

    with open('rews_1.txt', 'w') as file:
        for item in rews:
            file.write(str(item) + '\n')      
    
    with open('actor_1.pkl', 'wb') as file:
        pickle.dump(actor.model, file)

    with open('movingAverage_1.txt', 'w') as file:
        for item in movingAverage:
            file.write(str(item) + '\n')

        

    def step(self, action):
        print(
            "step {}/{}: {}".format(
                self.env.step_count + 1,
                self.env.max_episode_steps,
                self.env.actions(action).name,
            )
        )

        obs, reward, termination, truncation, info = self.env.step(action)
        print(f"reward={reward:.2f}")
        # print(obs.shape)

        if reward > 0:
            print(f"reward={reward:.2f}")

        if termination or truncation:
        # if termination:
            print("done!")
            self.env.reset(seed=0)

        self.env.render()
        return(obs,reward,termination, truncation)
