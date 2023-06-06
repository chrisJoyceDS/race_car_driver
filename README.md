# race_car_driver
Using PyTorch, Gym, Visdom, and reinforcement learning to teach a car how to self-drive a track


Deep Learning Fall backs:
- Actor Critic methods are sensitive to perturbations/small changes in the environment
    - Models can simply transition from working to not working

PPO Models:
- Limits update to policy network
    - base the update on the ratio of new policy to old policy
    - have to account for goodness of state (advantage)
    - clip loss function and take lower bound with min
- Keeps track of a fixed length trajectory of memories
- multiple network updates per data sample
    - minibatch stochastic gradient descent
- can also use multiple parallel actors (CPUS)

multibatch stochastic gradient descent:
- Memory indices = [0,1,2,...,19]
- Batches start at multiples of batch_size [0,5,10,15]
    - Shuffle memories then take batch size chunks

Other:
- two distinct networks instead of shared inputs for actor and critic
    - Critic evaluates states the agent encounters
    - Actor evaluates based on its current state
        - Network outputs probs (softmax) for a distribution
        - Exploration due to nature of distribution, probabilistic
- Memory is fixed to length T (say, 20 steps)
- Track states, actions, rewards, dones, values, log probs
- Shuffle memories and sample batch size of 5
- Perform 4 epochs of updates on each batch
