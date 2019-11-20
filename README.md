# Adversarial Inverse Reinforcement Learning implementation for Mountain Car

## Abstract
This project use [Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248) (AIRL) to learn a optimal policy and a optimal reward function for a basic control problem--[Mountain-Car](https://github.com/openai/gym/wiki/MountainCar-v0). It's important to note that **the reward function that AIRL learned can be transferred to learn a optimal policy from zero, while [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476) (GAIL) can't.** The [original implementation](https://github.com/justinjfu/inverse_rl) of AIRL was use **rllab**, which is not maintained anymore. In this work, OpenAI gym environment was used for simplicity.


## Introduction 
AIRL putted two forms of reward function:

1. **State-Action reward function**: $g_{\theta}(s,a)$
2. **State-Only reward function**: $g_{\theta}(s)$

It has been showed in the paper of AIRL that the State-Only reward function is **more easily transferred** to a different dynamics of environment. For example, the State-Only reward function learned in the four legs ant can be used to train a disabled two legs ant while State-Action reward function can't.

Although the dynamics of environment doesn't change in this work, I still used State-Only reward function. 
In mountain car problem, the state(observation) of agent has the car's position and the car's velocity alone the track. 

For more simplicity, and to make the learned reward function can be easily explained, I **just use the car's position as the input of reward function**, and ignore the velocity. This **Partial-State** reward function even simpler than the State-Only reward function.


When it comes to the training process, AIRL provide two ways to get score(the reward feed back to agent) from discriminator:

1. Score is the learned reward function $g_{\theta}(s)$
2. Score is the result of $\log D - log(1-D)$

Both ways worked. From experiments, I found:
- The reward function will become meaningless at the very end if training using  $g_{\theta}(s)$ as score.
- The reward function $g_{\theta}(s)$ is much much more stable if training using  $\log D - log(1-D)$ as score.

**The purpose of inverse reinforcement learning (IRL) should be focused more on getting a robust and stable reward function, rather than just getting a optimal policy, which imitation learning (IL) does.** 

From this point, I chose the $\log D - log(1-D)$ as the score of reinforcement learning (RL) agent. In this case, it should be noted that the learned reward function $g_{\theta}(s)$ is not used in the policy's training process. But using the learned reward function $g_{\theta}(s)$, we can train the optimal policy from zero.


## Experiments
### Get expert demonstrations
I designed a very simple rule to get expert demonstrations. 

**If the speed of the car is negative, the car is running to the left, then the agent will take action
`０`(full throttle reverse), else if the car speed is positive, take action `2`(full throttle forward), 
else take action `1`(zero throttle).**

The code like bellow, and it is efficient to get the expert demonstration.
```python
def get_action(state):
    if state[-1] < 0:
        action = 0 # Left
    elif state[-1] > 0:
        action = 2 # Right
    else:
        action = 1
    return action
```


I used 20, 50, 200 expert demonstrations to get different experimental results, and found that 20 expert demonstrations are totally enough.

### AIRL train process
!['Train Process'](./data/AIRL_MCarV0_Training_Log.png)

Score from discriminator:
- Yellow: The mean return of 20 expert demonstrations;
- Blue: The return of train policy demonstration;

The discriminator is trying to classify samples from experts and from generator, and the generator is trying it's best to generate samples similar to the experts, so that it can get high score from discriminator.

It's clear that in the first 25000 episodes, the discriminator is learning very fast, it can tell expert samples and generator samples very easily.

With the learning process of generator keep going, about 2000 episodes later, the difference between expert return and generator return is getting smaller.

Score from openAI:
- Red: The mean return of 20 expert demonstrations;
- Green: The return of train policy demonstration.

Note: OpenAI give each step a `-1` as score, if `200` steps pass the car can't reach to the destination, this episode end, so that it will get `-200` total scores (return). Only if the car reach the destination within 200 steps, the return will greater than `-200`.

Before 4000 episode, although the generator have learned some, but that was not enough to let the car reach to the destination, so that the green line was keeping at `-200`.

About 5000 episodes latter, the generator was good enough to let the car reach to the destination occasionally. At the same time, some parts of the blue line and the yellow line are overlapped, which indicating that sometimes the generator can perform as better as expert.

Although the mean return of generator can't reach to the expert's. But in our case, we think it was good enough. Because our main purpose was a robust reward function. 

### The learned reward function
The learned reward function $g_{\theta}(s)$ at the very end:
![](./data/23950_learned_rewards.png)

For better explaining the reward, the track of
mountain car problem is as follow:
![](./data/MountainCarTrack.png).

Basically, the reward function was telling us **the car get higher position, it can get higher reward.** This result was reasonable to our intuition.

Besides, the shape of learned reward function doesn't change much as the training the reward, so that a robust reward can be obtained in the end. 


### Use the learned reward function to train a policy
To test the learned reward function was robust and transferable, I fixed the reward function at the very end to train a new policy from zero.
![](./data/AIRL_MCarV0_Training_Log_G.png)

After 7000 episode, the generator's behavior was very close to the expert.


### Conclusion
This work used a simplified version AIRL to solve OpenAI Mountain Car problem, and got a robust and transferable reward function. 

# Appendix A: Some snapshots of learned reward function overtime
Episode 500:
![](./data/500_learned_rewards.png)
Episode 1000:
![](./data/1000_learned_rewards.png)
Episode 2000:
![](./data/2000_learned_rewards.png)
Episode 4000:
![](./data/4000_learned_rewards.png)
Episode 8000:
![](./data/8000_learned_rewards.png)
Episode 16000:
![](./data/16000_learned_rewards.png)
Episode 23000:
![](./data/23000_learned_rewards.png)

