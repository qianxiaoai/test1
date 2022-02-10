Robust Adversarial Reinforcement Learning 

鲁棒对抗强化学习



 Lerrel Pinto 1 James Davidson 2

Rahul Sukthankar 3 Abhinav Gupta 1 3



## Abstract 



Deep neural networks coupled with fast simula- tion and improved computation have led to re- cent successes in the field of reinforcement learn- ing (RL). However, most current RL-based ap- proaches fail to generalize since: (a) the gap be- tween simulation and real world is so large that policy-learning approaches fail to transfer; (b) even if policy learning is done in real world, the data scarcity leads to failed generalization from training to test scenarios (e.g., due to different friction or object masses). Inspired from H∞ control methods, we note that both modeling er- rors and differences in training and test scenar- ios can be viewed as extra forces/disturbances in the system. This paper proposes the idea of ro- bust adversarial reinforcement learning (RARL), where we train an agent to operate in the pres- ence of a destabilizing adversary that applies dis- turbance forces to the system. The jointly trained adversary is reinforced – that is, it learns an op- timal destabilization policy. We formulate the policy learning as a zero-sum, minimax objec- tive function. Extensive experiments in multiple environments (InvertedPendulum, HalfCheetah, Swimmer, Hopper and Walker2d) conclusively demonstrate that our method (a) improves train- ing stability; (b) is robust to differences in train- ing/test conditions; and c) outperform the base- line even in the absence of the adversary. 

 

摘要

深度神经网络结合快速模拟和改进的计算，在强化学习(RL)领域取得了最近的成功。然而，目前大多数基于rl的方法未能得到推广，因为:(a)仿真和现实世界之间的差距太大，策略学习方法无法迁移;(b)即使政策学习是在现实世界中进行的，数据缺乏导致从训练到测试场景的归纳失败(例如，由于摩擦或物体质量不同)。受H∞控制方法的启发，我们注意到建模误差和训练和测试场景的差异都可以视为系统中的额外力/干扰。本文提出了突破对抗强化学习(RARL)的思想，其中我们训练一个代理在不稳定的对手存在时操作，该对手对系统施加干扰力量。联合训练的对手得到加强——也就是说，它学会了一种最佳的破坏稳定政策。我们将策略学习定义为一个零和、极大极小目标函数。在多种环境下(InvertedPendulum, HalfCheetah, Swimmer, Hopper和Walker2d)的大量实验最终证明了我们的方法(a)提高了训练的稳定性;(b)对训练/测试条件的差异具有稳健性;c)即使在没有对手的情况下也能超越底线。

 

##  1. Introduction 



High-capacity function approximators such as deep neu- ral networks have led to increased success in the field of reinforcement learning (Mnih et al., 2015; Silver et al., 2016; Gu et al., 2016; Lillicrap et al., 2015; Mordatch et al., 2015). However, a major bottleneck for such policy-learning methods is their reliance on data: train- ing high-capacity models requires huge amounts of training data/trajectories. While this training data can be easily obtained for tasks like games (e.g., Doom, Montezuma’s Revenge) (Mnih et al., 2015), data-collection and policy learning for real-world physical tasks are significantly more challenging. 

There are two possible ways to perform policy learning for real-world physical tasks: 

• Real-world Policy Learning: The first approach is to learn the agent’s policy in the real-world. However, training in the real-world is too expensive, dangerous and time-intensive leading to scarcity of data. Due to scarcity of data, training is often restricted to a limited set of training scenarios, causing overfitting. If the test scenario is different (e.g., different friction coef- ficient), the learned policy fails to generalize. There- fore, we need a learned policy that is robust and gen- eralizes well across a range of scenarios. 

• Learning in simulation: One way of escaping the data scarcity in the real-world is to transfer a policy learned in a simulator to the real world. However the environment and physics of the simulator are not ex- actly the same as the real world. This reality gap often results in unsuccessful transfer if the learned policy isn’t robust to modeling errors (Christiano et al., 2016; Rusu et al., 2016). 

Both the test-generalization and simulation-transfer issues are further exacerbated by the fact that many policy- learning algorithms are stochastic in nature. For many hard physical tasks such as Walker2D (Brockman et al., 2016), only a small fraction of runs leads to stable walking poli- cies. This makes these approaches even more time and data-intensive. What we need is an approach that is signifi- cantly more stable/robust in learning policies across differ- ent runs and initializations while requiring less data during training. 

So, how can we model uncertainties and learn a pol- icy robust to all uncertainties? How can we model the gap between simulations and real-world? We begin with the insight that modeling errors can be viewed as ex- tra forces/disturbances in the system (Bas ̧ar & Bernhard, 2008). For example, high friction at test time might be modeled as extra forces at contact points against the di- rection of motion. Inspired by this observation, this paper proposes the idea of modeling uncertainties via an adver- sarial agent that applies disturbance forces to the system. Moreover, the adversary is reinforced – that is, it learns an optimal policy to thwart the original agent’s goal. Our pro- posed method, Robust Adversarial Reinforcement Learn- ing (RARL), jointly trains a pair of agents, a protagonist and an adversary, where the protagonist learns to fulfil the original task goals while being robust to the disruptions generated by its adversary. 

We perform extensive experiments to evaluate RARL on multiple OpenAI gym environments like InvertedPendu- lum, HalfCheetah, Swimmer, Hopper and Walker2d (see Figure 1). We demonstrate that our proposed approach is: (a) Robust to model initializations: The learned policy performs better given different model parameter initializa- tions and random seeds. This alleviates the data scarcity issue by reducing sensitivity of learning. (b) Robust to modeling errors and uncertainties: The learned policy generalizes significantly better to different test environment settings (e.g., with different mass and friction values). 





## 1. 介绍

深度神经网络等高容量函数逼近器在强化学习领域取得了越来越大的成功(Mnih等人，2015;Silver等人，2016;Gu等人，2016;Lillicrap等人，2015;Mordatch等人，2015)。然而，这种策略学习方法的一个主要瓶颈是对数据的依赖:训练高容量模型需要大量的训练数据/轨迹。虽然在游戏(游戏邦注:如《毁灭战士》、《Montezuma’s Revenge》等)等任务中可以轻松获得这些训练数据，但在现实世界的物理任务中，数据收集和策略学习则更具挑战性。

对于现实世界的物理任务，有两种可能的策略学习方式:

o真实世界的策略学习:第一种方法是在真实世界中学习代理的策略。然而，现实世界的训练过于昂贵、危险和耗时，导致数据匮乏。由于数据的缺乏，训练往往被限制在有限的一组训练场景中，导致过拟合。如果测试场景不同(例如，不同的摩擦系数)，则学习到的策略不能一般化。因此，我们需要一种经过学习的、稳健的、能够很好地适用于一系列场景的政策。

o在模拟中学习:逃避现实世界中数据匮乏的一种方法是将在模拟器中学习到的策略转移到现实世界中。然而，环境和物理模拟器不是完全相同的现实世界。如果学习到的策略对建模错误不够稳健，这种现实差距常常导致迁移不成功(Christiano et al.， 2016;Rusu等人，2016)。

许多策略学习算法本质上都是随机的，这使得测试泛化和模拟转移问题进一步恶化。对于许多高强度的体力活动，如Walker2D (Brockman等人，2016)，只有一小部分的跑步会导致稳定的步行政策。这使得这些方法更加耗时和数据密集。我们需要的是一种方法，在不同的运行和初始化的学习策略中更加稳定/健壮，同时在培训期间需要更少的数据。

那么，我们如何对不确定性建模并学习对所有不确定性具有鲁棒性的策略呢?我们如何模拟模拟和现实世界之间的差距?我们首先认识到，建模误差可以被视为系统中的额外力/干扰(Başar & Bernhard, 2008)。例如，测试时的高摩擦力可以被建模为对抗运动方向的接触点的额外力。受此启发，本文提出了利用对系统施加扰动力的对抗代理来建模不确定性的思想。此外，对手也得到了强化——也就是说，它学会了一种最优的策略，以挫败原代理人的目标。我们提出的方法，鲁棒对抗强化学习(RARL)，联合训练一对代理，一个主角和一个对手，其中主角学习完成最初的任务目标，同时对其对手产生的破坏具有鲁棒性。

我们在多个OpenAI-gym环境(如InvertedPendu- lum、HalfCheetah、Swimmer、Hopper和Walker2d)上进行了广泛的实验来评估RARL(见图1)。我们证明了我们提出的方法是:(a)对模型初始化具有鲁棒性:在给定不同的模型参数初始化和随机种子的情况下，学习策略的性能更好。这降低了学习的敏感性，从而缓解了数据稀缺问题。(b)对建模误差和不确定性的鲁棒性:学习到的策略对不同的测试环境设置(例如，具有不同的质量和摩擦值)具有显著的更好的通用性。

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image001.png)

Figure 1. We evaluate RARL on a variety of OpenAI gym problems. The adversary learns to apply destabilizing forces on specific points (denoted by red arrows) on the system, encouraging the protagonist to learn a robust control policy. These policies also transfer better to new test environments, with different environmental conditions and where the adversary may or may not be present. 

图1所示。我们评估RARL在OpenAI-gym的各种问题。对手学会在系统上的特定点(用红色箭头表示)上应用不稳定的力量，鼓励主角学习健壮的控制策略。这些策略还可以更好地转移到新的测试环境中，这些环境条件不同，对手可能出现，也可能不出现。

 

### 1.1. Overview of RARL 



Our goal is to learn a policy that is robust to modeling er- rors in simulation or mismatch between training and test scenarios. For example, we would like to learn policy for Walker2D that works not only on carpet (training scenario) but also generalizes to walking on ice (test scenario). Simi- larly, other parameters such as the mass of the walker might vary during training and test. One possibility is to list all such parameters (mass, friction etc.) and learn an ensem- ble of policies for different possible variations (Rajeswaran et al., 2016). But explicit consideration of all possible pa- rameters of how simulation and real world might differ or what parameters can change between training/test is infea- sible. 

Our core idea is to model the differences during training and test scenarios via extra forces/disturbances in the sys- tem. Our hypothesis is that if we can learn a policy that is robust to all disturbances, then this policy will be robust to changes in training/test situations; and hence generalize well. But is it possible to sample trajectories under all pos- sible disturbances? In unconstrained scenarios, the space of possible disturbances could be larger than the space of possible actions, which makes sampled trajectories even sparser in the joint space. 

To overcome this problem, we advocate a two-pronged ap- proach: 

(a) Adversarial agents for modeling disturbances: In- stead of sampling all possible disturbances, we jointly train a second agent (termed the adversary), whose goal is to im- pede the original agent (termed the protagonist) by apply- ing destabilizing forces. The adversary is rewarded only for the failure of the protagonist. Therefore, the adversary learns to sample hard examples: disturbances which will make original agent fail; the protagonist learns a policy that is robust to any disturbances created by the adversary. 

(b) Adversaries that incorporate domain knowledge: 

The naive way of developing an adversary would be to sim- ply give it the same action space as the protagonist – like a driving student and driving instructor fighting for control of a dual-control car. However, our proposed approach is much richer and is not limited to symmetric action spaces – we can exploit domain knowledge to: focus the adversary on the protagonist’s weak points; and since the adversary is in a simulated environment, we can give the adversary “super-powers” – the ability to affect the robot or environ- ment in ways the protagonist cannot (e.g., suddenly change a physical parameter like frictional coefficient or mass ).

 

### 1．1. 概述RARL

我们的目标是学习一个策略，是稳健的建模er- errors在模拟或不匹配的训练和测试场景。例如，我们想学习Walker2D的策略，它不仅适用于地毯(训练场景)，也适用于冰上行走(测试场景)。类似地，其他参数，如步行者的质量可能会在训练和测试中发生变化。一种可能性是列出所有这些参数(质量、摩擦等)，并学习针对不同可能变化的一整套政策(Rajeswaran等人，2016)。但是，明确地考虑模拟世界和现实世界可能如何不同，或者训练/测试之间的参数可以改变的所有可能的参数是不可能的。

我们的核心理念是通过系统中的额外力量/干扰来模拟训练和测试场景中的差异。我们的假设是，如果我们能学习到一个对所有干扰都具有鲁棒性的策略，那么这个策略将对训练/测试情境的变化具有鲁棒性;因此可以很好地概括。但是有可能对所有可能的扰动下的轨迹进行采样吗?在无约束的情况下，可能的扰动空间可能大于可能的动作空间，这使得采样轨迹在关节空间中更加稀疏。

为解决这一问题，我们提倡双管齐下:

(a)对干扰进行建模的对抗性代理:我们不是对所有可能的干扰进行采样，而是联合训练第二个代理(称为对手)，它的目标是通过施加不稳定力来阻止原始代理(称为主角)。对手只会因为主角的失败而获得奖励。因此，对手学会了采样困难的例子:干扰将使原始代理失败;主人公学会了一种政策，这种政策对对手制造的任何干扰都是有效的。

(b)包含领域知识的对手:

开发一个对手的简单方法是——给它和主角一样的行动空间——就像一个驾驶学生和驾驶教练为了控制一辆双控车而战斗。然而，我们提出的方法要丰富得多，而且不局限于对称的行动空间——我们可以利用领域知识:让对手关注主角的弱点;因为对手处在一个模拟的环境中，我们可以赋予对手“超能力”——影响机器人或环境的能力——以主角无法做到的方式(例如，突然改变摩擦系数或质量等物理参数)

\2. Background 

Before we delve into the details of RARL, we first out- line our terminology, standard reinforcement learning set- ting and two-player zero-sum games from which our paper is inspired. 

2.1. Standard reinforcement learning on MDPs 

In this paper we examine continuous space MDPs that are represented by the tuple: (S, A, P, r, γ, s0), where S is a set of continuous states and A is a set of continuous actions, P : S×A×S → Risthetransitionprobability,r : S × A → R is the reward function, γ is the discount factor, and s0 is the initial state distribution. 

Batch policy algorithms like (Williams, 1992; Kakade, 

2002; Schulman et al., 2015) attempt to learn a stochas- 

tic policy πθ : S × A → R that maximizes the cumula- 

tive discounted reward 􏰂T −1 γtr(s , a ). Here, θ denotes t=0 tt

the parameters for the policy π which takes action at given state st at timestep t. 

2.2. Two-player zero-sum discounted games 

The adversarial setting we propose can be expressed as a two player γ discounted zero-sum Markov game (Littman, 1994; Perolat et al., 2015). This game MDP can be ex- pressed as the tuple: (S, A1, A2, P, r, γ, s0) where A1 and A2 are the continuous set of actions the players can take. P : S ×A1 ×A2 ×S → R is the transition probability den- sity and r : S ×A1 ×A2 → R is the reward of both players. If player 1 (protagonist) is playing strategy μ and player 2 (adversary) is playing the strategy ν, the reward function is rμ,ν = Ea1∼μ(.|s),a2∼ν(.|s)[r(s, a1, a2)]. A zero-sum two-player game can be seen as player 1 maximizing the γ discounted reward while player 2 is minimizing it. 

 

## 2. 背景

在我们深入研究RARL的细节之前，我们首先列出了我们的术语、标准强化学习设置和我们的论文的灵感来自于二人零和游戏。

２．１． MDPs上的标准强化学习

在本文中,我们检查连续空间mdp所代表的元组:(S, P, r,γ,s0), S是一组连续状态和一个是一组连续的动作,P: S××S→Risthetransitionprobability, r: S×→r是奖励功能,γ是折扣因素,s0初始状态分布。

批处理策略算法如(Williams, 1992;祷告,

2002;Schulman等人，2015)尝试学习一个随机-

tic策略πθ: S × A→R使累积-最大化

\- 1 γtr(s, a)。这里θ表示t= 0tt

策略π在时间步t的给定状态st处采取行动的参数。

2．2． 2人零和折扣游戏

我们所提出的对抗设置可以表示为一个双参与人γ贴现零和马尔科夫博弈(Littman, 1994;Perolat等人，2015)。这个游戏MDP可以表示为元组:(S, A1, A2, P, r， γ， s0)，其中A1和A2是玩家可以采取的连续动作集。P: S ×A1 ×A2 ×S→R是转移概率密度，R: S ×A1 ×A2→R是双方玩家的奖励。如果玩家1(主角)执行策略μ，而玩家2(对手)执行策略ν，则奖励函数是rμ，ν = Ea1 ~ ν(.|s)，a2 ~ ν(.|s)[r(s, a1, a2)]。零和博弈可以看作是参与人1最大化γ折扣奖励，而参与人2最小化它。

 

\3. Robust Adversarial RL 

3.1. Robust Control via Adversarial Agents 

Our goal is to learn the policy of the protagonist (denoted by μ) such that it is better (higher reward) and robust (gen- eralizes better to variations in test settings). In the standard reinforcement learning setting, for a given transition function P, we can learn policy parameters θμ such that the expected reward is maximized where expected reward for policy μ from the start s0 is (xx1xx) .

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image002.png)

 

Note that in this formulation the expected reward is conditioned on the transition function since the the transition function defines the roll-out of states. In standard-RL set- tings, the transition function is fixed (since the physics en-gine and parameters such as mass, friction are fixed). However, in our setting, we assume that the transition function will have modeling errors and that there will be differences between training and test conditions. Therefore, in our gen- eral setting, we should estimate policy parameters θμ such that we maximize the expected reward over different possible transition functions as well. Therefore, (xx2xx)

 ![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image003.png)

Optimizing for the expected reward over *all* transition functions optimizes *mean performance*, which is a risk neutral formulation that assumes a known distribution over model parameters. A large fraction of policies learned under such a formulation are likely to fail in a different environment. Instead, inspired by work in robust control (Tamar et al., 2014; Rajeswaran et al., 2016), we choose to optimize for conditional value at risk (CVaR): (xx3xx)

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image004.png)

where Qα(ρ) is the α-quantile of ρ-values. Intuitively, in robust control, we want to maximize the worst-possible ρ- values. But how do you tractably sample trajectories that are in worst α-percentile? Approaches like EP-Opt (Ra- jeswaran et al., 2016) sample these worst percentile trajec- tories by changing parameters such as friction, mass of ob- jects, etc. during rollouts. 

Instead, we introduce an adversarial agent that applies forces on pre-defined locations, and this agent tries to change the trajectories such that reward of the protago- nist is minimized. Note that since the adversary tries to minimize the protagonist’s reward, it ends up sampling tra- jectories from worst-percentile leading to robust control- learning for the protagonist. If the adversary is kept fixed, the protagonist could learn to overfit to its adversarial ac- tions. Therefore, instead of using either a random or a fixed-adversary, we advocate generating the adversarial ac- tions using a learned policy ν. We would also like to point out the connection between our proposed approach and the practice of hard-example mining (Sung & Poggio, 1994; Shrivastava et al., 2016). The adversary in RARL learns to sample hard-examples (worst-trajectories) for the protag- onist to learn. Finally, instead of using α as percentile- parameter, RARL is parameterized by the magnitude of force available to the adversary. As the adversary becomes stronger, RARL optimizes for lower percentiles. However, very high magnitude forces lead to very biased sampling and make the learning unstable. In the extreme case, an un- reasonably strong adversary can always prevent the protag- onist from achieving the task. Analogously, the traditional RL baseline is equivalent to training with an impotent (zero strength) adversary. 

 

3.强大的敌对的RL

３.１. 通过对抗代理的鲁棒控制

我们的目标是学习主角的策略(用μ表示)，使其更好(更高的奖励)和健壮(更好地推广到测试设置的变化)。在标准的强化学习设置中，对于给定的过渡函数P，我们可以学习策略参数θμ，使策略μ从s0开始的期望报酬最大化，其中策略μ的期望报酬为(xx1xx)。

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image002.png)

 

注意，在这个公式中，期望的回报是基于转换函数的，因为转换函数定义了滚出状态。在标准rl设置中，过渡函数是固定的(因为物理引擎和质量、摩擦等参数是固定的)。然而，在我们的设置中，我们假设过渡函数会有建模误差，并且在训练和测试条件之间会有差异。因此，在我们的一般设置中，我们应该估计政策参数θμ，以便在不同可能的过渡函数上也使期望报酬最大化。因此,(xx2xx)

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image003.png)

对所有过渡函数的预期报酬进行优化可优化平均性能，这是一个假设模型参数分布已知的风险中性公式。在这种构想下学到的政策，有很大一部分很可能在不同的环境中失败。相反，灵感来自于鲁棒控制(Tamar et al.， 2014;Rajeswaran et al.， 2016)，我们选择优化风险条件价值(CVaR):(xx3xx)

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image004.png)

其中Qα(ρ)是ρ值的α分位数。直观地说，在鲁棒控制中，我们想使最坏的可能ρ-值最大化。但是如何对处于最差α-百分位的轨迹进行可追踪采样?像EP-Opt (Ra- jeswaran et al.， 2016)这样的方法通过在推出过程中改变摩擦、物体质量等参数来对这些最糟糕的百分位轨迹进行抽样。

相反，我们引入了一个对抗性代理，它在预先定义的位置施加力量，并且这个代理试图改变轨迹，从而使主角的奖励最小化。需要注意的是，因为对手试图最小化主角的奖励，所以它最终会从最糟糕的百分比中采样轨迹，从而导致主角的鲁棒控制学习。如果对手是固定的，主角可以学会过度适应它的对抗性行动。因此，我们主张使用学习到的策略ν来生成对抗性交互，而不是使用随机的或固定的对手。我们还想指出我们所建议的方法与硬例采矿的实践之间的联系(Sung & Poggio, 1994;Shrivastava等人，2016)。RARL中的对手学会了采样难的例子(最坏的轨迹)，让主控者去学习。最后，RARL不是使用α作为百分位参数，而是通过对手可用的力量的大小来参数化。当对手变得更强大时，RARL会针对更低的百分比进行优化。然而，非常大的力导致非常偏置的采样，使学习不稳定。在极端情况下，一个不合理的强大对手总能阻止主角完成任务。类似地，传统的RL基线相当于训练一个无能(零力量)的对手。

 

3.2. Formulating Adversarial Reinforcement Learning 

In our adversarial game, at every timestep t both play- ers observe the state st and take actions a1t ∼ μ(st) and a2t ∼ ν(st). The state transitions st+1 = P(st, a1t , a2t ) and a reward rt = r(st, a1t , a2t ) is obtained from the en- vironment. In our zero-sum game, the protagonist gets a reward rt1 = rt while the adversary gets a reward rt2 = −rt. Hence each step of this MDP can be represented as (st, a1t , a2t , rt1, rt2, st+1). 

The protagonist seeks to maximize the following reward function, (xx4xx)

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image005.png)

Since, the policies μ and ν are the only learnable components, R ≡ R (μ, ν). Similarly the adversary attempts to maximize its own reward: R2 ≡ R2(μ, ν) = −R1(μ, ν). One way to solve this MDP game is by discretizing the continuous state and action spaces and using dynamic pro- gramming to solve. (Perolat et al., 2015; Patek, 1997) show that notions of minimax equilibrium and Nash equilibrium are equivalent for this game with optimal equilibrium re- ward: (xx5xx)

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image006.png)

However solutions to finding the Nash equilibria strategies often involve greedily solving N minimax equilibria for a zero-sum matrix game, with N equal to the number of ob- served datapoints. The complexity of this greedy solution is exponential in the cardinality of the action spaces, which makes it prohibitive (Perolat et al., 2015). 

Most Markov Game approaches require solving for the equilibrium solution for a multiplayer value or minimax-Q function at each iteration. This requires evaluating a typ- ically intractable minimax optimization problem. Instead, we focus on learning stationary policies μ∗ and ν∗ such that R1(μ∗, ν∗) → R1∗. This way we can avoid this costly op- timization at each iteration as we just need to approximate the advantage function and not determine the equilibrium solution at each iteration. 

3.3. Proposed Method: RARL 

Our algorithm (RARL) optimizes both of the agents using the following alternating procedure. In the first phase, we learn the protagonist’s policy while holding the adversary’s policy fixed. Next, the protagonist’s policy is held constant and the adversary’s policy is learned. This sequence is re- peated until convergence. 

Algorithm 1 outlines our approach in detail. The initial pa- rameters for both players’ policies are sampled from a ran- dom distribution. In each of the Niter iterations, we carry out a two-step (alternating) optimization procedure. First, for Nμ iterations, the parameters of the adversary θν are held constant while the parameters θμ of the protagonist are optimized to maximize R1 (Equation 4). The *roll* function samples Ntraj trajectories given the environment definition E and the policies for both the players. Note that E contains the transition function P and the reward functions r1 and r2 to generate the trajectories. The tth element of the ith trajectory is of the form (si, a1i, a2i, r1i, r2i). These trajectories are then split such that the tth element of the ith trajectory is of the form (si, ai = a1i, ri = r1i). The protagonist’s parameters θμ are then optimized using a policy optimizer. For the second step, player 1’s parameters θμ are held constant for the next Nν iterations. Ntraj Trajectories are sampled and split into trajectories such that tth element of the ith trajectory is of the form (si, ai = a2i, ri = r2i). 

Player 2’s parameters θν are then optimized. This alternat- ing procedure is repeated for Niter iterations. 

 

3．2. 形成对抗强化学习

在我们的对弈游戏中，在每个时间步t上，两个玩家都观察状态st，并采取行动a1t ~ μ(st)和a2t ~ ν(st)。从环境中获得状态转换st+1 = P(st, a1t, a2t)和奖励rt = r(st, a1t, a2t)。在我们的零和游戏中，主角获得奖励rt1 = rt，而对手获得奖励rt2 =−rt。因此，MDP的每一步都可以表示为(st, a1t, a2t, rt1, rt2, st+1)。

主角寻求以下奖励功能最大化，(xx4xx)

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image005.png)

因为，策略μ和ν是唯一可学习的组件，R≡R (μ， ν)。类似地，对手试图使自己的奖赏最大化:R2≡R2(μ， ν) =−R1(μ， ν)。解决这一问题的一种方法是将连续的状态和行动空间离散化，并采用动态规划方法求解。(Perolat等人，2015;Patek, 1997)证明了该博弈在最优均衡条件下的极小极大均衡和纳什均衡概念是等价的:(xx5xx)

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image006.png)

然而，寻找纳什均衡策略的解往往涉及贪婪地求解N个极大极小均衡的零和矩阵博弈，其中N等于观测数据点的数量。这个贪婪解决方案的复杂性在动作空间的基数上是指数级的，这使得它令人望而却步(Perolat et al.， 2015)。

大多数马尔可夫对策方法要求在每次迭代时求解多人值或极小极大- q函数的均衡解。这需要评估一个典型的难以处理的极小极大优化问题。相反，我们专注于学习静止政策μ∗和ν∗使R1(μ∗，ν∗)→R1∗。通过这种方法，我们可以避免每次迭代时代价高昂的优化，因为我们只需要逼近优势函数，而不需要在每次迭代时确定均衡解。

3．3. 方法:RARL

我们的算法(RARL)使用以下交替过程优化这两个代理。在第一阶段，我们学习主角的政策，同时保持对手的政策不变。其次，保持主角的政策不变，学习对手的政策。重复这个序列直到收敛为止。

算法1详细描述了我们的方法。两个参与者策略的初始参数都是从随机分布中取样的。在每次Niter迭代中，我们执行两步(交替)优化过程。首先，对于n次迭代，对手的参数θν保持不变，而主角的参数θμ进行优化，使R1最大化(方程4)。在给定环境定义E和双方参与者的策略的情况下，滚动函数对Ntraj轨迹进行采样。注意E包含过渡函数P和奖励函数r1和r2来产生轨迹。第i个轨迹的第n个元素的形式为(si, a1i, a2i, r1i, r2i)。然后对这些轨迹进行分割，使第i个轨迹的第t个元素的形式为(si, ai = a1i, ri = r1i)。然后使用策略优化器优化主角的参数θμ。对于第二步，玩家1的参数θμ在下一次Nν迭代中保持不变。Ntraj轨迹被采样并分解成轨迹，使得第i个轨迹的第t个元素的形式为(si, ai = a2i, ri = r2i)。

然后对参与人2的参数θν进行优化。这种交替过程是重复的Niter迭代。

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image007.png)

 

 

\4. Experimental Evaluation 

We now demonstrate the robustness of the RARL algorithm: (a) for training with different initializations; (b) for testing with different conditions; (c) for adversarial disturbances in the testing environment. But first we will describe our implementation and test setting followed by evaluations and results of our algorithm. 

4.1. Implementation 

Our implementation of the adversarial environments build on OpenAI gym’s (Brockman et al., 2016) control environments with the MuJoCo (Todorov et al., 2012) physics sim- ulator. Details of the environments and their corresponding adversarial disturbances are (also see Figure 1): 

InvertedPendulum: The inverted pendulum is mounted on a pivot point on a cart, with the cart restricted to linear movement in a plane. The state space is 4D: position and velocity for both the cart and the pendulum. The pro- tagonist can apply 1D forces to keep the pendulum upright. The adversary applies a 2D force on the center of pendulum in order to destabilize it. 

HalfCheetah: The half-cheetah is a planar biped robot with 8 rigid links, including two legs and a torso, along with 6 actuated joints. The 17D state space includes joint angles and joint velocities. The adversary applies a 6D ac- tion with 2D forces on the torso and both feet in order to destabilize it. 

Swimmer: The swimmer is a planar robot with 3 links and 2 actuated joints in a viscous container, with the goal of moving forward. The 8D state space includes joint angles and joint velocities. The adversary applies a 3D force to the center of the swimmer. 

Hopper: The hopper is a planar monopod robot with 4 rigid links, corresponding to the torso, upper leg, lower leg, and foot, along with 3 actuated joints. The 11D state space includes joint angles and joint velocities. The adversary applies a 2D force on the foot. 

Walker2D: The walker is a planar biped robot consisting of 7 links, corresponding to two legs and a torso, along with 6 actuated joints. The 17D state space includes joint angles and joint velocities. The adversary applies a 4D action with 2D forces on both the feet. 

Our implementation of RARL is built on top of rllab (Duan et al., 2016) and uses Trust Region Policy Optimization (TRPO) (Schulman et al., 2015) as the policy optimizer. For all the tasks and for both the protagonist and adversary, we use a policy network with two hidden layers with 64 neurons each. We train both RARL and the baseline for 100 iterations on InvertedPendulum and for 500 iterations on the other tasks. Hyperparameters of TRPO are selected by grid search. 

4.2. Evaluating Learned Policies 

We evaluate the robustness of our RARL approach com- pared to the strong TRPO baseline. Since our policies are stochastic in nature and the starting state is also drawn from a distribution, we learn 50 policies for each task with dif- ferent seeds/initializations. First, we report the mean and variance of cumulative reward (over 50 policies) as a func- tion of the training iterations. Figure 2 shows the mean and variance of the rewards of learned policies for the task of HalfCheetah, Swimmer, Hopper and Walker2D. We omit the graph for InvertedPendulum because the task is easy and both TRPO and RARL show similar performance and similar rewards. As we can see from the figure, for all the four tasks RARL learns a better policy in terms of mean reward and variance as well. This clearly shows that the policy learned by RARL is better than the policy learned by TRPO even when there is no disturbance or change of settings between training and test conditions. Table 1 re- ports the average rewards with their standard deviations for the best learned policy. 

However, the primary focus of this paper is to show robust- ness in training these control policies. One way of visual- izing this is by plotting the average rewards for the nth per- centile of trained policies. Figure 3 plots these percentile curves and highlight the significant gains in robustness for training for the HalfCheetah, Swimmer and Hopper tasks. 

 

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image008.png)

Figure 2. Cumulative reward curves for RARL trained policies versus the baseline (TRPO) when tested without any disturbance. For all the tasks, RARL achieves a better mean than the base- line. For tasks like Hopper, we also see a significant reduction of variance across runs. 

 

\4. 实验评价

我们现在演示了RARL算法的鲁棒性:(a)用于不同初始化的训练;(b)在不同条件下进行测试;(c)测试环境中的对抗性干扰。但是首先我们将描述我们的实现和测试设置，然后是算法的评估和结果。

4．1. 实现

我们的对抗性环境的实现建立在OpenAI健身房(Brockman等人，2016)的控制环境与MuJoCo (Todorov等人，2012)物理模拟器。环境及其对应的对抗性干扰的细节如下(也见图1):

倒立摆:倒立摆安装在小车的支点上，小车只能在平面内作直线运动。状态空间是4D:小车和摆的位置和速度。支持者可以施加一维力使钟摆保持直立。对手在钟摆的中心施加一个二维的力以使它不稳定。

半猎豹:半猎豹是一个平面两足机器人，有8个刚性链接，包括两条腿和一个躯干，以及6个驱动关节。17D状态空间包括关节角和关节速度。对手用2D的力量对躯干和双脚施加6D的作用力以使其不稳定。

游泳者:游泳者是一个平面机器人，在一个粘性容器中有3个连杆和2个驱动关节，目标是前进。8D状态空间包括关节角度和关节速度。对手对游泳者的中心施加三维力。

料斗:料斗是一个平面单脚机器人，有4个刚性连杆，分别对应躯干、小腿、小腿和脚，以及3个驱动关节。11D状态空间包括关节角和关节速度。对手对脚施加二维力。

walker 2d: walker是一个平面两足机器人，由7个连杆组成，分别对应两条腿和一个躯干，以及6个驱动关节。17D状态空间包括关节角和关节速度。对手对双脚施加4D动作和2D力量。

我们的RARL实现建立在rllab (Duan et al.， 2016)之上，并使用Trust Region Policy Optimization (TRPO) (Schulman et al.， 2015)作为策略优化器。对于所有的任务，以及对于主角和对手，我们使用了一个带有两个隐含层的策略网络，每个隐含层有64个神经元。我们对RARL和基线进行了培训，在InvertedPendulum上进行了100次迭代，在其他任务上进行了500次迭代。通过网格搜索选择TRPO的超参数。

4．2. 评估学习政策

与强大的TRPO基线相比，我们评估了RARL方法的稳健性。由于我们的策略在本质上是随机的，而且起始状态也是从分布中提取的，所以我们为每个任务学习了50个策略，这些策略具有不同的种子/初始化。首先，我们报告累积奖励(超过50个策略)的平均值和方差作为训练迭代的函数。图2显示了HalfCheetah、Swimmer、Hopper和Walker2D在任务中学习策略所获得奖励的平均值和方差。由于任务简单，TRPO和RARL表现出相似的性能和奖励，我们省略了InvertedPendulum的图表。从图中可以看出，对于所有的四个任务，RARL在平均奖励和方差方面都学习到了更好的策略。这清楚地表明，即使在训练和测试条件之间没有干扰或设置变化时，RARL学习到的策略也比TRPO学习到的策略好。表1报告了最佳学习策略的平均奖励及其标准差。

然而，本文的主要焦点是在训练这些控制策略时显示鲁棒性。一种可视化的方法是绘制出第n个百分之一的培训政策的平均回报。图3绘制了这些百分比曲线，并突出了HalfCheetah、Swimmer和Hopper任务的健壮性显著提高。

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image008.png)

图2。在没有任何干扰的情况下，RARL训练策略相对于基线(TRPO)的累积奖励曲线。对于所有的任务，RARL达到了一个比基线更好的平均值。对于像Hopper这样的任务，我们也看到了运行中方差的显著减少。

 

4.3. Robustness under Adversarial Disturbances 

While deploying controllers in the real world, unmodeled environmental effects can cause controllers to fail. One way of measuring robustness to such effects is by measur- ing the performance of our learned control polices in the presence of an adversarial disturbance. For this purpose, we train an adversary to apply a disturbance while holding the protagonist’s policy constant. We again show the per- centile graphs as described in the section above. RARL’s control policy, since it was trained on similar adversaries, performs better, as seen in Figure 4. 

4.4. Robustness to Test Conditions 

Finally, we evaluate the robustness and generalization of the learned policy with respect to varying test conditions. In this section, we train the policy based on certain mass and friction values; however at test time we evaluate the policy when different mass and friction values are used in the environment. Note we omit evaluation of Swimmer since the policy for the swimming task is not significantly impacted by a change mass or friction. 

4.4.1. EVALUATION WITH CHANGING MASS 

We describe the results of training with the standard mass variables in OpenAI gym while testing it with differ- ent mass. Specifically, the mass of InvertedPendulum, HalfCheetah, Hopper and Walker2D were 4.89, 6.36, 3.53 and 3.53 respectively. At test time, we evaluated the learned policies by changing mass values and estimating the average cumulative rewards. Figure 5 plots the average rewards and their standard deviations against a given torso mass (horizontal axis). As seen in these graphs, RARL policies generalize significantly better. 

4.4.2. EVALUATION WITH CHANGING FRICTION 

Since several of the control tasks involve contacts and fric- tion (which is often poorly modeled), we evaluate robust- ness to different friction coefficients in testing. Similar to the evaluation of robustness to mass, the model is trained with the standard variables in OpenAI gym. Figure 6 shows the average reward values with different friction coeffi- cients at test time. It can be seen that the baseline policies fail to generalize and the performance falls significantly when the test friction is different from training. On the other hand RARL shows more resilience to changing fric- tion values. 

We visualize the increased robustness of RARL in Fig- ure7,wherewetestwithjointlyvaryingbothmassand friction coefficient. As observed from the figure, for most combinations of mass and friction values RARL leads sig- nificantly higher reward values compared to the baseline. 

4．3． 对抗干扰下的鲁棒性

在现实世界中部署控制器时，未建模的环境影响可能导致控制器失败。一种测量这种影响的鲁棒性的方法是测量我们的学习控制策略在对抗干扰的存在下的性能。为此，我们训练对手在保持主角政策不变的情况下应用干扰。我们再次显示上面一节中描述的百分位图。RARL的控制策略，因为它是在类似的对手上训练的，所以表现得更好，如图4所示。

4.4。对测试条件的稳健性

最后，我们评估了学习策略在不同测试条件下的鲁棒性和泛化性。在这一节中，我们根据一定的质量和摩擦值训练策略;然而，在测试时，我们评估了在环境中使用不同质量和摩擦值时的策略。注意，我们省略了对Swimmer的评估，因为游泳任务的策略不会受到质量变化或摩擦的显著影响。

4.1.1。质量变化评价

我们描述了标准质量变量在OpenAI体育馆的训练结果，并对不同质量进行了测试。其中，倒摆、HalfCheetah、Hopper和Walker2D的质量分别为4.89、6.36、3.53和3.53。在测试时，我们通过改变质量值和估计平均累积奖励来评估学习策略。图5绘制了平均奖励和他们的标准偏差相对于给定的躯干质量(水平轴)。从这些图中可以看出，RARL策略的通用性明显更好。

10/24/11。根据摩擦变化进行评价

由于一些控制任务涉及接触和摩擦(这往往是很糟糕的模型)，我们在测试中评估不同摩擦系数的鲁棒性。与对质量的稳健性评价类似，模型使用OpenAI gym中的标准变量进行训练。图6显示了测试时不同摩擦系数下的平均奖励值。可以看出，当测试摩擦不同于训练时，基线策略没有普遍化，表现明显下降。另一方面，RARL对摩擦值的变化表现出更强的弹性。

我们在图- ure7中可视化了RARL增强的鲁棒性，其中质量和摩擦系数共同变化最明显。从图中可以观察到，对于大多数质量和摩擦值的组合，RARL导致的奖励值明显高于基线值。

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image009.png)

表1。RARL学习到的最佳策略与基线的比较(平均值±一个标准差)

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image010.png)

图3。我们显示没有任何干扰的百分位图来显示RARL与基线相比的稳健性。在这里，算法在多个初始化中运行，然后进行排序以显示累积最终奖励的第n个百分位。

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image011.png)

Figure 4. Percentile plots with a learned adversarial distur- bance show the robustness of RARL compared to the baseline in the presence of an adversary. Here the algorithms are run on multiple initializations followed by learning an adversarial distur- bance that is applied at test time. 

图4。百分位图与学习对抗性干扰显示RARL相比的鲁棒性在存在一个对手的基线。在这里，算法在多个初始化上运行，然后学习在测试时应用的对抗性干扰。

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image012.png)

Figure5.The graphs show robustness of RARL policies to changing mass between training and testing. For the Inverted- Pendulum the mass of the pendulum is varied, while for the other tasks, the mass of the torso is varied. 

Figure5。这些图显示了RARL策略对训练和测试之间质量变化的鲁棒性。对于倒立摆的质量是变化的，而对于其他任务，躯干的质量是变化的。

 

4.5. Visualizing the Adversarial Policy 

Finally, we visualize the adversarial policy for the case of InvertedPendulum and Hopper to see whether the learned policies are human interpretable. As shown in Figure 8, the direction of the force applied by the adversary agrees with human intuition: specifically, when the cart is station- ary and the pole is already tilted (top row), the adversary attempts to accentuate the tilt. Similarly, when the cart is moving swiftly and the pole is vertical (bottom row), the adversary applies a force in the direction of the cart’s mo- tion. The pole will fall unless the cart speeds up further (which can also cause the cart to go out of bounds). Note that the naive policy of pushing in the opposite direction would be less effective since the protagonist could slow the cart to stabilize the pole. 

Similarly for the Hopper task in Figure 9, the adversary ap- plies horizontal forces to impede the motion when the Hop- per is in the air (left) while applying forces to counteract gravity and reduce friction when the Hopper is interacting with the ground (right). 

4．5． 可视化对抗政策

最后，我们将对抗性策略可视化为倒摆和Hopper的情况，看看学习到的策略是否人类可解释。如图8所示，对手施加的力的方向与人类的直觉相一致:具体来说，当马车站好，柱子已经倾斜(上一排)，对手试图加大倾斜。同样地，当大车快速移动，柱子是垂直的(最下面一排)，对手对大车运动的方向施加一个力。除非车进一步加速(这也可能导致车越界)，否则柱子会掉下来。需要注意的是，向相反方向推的天真策略效果会更差，因为主角可以让车慢下来以稳定杆子。

与图9中的Hopper任务类似，当Hop- per在空中时(左)，对手施加水平力来阻碍其运动，而当Hopper与地面交互时(右)，对手施加力来抵消重力并减少摩擦力。

\5. Related Research 

Recent applications of deep reinforcement learning (deep RL) have shown great success in a variety of tasks rang- ing from games (Mnih et al., 2015; Silver et al., 2016), robot control (Gu et al., 2016; Lillicrap et al., 2015; Mor- datch et al., 2015), to meta learning (Zoph & Le, 2016). An overview of recent advances in deep RL is presented in (Li, 2017) and (Kaelbling et al., 1996; Kober & Peters, 2012) provide a comprehensive history of RL research. 

Learned policies should be robust to uncertainty and pa- rameter variation to ensure predicable behavior, which is essential for many practical applications of RL includ- ing robotics. Furthermore, the process of learning poli- cies should employ safe and effective exploration with im- proved sample efficiency to reduce risk of costly failure. These issues have long been recognized and investigated in reinforcement learning (Garcıa & Ferna ́ndez, 2015) and have an even longer history in control theory research (Zhou & Doyle, 1998). These issues are exacerbated in deep RL by using neural networks, which while more ex- pressible and flexible, often require significantly more data to train and produce potentially unstable policies. 

In terms of (Garcıa & Ferna ́ndez, 2015) taxonomy, our ap- proach lies in the class of worst-case formulations. We model the problem as an H∞ optimal control problem (Bas ̧ar&Bernhard,2008). Inthisformulation,nature (which may represent input, transition or model uncer- tainty) is treated as an adversary in a continuous dynamic zero-sum game. We attempt to find the minimax solu- tion to the reward optimization problem. This formulation was introduced as robust RL (RRL) in (Morimoto & Doya, 2005). RRL proposes a model-free an actor-disturber-critic method. Solving for the optimal strategy for general non- linear systems requires is often analytically infeasible for most problems. To address this, we extend RRL’s model- free formulation using deep RL via TRPO (Schulman et al., 2015) with neural networks as the function approximator. 

Other worst-case formulations have been introduced. (Nilim & El Ghaoui, 2005) solve finite horizon tabular MDPs using a minimax form of dynamic programming. Using a similar game theoretic formulation (Littman, 1994) introduces the notion of a Markov Game to solve tabu- lar problems, which involves linear program (LP) to solve the game optimization problem. (Sharma & Gopal, 2007) extend the Markov game formulation using a trained neu- ral network for the policy and approximating the game to continue using LP to solve the game. (Wiesemann et al., 2013) present an enhancement to standard MDP that pro- vides probabilistic guarantees to unknown model param- eters. Other approaches are risk-based including (Tamar et al., 2014; Delage & Mannor, 2010), which formulate various mechanisms of percentile risk into the formulation. Our approach focuses on continuous space problems and is a model-free approach that requires explicit parametric formulation of model uncertainty. 

Adversarial methods have been used in other learning prob- lems including (Goodfellow et al., 2015), which leverages adversarial examples to train a more robust classifiers and (Goodfellow et al., 2014; Dumoulin et al., 2016), which uses an adversarial lost function for a discriminator to train a generative model. In (Pinto et al., 2016) two supervised agents were trained with one acting as an adversary for self- supervised learning which showed improved robot grasp- ing. Other adversarial multiplayer approaches have been proposed including (Heinrich & Silver, 2016) to perform self-play or fictitious play. Refer to (Bus ̧oniu et al., 2010) for an review of multiagent RL techniques. 

Recent deep RL approaches to the problem focus on ex- plicit parametric model uncertainty. (Heess et al., 2015) use recurrent neural networks to perform direct adaptive control. Indirect adaptive control was applied in (Yu et al., 2017) for online parameter identification. (Rajeswaran et al., 2016) learn a robust policy by sampling the worst case trajectories from a class of parametrized models, to learn a robust policy. 

\5. 相关研究

近年来，深度强化学习(deep reinforcement learning, deep RL)在游戏等各种任务中取得了巨大成功(Mnih等人，2015;Silver等人，2016)，机器人控制(Gu等人，2016;Lillicrap等人，2015;Mor- datch等人，2015)，元学习(Zoph & Le, 2016)。(Li, 2017)和(Kaelbling et al.， 1996;Kober & Peters, 2012)提供了RL研究的全面历史。

学习到的策略应该对不确定性和参数变化具有鲁棒性，以确保行为的可预测性，这对于RL的许多实际应用包括机器人。此外，学习策略的过程应采用安全有效的探索，提高样本效率，以减少代价高昂的失败风险。这些问题早已在强化学习中被认识和研究(Garcıa & Fernández, 2015)，在控制理论研究中有更长的历史(Zhou & Doyle, 1998)。在深度RL中，神经网络的使用加剧了这些问题。神经网络虽然更具有可表达性和灵活性，但通常需要大量的数据来训练和产生潜在的不稳定策略。

根据(Garcıa & Fernández, 2015)分类，我们的方法在于最坏情况公式的类别。我们将该问题建模为H∞最优控制问题(Başar&Bernhard,2008)。在这个公式中，自然(可能代表输入、过渡或模型的不确定性)被视为一个持续动态零和博弈中的对手。我们试图找到报酬优化问题的极大极小解。该配方是在(Morimoto & Doya, 2005)中引入的稳健RL (RRL)。RRL提出了一种无模型的行动者-干扰者-批评方法。求解一般非线性系统的最优策略对于大多数问题来说通常是解析上不可行的。为了解决这个问题，我们通过TRPO (Schulman et al.， 2015)扩展了RRL的无模型公式，并以神经网络作为函数逼近器。

还引入了其他最坏情况的公式。(Nilim & El Ghaoui, 2005)利用动态规划的极大极小形式求解有限水平表mdp。利用类似的博弈论公式(Littman, 1994)引入了马尔科夫博弈论的概念来解决禁忌问题，它涉及到线性规划(LP)来解决博弈优化问题。(Sharma & Gopal, 2007)使用训练过的神经网络来扩展马尔可夫博弈公式，并对博弈进行逼近，继续使用LP来求解博弈。(Wiesemann等人，2013)提出了对标准MDP的一种增强，为未知模型参数提供概率保证。其他方法是基于风险的，包括(Tamar等人，2014;Delage & Mannor, 2010)，其中制定了各种机制的风险百分比进入制定。我们的方法集中在连续空间问题，是一个无模型的方法，需要明确的参数形式的模型不确定性。

对抗方法已经被用于其他学习问题，包括(Goodfellow等，2015)，它利用对抗例子来训练更健壮的分类器和(Goodfellow等，2014;Dumoulin et al.， 2016)，该算法使用一个对抗性损失函数作为鉴别器来训练生成模型。在(Pinto et al.， 2016)两个有监督的代理被训练，其中一个作为对手进行自我监督学习，这显示了改进的机器人抓取。还提出了其他对抗性多人游戏方法，包括(Heinrich & Silver, 2016)执行自我游戏或虚拟游戏。参考(Buşoniu et al.， 2010)对多agent RL技术的综述。

最近针对这一问题的深度RL方法主要关注显性参数模型的不确定性。(Heess et al.， 2015)利用递归神经网络进行直接自适应控制。(Yu et al.， 2017)采用间接自适应控制进行在线参数辨识。(Rajeswaran et al.， 2016)通过从一类参数化模型中取样最坏情况的轨迹来学习稳健的政策，从而学习稳健的政策。

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image013.png)

图6。这些图显示了RARL策略对改变训练和测试之间摩擦的鲁棒性。注意，我们排除了倒摆和游泳者的结果，因为摩擦与这些任务无关。

 

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image014.png)

图7。热图显示了RARL策略在训练和测试之间改变摩擦和质量的鲁棒性。对于Hopper和HalfCheetah的任务，我们观察到鲁棒性的显著增加。

 

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image015.png)

图8。可视化对手施加在垂直摆上的力。在(a)和(b)中，小车是静止的，而在(c)和(d)中，小车以垂直摆运动。

![img](file:////Users/babytree/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image016.png)

图9。可视化的力量应用的对手对跳。在左边，漏斗的脚在空中，而在右边，脚与地面互动。

 

6. Conclusion 

We have presented a novel adversarial reinforcement learn- ing framework, RARL, that is: (a) robust to training ini- tializations; (b) generalizes better and is robust to environ- mental changes between training and test conditions; (c) robust to disturbances in the test environment that are hard to model during training. Our core idea is that modeling errors should be viewed as extra forces/disturbances in the system. Inspired by this insight, we propose modeling un- certainties via an adversary that applies disturbances to the system. Instead of using a fixed policy, the adversary is re- inforced and learns an optimal policy to optimally thwart the protagonist. Our work shows that the adversary effec- tively samples hard examples (trajectories with worst re- wards) leading to a more robust control strategy. 

 

## 6. 结论

我们提出了一种新的对抗强化学习框架RARL，即:(a)对训练ini化具有鲁棒性;(b)能更好地概括和适应训练和考试条件之间的环境变化;(c)对训练中难以建模的测试环境中的干扰具有鲁棒性。我们的核心思想是，建模误差应该被视为系统中的额外力/干扰。受此启发，我们提出通过一个对手对系统应用干扰来建模不确定性。对手不再使用固定的策略，而是得到了强化，并学会了最优策略，以最佳方式挫败主角。我们的工作表明，对手有效地采样困难的例子(最坏的奖励轨迹)导致一个更鲁棒的控制策略。

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 