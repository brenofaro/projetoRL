# %%
%pip install torch torchvision gymnasium[atari] numpy scikit-image ale_py tensorboard 

# %% [markdown]
# ## Agente CNN com DQN

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import time
import random
from collections import deque
import math
from skimage.transform import resize
from torch.utils.tensorboard import SummaryWriter  
import os

# imprimir se a gpu esta disponivel
print(torch.cuda.is_available())

# %%
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def feature_size(self, input_shape):
        return self.conv(torch.zeros(1, *input_shape)).view(1, -1).size(1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# %%
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(np.array(actions)),
                torch.FloatTensor(np.array(rewards)),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(np.array(dones)))

    def __len__(self):
        return len(self.memory)

# %%
class DQNAgent:
    def __init__(self, state_shape, num_actions, learning_rate=0.00025, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9999, batch_size=32, memory_size=50000, update_target_freq=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_shape, num_actions).to(self.device)
        self.target_model = DQN(state_shape, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # Inicializa o target model com os mesmos pesos
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.train_step = 0
        self.num_actions = num_actions
        self.loss_fn = nn.MSELoss() # Salva o Loss
        self.episode_q_values = []  # Para a média dos Q-values


    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            self.episode_q_values.append(q_values.mean().item())  #  Salva o Q-value
            return q_values.argmax().item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item(), q_values.mean().item()

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

def preprocess_state(state):
    state = resize(state, (84, 84), anti_aliasing=True)
    state = state.astype(np.float32)
    return state

# %%
def create_writer(log_metrics=False, log_dir="tensorboardteste/dqn_training"):
    return SummaryWriter(log_dir) if log_metrics else None

def train_agent(env, agent, episodes=1000, save_freq=20, log_dir="tensorboardteste/dqn_training", log_metrics=False):
    """
    Treina o agente DQN e registra métricas no TensorBoard.

    Args:
        env: Ambiente Gym.
        agent: Agente DQN.
        episodes: Número de episódios para treinamento.
        save_freq: Frequência para salvar o modelo.
        log_dir: Diretório para salvar os logs do TensorBoard.
    """
    try:

        writer = create_writer (log_metrics=log_metrics, log_dir=log_dir)
        total_steps = 0
        running_rewards = deque(maxlen=100) # Salvar as recompensas dos ultimos 100 episodios

        for episode in range(episodes):
            state, _ = env.reset()
            state = preprocess_state(state)

            # Frame Stacking
            state = np.transpose(state, (2, 0, 1))
            stacked_frames = deque([state] * 4, maxlen=4)
            state = np.concatenate(stacked_frames, axis=0)

            episode_reward = 0
            episode_steps = 0
            episode_start_time = time.time()
            done = False
            episode_actions = []  # Salvar ações do episódio
            episode_losses = []  # Salvar o loss do episódio


            while not done:
                action = agent.choose_action(state)
                episode_actions.append(action)  # Salvar ação
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = preprocess_state(next_state)

                # Frame Stacking - Update
                next_state = np.transpose(next_state, (2, 0, 1))
                stacked_frames.append(next_state)
                next_state = np.concatenate(stacked_frames, axis=0)

                done = terminated or truncated
                agent.memory.push(state, action, reward, next_state, done)
                loss, mean_q = agent.learn()

                if loss is not None:
                    episode_losses.append(loss) # Salvar loss
                    if writer:
                        writer.add_scalar("Loss/step", loss, total_steps)  # Salvar a loss no tensorboard
                        writer.add_scalar("Mean_Q/step", mean_q, total_steps) # Salvar a media dos q_values no tensorboard

                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1


            # Calculo das métricas do episódio
            episode_duration = time.time() - episode_start_time
            steps_per_second = episode_steps / episode_duration
            running_rewards.append(episode_reward)
            running_mean_reward = np.mean(running_rewards)
            mean_reward_per_step = episode_reward / episode_steps if episode_steps > 0 else 0.0

             # Calcular a ação média
            action_counts = np.bincount(episode_actions, minlength=agent.num_actions)
            avg_action = np.average(np.arange(agent.num_actions), weights=action_counts)
            if writer:
                # Salvar as métricas no tensorboard
                writer.add_scalar("Episode_Reward/episode", episode_reward, episode)
                writer.add_scalar("Running_Mean_Reward/episode", running_mean_reward, episode)
                writer.add_scalar("Mean_Reward_Per_Step/episode", mean_reward_per_step, episode)
                writer.add_scalar("Episode_Duration/episode", episode_duration, episode)
                writer.add_scalar("Steps_Per_Second/episode", steps_per_second, episode)
                writer.add_scalar("Average_Action/episode", avg_action, episode) #colocar no tensorboard a media de acoes
                writer.add_scalar("Epsilon/episode", agent.epsilon, episode)
                if episode_losses: 
                    writer.add_scalar("Mean_Loss/episode", np.mean(episode_losses), episode) # Salvar a loss media no tensorboard
            if agent.episode_q_values: # Salvar os q_values medios
                if writer:
                    writer.add_scalar("Mean_Q_Values/episode", np.mean(agent.episode_q_values), episode)
                agent.episode_q_values = [] # Resetar a lista de q_values
            
            print(f"{total_steps}/{1000000}: episódio: {episode + 1}, duração: {episode_duration:.3f}s, passos no episódio: {episode_steps}, passos por segundo: {steps_per_second:.1f}, recompensa do episódio: {episode_reward:.3f}, recompensa média por passo: {mean_reward_per_step:.3f}, ação média: {avg_action:.3f}, epsilon: {agent.epsilon:.6f}, running_mean_reward: {running_mean_reward:.6f}")


            if (episode + 1) % save_freq == 0:
                agent.save(f'dqn_model_episode_{episode + 1}.pth')
        if writer:
            writer.close()
        return agent

    except Exception as e:
        print(f"Error during training: {e}")
        raise

# %% [markdown]
# ## Rodando o treinamento

# %% [markdown]
# #### Rodando o treinamento 

# %%
from ale_py import ALEInterface
from torchsummary import summary

ale = ALEInterface()

if __name__ == "__main__":
    env = gym.make('ALE/SpaceInvaders-v5', render_mode="rgb_array")
    state_shape = (12, 84, 84)
    num_actions = env.action_space.n

    agent = DQNAgent(state_shape, num_actions)
    # Resumo do modelo
    summary(agent.model, state_shape, device="cuda" if torch.cuda.is_available() else "cpu")

    train_agent(env, agent, episodes=100000, log_metrics=False)
    env.close()

# %% [markdown]
# ## Rodando o modelo treinado

# %%
import torch
import gymnasium as gym
import numpy as np
import time
from collections import deque
from skimage.transform import resize
import torch.nn as nn
import random


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def feature_size(self, input_shape):
        return self.conv(torch.zeros(1, *input_shape)).view(1, -1).size(1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Agent:
    def __init__(self, state_shape, num_actions, epsilon=0.05):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_shape, num_actions).to(self.device)
        self.epsilon = epsilon
        self.num_actions = num_actions

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()

    def choose_action(self, state):
        if random.random() < self.epsilon: # Escolha aleatória
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.model(state)
                return q_values.argmax().item()

def preprocess_state(state):
    """Resize and normalize the RGB state."""
    state = resize(state, (84, 84), anti_aliasing=True)
    state = state.astype(np.float32)
    return state

def run_agent(env, agent, model_path, episodes=10):
    agent.load(model_path)  # Carregar o modelo treinado
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_state(state)

        # Frame Stacking (RGB)
        state = np.transpose(state, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        stacked_frames = deque([state] * 4, maxlen=4)
        state = np.concatenate(stacked_frames, axis=0)

        done = False
        episode_reward = 0
        episode_steps = 0
        start_time = time.time()

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess_state(next_state)

            next_state = np.transpose(next_state, (2, 0, 1))
            stacked_frames.append(next_state)
            state = np.concatenate(stacked_frames, axis=0)

            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1

            env.render()


        episode_time = time.time() - start_time
        steps_per_second = episode_steps / episode_time

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}, Steps: {episode_steps}, Time: {episode_time:.2f}s, Steps/s: {steps_per_second:.2f}")

    env.close()
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward:.2f}")
    print(f"Standard Deviation of Rewards: {std_reward:.2f}")

# %%
from ale_py import ALEInterface

ale = ALEInterface()

if __name__ == "__main__":
    env = gym.make('ALE/SpaceInvaders-v5', render_mode="human")
    state_shape = (12, 84, 84) 
    num_actions = env.action_space.n

    agent = Agent(state_shape, num_actions)
    model_file = "dqn_model_episode_2440.pth" # Carregar o modelo treinado
    run_agent(env, agent, model_file, episodes=5)


