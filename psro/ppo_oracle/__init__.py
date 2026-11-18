"""PPO best-response implementations and actor/critic networks."""
from .red_ppo import PPOConfig, PPOOracleStats, train_ppo_best_response
from .blue_ppo import train_relgnn_blue_best_response
