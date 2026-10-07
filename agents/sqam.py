import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class SQAMAgent(flax.struct.PyTreeNode):
    """Scalar Q Adjoint Matching (SQAM)."""

    rng: Any
    network: Any
    lam: float      # trust-region dual variable
    kl_ema: float   # EMA of the path-space KL estimate
    config: Any = nonpytree_field()

    def _batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def critic_loss(self, batch, grad_params, rng, a_pi):
        """TD loss on the ensemble, plus the value penalty at the policy endpoint `a_pi`."""
        batch_actions = self._batch_actions(batch)

        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=rng)
        next_actions = jnp.clip(next_actions, -1, 1)
        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        info = {'critic_loss': critic_loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min()}

        # Value penalty: Q(s, a_pi) - Q(s, a_data) at the policy's own endpoints.
        if self.config["gap_reg_coef"] > 0.0:
            a_pi = jax.lax.stop_gradient(jnp.clip(a_pi, -1, 1))
            q_pi = self.network.select('critic')(batch['observations'], a_pi, params=grad_params)
            gap = (q_pi.mean(axis=0) - q.mean(axis=0)).mean()
            critic_loss = critic_loss + self.config["gap_reg_coef"] * gap
            info['gap_reg_term'] = gap
            info['critic_loss_total'] = critic_loss

        return critic_loss, info

    def get_lambda(self):
        return self.config["lam_scale"] * self.lam

    def get_sigma_scale(self):
        """Trust-region diffusion scale, 1 / sqrt(lambda)."""
        return 1.0 / jnp.sqrt(jnp.maximum(self.get_lambda(), 1e-8))

    @jax.jit
    def estimate_adjoint(self, obs):
        """Return (xs, adjs, ts, info) for the scalar adjoint.

        The policy endpoint a_hat comes from an Euler rollout of `actor_fast`. The K bridge states
        are x_tau = (1 - tau) * eps + tau * a_hat on the grid tau = 1/K, ..., 1, and the adjoint at
        each is -tau * grad_a Q(s, a_hat), with grad_a Q computed once at the endpoint.
        """
        flow_steps = self.config["flow_steps"]
        action_dim = self.config['action_dim'] * \
            (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        actor_fast = self.network.select("actor_fast")

        h = 1.0 / flow_steps
        rng, x_rng = jax.random.split(self.rng)
        x = jax.random.normal(x_rng, shape=obs.shape[:-1] + (action_dim,))
        for i in range(flow_steps):
            t = i / flow_steps * jnp.ones_like(x[..., 0:1])
            x = x + h * actor_fast(obs, x, t)
        a_hat = jax.lax.stop_gradient(x)                       # policy endpoint, tau = 1

        # Bridge states from the pretraining noising kernel, with fresh noise.
        K = self.config["n_bridge"]
        tshape = (K,) + a_hat.shape[:-1] + (1,)
        xshape = (K,) + a_hat.shape
        ts_stack = (jnp.arange(1, K + 1) / K).reshape((K,) + (1,) * (len(tshape) - 1)) * jnp.ones(tshape)
        eps = jax.random.normal(jax.random.fold_in(self.rng, 1), xshape)
        xs_stack = (1.0 - ts_stack) * eps + ts_stack * a_hat[None]

        # One critic gradient at the endpoint, shared by every bridge state.
        def critic_scalar(a_end):
            a_end = jnp.clip(a_end, -1.0, 1.0)
            return self.network.select("target_critic")(obs, a_end).mean(axis=0).sum()
        gQ = jax.grad(critic_scalar)(x)

        adj_stack = -gQ[None] * ts_stack                       # scalar adjoint, -tau * grad_a Q
        info = {
            "adj_mean": jnp.abs(adj_stack).mean(),
            "adj_max": jnp.abs(adj_stack).max(),
            "adj_std": jnp.abs(adj_stack).std(),
        }
        return xs_stack, adj_stack, ts_stack, info

    def actor_loss(self, batch, grad_params, rng, adjoint=None):
        """BC flow matching on `actor_slow`, and adjoint matching on `actor_fast`."""
        batch_actions = self._batch_actions(batch)
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow-matching loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('actor_slow')(batch['observations'], x_t, t, params=grad_params)
        flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])
        actor_loss = flow_loss

        if self.config["bc_only"]:
            return actor_loss, {'flow_loss': flow_loss, "fast_loss": 0.0}

        xs, adjs, ts, adj_info = adjoint if adjoint is not None else self.estimate_adjoint(batch["observations"])
        adjs = jax.lax.stop_gradient(adjs)
        n_states = xs.shape[0]
        h = 1.0 / n_states

        sigma_scale = self.get_sigma_scale()
        g_t_sq = 2 * (1 - ts + h) / (ts + h)
        g_t = jnp.sqrt(g_t_sq)
        sigmas = sigma_scale * g_t

        observations = jnp.repeat(batch["observations"][None], n_states, axis=0)
        vf_fine = self.network.select("actor_fast")(observations, xs, ts, params=grad_params)
        actor_slow = self.network.select(
            "target_actor_slow" if self.config["target_actor"] else "actor_slow")
        vf_base = actor_slow(observations, xs, ts)

        # Path-space KL to the pretrained flow, per action-chunk step.
        vel_diff_sq = jnp.sum((vf_fine - vf_base) ** 2, axis=-1)
        kl_per_step = (2 * h / g_t_sq[..., 0]) * vel_diff_sq
        path_kl = jnp.mean(jnp.sum(kl_per_step, axis=0))
        horizon = self.config['horizon_length'] if self.config["action_chunking"] else 1
        path_kl = path_kl / horizon

        # Adjoint matching
        adj_loss = jnp.sum(jnp.square((vf_fine - vf_base) * 2 / sigmas + sigmas * adjs), axis=-1)
        adj_loss = jnp.mean(jnp.sum(adj_loss, axis=0))

        info = {'flow_loss': flow_loss, "fast_loss": adj_loss, "adj_loss": adj_loss, "path_kl": path_kl,
                **adj_info}
        return actor_loss + adj_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        if self.config["bc_only"]:
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
            return actor_loss, info

        # One rollout serves both losses
        adjoint = self.estimate_adjoint(batch["observations"])
        a_pi = adjoint[0][-1]                                  # bridge state at tau = 1 is a_hat

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng, a_pi)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng, adjoint=adjoint)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    def dual_update(self, kl_estimate):
        kl_clip_max = self.config["kl_clip_coef"] * self.config["kl_budget"]
        kl_estimate_clipped = jnp.minimum(kl_estimate, kl_clip_max)
        new_kl_ema = (1 - self.config["kl_ema_coef"]) * self.kl_ema + \
                     self.config["kl_ema_coef"] * kl_estimate_clipped
        constraint_violation = new_kl_ema - self.config["kl_budget"]
        new_lam = jnp.maximum(self.config["lambda_min"],
                              self.lam + self.config["eta_lambda"] * constraint_violation)
        new_lam = jnp.minimum(new_lam, self.config["lambda_max"])
        return new_lam, new_kl_ema

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)

        if agent.config["bc_only"]:
            return agent.replace(network=new_network, rng=new_rng), info

        agent.target_update(new_network, 'critic')
        agent.target_update(new_network, 'actor_slow')
        new_lam, new_kl_ema = agent.dual_update(info['actor/path_kl'])

        lambda_value = agent.config["lam_scale"] * new_lam
        info['dual/lambda'] = lambda_value
        info['dual/sigma_scale'] = 1.0 / jnp.sqrt(jnp.maximum(lambda_value, 1e-8))
        info['dual/kl_ema'] = new_kl_ema
        info['dual/kl_budget'] = agent.config["kl_budget"]

        return agent.replace(network=new_network, rng=new_rng,
                             lam=new_lam, kl_ema=new_kl_ema), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(self, observations, rng):
        action_dim = self.config['action_dim'] * \
            (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        noises = jax.random.normal(
            rng, (*observations.shape[: -len(self.config['ob_dims'])],
                  self.config["best_of_n"], action_dim))
        observations = jnp.repeat(observations[..., None, :], self.config["best_of_n"], axis=-2)
        actions = self.compute_flow_actions(observations, noises, model="fast")
        actions = jnp.clip(actions, -1, 1)
        q = self.network.select("critic")(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)
        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[
            jnp.arange(bsize), indices, :].reshape(bshape + (action_dim,))
        return actions

    @partial(jax.jit, static_argnames="model")
    def compute_flow_actions(self, observations, noises, model="slow"):
        actions = noises
        networks = [self.network.select(f'actor_{m}') for m in model.split(",")]
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = sum([network(observations, actions, t) for network in networks])
            actions = actions + vels / self.config['flow_steps']
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        critic_def = Value(hidden_dims=config['value_hidden_dims'],
                           layer_norm=config['value_layer_norm'], num_ensembles=config['num_qs'])
        actor_def = ActorVectorField(hidden_dims=config['actor_hidden_dims'],
                                     layer_norm=config['actor_layer_norm'], action_dim=full_action_dim)

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)

        if config["clip_grad"]:
            network_tx = optax.chain(optax.clip_by_global_norm(max_norm=1.0),
                                     optax.adam(learning_rate=config["lr"]))
        else:
            network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor_slow'] = params['modules_actor_slow']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, lam=1.0, kl_ema=0.0,
                   config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='sqam',  # Agent name.
        ob_dims=ml_collections.config_dict.placeholder(list),   # Observation dimensions (will be set automatically).
        action_dim=ml_collections.config_dict.placeholder(int), # Action dimension (will be set automatically).
        
        ## Common hyperparamters
        lr=3e-4,  # Learning rate.
        batch_size=256,  # Batch size.
        actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
        actor_layer_norm=False,
        value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
        value_layer_norm=True,

        ## Q-chunking hyperparameters
        horizon_length=ml_collections.config_dict.placeholder(int), # Will be set
        action_chunking=False,                                      # Use Q-chunking or just n-step return

        ## RL hyperparameters
        num_qs=10,      # Critic ensemble size
        rho=0.5,        # Pessimistic backup

        discount=0.995,  # Discount factor.
        tau=0.005,      # Target network update rate.
        flow_steps=10,  # Number of flow steps.

        best_of_n=1,    # Best-of-n for computing Q-targets and sampling actions.
        
        # Trust region.
        lam_scale=3.0, kl_budget=0.5, eta_lambda=0.01, kl_ema_coef=0.1,
        lambda_min=0.01, lambda_max=100.0, kl_clip_coef=2.0,
        target_actor=True, clip_grad=True,
        n_bridge=10,         # number of bridge states K per endpoint
        gap_reg_coef=0.3,    # value penalty coefficient c (0 turns it off)
        bc_only=False,       # BC pretraining only
    ))
    return config
