import config as cfg
from models.transformer_actor import TransformerActor
from routing.dlbh import DLBH
from algorithms.ppo_ctde import Transition
import torch

actor = TransformerActor()
f = torch.zeros(1, 4, cfg.DIM_IN)
m = torch.zeros(1, 4, 4)
v = torch.ones(1, 4)
d = torch.ones(1, 4) * 0.5
out = actor(f, m, v, d)
print('Actor OK:', out.shape)

dlbh = DLBH(10)
print('DLBH OK')

t = Transition(feats=torch.zeros(4,6), mob=torch.zeros(4,4),
    mask=torch.ones(4), action=torch.tensor(0), logp=torch.tensor(0.0),
    reward=0.0, done=False, global_state=torch.zeros(10), dist=torch.ones(4)*0.5)
print('Transition.dist OK:', t.dist.shape)

logp, ent = actor.evaluate_actions(f, m, v, torch.tensor([0]), d)
print('evaluate_actions OK:', logp.shape)

print('ALL PASS')
