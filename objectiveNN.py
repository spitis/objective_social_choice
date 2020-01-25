import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EquivariantLayer(nn.Module):
  """basic permutation equivariant layer: which is an independent linear layer of 
  each item, minus a permutation invariant function (max or mean) of each item """

  def __init__(self, in_dim, out_dim, mode='max'):
    super().__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.mode = mode

  def forward(self, x):
    if self.mode == 'max':
      xm, _ = x.max(-2, keepdim=True)
    elif self.mode == 'mean':
      xm = x.mean(-2, keepdim=True)
    else:
      raise NotImplementedError
    x = self.Gamma(x-xm)
    return x

class DeepSet(nn.Module):
  # Basic Deep Set NN; https://arxiv.org/abs/1703.06114
  def __init__(self, phi: nn.Module, rho: nn.Module, mode='sum'):
    super().__init__()
    self.phi = phi
    self.rho = rho
    self.mode = mode

  def forward(self, x):
    x = self.phi.forward(x) # x should be [|batch| x] |set| x |features|
    if self.mode == 'sum':
      x = torch.sum(x, dim=-2) # sum across set
    elif self.mode =='mean':
      x = torch.mean(x, dim=-2)
    else:
      raise NotImplementedError
    return F.log_softmax(self.rho.forward(x), dim=-1) # map to output


def make_mlp(input_size, hidden_size, output_size, module=nn.Linear, module_kwargs={}, num_hidden_layers=2):
  """ function to make a feedforward network """
  layers = [module(input_size, hidden_size), nn.ReLU()]
  for layer in range(num_hidden_layers - 1):
    layers += [module(hidden_size, hidden_size, **module_kwargs), nn.ReLU()]
  layers += [module(hidden_size, output_size, **module_kwargs)]
  return nn.Sequential(*layers)

class AggregationNN(nn.Module):
  """ This is same thing as DeepSet but implemented slightly differently using 1D conv. """
  def __init__(self, len_inputs, num_alternatives, hidden=30, embedding=30):
    super(AggregationNN, self).__init__()
    # 1x1 convolutions change just the channel size and preserve the spatial dimension
    self.input_to_hidden = nn.Conv2d(in_channels=len_inputs, out_channels=embedding, kernel_size=1)
    self.hidden_to_embedding = nn.Conv2d(in_channels=embedding, out_channels=hidden, kernel_size=1)
    self.embed_to_hidden = nn.Linear(hidden, hidden)
    self.hidden_to_output = nn.Linear(hidden, num_alternatives)

  def forward(self, inputs):
    # ipdb.set_trace()
    h1 = F.relu(self.input_to_hidden(inputs))
    embeddings = F.relu(self.hidden_to_embedding(h1))
    embeddings = torch.squeeze(embeddings, -1) # get rid of dummy spatial dim
    embeddings = torch.sum(embeddings, -1) # sum across voter set

    h2 = F.relu(self.embed_to_hidden(embeddings))
    out = F.log_softmax(self.hidden_to_output(h2), dim=1)
    return out

def get_targets(means, target_type = 'hard'):
  if target_type == "soft":
    return F.softmax(means,dim=1)
  elif target_type == "hard":
    return torch.argmax(means, dim=1)
  raise NotImplementedError

def process_rankings(rankings, ranking_process):
  batch_size, num_voters, num_alternatives = rankings.shape

  if ranking_process == "pairwise_old":
    processed_rankings = np.zeros((batch_size, num_voters, num_alternatives * (num_alternatives - 1) // 2))
    for n in range(batch_size):
      for v in range(num_voters):
        count = 0
        # ipdb.set_trace()
        for i in range(num_alternatives):
          for j in range(i + 1, num_alternatives):
            processed_rankings[n][v][count] = rankings[n][v][i] > rankings[n][v][j]
            count += 1

  elif ranking_process =='pairwise':
    pairwise = (np.expand_dims(rankings, -1) > np.expand_dims(rankings, -2))
    shape = pairwise.shape[:-2] + (-1,)
    pairwise = pairwise[np.triu(np.ones_like(pairwise), 1)]
    pairwise = pairwise.reshape(*shape)
    processed_rankings = pairwise.astype(np.float32)

  elif ranking_process == "normalize":
    processed_rankings = (num_alternatives - rankings) / (num_alternatives - 1) 

  else:
    raise NotImplementedError
  return processed_rankings

class UnsqueezeLast(nn.Module):
  def forward(self, x):
    return x.unsqueeze(-1)

def make_net(num_input_feats, target_type, lr=3e-4, enc_layers=4, dec_layers=2, hidden_size=64,
  equivariant_fn='max', invariant_fn='sum'):
  """Returns prediction_fn, loss function, optimizer

  here the input to net2 will be [batch_size x num_voters x num_alts x num_input features]
  """

  if target_type == "soft":
    loss_fn = torch.nn.KLDivLoss() # inputs should be log probabilities, targets should be
  elif target_type == "hard":
    loss_fn = torch.nn.NLLLoss()
  else:
    raise NotImplementedError
  
  if dec_layers > 0:
    decoder = [UnsqueezeLast(), make_mlp(input_size=1, hidden_size=hidden_size, output_size=1, num_hidden_layers=dec_layers, 
               module=EquivariantLayer, module_kwargs={'mode':equivariant_fn}), nn.Flatten(-2, -1)]

  net = DeepSet(
    nn.Sequential(
      make_mlp(input_size=num_input_feats, hidden_size=hidden_size, output_size=1, num_hidden_layers=enc_layers, 
               module=EquivariantLayer, module_kwargs={'mode':equivariant_fn}), 
      nn.Flatten(-2, -1)
    ),
    nn.Identity() if dec_layers == 0 else nn.Sequential(*decoder),
    mode=invariant_fn
  ).to(dev)

  def pred_fn(inputs):
    return net(inputs)

  optimizer = torch.optim.Adam(net.parameters(), lr=lr)

  return net, pred_fn, loss_fn, optimizer
  