
import numpy as np
import torch
from objectiveNN import make_net, process_rankings
from scipy.stats import norm

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_weight(c1, c2, sqrt_weight=True):
  weight = (1.0 * c1 * c2) / (c1 + c2)
  if sqrt_weight:
    return weight ** 0.5
  else:
    return weight

def pairwise_wrapper(sqrt_weight, vectorized=False, elimination=False, abs_normalize=False):
  if vectorized:
    if elimination:
      def f(c, rankings):
        return vec_pairwise_weighted_elimination(c, rankings, sqrt_weight)
    else:
      def f(c, rankings):
        return vec_pairwise_weighted(c, rankings, sqrt_weight, abs_normalize=abs_normalize)
  else:
    if elimination:
      def f(c, rankings):
        return pairwise_weighted_elimination(c, rankings, sqrt_weight)
    else:
      def f(c, rankings):
        return pairwise_weighted(c, rankings, sqrt_weight)
  return f

def pairwise_weighted(c, rankings, sqrt_weight=True, return_loser=False):
  voters, alternatives = c.shape
  scores = np.zeros(alternatives)
  for i in range(alternatives):
    for j in range(i + 1, alternatives):
      for k in range(voters):
        score = get_weight(c[k][i], c[k][j], sqrt_weight)
        if rankings[k][i] < rankings[k][j]:
          pass # alternative i wins
        else:
          score = - score
        scores[i] += score
        scores[j] -= score
  if return_loser:
    return np.argmin(scores)
  return np.argmax(scores)

def vec_pairwise_weighted(vec_c, vec_rankings, sqrt_weight=True, return_loser=False, abs_normalize=False):
  voters, alternatives = vec_c.shape[1:]
  #totals = np.zeros((vec_c.shape[0], alternatives))
  totals = np.zeros(vec_c.shape + (alternatives,))
  for i in range(alternatives):
    for j in range(i + 1, alternatives):
      for k in range(voters):
        scores = get_weight(vec_c[:,k,i], vec_c[:,k,j], sqrt_weight)
        scores *= ((vec_rankings[:,k,i] < vec_rankings[:, k, j]).astype(np.float32) * 2. - 1.)
        totals[:,k,i,j] = scores
        totals[:,k,j,i] = -scores
  if abs_normalize:
    total_abs = np.sum(np.abs(totals), axis=-1)
    total_abs = np.sum(total_abs, axis=1)
    totals = np.sum(totals, axis=-1)
    totals = np.sum(totals, axis=1) / total_abs
  else:
    totals = np.sum(totals, axis=-1)
    totals = np.sum(totals, axis=1)

  if return_loser:
    return np.argmin(totals, axis=1)
  return np.argmax(totals, axis=1)

def pairwise_weighted_elimination(c, rankings, sqrt_weight=True):
  # if only 2 alts return top choice
  if rankings.shape[-1] == 2:
    return pairwise_weighted(c, rankings, sqrt_weight)

  # else, eliminate one and recurse
  losing_idx = pairwise_weighted(c, rankings, sqrt_weight=sqrt_weight, return_loser=True)

  new_alts = np.delete(np.arange(rankings.shape[-1]), losing_idx)
  new_c = np.delete(c, losing_idx, axis=1)
  new_rankings = np.delete(rankings, losing_idx, axis=1)
  return new_alts[pairwise_weighted_elimination(new_c, new_rankings, sqrt_weight)]


def vec_pairwise_weighted_elimination(c, rankings, sqrt_weight=True):
  batch_size, num_voters, num_alts = rankings.shape

  # if only 2 alts return top choice
  if num_alts == 2:
    return vec_pairwise_weighted(c, rankings, sqrt_weight)

  # else, eliminate one and recurse
  losing_idx = vec_pairwise_weighted(c, rankings, sqrt_weight=sqrt_weight, return_loser=True)
  old_alts = np.broadcast_to(np.arange(num_alts)[None], (len(rankings), num_alts))
  
  a = np.ones_like(old_alts)
  a[np.arange(batch_size), losing_idx] = 0
  bool_mask = a.astype(np.bool)

  new_alts = old_alts[bool_mask].reshape(batch_size, -1)
  
  bool_mask = np.broadcast_to(bool_mask[:, None, :], rankings.shape)
  new_c = c[bool_mask].reshape(batch_size, num_voters, -1)
  new_rankings = rankings[bool_mask].reshape(batch_size, num_voters, -1)
  return new_alts[np.arange(batch_size), vec_pairwise_weighted_elimination(new_c, new_rankings, sqrt_weight)]

def harmonic_borda(c, rankings):
    # Each voter is weighed by their harmonic mean
    # not well justified, just another baseline
    voters, alternatives = c.shape
    harmonic_mean = (alternatives * 1.0) / np.sum(1. / c, axis=1).reshape(voters, 1)
    points = alternatives - 1 * rankings
    points = points * harmonic_mean
    total_points = np.sum(points, axis=0)
    return np.argmax(total_points)

def harmonic_plurality(c, rankings):
    # Each voter is weighed by their harmonic mean
    # not well justified, just another baseline
    voters, alternatives = c.shape
    harmonic_mean = (alternatives * 1.0) / np.sum(1. / c, axis=1).reshape(voters, 1)
    points = (rankings == 1)
    points = points * harmonic_mean
    total_points = np.sum(points, axis=0)
    return np.argmax(total_points)

def irv(_, rankings):
  votes = np.sum(rankings == np.min(rankings, axis=-1, keepdims=True), axis=0)

  # if only 2 alts return top choice
  if rankings.shape[-1] == 2:
    return np.argmax(votes) 

  # else, eliminate one and recurse
  losing_idx = np.argmin(votes)
  new_alts = np.delete(np.arange(len(votes)), losing_idx)
  new_rankings = np.delete(rankings, losing_idx, axis=1)
  return new_alts[irv(_, new_rankings)]

def vec_irv(_, rankings):
  batch_size, num_voters, num_alts = rankings.shape

  votes = np.sum(rankings == np.min(rankings, axis=-1, keepdims=True), axis=1)

  # if only 2 alts return top choice
  if num_alts == 2:
    return np.argmax(votes, axis=1) 

  # else, eliminate one and recurse
  losing_idx = np.argmin(votes, axis=1)
  old_alts = np.broadcast_to(np.arange(num_alts)[None], (len(rankings), num_alts))
  
  a = np.ones_like(old_alts)
  a[np.arange(batch_size), losing_idx] = 0
  bool_mask = a.astype(np.bool)

  new_alts = old_alts[bool_mask].reshape(batch_size, -1)
  
  bool_mask = np.broadcast_to(bool_mask[:, None, :], rankings.shape)
  new_rankings = rankings[bool_mask].reshape(batch_size, num_voters, -1)
  return new_alts[np.arange(batch_size), vec_irv(_, new_rankings)]

def plurality(_, rankings):
  votes = np.sum(rankings == 1, axis=0)
  return np.argmax(votes)

def vec_plurality(_, vec_rankings):
  votes = np.sum(vec_rankings == 1, axis=1)
  return np.argmax(votes, axis=1)

def vec_naive_baselines_wrapper(type='borda', naive_type='amean', fn=np.sqrt):
  def f(c, rankings):
    batch, voters, alternatives = c.shape
    if naive_type=='amean':
      # arithmetic mean of each voter's counts
      factor = np.sum(c, axis=-1, keepdims=True)/ (alternatives * 1.0)
    elif naive_type=='hmean':
      # harmonic mean of each voter's counts
      factor = (alternatives * 1.0) / np.sum(1. / c, axis=-1, keepdims=True)
    elif naive_type=='counts':
      # use the counts directly
      factor = c
    elif naive_type=='countsmax':
      # use the counts directly
      added = np.sum(c, axis=-1, keepdims=True)
      factor = (c * added) /(c + added)
    else:
      raise NotImplementedError

    if fn is not None:
      factor = fn(factor)
    
    if type=='borda':
      points = alternatives - 1 * rankings
    elif type=='plurality':
      points = (rankings == 1)
    elif type=='mplurality':
      points = (rankings == 1)
      points = points * (1 + 1/(alternatives - 1)) - (1/(alternatives - 1))
      factor = (rankings == 1) * factor
      factor = np.sum(factor, axis=-1, keepdims=True)
    else:
      raise NotImplementedError

    points = points*factor
    total_points = np.sum(points, axis = 1)
    return np.argmax(total_points, axis = 1)
  
  return f

def borda(_, rankings):
  points = -1 * rankings
  total_points = np.sum(points, axis = 0)
  return np.argmax(total_points)

def vec_borda(_, vec_rankings):
  points = -1 * vec_rankings
  total_points = np.sum(points, axis = 1)
  return np.argmax(total_points, axis = 1)

def oracle(c, observations):
  return np.argmax(np.sum(c * observations * 1.0, axis = 0) / np.sum(c, axis=0))

def vec_oracle(vec_c, vec_observations):
  return np.argmax(np.sum(vec_c * vec_observations * 1.0, axis = 1) / np.sum(vec_c, axis=1), axis=1)

def vec_cardinal(_, vec_observations):
  return np.argmax(np.sum(vec_observations, axis = 1), axis=1)

def case5_wrapper(jensen=True, vectorized=True, sqrt=True):
  if vectorized:
    def f(c, rankings):
      return vec_case5(c, rankings, jensen, sqrt)
  else:
    def f(c, rankings):
      return case5(c, rankings, jensen, sqrt)
  return f


def case5(c, rankings, jensen=True, sqrt=True):
  rankings = (rankings == 1).astype(np.float32)

  outerprod = c[:,:,None]*c[:,None,:]
  outersum = c[:,:,None]+c[:,None,:]
  weight = outerprod/(outersum) * (1 - np.eye(rankings.shape[-1])[None, :, :])
  if sqrt:
    weight = np.sqrt(weight)
  weight *= -1
  if not jensen:
    weight += np.sum(weight, axis=2, keepdims=True) * (-np.eye(rankings.shape[-1])[None, :, :])
  scores = (rankings[:,:,None] * weight).sum(axis=1)
  case5_votes = np.argmax(np.sum(scores, 0), 0)
  return case5_votes

def vec_case5(c, rankings, jensen=True, sqrt=True):
  rankings = (rankings == 1).astype(np.float32)

  outerprod = c[:,:,:,None]*c[:,:,None,:]
  outersum = c[:,:,:,None]+c[:,:,None,:]
  weight = outerprod/(outersum) * (1 - np.eye(rankings.shape[-1])[None, None, :, :])
  if sqrt:
    weight = np.sqrt(weight)
  weight *= -1

  if not jensen:
    weight += np.sum(weight, axis=2, keepdims=True) * (-np.eye(rankings.shape[-1])[None, None, :, :])
  else:
    weight += np.sum(weight, axis=2, keepdims=True) * (-np.eye(rankings.shape[-1])[None, None, :, :])
    weight *= np.eye(rankings.shape[-1])[None, None, :, :]

  scores = (rankings[:,:,:,None] * weight).sum(axis=2)
  case5_votes = np.argmax(np.sum(scores, 1), 1)
  return case5_votes

def make_deepset_rule(max_experience=50, saved_model = 'voting_rule_model.pt', vectorized=False):
  net, pred_fn, _, _ = make_net(2, 'hard')
  net.load_state_dict(torch.load(saved_model))
  net.eval()

  def rule(c, rankings):
    if not vectorized:
      rankings = rankings[np.newaxis]
      c = c[np.newaxis]
    r = process_rankings(rankings, 'normalize')
    c = c / max_experience
    inputs = torch.stack((torch.tensor(c, dtype=torch.float32), torch.tensor(r, dtype=torch.float32)), dim=-1).to(dev)
    if not vectorized:
      with torch.no_grad():
        return np.argmax(pred_fn(inputs).flatten().cpu().detach().numpy())
    else:
      with torch.no_grad():
        return np.argmax(pred_fn(inputs).cpu().detach().numpy(), axis=1)
  return rule

def gradient_rule(c, rankings, est_pull_variance=10., samples=100):
  votes = (rankings == 1).astype(np.float32)
  batch_size, num_voters, num_alternatives = rankings.shape
  predicted_winners = np.zeros(batch_size)

  counts = c
  gradients = np.zeros((batch_size, num_voters, num_alternatives))
  sum_f_i = np.zeros((batch_size, num_voters))

  for s in range(samples):
    raw_sample = np.random.randn(batch_size, num_voters) # batch x voters
    winner_counts = np.sum(counts*votes, axis=-1) # batch x voters
    sample = raw_sample * np.sqrt(est_pull_variance / winner_counts) # batch x voters ;;   This is the "x" in the math

    # accumulate f_i
    cdfs_at_x = norm.cdf(sample[:,:,None], loc=0, scale=np.sqrt(est_pull_variance / counts)) # batch x votes x alts
    cdfs_at_x = votes + (1 - votes)*cdfs_at_x # batch x votes x alts
    s_i_x = np.prod(cdfs_at_x, axis = -1) # batch x voters
    sum_f_i += s_i_x

    # compute g(i, x) * s(i, x)
    g_i_x = sample * winner_counts / est_pull_variance # batch x voters ;;; 1/sigma^2 *  (x - mu_j)
    g_i_x = g_i_x[:,:,None] * votes # gradient is 0 everywhere except winner

    g_i_x_times_s_i_x = s_i_x[:,:,None] * g_i_x # batch x voters x alts

    # compute grad g(i, x)
    grad_s_i_x = s_i_x[:,:,None] / (cdfs_at_x + 1e-8) * (1 - votes)# batch x voters x alts ;;; winner is set to 0
    grad_s_i_x *= norm.pdf(sample[:, :, None], loc=0, scale=np.sqrt(est_pull_variance / counts))
    grad_s_i_x *= -1 # since we are taking derivative w/r/t mu instead of x. 
    
    gradients += g_i_x_times_s_i_x + grad_s_i_x # batch x voters x alts

  f_i = (sum_f_i / samples)[:,:,None] # batch x voters x alts
  gradients /= f_i

  gradient = np.sum(gradients, axis=1) # batch x alts

  predicted_winners = np.argmax(gradient, axis=1) # batch
  return predicted_winners.astype(np.int)