
import argparse
import numpy as np
from vote_rules import *
import tqdm
import os, pickle
from copy import deepcopy

VOTE_RULES = {
  # anonymous baselines
  "oracle": vec_oracle,
  "borda": vec_borda,
  "plurality": vec_plurality,
  # our rules (learned rule set below)
  "case4": pairwise_wrapper(sqrt_weight=True, vectorized=True),
  "case4norm": pairwise_wrapper(sqrt_weight=True, vectorized=True, abs_normalize=True),
  "case5lower": case5_wrapper(jensen=False, vectorized=True),
  "case5zero": vec_naive_baselines_wrapper('plurality', 'counts', fn=np.sqrt),
  "case5monte": gradient_rule,
  # naively non-anonymous baselines
  "hplurality": vec_naive_baselines_wrapper('plurality', 'hmean', fn=None),
  "hplurality_sqrt": vec_naive_baselines_wrapper('plurality', 'hmean', fn=np.sqrt),
  "hplurality_log": vec_naive_baselines_wrapper('plurality', 'hmean', fn=np.log),
  "aplurality": vec_naive_baselines_wrapper('plurality', 'amean', fn=None),
  "aplurality_sqrt": vec_naive_baselines_wrapper('plurality', 'amean', fn=np.sqrt),
  "aplurality_log": vec_naive_baselines_wrapper('plurality', 'amean', fn=np.log),
  "hborda": vec_naive_baselines_wrapper('borda', 'hmean', fn=None),
  "hborda_sqrt": vec_naive_baselines_wrapper('borda', 'hmean', fn=np.sqrt),
  "hborda_log": vec_naive_baselines_wrapper('borda', 'hmean', fn=np.log),
  "aborda": vec_naive_baselines_wrapper('borda', 'amean', fn=None),
  "aborda_sqrt": vec_naive_baselines_wrapper('borda', 'amean', fn=np.sqrt),
  "aborda_log": vec_naive_baselines_wrapper('borda', 'amean', fn=np.log),
}

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def get_arm_means(num_alternatives, mean=0, sd=1):
  arm_means = np.random.normal(loc=mean, scale=sd, size=(num_alternatives))
  return arm_means

def vec_get_arm_means(batch_size, num_alternatives, mean=0, sd=1):
  arm_means = np.random.normal(loc=mean, scale=sd, size=(batch_size, num_alternatives))
  return arm_means

def get_observation_counts(num_voters, num_alternatives, voter_experience='default', max_experience=50):
  """voter_experience specifies how voter experience is generated

  ``default": each voter has experience for a given alternative uniformly sampled from 1 to `max_experience'
  ``stratified": each voter has an individual max experience. Their pulls are uniformly sampled
  ``diverse": TODO (currently unused, need to update temperature faction of 3)
  """
  if voter_experience == "default":
    return np.random.randint(1, max_experience, size=(num_voters, num_alternatives))
  # choose bound for each voter first (so some voters are weighed more heavily)
  elif voter_experience == "stratified":
    obs_count = np.zeros((num_voters, num_alternatives))
    for i in range(num_voters):
      max_exp_i = np.random.randint(2, max_experience)
      obs_count[i] = np.random.randint(1, max_exp_i, size=(num_alternatives))
    return obs_count
  elif voter_experience == 'diverse':
    raise NotImplementedError("fix temperature")
    experience_partition = softmax(np.random.random((num_voters, num_alternatives))*3)
    total_experiences = np.random.randint(max_experience//3, max_experience, size=(num_voters, 1))
    return np.maximum(np.ceil(total_experiences * experience_partition), 1).astype(np.int64)
  else:
    raise NotImplementedError

def vec_get_observation_counts(batch_size, num_voters, num_alternatives, 
                               voter_experience='default', max_experience=50):
  assert voter_experience == 'default'
  return np.random.randint(1, max_experience, size=(batch_size, num_voters, num_alternatives))
 
def generate_observations(num_voters, num_alternatives, arm_means, c, pull_variance):
  """Given arm means and number of observations made by each voter,
  generate experience that each voter observes
  Returns:
      observations (np.array num_voters x num_alternatives):
            observations[i][j] is average pull experienced by voter i on alternative j
      rankings (np.array num_voters x num_alternatives):
            rankings[i][j] is ranking of voter i for alternative j; lower rank is better (1 is best)
  """
  if len(arm_means.shape) == 1:
    a = arm_means[None]
  else:
    a = arm_means[:,None]
  observations = np.random.normal(a, np.sqrt(pull_variance / c))
  rankings = num_alternatives - observations.argsort().argsort()
  
  return observations, rankings

vec_generate_observations = generate_observations

def gen_trial(num_voters, num_alternatives, voter_experience='default', 
              max_experience=50, pull_variance=100., count_noise=0., random_count_prob=0.):
  """Generates data for a single trial
  Adds count noise and random count probabilities.
  """
  arm_means = get_arm_means(num_alternatives)
  c = get_observation_counts(num_voters, num_alternatives, voter_experience, max_experience) # v x a
  observations, rankings = generate_observations(num_voters, num_alternatives, arm_means, c, pull_variance)

  # get noisy counts
  noise = np.random.uniform(high=2*count_noise, size=c.shape) + (1 - count_noise)
  c = np.clip(np.round(c * noise), 1, None).astype(np.int64)

  # get replaced counts
  alt_c = get_observation_counts(num_voters, num_alternatives, voter_experience, max_experience) # v x a
  mask = (np.random.uniform(size=c.shape) > random_count_prob).astype(np.float32)
  c = c * mask + alt_c * (1. - mask)

  return arm_means, c, observations, rankings

def vec_gen_trial(batch_size, num_voters, num_alternatives, voter_experience='default', 
              max_experience=50, pull_variance=100., count_noise=0., random_count_prob=0.):
  
  arm_means = vec_get_arm_means(batch_size, num_alternatives)
  c = vec_get_observation_counts(batch_size, num_voters, num_alternatives, voter_experience, max_experience) # v x a
  observations, rankings = vec_generate_observations(num_voters, num_alternatives, 
                                                     arm_means, c, pull_variance)
  
  # get noisy counts
  noise = np.random.uniform(high=2*count_noise, size=c.shape) + (1 - count_noise)
  c = np.clip(np.round(c * noise), 1, None).astype(np.int64)

  # get replaced counts
  alt_c = vec_get_observation_counts(batch_size, num_voters, num_alternatives, voter_experience, max_experience) # v x a
  mask = (np.random.uniform(size=c.shape) > random_count_prob).astype(np.float32)
  c = c * mask + alt_c * (1. - mask)  
  
  return arm_means, c, observations, rankings

def simulate_all_rules(trials, batch_size, num_voters, num_alternatives, voter_experience = 'default', 
                       max_experience = 50, pull_variance = 100., count_noise = 0., random_count_prob = 0., old_results=None):

  VOTE_RULES['learned'] = make_deepset_rule(max_experience,
      vectorized=True, saved_model='models/voting_rule_modelcn-0_rcp-0___0.0003-4-2-max-sum.pt')
  VOTE_RULES['learned_noisy'] = make_deepset_rule(max_experience,
      vectorized=True, saved_model='models/voting_rule_modelcn-50_rcp-0___0.0003-4-2-max-sum.pt')
    
  correct_prediction = {k:0.0 for k in VOTE_RULES}
  regret = {k: 0.0 for k in VOTE_RULES}

  if old_results is not None:
    DONE_RULES = set(old_results['correct_prediction'].keys())
    for rule in DONE_RULES:
      correct_prediction[rule] = old_results['correct_prediction'][rule]
      regret[rule] = old_results['regret'][rule]
    ACTIVE_RULES = {k:v for k, v in VOTE_RULES.items() if not k in DONE_RULES}
    
  else:
    ACTIVE_RULES = VOTE_RULES

  gt = lambda : gen_trial(num_voters, num_alternatives, voter_experience, 
                                                      max_experience, pull_variance, count_noise, random_count_prob)

  for _ in tqdm.tqdm(range(trials // batch_size)):
      
    #arm_means, c, observations, rankings = [np.array(a) for a in zip(*[gt() for _ in range(batch_size)])]
    #this is only really faster for low voter count:
    arm_means, c, observations, rankings = vec_gen_trial(batch_size, num_voters, num_alternatives, voter_experience, 
                                                      max_experience, pull_variance, count_noise, random_count_prob)
    
    # get winners
    winners = np.argmax(arm_means, axis = 1)

    for vote_rule in ACTIVE_RULES:
      if vote_rule == "oracle":
        predictions = ACTIVE_RULES[vote_rule](c, observations)
      else:
        predictions = ACTIVE_RULES[vote_rule](c, rankings)

      correct_prediction[vote_rule] += np.sum(predictions == winners)
      regret[vote_rule] += np.sum(arm_means[np.arange(len(arm_means)), winners] - arm_means[np.arange(len(arm_means)), predictions])

  for vote_rule in ACTIVE_RULES:
    correct_prediction[vote_rule] = round(correct_prediction[vote_rule] / trials, 5)
    regret[vote_rule] = round(regret[vote_rule] / trials, 5)

  return {"correct_prediction": correct_prediction,
          "regret": regret}

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Bandit simulation")
  parser.add_argument('--trials', default=1000000, type=int, help="number of simulations to run")
  parser.add_argument('--batch_size', default=500, type=int, help="size of each batch for evaluation")
  parser.add_argument('--num_voters', default=10, type=int, help="number of voters")
  parser.add_argument('--num_alternatives', default=10, type=int, help="number of alternatives")
  parser.add_argument('--ve', default="default", type=str, help="voter experience scheme")
  parser.add_argument('--max_experience', default=50, type=int, help="max pulls for a single voter. Interacts with voter experience scheme")
  parser.add_argument('--pull_variance', default=1000.0, type=float, help="variance of a given pull")
  parser.add_argument('--count_noise', default=0.0, type=float, help="max absolute percentage count noise")
  parser.add_argument('--random_count_prob', default=0.0, type=float, help="probability of random count info")
  parser.add_argument('--redo', action='store_true', help="whether to perform a fresh run, even if results saved")
  parser.add_argument('--save', action='store_true')

  args = parser.parse_args()

  # Load results if they do not exist
  if os.path.exists('results.pickle'):
    print("Loading previous results..")
    with open('results.pickle', 'rb') as f:
      results = pickle.load(f)
  else:
    results = {}

  exp_key = (args.trials, args.num_voters, round(100*args.count_noise), round(100*args.random_count_prob),\
        round(args.pull_variance))

  if exp_key in results and not args.redo:
    out = simulate_all_rules(args.trials, args.batch_size, args.num_voters, args.num_alternatives, args.ve, 
                            args.max_experience, args.pull_variance, args.count_noise, args.random_count_prob, results[exp_key])
  else:
    out = simulate_all_rules(args.trials, args.batch_size, args.num_voters, args.num_alternatives, args.ve,
                            args.max_experience, args.pull_variance, args.count_noise, args.random_count_prob)

  if os.path.exists('results.pickle'):
    with open('results.pickle', 'rb') as f:
      results = pickle.load(f)
  else:
    results = {}

  results[exp_key] = out

  if args.save:
    with open('results.pickle', 'wb') as f:
      pickle.dump(results, f)

  print(out)