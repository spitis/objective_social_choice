import argparse
from objectiveNN import *
import torch
from simulate_bandits import gen_trial


def gen_batch(batch_size,
              num_voters,
              num_alternatives,
              voter_experience='default',
              max_experience=50,
              pull_variance=100.,
              count_noise=0.,
              random_count_prob=0.):
  """generates a batch of trials of size batch_size"""
  gt = lambda: gen_trial(num_voters, num_alternatives, voter_experience, max_experience, pull_variance, count_noise, random_count_prob)
  arm_means, c, _, rankings = list(zip(*[gt() for _ in range(batch_size)]))
  return torch.tensor(arm_means).float(), torch.tensor(c).float(), torch.tensor(rankings).float()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Training neural nets to predict objective social choice")

  # data generation hyperparameters
  parser.add_argument('--iterations', default=5001, type=int)
  parser.add_argument('--batch_size', default=128, type=int, help="batch_size")
  parser.add_argument('--num_voters', default=-1, type=int, help="number of voters")
  parser.add_argument('--num_alternatives', default=-1, type=int, help="number of alternatives")
  parser.add_argument('--ve', default="default", type=str, help="voter experience scheme")
  parser.add_argument('--max_experience', default=50, type=int, help="max pulls a voter can experience")
  parser.add_argument('--pull_variance', default=1000., type=int, help="variance of a given pull")
  parser.add_argument('--count_noise', default=0., type=float, help="max absolute percentage count noise")
  parser.add_argument('--random_count_prob', default=0.0, type=float, help="probability of random count info")

  # preprocessing hyperparameters
  # parser.add_argument('--aggregation', default="augment", type=str, help="how to input attention weights") #todo: not used
  parser.add_argument('--target_type', default="hard", type=str, help="how targets are processed for loss function")
  parser.add_argument('--ranking_process', default="normalize", type=str, help="how preference profiles are fed")

  # network/training hyperparams
  parser.add_argument('--network_type', default='deepset', type=str, help="use deepset vs aggnet")
  parser.add_argument('--lr', default=3e-4, type=float, help="learning rate")
  parser.add_argument('--enc_layers', default=4, type=int, help="encoder layers")
  parser.add_argument('--dec_layers', default=2, type=int, help="decoder layers")
  parser.add_argument('--hidden_size', default=64, type=int, help="hidden size")
  parser.add_argument('--equivariant_fn', default='max', type=str)
  parser.add_argument('--invariant_fn', default='sum', type=str)

  parser.add_argument('--random_params', action='store_true')
  parser.add_argument('--print_every', default=100, type=int)
  parser.add_argument('--tag', default='', type=str, help="additional tag for model name")

  args = parser.parse_args()
  num_voters, num_alternatives = args.num_voters, args.num_alternatives

  if args.random_params:
    np.random.seed()
    args.lr = np.random.choice([3e-3, 1e-3, 3e-4, 1e-4])
    args.enc_layers = np.random.choice([2, 3, 4])
    args.dec_layers = np.random.choice([0, 1, 2])
    args.equivariant_fn = np.random.choice(['mean', 'max'])
    args.invariant_fn = np.random.choice(['mean', 'sum'])
    # 144 Total configs

  if args.ranking_process == "pairwise":
    assert num_alternatives > 0
    # pairwise preferences and count information
    num_input_feats = num_alternatives * (num_alternatives - 1) // 2 + num_alternatives
  elif args.ranking_process == 'normalize':
    num_input_feats = 2
  else:
    raise NotImplementedError

  net, pred_fn, loss_fn, optimizer = make_net(num_input_feats, args.target_type, args.lr, args.enc_layers,
                                              args.dec_layers, args.hidden_size, args.equivariant_fn, args.invariant_fn)

  # initial values for exponential moving average
  _accuracy = 0.1
  _loss = 10.

  best_loss = np.inf
  for i in range(args.iterations):
    if num_voters < 0:
      _num_voters = np.random.randint(5, 350)
    else:
      _num_voters = num_voters

    if num_alternatives < 0:
      _num_alts = np.random.randint(5, 15)
    else:
      _num_alts = num_alternatives

    train_means, train_c, train_rankings = gen_batch(args.batch_size, _num_voters, _num_alts, args.ve,
                                                     args.max_experience, args.pull_variance, args.count_noise,
                                                     args.random_count_prob)
    targets = get_targets(train_means, args.target_type).to(dev)
    train_rankings = process_rankings(train_rankings, args.ranking_process)
    train_c = train_c / args.max_experience

    # want bs x (ranking + count info) x voter x 1
    inputs = torch.stack((train_c, train_rankings), dim=-1).to(dev)
    predictions = pred_fn(inputs)

    optimizer.zero_grad()
    loss = loss_fn(predictions, targets)
    loss.backward()
    optimizer.step()

    if _loss < 0:
      _loss = loss.item()
    else:
      _loss = 0.99 * _loss + 0.01 * loss.item()
    _accuracy = 0.99 * _accuracy + 0.01 * torch.sum(
        torch.argmax(predictions, dim=1) == targets).item() * 1.0 / args.batch_size

    if i % args.print_every == 0:
      print(i)
      print("Loss: ", _loss)
      print("Accuracy: ", _accuracy)
      if _loss < best_loss:
        print("New best loss! Saving model ... ")
        best_loss = _loss
        model_name = 'models/voting_rule_modelcn-{}_rcp-{}_{}__{}.pt'.format(
                round(args.count_noise * 100), round(args.random_count_prob * 100), args.tag,
                '{}-{}-{}-{}-{}'.format(args.lr, args.enc_layers, args.dec_layers, args.equivariant_fn, args.invariant_fn))
        torch.save(net.state_dict(), model_name)
        with open('{}.txt'.format(model_name), 'w') as f:
          f.write(model_name)
          f.write('\nAccuracy: {}'.format(_accuracy))
          f.write('\nLoss: {}'.format(_loss))
