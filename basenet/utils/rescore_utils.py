import math
from collections import defaultdict

def log_add(args) -> float:

    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp

def ctc_beam_search(ctc_probs, beam_size = 5):
    maxlen = ctc_probs.shape[0]
    cur_hyps = [(tuple(), (0.0, -float('inf')))]
    # 2. CTC beam search step by step
    for t in range(0, maxlen):
        logp = ctc_probs[t]  # (vocab_size,)
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
        # 2.1 First beam prune: select topk best
        top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
        for s in top_k_index:
            s = s.item()
            ps = logp[s].item()
            for prefix, (pb, pnb) in cur_hyps:
                last = prefix[-1] if len(prefix) > 0 else None
                if s == 0:  # blank
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pb = log_add([n_pb, pb + ps, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                elif s == last:
                    #  Update *ss -> *s;
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pnb = log_add([n_pnb, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                    # Update *s-s -> *ss, - is for blank
                    n_prefix = prefix + (s, )
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)
                else:
                    n_prefix = prefix + (s, )
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)

        # 2.2 Second beam prune
        next_hyps = sorted(next_hyps.items(),
                            key=lambda x: log_add(list(x[1])),
                            reverse=True)
        cur_hyps = next_hyps[:beam_size]
    hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
    return hyps