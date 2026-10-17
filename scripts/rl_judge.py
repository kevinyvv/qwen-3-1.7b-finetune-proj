"""GRPO with the LLM judge as the reward (docs/experiment_history.md, section 13).

    python -m scripts.rl_judge --run_dir=runs/rl_base_masked --prompts=data/prompts/rl_pool.jsonl
    python -m scripts.rl_judge --run_dir=runs/rl_sft180_masked --prompts=data/prompts/rl_pool.jsonl \
        --start_adapter=runs/sft_sizes_20261016/train_180/checkpoints/best.pt

Each step samples group_size answers for each of prompts_per_step prompts with the
mask on, scores every answer with the judge (both scores combined into 0..1, by
their sum or, with --reward_combine=lowest, by the lower one), and makes one GRPO
update. --fixed_length_loss divides each answer's loss by max_new_tokens instead of
by its own length. Answers whose judge call failed are left out of the update.
Every eval_every steps it saves the adapter, runs the short eval and saves training
answers to read; it stops after max_steps, or when the usable rate falls at two
evals in a row. The drift penalty anchors to the starting point (base, or the SFT
adapter when one is given).
"""
import json
from pathlib import Path
import random
import time

import torch

from core.checks import e_pass
from core.judge import judge_many, make_judge_client, reward, usable
from core.model import Qwen3ForCausalLM
from core.sampling import build_forbidden_token_mask
from core.utils import get_encoder, load_qwen3_weights
from scripts.evaluate import NO_E_INSTRUCTION, evaluate
from scripts.judge_eval import judge_run
from training.checkpointing import load_checkpoint
from training.rl.grpo import GRPOEngine

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_model(start_adapter=None):
    cfg, state = load_qwen3_weights()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen3ForCausalLM(cfg, use_sdpa=True)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.load_state_dict(state, strict=False)
    del state
    if start_adapter is not None:
        load_checkpoint(start_adapter, model, None)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.endswith(('.A', '.B'))
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return model.cuda()


def prompt_batches(rows, tokenizer, prompts_per_step, seed):
    """Endless shuffled batches; each pass over the pool uses a new order."""
    rng = random.Random(seed)
    while True:
        order = rows[:]
        rng.shuffle(order)
        for start in range(0, len(order) - prompts_per_step + 1, prompts_per_step):
            chunk = order[start:start + prompts_per_step]
            texts = [tokenizer.apply_chat_template(
                [{'role': 'system', 'content': NO_E_INSTRUCTION}, {'role': 'user', 'content': row['prompt']}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in chunk]
            batch = tokenizer(texts, padding=True, return_tensors='pt')
            yield dict(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                       prompts=[row['prompt'] for row in chunk], ids=[row['id'] for row in chunk])


class JudgeReward:
    """reward_fn for GRPOEngine: decodes answers, judges them in parallel, logs every answer."""

    def __init__(self, tokenizer, client, log_path, masked, workers=16, combine='sum'):
        self.tokenizer, self.client, self.masked, self.workers = tokenizer, client, masked, workers
        self.combine = combine
        self.log = Path(log_path).open('a')
        self.step, self.ids, self.last, self.judge_seconds = 0, None, [], 0.0

    def __call__(self, tokens, reward_mask, prompts):
        groups, batch = tokens.shape[:2]
        eos = self.tokenizer.eos_token_id
        answers, items = {}, []
        for g in range(groups):
            for b in range(batch):
                answer = self.tokenizer.decode(tokens[g, b][reward_mask[g, b]].tolist(),
                                               skip_special_tokens=True).strip()
                cut_off = not bool((tokens[g, b] == eos).any())
                answers[(g, b)] = (answer, cut_off)
                # without the mask, an answer with an e gets 0 and is not judged
                if self.masked or e_pass(answer):
                    items.append(((g, b), prompts[b], answer, cut_off))
        started = time.monotonic()
        judgments = judge_many(self.client, items, workers=self.workers)
        self.judge_seconds = time.monotonic() - started
        scores = torch.full((groups, batch), float('nan'))
        self.last = []
        for (g, b), (answer, cut_off) in answers.items():
            judgment = judgments.get((g, b))
            ok = e_pass(answer)
            if judgment is None:
                scores[g, b] = 0.0 # contains e, unmasked run
            elif not judgment['error']:
                scores[g, b] = reward(judgment, self.combine)
            record = dict(step=self.step, id=self.ids[b] if self.ids else None, group_index=g,
                          prompt=prompts[b], answer=answer, cut_off=cut_off, e_pass=ok,
                          reward=None if torch.isnan(scores[g, b]) else scores[g, b].item(),
                          usable=usable(ok, True, judgment), judgment=judgment)
            self.last.append(record)
            self.log.write(json.dumps(record) + '\n')
        self.log.flush()
        return scores


def main(run_dir, prompts, start_adapter=None, masked=True, max_steps=300, prompts_per_step=4,
         group_size=8, temperature=0.7, max_new_tokens=300, learning_rate=2e-5, kl_beta=0.02,
         eps=0.2, micro_batch_size=4, eval_every=25, eval_prompts='data/prompts/short_eval_50.jsonl',
         eval_seeds=(0, 1, 2, 3), seed=0, judge_workers=16, start_eval=None, final_eval=True,
         reward_combine='sum', fixed_length_loss=False):
    """start_eval: an existing short-eval run of the starting point with the same eval
    settings; its usable rate is recorded as step 0. final_eval=False skips the eval at
    max_steps (for short smoke tests)."""
    reward(dict(coherence=5, responds=5), reward_combine)  # an unknown name fails before any work
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f'Choose a new run_dir: {run_dir}')
    run_dir.mkdir(parents=True)
    pool = read_rows(prompts)
    evals = read_rows(eval_prompts)
    overlap = {row['prompt'] for row in pool} & {row['prompt'] for row in evals}
    if overlap:
        raise ValueError(f'{len(overlap)} training prompts are eval prompts')
    config = dict(prompts=str(prompts), pool_size=len(pool), start_adapter=start_adapter, masked=masked,
                  max_steps=max_steps, prompts_per_step=prompts_per_step, group_size=group_size,
                  temperature=temperature, max_new_tokens=max_new_tokens, learning_rate=learning_rate,
                  kl_beta=kl_beta, eps=eps, micro_batch_size=micro_batch_size, eval_every=eval_every,
                  eval_prompts=str(eval_prompts), eval_seeds=list(eval_seeds), seed=seed,
                  start_eval=start_eval, reward_combine=reward_combine,
                  loss_length=max_new_tokens if fixed_length_loss else 'each answer\'s own length',
                  system_prompt=NO_E_INSTRUCTION, thinking=False, ascii_only_mask=masked,
                  drift_anchor='start adapter' if start_adapter else 'base model (adapters off)')
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2))

    # On WSL, going past the GPU's memory spills into system memory and slows every step
    # several-fold instead of failing; cap it so that shows up as an out-of-memory error.
    free, total = torch.cuda.mem_get_info()
    cap = min(14 * 1024**3, free - 1024**3)
    torch.cuda.set_per_process_memory_fraction(cap / total)
    torch.manual_seed(seed)
    tokenizer = get_encoder()
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(start_adapter)
    # anchor the drift penalty to the starting adapters by swapping their values in,
    # instead of a second copy of the model (which does not fit next to training)
    ref_adapter_state = None
    if start_adapter is not None:
        ref_adapter_state = {name: p.detach().clone() for name, p in model.named_parameters()
                             if name.endswith(('.A', '.B'))}
    forbidden = build_forbidden_token_mask(tokenizer, model.config.vocab_size, 'eE', ascii_only=True,
                                           device='cuda') if masked else None
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    engine = GRPOEngine(model, tokenizer, optimizer, loss_backend='torch',
                        ref_adapter_state=ref_adapter_state, forbidden_token_mask=forbidden)
    engine.epoch, engine.global_step = 0, 0
    reward_fn = JudgeReward(tokenizer, make_judge_client(), run_dir / 'answers.jsonl', masked, judge_workers,
                            reward_combine)
    batches = prompt_batches(pool, tokenizer, prompts_per_step, seed)
    metrics = (run_dir / 'metrics.jsonl').open('a')
    state = dict(step=0, evals=[], started=time.monotonic())
    if start_eval is not None:
        start = json.loads((Path(start_eval) / 'short_eval_report.json').read_text())
        state['evals'].append(dict(step=0, usable_rate=start['usable_rate'], readable_rate=start['readable_rate'],
                                   middle_90=start['usable_rate_middle_90'], source=str(start_eval)))

    def run_eval(step):
        eval_dir = run_dir / 'eval' / f'step_{step:04d}'
        with torch.autocast('cuda', dtype=torch.bfloat16):
            evaluate(model, tokenizer, evals, eval_dir, max_new_tokens=max_new_tokens, temperature=0.7,
                     seeds=list(eval_seeds), batch_size=8, no_e=True, mask_e=masked, ascii_only=masked,
                     metadata=dict(run_dir=str(run_dir), step=step))
        model.train()
        summary = judge_run(eval_dir, workers=judge_workers)
        state['evals'].append(dict(step=step, usable_rate=summary['usable_rate'],
                                   readable_rate=summary['readable_rate'],
                                   middle_90=summary['usable_rate_middle_90']))
        (run_dir / 'evals.json').write_text(json.dumps(state['evals'], indent=2))
        return summary['usable_rate']

    def on_step(info):
        state['step'] += 1
        step = state['step']
        engine.global_step = step
        reward_fn.step = step
        records = reward_fn.last
        scored = [r['reward'] for r in records if r['reward'] is not None]
        metrics.write(json.dumps(dict(
            step=step, seconds=round(time.monotonic() - state['started'], 1),
            judge_seconds=round(reward_fn.judge_seconds, 1),
            mean_reward=sum(scored) / len(scored) if scored else None,
            usable=sum(bool(r['usable']) for r in records), answers=len(records),
            judge_errors=len(records) - len(scored), kl=info['kl'],
            sampling_vs_training_logprob_gap=info['logprob_gap'],
            peak_gpu_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            mean_answer_chars=sum(len(r['answer']) for r in records) / len(records),
            cut_off=sum(r['cut_off'] for r in records))) + '\n')
        metrics.flush()
        torch.cuda.reset_peak_memory_stats()
        if step % eval_every == 0 or (step >= max_steps and final_eval):
            engine.save_checkpoint(run_dir / 'checkpoints' / f'step_{step:04d}.pt')
            (run_dir / f'answers_step_{step:04d}.txt').write_text('\n\n'.join(
                f'reward={r["reward"]} usable={r["usable"]} | {r["prompt"][:100]}\n{r["answer"]}'
                for r in random.Random(step).sample(records, min(10, len(records)))) + '\n')
            rates = [e['usable_rate'] for e in state['evals']] + [run_eval(step)]
            print(f'step {step}: eval usable {rates[-1]:.1%}', flush=True)
            if len(rates) >= 3 and rates[-1] < rates[-2] < rates[-3]:
                print('Stopping: usable rate fell at two evals in a row', flush=True)
                return True
        return step >= max_steps

    class Loader:
        """Feeds endless batches and tells the reward which prompt ids it is scoring."""
        def __iter__(self):
            for batch in batches:
                reward_fn.ids = batch.pop('ids')
                yield batch

    with torch.autocast('cuda', dtype=torch.bfloat16):
        engine.fit(Loader(), max_new_tokens=max_new_tokens, temperature=temperature, epochs=1,
                   group_size=group_size, reward_fn=reward_fn, kl_beta=kl_beta, eps=eps,
                   micro_batch_size=micro_batch_size, on_step=on_step, recompute_old_log_probs=True,
                   parallel_groups=True, fixed_loss_length=max_new_tokens if fixed_length_loss else None)
    print(json.dumps(state['evals'], indent=2))
    return state['evals']


if __name__ == '__main__':
    import fire

    fire.Fire(main)
