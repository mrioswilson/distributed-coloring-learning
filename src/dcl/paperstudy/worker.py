"""One sequential GPU task. Parent records budget and owns only this child."""
import argparse
import os
import signal

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate", "mechanisms"))
    parser.add_argument("--treatment")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-spec")
    args = parser.parse_args()
    import torch
    from dcl.study.storage import training_lock, resources
    from .config import ROOT, Run
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: no automatic CPU substitution")
    torch.cuda.set_per_process_memory_fraction(min(.5, 4 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__(0, True))
    with training_lock(ROOT):
        resources(ROOT)
        free, _ = torch.cuda.mem_get_info()
        if free < 1024 * 2**20:
            raise RuntimeError("External GPU reserve below 1 GiB")
        if args.action == "train":
            from .training import train
            result = train(Run(args.treatment, args.seed), stop_requested=lambda: stop[0])
            print(f"TASK_RESULT {result}", flush=True)
        elif args.action == "evaluate":
            from .evaluation import evaluate_spec
            evaluate_spec(args.model_spec, stop_requested=lambda: stop[0])
        else:
            from .mechanisms import run_all
            run_all(stop_requested=lambda: stop[0],only_spec=args.model_spec)


if __name__ == "__main__":
    main()
