"""
Do zero-shot classification on IID data with both seen and unseen classes.

Single-process. If you want to run all evaluations of a single model at once, look
in scripts/.

Writes the output to a plaintext and JSON format in the logs directory.
"""

import datetime
import logging
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..open_clip import (
    create_model_and_transforms,
    get_cast_dtype,
    get_tokenizer,
    trace_model,
)
from ..training.imagenet_zeroshot_data import openai_imagenet_template
from ..training.logger import setup_logging
from ..training.precision import get_autocast

from .data import DatasetFromFile
from .params import parse_args
from .utils import init_device, random_seed



def get_dataloader(dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=None,
    )


def zero_shot_classifier(model, classnames, templates, args):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(args.device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return dict([
        (k,float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()))
        for k in topk
    ])


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    dump = args.dump_predictions is not None
    dump_topk = max(1, int(getattr(args, 'dump_topk', 5)))
    dump_limit = int(getattr(args, 'dump_limit', 0))
    dumped = 0
    dump_rows = []
    classes = list(getattr(dataloader.dataset, 'classes', []))
    has_paths = hasattr(dataloader.dataset, 'paths')
    has_samples = hasattr(dataloader.dataset, 'samples')
    global_index = 0
    with torch.no_grad():
        n = 0.0
        topk = dict()
        for i in (1,min(len(dataloader.dataset.classes),3), min(len(dataloader.dataset.classes),5)):
            topk[i] = 0.0
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device) #images.shape: torch.Size([batch_size, 3 rgb channels, image_height, image_width])
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                image_features, _ = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = model.logit_scale.exp() * image_features @ classifier

            # measure accuracy
            acc = accuracy(logits, target, topk=topk.keys())
            for k,v in acc.items():
                topk[k] += v
            n += images.size(0)
            if dump and (dump_limit == 0 or dumped < dump_limit):
                probs = F.softmax(logits, dim=-1)
                batch_size = images.size(0)
                for bi in range(batch_size):
                    if dump_limit and dumped >= dump_limit:
                        break
                    # derive path if available
                    path = None
                    if has_paths:
                        path = dataloader.dataset.paths[global_index + bi]
                    elif has_samples:
                        try:
                            path = dataloader.dataset.samples[global_index + bi]
                        except Exception:
                            path = None
                    true_idx = int(target[bi].item())
                    true_label = classes[true_idx] if classes and 0 <= true_idx < len(classes) else true_idx
                    pvals, pidx = torch.topk(probs[bi], k=min(dump_topk, probs.shape[-1]))
                    preds = []
                    for j in range(pidx.numel()):
                        idx = int(pidx[j].item())
                        score = float(pvals[j].item())
                        label = classes[idx] if classes and 0 <= idx < len(classes) else idx
                        preds.append({"rank": j+1, "index": idx, "label": label, "prob": score})
                    dump_rows.append({
                        "index": global_index + bi,
                        "path": path,
                        "true_index": true_idx,
                        "true_label": true_label,
                        "topk": preds,
                    })
                    dumped += 1
                global_index += batch_size

    for k,v in acc.items():
        topk[k] /= n
    if dump and len(dump_rows) > 0:
        try:
            import json, os
            os.makedirs(os.path.dirname(args.dump_predictions), exist_ok=True)
            with open(args.dump_predictions, 'w') as f:
                json.dump(dump_rows, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save dump_predictions to {args.dump_predictions}: {e}")
    return topk


def build_species_aggregation(names_json_path, target_level: str, rank_only: bool, allowed_labels: list[str]):
    import json
    with open(names_json_path, 'r') as f:
        names = json.load(f)

    level_map = {"kingdom": 0, "phylum": 1, "cls": 2, "class": 2, "order": 3, "family": 4, "genus": 5, "species": 6}
    idx = level_map[target_level.lower()]

    def agg_label(entry):
        tax = entry[0]
        if rank_only:
            if idx == 6:
                if len(tax) >= 7:
                    return f"{tax[5]} {tax[6]}".strip()
                return tax[idx]
            return tax[idx]
        else:
            if idx == 6 and len(tax) >= 7:
                return " ".join(tax[:6] + [f"{tax[5]} {tax[6]}"])
            return " ".join(tax[: idx + 1])

    label_per_species = [agg_label(e) for e in names]
    mapping = {}
    for i, lab in enumerate(label_per_species):
        mapping.setdefault(lab, []).append(i)

    if allowed_labels is not None:
        allowed = set(allowed_labels)
        mapping = {lab: idxs for lab, idxs in mapping.items() if lab in allowed}

    if allowed_labels is not None:
        labels = [lab for lab in allowed_labels if lab in mapping]
    else:
        labels = sorted(mapping.keys())

    return mapping, labels


def run_aggregate_species(model, species_emb, mapping, label_order, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    device = args.device

    A = torch.zeros((species_emb.shape[1], len(label_order)), dtype=torch.float32, device=device)
    for j, lab in enumerate(label_order):
        idxs = mapping[lab]
        A[idxs, j] = 1.0

    with torch.no_grad():
        n = 0.0
        topk = dict()
        for i in (1, min(len(label_order), 3), min(len(label_order), 5)):
            topk[i] = 0.0

        dump = args.dump_predictions is not None
        dump_topk = max(1, int(getattr(args, 'dump_topk', 5)))
        dump_limit = int(getattr(args, 'dump_limit', 0))
        dumped = 0
        dump_rows = []
        has_paths = hasattr(dataloader.dataset, 'paths')
        has_samples = hasattr(dataloader.dataset, 'samples')
        global_index = 0

        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(device)

            with autocast():
                img_feat, _ = model.encode_image(images)
                img_feat = F.normalize(img_feat, dim=-1)
                logits_species = model.logit_scale.exp() * img_feat @ species_emb
                probs_species = F.softmax(logits_species, dim=-1)
                probs_rank = probs_species @ A

            acc = accuracy(probs_rank, target, topk=topk.keys())
            for k, v in acc.items():
                topk[k] += v
            n += images.size(0)

            if dump and (dump_limit == 0 or dumped < dump_limit):
                batch_size = images.size(0)
                for bi in range(batch_size):
                    if dump_limit and dumped >= dump_limit:
                        break
                    path = None
                    if has_paths:
                        path = dataloader.dataset.paths[global_index + bi]
                    elif has_samples:
                        try:
                            path = dataloader.dataset.samples[global_index + bi]
                        except Exception:
                            path = None
                    true_idx = int(target[bi].item())
                    true_label = label_order[true_idx] if 0 <= true_idx < len(label_order) else true_idx
                    pvals, pidx = torch.topk(probs_rank[bi], k=min(dump_topk, probs_rank.shape[-1]))
                    preds = []
                    for j in range(pidx.numel()):
                        idx = int(pidx[j].item())
                        score = float(pvals[j].item())
                        label = label_order[idx]
                        preds.append({"rank": j + 1, "index": idx, "label": label, "prob": score})
                    dump_rows.append({
                        "index": global_index + bi,
                        "path": path,
                        "true_index": true_idx,
                        "true_label": true_label,
                        "topk": preds,
                    })
                    dumped += 1
                global_index += batch_size

    for k, v in acc.items():
        topk[k] /= n

    if dump and len(dump_rows) > 0:
        try:
            import json, os
            os.makedirs(os.path.dirname(args.dump_predictions), exist_ok=True)
            with open(args.dump_predictions, 'w') as f:
                json.dump(dump_rows, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save dump_predictions to {args.dump_predictions}: {e}")
    return topk


def zero_shot_eval(model, data, args):
    results = {}

    logging.info("Starting zero-shot.")

    for split in data:
        logging.info("Building zero-shot %s classifier.", split)
        classnames = [c for c in data[split].dataset.classes]

        if getattr(args, 'aggregate_to_rank', False):
            if not args.species_emb_npy or not args.species_names_json:
                raise RuntimeError("--aggregate_to_rank requires --species_emb_npy and --species_names_json")
            import numpy as np
            txt_emb_np = np.load(args.species_emb_npy)
            species_emb = torch.from_numpy(txt_emb_np).to(args.device)
            cast_dtype = get_cast_dtype(args.precision)
            if cast_dtype is not None:
                species_emb = species_emb.to(dtype=cast_dtype)
            mapping, label_order = build_species_aggregation(
                args.species_names_json, args.taxonomic_level, getattr(args, 'rank_only', False), classnames
            )
            # align dataset classes to label_order
            data[split].dataset.classes = label_order
            topk = run_aggregate_species(model, species_emb, mapping, label_order, data[split], args)
        else:
            # choose template style
            if getattr(args, 'template_style', 'openai') == 'plain':
                templates = [lambda c: f"{c}"]
            elif getattr(args, 'template_style', 'openai') == 'bio':
                templates = [
                    lambda c: f"{c}",
                    lambda c: f"a photo of {c}",
                    lambda c: f"an image of {c}",
                    lambda c: f"a biological photograph of {c}",
                ]
            else:
                templates = openai_imagenet_template

            classifier = zero_shot_classifier(
                model, classnames, templates, args
            )

            topk = run(model, classifier, data[split], args)

        for k,v in topk.items():
            results[f"{split}-top{k}"] = v

        logging.info("Finished zero-shot %s with total %d classes.", split, len(data[split].dataset.classes))

    logging.info("Finished zero-shot.")

    return results


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_device(args)

    args.save_logs = args.logs and args.logs.lower() != "none"

    # get the name of the experiments
    if args.save_logs and args.name is None:
        # sanitize model name for filesystem/uri use
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
                "zero_shot_iid",
            ]
        )
    if args.save_logs is None:
        args.log_path = None
    else:
        log_base_path = os.path.join(args.logs, args.name)
        args.log_path = None
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    if (
        isinstance(args.force_image_size, (tuple, list))
        and len(args.force_image_size) == 1
    ):
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=None,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
        load_weights_only=False,
    )

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    logging.info("Model:")
    logging.info(f"{str(model)}")
    logging.info("Params:")
    if  args.save_logs is None:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
    else:
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # initialize datasets
    if args.data_loader == "catalogue":
        from dataset_catalogue import FranCatalogueLoader
        csv_path = args.label_filename or os.path.join(args.data_root, "metadata.csv")
        ds = FranCatalogueLoader(
            csv_path,
            data_root=args.data_root,
            taxonomic_level=args.taxonomic_level,
            transform=preprocess_val,
            require_species=args.require_species,
            rank_only=getattr(args, 'rank_only', False),
        )
    else:
        ds = DatasetFromFile(args.data_root, args.label_filename, transform=preprocess_val, classes=args.text_type)

    data = {
        "val-unseen": get_dataloader(
            ds,
            batch_size=args.batch_size, num_workers=args.workers
        ),
    }

    model.eval()
    metrics = zero_shot_eval(model, data, args)
    
    logging.info("Results:")
    for key, value in metrics.items():
        logging.info(f"  {key}: {value*100:.2f}")
    logging.info("Done.")
