import numpy as np
from data_provider.data_loader import Dataset_original
from data_provider.data_loader import my_collate_fn_baseline, my_collate_fn_withId, my_collate_fn_multidomain
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, Dataset

data_dict = {
    'Dataset_original': Dataset_original
}

def data_provider_baseline_DA(args, flag, tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None, total_prompts=None, 
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, sample_weighted=False, target_dataset='None'):
    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    if flag == 'test' or flag == 'val':
        data_set = Data(args=args,
                flag=flag,
                tokenizer=tokenizer,
                label_scaler=label_scaler,
                eval_cycle_min=eval_cycle_min,
                eval_cycle_max=eval_cycle_max,
                total_prompts=total_prompts, 
                total_charge_discharge_curves=total_charge_discharge_curves, 
                total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
                class_labels=class_labels,
                life_class_scaler=life_class_scaler,
                use_target_dataset=True
            )
    else:
        data_set = Data(args=args,
                flag=flag,
                tokenizer=tokenizer,
                label_scaler=label_scaler,
                eval_cycle_min=eval_cycle_min,
                eval_cycle_max=eval_cycle_max,
                total_prompts=total_prompts, 
                total_charge_discharge_curves=total_charge_discharge_curves, 
                total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
                class_labels=class_labels,
                life_class_scaler=life_class_scaler,
                use_target_dataset=False
            )

    data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=my_collate_fn_baseline)
    
    if target_dataset != 'None' and flag=='train':
        target_data_set = Data(args=args,
                flag=flag,
                tokenizer=tokenizer,
                label_scaler=data_set.return_label_scaler(),
                eval_cycle_min=eval_cycle_min,
                eval_cycle_max=eval_cycle_max,
                total_prompts=total_prompts, 
                total_charge_discharge_curves=total_charge_discharge_curves, 
                total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
                class_labels=class_labels,
                life_class_scaler=data_set.return_life_class_scaler(),
                use_target_dataset=True
            )

        target_data_loader = DataLoader(
                    target_data_set,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=drop_last,
                    collate_fn=my_collate_fn_baseline)
        target_sampler = RandomSampler(target_data_loader.dataset, replacement=True, num_samples=len(data_loader.dataset))
        target_resampled_dataloader = DataLoader(target_data_loader.dataset, batch_size=batch_size, sampler=target_sampler, collate_fn=my_collate_fn_baseline)
        return data_set, data_loader, target_data_set, target_resampled_dataloader
    else:
        return data_set, data_loader

def data_provider_baseline(args, flag, tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None, total_prompts=None, 
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, sample_weighted=False):
    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(args=args,
            flag=flag,
            tokenizer=tokenizer,
            label_scaler=label_scaler,
            eval_cycle_min=eval_cycle_min,
            eval_cycle_max=eval_cycle_max,
            total_prompts=total_prompts,
            total_charge_discharge_curves=total_charge_discharge_curves,
            total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
            class_labels=class_labels,
            life_class_scaler=life_class_scaler
        )

    # [H3] when joint multi-domain training, use the domain-aware collate and (on train) a
    # domain-balanced sampler so scarce domains aren't swamped by Li-ion. Default path unchanged.
    multidomain = getattr(args, 'multidomain', 'off') == 'on'
    collate = my_collate_fn_multidomain if multidomain else my_collate_fn_baseline
    sampler = None
    if multidomain and flag == 'train' and getattr(args, 'weighted_sampling', False):
        dom = np.asarray(data_set.total_domain_ids, dtype=int)
        counts = np.bincount(dom, minlength=getattr(args, 'num_domains', 4)).astype(float)
        inv = np.where(counts > 0, 1.0 / counts, 0.0)
        sample_weights = inv[dom]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=(shuffle_flag and sampler is None),
                sampler=sampler,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=collate)

    return data_set, data_loader


def data_provider_evaluate(args, flag, tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None, total_prompts=None, 
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, sample_weighted=False):
    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(args=args,
            flag=flag,
            tokenizer=tokenizer,
            label_scaler=label_scaler,
            eval_cycle_min=eval_cycle_min,
            eval_cycle_max=eval_cycle_max,
            total_prompts=total_prompts, 
            total_charge_discharge_curves=total_charge_discharge_curves, 
            total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
            class_labels=class_labels,
            life_class_scaler=life_class_scaler
        )


    data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=my_collate_fn_withId)
    return data_set, data_loader