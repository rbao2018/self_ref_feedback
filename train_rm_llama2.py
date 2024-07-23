import math
import os

from transformers.trainer import get_scheduler

from RLXF.dataset.reward_dataset import RewardDataset
from RLXF.fsdp_strategy import FSDPStrategy
from RLXF.model.reward_model import get_llm_for_sequence_regression
from RLXF.trainer.rm_trainer import RewardModelTrainer
from RLXF.parse import parse_args
from RLXF.utils import blending_datasets, get_tokenizer

def main(args):
    # configure strategy
    strategy = FSDPStrategy(
        seed=args.seed,
        micro_train_batch_size=args.critic_train_batch_size,
        train_batch_size=args.train_batch_size,
        sharding_strategy="FULL_SHARD" if args.zero_stage == 3 else "SHARD_GRAD_OP",
        fsdp_activation_checkpointing=args.gradient_checkpointing,
        gradient_clipping_threshold=args.max_norm,
        args=args
    )
    strategy.setup_distributed()
    strategy.print(args)
    # configure model
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        bf16=strategy.args.bf16,
        global_rank=strategy.get_rank(),
        lora_rank=strategy.args.lora_rank,
        lora_alpha=strategy.args.lora_alpha,
        target_modules=strategy.args.target_modules,
        use_flash_attention_2=strategy.args.flash_attn,
        init_value_head=not args.skip_train,
        packing_samples=args.packing_samples
    )
    args.input_template = args.input_template.replace('\\n', '\n')

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model)

    strategy.print(model)
    model = strategy.prepare_model(model)

    # configure optimizer
    optim = strategy.create_optimizer(
        model, 
        lr=args.learning_rate, 
        betas=(0.9, 0.95), 
        weight_decay=args.l2
    )

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    train_dataset = RewardDataset(train_data, tokenizer, args.max_len, strategy)
    eval_dataset = RewardDataset(eval_data, tokenizer, args.max_len, strategy)
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset, 
        args.micro_train_batch_size, 
        pin_memory=True, 
        shuffle=False, 
        collate_fn=train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn
    )
    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) * args.max_epochs // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        "cosine",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward

    trainer = RewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        max_epochs=args.max_epochs,
        loss=args.loss,
    )
    if not args.skip_train:
        trainer.train(args)
    trainer.evaluate(eval_dataloader, steps=0)



if __name__ == "__main__":
    args = parse_args()
    main(args)
