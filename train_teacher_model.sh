#! /bin/bash
export MODEL_DIR=model
export DATA_DIR=data

CUDA_VISIBLE_DEVICES=1 python bert/run_classifier.py \
	--task_name=sd \
	--do_train=true \
	--do_eval=true \
	--do_distill=true \
	--do_predict=true \
	--data_dir=data \
	--vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
	--bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
	--init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt \
	--max_seq_length=128 \
	--train_batch_size=8 \
	--learning_rate=0.00005 \
	--num_train_epochs=50.0 \
	--output_dir=output/teacher \
	--temperature=1