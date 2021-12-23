#! /bin/bash
export MODEL_DIR=model
export DATA_DIR=data

CUDA_VISIBLE_DEVICES=0 python bert/run_classifier.py \
	--task_name=sd \
	--do_predict=true \
	--data_dir=data \
	--vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
	--bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
	--init_checkpoint=output/teacher/model.ckpt-4484\
	--output_dir=output/distill
