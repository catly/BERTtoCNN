#! /bin/bash
export MODEL_DIR=model
export DATA_DIR=data

CUDA_VISIBLE_DEVICES=0 python bert/run_classifier.py \
	--task_name=sd \
	--do_export=true \
	--do_distill=true \
	--temperature=30 \
	--data_dir=data \
	--vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
	--bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
	--init_checkpoint=output/teacher/model.ckpt-10700 \
	--output_dir=output/teacher \
	--export_dir=export
