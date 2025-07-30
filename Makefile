# Makefile to train Classifiers checkpoints

# Default Python interpreter
PYTHON ?= python

# Training script relative path
TRAIN_SCRIPT := Classifiers/train_glmnet.py

# Categories excluding raw label which is trained per cluster
CATEGORIES := color color_binary face_appearance human_appearance label_cluster obj_number optical_flow_score

# Cluster indices for the label category
LABEL_CLUSTERS := 0 1 2 3 4 5 6 7 8

# Optional argument to enable wandb logging
WANDB_ARG := $(if $(use_wandb),--use_wandb)
MODEL ?= deepnet
SPLIT_SEED ?= 0

# Directory for checkpoints
CKPT_ROOT := Checkpoints
CKPT_ROOT_NET := Checkpoints
TRAIN_NET_SCRIPT := Classifiers/train_net.py


.PHONY: checkpoints
checkpoints:
	@set -e; \
	for c in $(CATEGORIES); do \
		ckpt="$(CKPT_ROOT)/multi/$(SPLIT_SEED)/glmnet/$$c/glmnet_best.pt"; \
		if [ ! -f $$ckpt ]; then \
			$(PYTHON) $(TRAIN_SCRIPT) --category $$c --seed $(SPLIT_SEED) --save_dir $(CKPT_ROOT) $(WANDB_ARG); \
		else \
			echo "[Makefile] Skip $$c: checkpoint already exists"; \
		fi; \
	done; \
	for cl in $(LABEL_CLUSTERS); do \
		ckpt="$(CKPT_ROOT)/multi/$(SPLIT_SEED)/glmnet/label_cluster$$cl/glmnet_best.pt"; \
		if [ ! -f $$ckpt ]; then \
			$(PYTHON) $(TRAIN_SCRIPT) --category label --cluster $$cl --seed $(SPLIT_SEED) --save_dir $(CKPT_ROOT) $(WANDB_ARG); \
		else \
			echo "[Makefile] Skip label cluster $$cl: checkpoint already exists"; \
		fi; \
	done

TRAIN_1SUB_SCRIPT := Classifiers/train_glmnet_1sub.py
SUBJECT_TO_TRAIN := sub3
.PHONY: checkpoints_1sub
checkpoints_1sub:
	@set -e; \
	for c in $(CATEGORIES); do \
		ckpt="$(CKPT_ROOT)/mono/$(SUBJECT_TO_TRAIN)/ordered/$(SPLIT_SEED)/glmnet/$$c/glmnet_best.pt"; \
		if [ ! -f $$ckpt ]; then \
			$(PYTHON) $(TRAIN_1SUB_SCRIPT) --category $$c $(WANDB_ARG) --subj_name $(SUBJECT_TO_TRAIN) --split_seed $(SPLIT_SEED) --save_dir $(CKPT_ROOT); \
		else \
			echo "[Makefile] Skip $$c: checkpoint already exists"; \
		fi; \
	done; \
	for cl in $(LABEL_CLUSTERS); do \
		ckpt="$(CKPT_ROOT)/mono/$(SUBJECT_TO_TRAIN)/ordered/$(SPLIT_SEED)/glmnet/label_cluster$$cl/glmnet_best.pt"; \
		if [ ! -f $$ckpt ]; then \
			$(PYTHON) $(TRAIN_1SUB_SCRIPT) --category label --cluster $$cl $(WANDB_ARG) --subj_name $(SUBJECT_TO_TRAIN) --split_seed $(SPLIT_SEED) --save_dir $(CKPT_ROOT); \
		else \
			echo "[Makefile] Skip label cluster $$cl: checkpoint already exists"; \
		fi; \
	done

.PHONY: checkpoints_net
checkpoints_net:
	@set -e; \
	for c in $(CATEGORIES); do \
		ckpt="$(CKPT_ROOT_NET)/multi/$(SPLIT_SEED)/$(MODEL)/$$c/$(MODEL)_best.pt"; \
		if [ ! -f $$ckpt ]; then \
			$(PYTHON) $(TRAIN_NET_SCRIPT) --category $$c --model $(MODEL) --seed $(SPLIT_SEED) --save_dir $(CKPT_ROOT_NET) $(WANDB_ARG); \
		else \
			echo "[Makefile] Skip $$c: checkpoint already exists"; \
		fi; \
	done; \
	for cl in $(LABEL_CLUSTERS); do \
		ckpt="$(CKPT_ROOT_NET)/multi/$(SPLIT_SEED)/$(MODEL)/label_cluster$$cl/$(MODEL)_best.pt"; \
		if [ ! -f $$ckpt ]; then \
			$(PYTHON) $(TRAIN_NET_SCRIPT) --category label --cluster $$cl --model $(MODEL) --seed $(SPLIT_SEED) --save_dir $(CKPT_ROOT_NET) $(WANDB_ARG); \
		else \
			echo "[Makefile] Skip label cluster $$cl: checkpoint already exists"; \
		fi; \
	done