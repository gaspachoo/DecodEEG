# Makefile to train Classifiers checkpoints

# Default Python interpreter
PYTHON ?= python

# Training script relative path
TRAIN_SCRIPT := Classifiers/train_Classifiers.py

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
CACHE_DIR := Classifiers/cache
CKPT_ROOT_NET := Checkpoints
TRAIN_NET_SCRIPT := Classifiers/train_net.py

# Directory name matching the default split with seed 0
SUB_DIR := sub_13_9_15_11_3_6_12_2_19_17_1_20_16

.PHONY: checkpoints
checkpoints:
	@set -e; \
	for c in $(CATEGORIES); do \
			ckpt="$(CKPT_ROOT)/$(SUB_DIR)/$$c/Classifiers_best.pt"; \
                       if [ ! -f $$ckpt ]; then \
                                       $(PYTHON) $(TRAIN_SCRIPT) --category $$c --seed $(SPLIT_SEED) --cache_dir $(CACHE_DIR) $(WANDB_ARG); \
			else \
					echo "[Makefile] Skip $$c: checkpoint already exists"; \
			fi; \
	done; \
	for cl in $(LABEL_CLUSTERS); do \
			ckpt="$(CKPT_ROOT)/$(SUB_DIR)/label_cluster$$cl/Classifiers_best.pt"; \
                       if [ ! -f $$ckpt ]; then \
                                       $(PYTHON) $(TRAIN_SCRIPT) --category label --cluster $$cl --seed $(SPLIT_SEED) --cache_dir $(CACHE_DIR) $(WANDB_ARG); \
			else \
					echo "[Makefile] Skip label cluster $$cl: checkpoint already exists"; \
			fi; \
	done

TRAIN_1SUB_SCRIPT := Classifiers/train_Classifiers_1sub.py
SUBJECT_TO_TRAIN := sub3
.PHONY: checkpoints_1sub
checkpoints_1sub:
	@set -e; \
       for c in $(CATEGORIES); do \
               $(PYTHON) $(TRAIN_1SUB_SCRIPT) --category $$c $(WANDB_ARG) --subj_name $(SUBJECT_TO_TRAIN) --split_seed $(SPLIT_SEED); \
       done; \
       for cl in $(LABEL_CLUSTERS); do \
               $(PYTHON) $(TRAIN_1SUB_SCRIPT) --category label --cluster $$cl $(WANDB_ARG) --subj_name $(SUBJECT_TO_TRAIN) --split_seed $(SPLIT_SEED); \
       done

.PHONY: checkpoints_net
checkpoints_net:
	@set -e; \
	for c in $(CATEGORIES); do \
		ckpt="$(CKPT_ROOT_NET)/$(SUB_DIR)/$$c/$(MODEL)_best.pt"; \
                if [ ! -f $$ckpt ]; then \
                        $(PYTHON) $(TRAIN_NET_SCRIPT) --category $$c --model $(MODEL) --seed $(SPLIT_SEED) $(WANDB_ARG); \
		else \
			echo "[Makefile] Skip $$c: checkpoint already exists"; \
		fi; \
	done; \
	for cl in $(LABEL_CLUSTERS); do \
		ckpt="$(CKPT_ROOT_NET)/$(SUB_DIR)/label_cluster$$cl/$(MODEL)_best.pt"; \
                if [ ! -f $$ckpt ]; then \
                        $(PYTHON) $(TRAIN_NET_SCRIPT) --category label --cluster $$cl --model $(MODEL) --seed $(SPLIT_SEED) $(WANDB_ARG); \
		else \
			echo "[Makefile] Skip label cluster $$cl: checkpoint already exists"; \
		fi; \
	done