# Makefile to train GLMNet checkpoints

# Default Python interpreter
PYTHON ?= python

# Training script relative path
TRAIN_SCRIPT := GLMNet/train_glmnet.py

# Categories excluding raw label which is trained per cluster
CATEGORIES := color color_binary face_appearance human_appearance label_cluster obj_number optical_flow_score

# Cluster indices for the label category
LABEL_CLUSTERS := 0 1 2 3 4 5 6 7 8

# Optional argument to enable wandb logging
WANDB_ARG := $(if $(use_wandb),--use_wandb)

# Directory for checkpoints
CKPT_ROOT := GLMNet/checkpoints
CACHE_DIR := GLMNet/cache

# Default training subjects used when calling ``make``
TRAIN_SUBJECTS := sub13 sub9 sub15 sub11 sub3 sub6 sub12 sub2 sub19 sub17 sub1 sub20 sub16
SUB_DIR := sub_13_9_15_11_3_6_12_2_19_17_1_20_16

.PHONY: checkpoints
checkpoints:
	@set -e; \
	for c in $(CATEGORIES); do \
			ckpt="$(CKPT_ROOT)/$(SUB_DIR)/$$c/glmnet_best.pt"; \
                       if [ ! -f $$ckpt ]; then \
                                       $(PYTHON) $(TRAIN_SCRIPT) --category $$c --train_subjects $(TRAIN_SUBJECTS) --cache_dir $(CACHE_DIR) $(WANDB_ARG); \
			else \
					echo "[Makefile] Skip $$c: checkpoint already exists"; \
			fi; \
	done; \
	for cl in $(LABEL_CLUSTERS); do \
			ckpt="$(CKPT_ROOT)/$(SUB_DIR)/label_cluster$$cl/glmnet_best.pt"; \
                       if [ ! -f $$ckpt ]; then \
                                       $(PYTHON) $(TRAIN_SCRIPT) --category label --cluster $$cl --train_subjects $(TRAIN_SUBJECTS) --cache_dir $(CACHE_DIR) $(WANDB_ARG); \
			else \
					echo "[Makefile] Skip label cluster $$cl: checkpoint already exists"; \
			fi; \
	done