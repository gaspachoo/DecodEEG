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

.PHONY: checkpoints
checkpoints:
	@set -e; \
	for c in $(CATEGORIES); do \
		$(PYTHON) $(TRAIN_SCRIPT) --category $$c $(WANDB_ARG); \
	done; \
	for cl in $(LABEL_CLUSTERS); do \
		$(PYTHON) $(TRAIN_SCRIPT) --category label --cluster $$cl $(WANDB_ARG); \
	done
