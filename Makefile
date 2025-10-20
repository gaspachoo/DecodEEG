# Makefile to train Classifiers checkpoints

RUN := uv run

# Training scripts used for training
TRAIN_MULTI_SCRIPT := Classifiers/train_classifier_multi.py
TRAIN_MONO_SCRIPT  := Classifiers/train_classifier_mono.py

# Categories excluding raw label which is trained per cluster
CATEGORIES := color color_binary face_appearance human_appearance label_cluster obj_number optical_flow_score

# Cluster indices for the label category
LABEL_CLUSTERS := 0 1 2 3 4 5 6 7 8

# Optional argument to enable wandb logging
WANDB_ARG   := $(if $(use_wandb),--use_wandb)
MODEL ?= glmnet
SEED  ?= 0
SUBJECT ?= sub3
# Flag to shuffle mono training instead of ordered mode
SHUFFLE_ARG := $(if $(shuffle),--shuffle)
MODE := $(if $(shuffle),shuffle,ordered)

# Directory for checkpoints
CKPT_ROOT := Classifiers/checkpoints


# Train checkpoints for the multi-subject setup
.PHONY: checkpoints_multi
checkpoints_multi:
	@set -e; \
	for c in $(CATEGORIES); do \
		ckpt="$(CKPT_ROOT)/multi/$(SEED)/$(MODEL)/$$c/$(MODEL)_best.pt"; \
		if [ ! -f $$ckpt ]; then \
			$(RUN) $(TRAIN_MULTI_SCRIPT) --category $$c --model $(MODEL) --seed $(SEED) --save_dir $(CKPT_ROOT) $(WANDB_ARG); \
		else \
			echo "[Makefile] Skip $$c: checkpoint already exists"; \
		fi; \
	done; \
	for cl in $(LABEL_CLUSTERS); do \
		ckpt="$(CKPT_ROOT)/multi/$(SEED)/$(MODEL)/label_cluster$$cl/$(MODEL)_best.pt"; \
		if [ ! -f $$ckpt ]; then \
			$(RUN) $(TRAIN_MULTI_SCRIPT) --category label --cluster $$cl --model $(MODEL) --seed $(SEED) --save_dir $(CKPT_ROOT) $(WANDB_ARG); \
		else \
			echo "[Makefile] Skip label cluster $$cl: checkpoint already exists"; \
		fi; \
	done

# Train checkpoints for the mono-subject setup
.PHONY: checkpoints_mono
checkpoints_mono:
	@set -e; \
	for c in $(CATEGORIES); do \
		ckpt="$(CKPT_ROOT)/mono/$(SUBJECT)/$(MODE)/seed$(SEED)/$(MODEL)/$$c/$(MODEL)_best.pt"; \
		if [ ! -f $$ckpt ]; then \
			$(RUN) $(TRAIN_MONO_SCRIPT) --category $$c --model $(MODEL) --seed $(SEED) --subj_name $(SUBJECT) --save_dir $(CKPT_ROOT) $(SHUFFLE_ARG) $(WANDB_ARG); \
		else \
			echo "[Makefile] Skip $$c: checkpoint already exists"; \
		fi; \
	done; \
	for cl in $(LABEL_CLUSTERS); do \
		ckpt="$(CKPT_ROOT)/mono/$(SUBJECT)/$(MODE)/seed$(SEED)/$(MODEL)/label_cluster$$cl/$(MODEL)_best.pt"; \
		if [ ! -f $$ckpt ]; then \
			$(RUN) $(TRAIN_MONO_SCRIPT) --category label --cluster $$cl --model $(MODEL) --seed $(SEED) --subj_name $(SUBJECT) --save_dir $(CKPT_ROOT) $(SHUFFLE_ARG) $(WANDB_ARG); \
		else \
			echo "[Makefile] Skip label cluster $$cl: checkpoint already exists"; \
		fi; \
	done
