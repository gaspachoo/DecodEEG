# Makefile to train Classifiers checkpoints

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


# Unified train checkpoints target (multi or mono)
.PHONY: checkpoints
# TARGET: multi (default) or mono
TARGET ?= multi

# Train checkpoints (multi or mono depending on TARGET)
checkpoints:
	@set -e; \
	if [ "$(TARGET)" = "multi" ]; then \
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
		done; \
	elif [ "$(TARGET)" = "mono" ]; then \
		for c in $(CATEGORIES); do \
			ckpt="$(CKPT_ROOT)/mono/$(SUBJECT)/$(MODE)/seed$(SEED)/$(MODEL)/$$c/$(MODEL)_best.pt"; \
			if [ ! -f $$ckpt ]; then \
				uv run $(TRAIN_MONO_SCRIPT) --category $$c --model $(MODEL) --seed $(SEED) --subj_name $(SUBJECT) --save_dir $(CKPT_ROOT) $(SHUFFLE_ARG) $(WANDB_ARG); \
			else \
				echo "[Makefile] Skip $$c: checkpoint already exists"; \
			fi; \
		done; \
		for cl in $(LABEL_CLUSTERS); do \
			ckpt="$(CKPT_ROOT)/mono/$(SUBJECT)/$(MODE)/seed$(SEED)/$(MODEL)/label_cluster$$cl/$(MODEL)_best.pt"; \
			if [ ! -f $$ckpt ]; then \
				uv run $(TRAIN_MONO_SCRIPT) --category label --cluster $$cl --model $(MODEL) --seed $(SEED) --subj_name $(SUBJECT) --save_dir $(CKPT_ROOT) $(SHUFFLE_ARG) $(WANDB_ARG); \
			else \
				echo "[Makefile] Skip label cluster $$cl: checkpoint already exists"; \
			fi; \
		done; \
	else \
		echo "[Makefile] Unknown TARGET '$(TARGET)'. Use TARGET=multi or TARGET=mono"; exit 1; \
	fi