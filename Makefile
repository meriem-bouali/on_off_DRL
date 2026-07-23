PYTHON         = uv run python


activate:    # is not required , and it can be activated using make
	. .venv/bin/activate

#! ── Train ─────────────────────────────────────────────────────────────────────
#### ── On-Rl
train-onrl:
	$(PYTHON) src/policy/on_rl/train_on_rl.py
#### ── BCQ
train-bcq-4e6:
	$(PYTHON) src/policy/bcq_4e6/train_bcq_4e6.py
	
train-bcq-4e6shuffle:
	$(PYTHON) src/policy/bcq_4e6_shuffle/train_bcq_4e6_shuffle.py

train-bcq-4e6-clean:
	$(PYTHON) src/policy/bcq_4e6_clean/train_bcq_4e6_clean.py

train-bcq-4e6-clean-shuffle:
	$(PYTHON) src/policy/bcq_4e6_clean_shuffle/train_bcq_4e6_clean_shuffle.py

train-bcq-8e6-clean:
	$(PYTHON) src/policy/bcq_8e6_clean/train_bcq_8e6_clean.py

### ── Off-Rl
train-offrl-4e6:
	$(PYTHON) src/policy/off_rl_4e6/train_off_rl_4e6.py

train-offrl-4e6-shuffle:
	$(PYTHON) src/policy/off_rl_4e6_shuffle/train_off_rl_4e6_shuffle.py

train-offrl-4e6-clean:
	$(PYTHON) src/policy/off_rl_4e6_clean/train_off_rl_4e6_clean.py
	
train-offrl-4e6-clean-shuffle:
	$(PYTHON) src/policy/off_rl_4e6_clean_shuffle/train_off_rl_4e6_clean_shuffle.py

train-offrl-4e8-clean:
	$(PYTHON) src/policy/off_rl_8e6_clean/train_off_rl_8e6_clean.py

#! ── Test ─────────────────────────────────────────────────────────────────────
#### ── On-Rl
test-onrl:
	$(PYTHON) src/policy/on_rl/test_on_rl.py
#### ── BCQ
test-bcq-4e6:
	$(PYTHON) src/policy/bcq_4e6/test_bcq_4e6.py
	
test-bcq-4e6shuffle:
	$(PYTHON) src/policy/bcq_4e6_shuffle/test_bcq_4e6_shuffle.py

test-bcq-4e6-clean:
	$(PYTHON) src/policy/bcq_4e6_clean/test_bcq_4e6_clean.py

test-bcq-4e6-clean-shuffle:
	$(PYTHON) src/policy/bcq_4e6_clean_shuffle/test_bcq_4e6_clean_shuffle.py

test-bcq-8e6-clean:
	$(PYTHON) src/policy/bcq_8e6_clean/test_bcq_8e6_clean.py

### ── Off-Rl
test-offrl-4e6:
	$(PYTHON) src/policy/off_rl_4e6/test_off_rl_4e6.py

test-offrl-4e6-shuffle:
	$(PYTHON) src/policy/off_rl_4e6_shuffle/test_off_rl_4e6_shuffle.py

test-offrl-4e6-clean:
	$(PYTHON) src/policy/off_rl_4e6_clean/test_off_rl_4e6_clean.py
	
test-offrl-4e6-clean-shuffle:
	$(PYTHON) src/policy/off_rl_4e6_clean_shuffle/test_off_rl_4e6_clean_shuffle.py

test-offrl-4e8-clean:
	$(PYTHON) src/policy/off_rl_8e6_clean/test_off_rl_8e6_clean.py



# # # ── Observe ───────────────────────────────────────────────────────────────────

# ── TensorBoard ───────────────────────────────────────────────────────────────
tensorboard:
# 	uv tool run tensorboard --logdir $(LOGDIR) --port 6007
	uv run tensorboard --logdir=src/
