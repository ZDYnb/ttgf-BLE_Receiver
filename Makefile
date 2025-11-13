# Makefile for Tiny Tapeout GF180MCU Hardening Flow
# Sets up environment and runs the hardening process

# Environment variables
export PDK_ROOT := $(HOME)/ttsetup/pdk
export PDK := gf180mcuD
export OPENLANE_TAG := 2.4.2

# Generate timestamp for log directory
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
LOG_DIR := logs/$(TIMESTAMP)

# Default target
.PHONY: all
all: harden

# Create user config
.PHONY: config
config:
	@echo "Creating user config..."
	./tt/tt_tool.py --gf --create-user-config

# Run hardening flow
.PHONY: harden
harden: config
	@echo "Creating log directory: $(LOG_DIR)"
	@mkdir -p $(LOG_DIR)
	@mkdir -p $(LOG_DIR)/src
	@echo "Saving environment to $(LOG_DIR)/env.txt"
	@echo "PDK_ROOT=$(PDK_ROOT)" > $(LOG_DIR)/env.txt
	@echo "PDK=$(PDK)" >> $(LOG_DIR)/env.txt
	@echo "OPENLANE_TAG=$(OPENLANE_TAG)" >> $(LOG_DIR)/env.txt
	@echo "TIMESTAMP=$(TIMESTAMP)" >> $(LOG_DIR)/env.txt
	@echo "./tt/tt_tool.py --gf --create-user-config" > $(LOG_DIR)/command.txt
	@echo "./tt/tt_tool.py --gf --harden" >> $(LOG_DIR)/command.txt
	@if [ -d .git ]; then \
		echo "Saving git context to $(LOG_DIR)/git_context.txt"; \
		git log -1 --oneline > $(LOG_DIR)/git_context.txt 2>/dev/null || echo "No git history" > $(LOG_DIR)/git_context.txt; \
		git status >> $(LOG_DIR)/git_context.txt 2>/dev/null || true; \
	else \
		echo "Not a git repository" > $(LOG_DIR)/git_context.txt; \
	fi
	@echo "Copying source files to $(LOG_DIR)/src/"
	@cp -r src/* $(LOG_DIR)/src/ 2>/dev/null || echo "No src files to copy"
	@echo ""
	@echo "Starting hardening flow..."
	@echo "PDK_ROOT = $(PDK_ROOT)"
	@echo "PDK = $(PDK)"
	@echo "OPENLANE_TAG = $(OPENLANE_TAG)"
	@echo "Logging to $(LOG_DIR)/run.log..."
	@./tt/tt_tool.py --gf --harden 2>&1 | tee $(LOG_DIR)/run.log
	@echo ""
	@echo "============================================="
	@echo "Hardening complete!"
	@echo "Logs saved to: $(LOG_DIR)"
	@echo "  - run.log: Full hardening output"
	@echo "  - command.txt: Commands executed"
	@echo "  - env.txt: Environment variables"
	@echo "  - git_context.txt: Git commit info"
	@echo "  - src/: Source code snapshot"
	@echo "============================================="

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning runs directory..."
	rm -rf runs/

# Clean logs (use with caution!)
.PHONY: clean-logs
clean-logs:
	@echo "WARNING: This will delete all logs in logs/ directory"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] && rm -rf logs/* || echo "Cancelled"

# Help target
.PHONY: help
help:
	@echo "Tiny Tapeout GF180 Hardening Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Run full hardening flow (config + harden)"
	@echo "  make config   - Create user config only"
	@echo "  make harden   - Run hardening (includes config)"
	@echo "  make clean    - Remove runs directory"
	@echo "  make clean-logs - Remove all log directories (interactive)"
	@echo "  make env      - Show environment variables"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Logs are saved to: logs/YYYYMMDD_HHMMSS/"
	@echo "  - run.log: Full hardening output"
	@echo "  - command.txt: Commands executed"
	@echo "  - env.txt: Environment variables"
	@echo "  - git_context.txt: Git commit info"
	@echo "  - src/: Source code snapshot"