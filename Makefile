CC ?= clang
CSTD ?= -std=c11
WARN ?= -Wall -Wextra -Wpedantic
OPT ?= -O3 -DNDEBUG -funroll-loops -mcpu=native
INCLUDES ?= -Iinclude
CFLAGS ?= $(CSTD) $(WARN) $(OPT) $(INCLUDES)
LDFLAGS ?=
LDLIBS ?= -lm

SUBMISSION ?= submissions/example_submission.c
SUBMISSION_FILES := $(sort $(wildcard submissions/*.c))
BENCHMARK_ALL_ARGS ?= --validate --sizes 128,256 --repeat 2 --warmup 1

BENCHMARK_BIN := benchmark
SOURCES := src/benchmark.c src/utils.c src/matmul_ref.c $(SUBMISSION)
HEADERS := include/matmul.h include/utils.h

.PHONY: all clean run run-both run-validate validate bench-all list-submissions

all: $(BENCHMARK_BIN)

$(BENCHMARK_BIN): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(SOURCES) -o $@ $(LDFLAGS) $(LDLIBS)

run: $(BENCHMARK_BIN)
	./$(BENCHMARK_BIN)

run-validate: $(BENCHMARK_BIN)
	./$(BENCHMARK_BIN) --validate

run-both: run-validate

validate: $(BENCHMARK_BIN)
	./$(BENCHMARK_BIN) --validate

list-submissions:
	@for file in $(SUBMISSION_FILES); do echo "$$file"; done

bench-all:
	@set -u; \
	total=0; \
	failed=0; \
	files="$(SUBMISSION_FILES)"; \
	if [ -z "$$files" ]; then \
		echo "No submission files found in submissions/"; \
		exit 1; \
	fi; \
	for file in $$files; do \
		total=$$((total + 1)); \
		name=$$(basename "$$file"); \
		echo "=== Benchmarking $$name ==="; \
		if $(MAKE) --no-print-directory clean all SUBMISSION="$$file" && ./$(BENCHMARK_BIN) $(BENCHMARK_ALL_ARGS); then \
			echo "RESULT: PASS ($$name)"; \
		else \
			echo "RESULT: FAIL ($$name)"; \
			failed=$$((failed + 1)); \
		fi; \
		echo; \
	done; \
	echo "Summary: total=$$total pass=$$((total - failed)) fail=$$failed"; \
	if [ $$failed -ne 0 ]; then \
		exit 1; \
	fi

clean:
	rm -f $(BENCHMARK_BIN)
