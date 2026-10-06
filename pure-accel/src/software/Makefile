CC ?= cc
AR ?= ar
OUT ?= ../../build/software
CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -Werror -Iinclude
PYTHON ?= python3

.PHONY: all test clean
all: $(OUT)/bitnetctl $(OUT)/libbitnet.a

$(OUT):
	mkdir -p $@
$(OUT)/bitnet.o: bitnet.c include/bitnet.h | $(OUT)
	$(CC) $(CFLAGS) -c $< -o $@
$(OUT)/libbitnet.a: $(OUT)/bitnet.o
	$(AR) rcs $@ $<
$(OUT)/bitnetctl: bitnet.c bitnet_linux.c bitnetctl.c include/bitnet.h include/bitnet_linux.h | $(OUT)
	$(CC) $(CFLAGS) bitnet.c bitnet_linux.c bitnetctl.c -o $@
$(OUT)/test_driver: bitnet.c tests/test_driver.c include/bitnet.h | $(OUT)
	$(CC) $(CFLAGS) bitnet.c tests/test_driver.c -o $@
test: all $(OUT)/test_driver
	$(OUT)/test_driver
	BITNETCTL=$(abspath $(OUT)/bitnetctl) $(PYTHON) -m unittest discover -s tests -v
clean:
	rm -f $(OUT)/bitnet.o $(OUT)/libbitnet.a $(OUT)/bitnetctl $(OUT)/test_driver
