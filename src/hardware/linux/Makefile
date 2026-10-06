CC ?= gcc
CORE_DIR ?= ../baremetal
BUILD_DIR ?= build
CFLAGS ?= -O3 -march=armv8-a+simd -flto -fPIE -Wall -Wextra
LDFLAGS ?= -static-pie
CPPFLAGS += -Icompat -I$(CORE_DIR) -D_GNU_SOURCE -DBITNET_LINUX
LDLIBS += -lm -pthread

OBJECTS := $(BUILD_DIR)/bitnet_linux.o \
           $(BUILD_DIR)/bitnet_client.o \
           $(BUILD_DIR)/prompt_source.o \
           $(BUILD_DIR)/linux_compat.o \
           $(BUILD_DIR)/core_main.o \
           $(BUILD_DIR)/bitnet_runtime.o \
           $(BUILD_DIR)/bitnet_kernels.o \
           $(BUILD_DIR)/bitnet_accel.o

.PHONY: all clean
all: $(BUILD_DIR)/bitnet-linux

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/bitnet_linux.o: bitnet_linux.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/bitnet_client.o: bitnet_client.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/prompt_source.o: prompt_source.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/linux_compat.o: compat/linux_compat.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/core_main.o: $(CORE_DIR)/main.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Dmain=bitnet_core_main -c $< -o $@

$(BUILD_DIR)/bitnet_runtime.o: $(CORE_DIR)/bitnet_runtime.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/bitnet_kernels.o: $(CORE_DIR)/bitnet_kernels.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/bitnet_accel.o: $(CORE_DIR)/bitnet_accel.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/bitnet-linux: $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -rf $(BUILD_DIR)
