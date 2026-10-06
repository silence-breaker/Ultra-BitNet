#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

typedef ap_axiu<64, 0, 0, 0> axis64_t;
typedef ap_axiu<32, 0, 0, 0> axis32_t;

extern "C" void bitnet_attention_axis(hls::stream<axis64_t> &S_AXIS,
                                       hls::stream<axis32_t> &M_AXIS);

static uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void push64(hls::stream<axis64_t> &stream, uint64_t value, bool last = false) {
    axis64_t word;
    word.data = value;
    word.keep = -1;
    word.strb = -1;
    word.last = last ? 1 : 0;
    stream.write(word);
}

static int quantize_reference(float value, float scale) {
    float scaled = value * scale;
    int result = (scaled >= 0.0f) ? static_cast<int>(scaled + 0.5f)
                                  : static_cast<int>(scaled - 0.5f);
    if (result > 127) result = 127;
    if (result < -128) result = -128;
    return result;
}

int main() {
    constexpr int hidden_size = 2560;
    constexpr int kv_size = 640;
    hls::stream<axis64_t> input;
    hls::stream<axis32_t> output;
    std::vector<int32_t> q(hidden_size);
    std::vector<int32_t> k(kv_size);
    std::vector<int32_t> v(kv_size);

    for (int i = 0; i < hidden_size; ++i) {
        q[i] = (i % 31) - 15;
    }
    for (int i = 0; i < kv_size; ++i) {
        k[i] = (i % 17) - 8;
        v[i] = (i % 23) - 11;
    }

    push64(input, (static_cast<uint64_t>(1u) << 32) | 0x41544e31u);
    push64(input, (static_cast<uint64_t>(float_bits(0.25f)) << 32) |
                      float_bits(0.125f));
    push64(input, (static_cast<uint64_t>(32767u) << 32) | float_bits(0.5f));

    for (int pair = 0; pair < 64; pair += 2) {
        uint32_t p0 = 32767u;
        uint32_t p1 = 32767u;
        push64(input, (static_cast<uint64_t>(p1) << 32) | p0);
    }
    for (int i = 0; i < hidden_size; i += 2) {
        push64(input, (static_cast<uint64_t>(static_cast<uint32_t>(q[i + 1])) << 32) |
                          static_cast<uint32_t>(q[i]));
    }
    for (int i = 0; i < kv_size; i += 2) {
        push64(input, (static_cast<uint64_t>(static_cast<uint32_t>(k[i + 1])) << 32) |
                          static_cast<uint32_t>(k[i]));
    }
    for (int i = 0; i < kv_size; i += 2) {
        push64(input, (static_cast<uint64_t>(static_cast<uint32_t>(v[i + 1])) << 32) |
                          static_cast<uint32_t>(v[i]), (i + 2) == kv_size);
    }

    bitnet_attention_axis(input, output);

    std::vector<uint32_t> words;
    bool saw_last = false;
    while (!output.empty()) {
        axis32_t word = output.read();
        words.push_back(static_cast<uint32_t>(word.data));
        saw_last = word.last != 0;
    }

    const size_t expected_words = 5u + 640u + 160u + 160u;
    if (words.size() != expected_words || !saw_last) {
        std::printf("FAIL: output words=%zu expected=%zu last=%d\n",
                    words.size(), expected_words, saw_last ? 1 : 0);
        return 1;
    }
    if (words[0] != 0x41544e4fu || words[1] != 0u) {
        std::printf("FAIL: output header %08x %08x\n", words[0], words[1]);
        return 1;
    }

    // A one-token attention must reproduce the dynamically quantized V vector
    // in every GQA query head assigned to that KV head.
    for (int q_head = 0; q_head < 20; ++q_head) {
        int kv_head = q_head / 4;
        for (int dim = 0; dim < 128; ++dim) {
            int output_index = q_head * 128 + dim;
            uint32_t packed = words[5u + static_cast<size_t>(output_index / 4)];
            int8_t actual = static_cast<int8_t>(packed >> ((output_index % 4) * 8));
            int source = v[kv_head * 128 + dim];
            int expected = static_cast<int>(std::round(static_cast<float>(source) * 127.0f / 11.0f));
            if (expected > 127) expected = 127;
            if (expected < -128) expected = -128;
            if (actual != expected) {
                std::printf("FAIL: attention[%d]=%d expected=%d\n",
                            output_index, static_cast<int>(actual), expected);
                return 1;
            }
        }
    }

    hls::stream<axis64_t> input2;
    hls::stream<axis32_t> output2;
    push64(input2, (static_cast<uint64_t>(2u) << 32) | 0x41544e31u);
    push64(input2, (static_cast<uint64_t>(float_bits(0.25f)) << 32) |
                       float_bits(0.125f));
    push64(input2, (static_cast<uint64_t>(32767u) << 32) | float_bits(0.5f));
    for (int pair = 0; pair < 64; pair += 2) {
        push64(input2, (static_cast<uint64_t>(32767u) << 32) | 32767u);
    }
    for (int i = 0; i < hidden_size; i += 2) {
        push64(input2, 0u);
    }
    for (int i = 0; i < kv_size; i += 2) {
        push64(input2, 0u);
    }
    for (int i = 0; i < kv_size; i += 2) {
        push64(input2,
               (static_cast<uint64_t>(static_cast<uint32_t>(v[i + 1])) << 32) |
                   static_cast<uint32_t>(v[i]));
    }
    push64(input2, (static_cast<uint64_t>(float_bits(0.1f)) << 32) |
                       float_bits(1.0f));
    for (int i = 0; i < kv_size; i += 8) {
        push64(input2, 0u);
    }
    for (int i = 0; i < kv_size; i += 8) {
        push64(input2, 0x1414141414141414ull, (i + 8) == kv_size);
    }

    bitnet_attention_axis(input2, output2);
    words.clear();
    while (!output2.empty()) {
        words.push_back(static_cast<uint32_t>(output2.read().data));
    }
    if (words.size() != expected_words || words[0] != 0x41544e4fu || words[1] != 0u) {
        std::printf("FAIL: two-token output header/length\n");
        return 1;
    }

    std::vector<float> expected_float(hidden_size);
    float current_v_scale = 0.5f * 11.0f / 127.0f;
    float max_abs = 0.0f;
    for (int q_head = 0; q_head < 20; ++q_head) {
        int kv_head = q_head / 4;
        for (int dim = 0; dim < 128; ++dim) {
            int output_index = q_head * 128 + dim;
            int source = v[kv_head * 128 + dim];
            int current_v_i8 = quantize_reference(static_cast<float>(source), 127.0f / 11.0f);
            float value = (2.0f + static_cast<float>(current_v_i8) * current_v_scale) * 0.5f;
            expected_float[output_index] = value;
            if (std::fabs(value) > max_abs) max_abs = std::fabs(value);
        }
    }
    float attention_quant_scale = 127.0f / max_abs;
    for (int i = 0; i < hidden_size; ++i) {
        uint32_t packed = words[5u + static_cast<size_t>(i / 4)];
        int8_t actual = static_cast<int8_t>(packed >> ((i % 4) * 8));
        int expected = quantize_reference(expected_float[i], attention_quant_scale);
        if (actual != expected) {
            std::printf("FAIL: two-token attention[%d]=%d expected=%d\n",
                        i, static_cast<int>(actual), expected);
            return 1;
        }
    }

    std::printf("PASS: HLS attention one/two-token RoPE/online-softmax/V path\n");
    return 0;
}
