#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <stdint.h>

namespace {

static const uint32_t kInputMagic = 0x41544e31u;  // ATN1
static const uint32_t kOutputMagic = 0x41544e4fu; // ATNO
static const int kQHeads = 20;
static const int kKvHeads = 5;
static const int kHeadDim = 128;
static const int kGroupSize = kQHeads / kKvHeads;
static const int kHiddenSize = kQHeads * kHeadDim;
static const int kKvSize = kKvHeads * kHeadDim;
static const int kRopePairs = kHeadDim / 2;
static const int kMaxContext = 4096;
static const float kInvSqrtHeadDim = 0.08838834764831843f;

typedef ap_axiu<64, 0, 0, 0> axis64_t;
typedef ap_axiu<32, 0, 0, 0> axis32_t;

union FloatBits {
    uint32_t u;
    float f;
};

uint64_t read_u64(hls::stream<axis64_t> &input) {
#pragma HLS INLINE
    axis64_t word = input.read();
    return static_cast<uint64_t>(word.data);
}

void write_u32(hls::stream<axis32_t> &output, uint32_t value, bool last) {
#pragma HLS INLINE
    axis32_t word;
    word.data = value;
    word.keep = -1;
    word.strb = -1;
    word.last = last ? 1 : 0;
    output.write(word);
}

int32_t rounded_q15_product(int64_t product) {
#pragma HLS INLINE
    if (product >= 0) {
        return static_cast<int32_t>((product + 16384) >> 15);
    } else {
        return -static_cast<int32_t>(((-product) + 16384) >> 15);
    }
}

int8_t quantize_value(float value, float scale) {
#pragma HLS INLINE
    float scaled_value = value * scale;
    int32_t rounded = (scaled_value >= 0.0f)
                          ? static_cast<int32_t>(scaled_value + 0.5f)
                          : static_cast<int32_t>(scaled_value - 0.5f);
    if (rounded > 127) {
        rounded = 127;
    } else if (rounded < -128) {
        rounded = -128;
    }
    return static_cast<int8_t>(rounded);
}

float quantize_i32(const int32_t *source, int8_t *destination, int count,
                   float source_factor) {
    uint32_t max_abs = 0u;

    for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        int64_t wide = source[i];
        uint32_t magnitude = static_cast<uint32_t>((wide < 0) ? -wide : wide);
        if (magnitude > max_abs) {
            max_abs = magnitude;
        }
    }

    if (max_abs == 0u) {
        for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
            destination[i] = 0;
        }
        return 1.0f;
    }

    float quant_scale = 127.0f / static_cast<float>(max_abs);
    for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        destination[i] = quantize_value(static_cast<float>(source[i]), quant_scale);
    }
    return source_factor / quant_scale;
}

void rotate_rope(int32_t *vector, int head_count, const int16_t *cos_table,
                 const int16_t *sin_table) {
    int64_t product_x0_cos[kRopePairs];
    int64_t product_x1_sin[kRopePairs];
    int64_t product_x1_cos[kRopePairs];
    int64_t product_x0_sin[kRopePairs];

    for (int head = 0; head < head_count; ++head) {
        int base = head * kHeadDim;
        for (int pair = 0; pair < kRopePairs; ++pair) {
#pragma HLS PIPELINE II=1
            int lo = base + pair;
            int hi = lo + kRopePairs;
            int32_t x0 = vector[lo];
            int32_t x1 = vector[hi];
            product_x0_cos[pair] = static_cast<int64_t>(x0) * cos_table[pair];
            product_x1_sin[pair] = static_cast<int64_t>(x1) * sin_table[pair];
            product_x1_cos[pair] = static_cast<int64_t>(x1) * cos_table[pair];
            product_x0_sin[pair] = static_cast<int64_t>(x0) * sin_table[pair];
        }
        for (int pair = 0; pair < kRopePairs; ++pair) {
#pragma HLS PIPELINE II=1
            int lo = base + pair;
            int hi = lo + kRopePairs;
            vector[lo] = rounded_q15_product(product_x0_cos[pair]) -
                         rounded_q15_product(product_x1_sin[pair]);
            vector[hi] = rounded_q15_product(product_x1_cos[pair]) +
                         rounded_q15_product(product_x0_sin[pair]);
        }
    }
}

void update_online_attention(const int8_t *q, const int8_t *key,
                             const int8_t *value, float q_scale,
                             float key_scale, float value_scale,
                             float *running_max, float *denominator,
    float *accumulator) {
    for (int q_head = 0; q_head < kQHeads; ++q_head) {
        int q_base = q_head * kHeadDim;
        int kv_base = (q_head / kGroupSize) * kHeadDim;
        int32_t dot = 0;

        for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=16
            dot += static_cast<int32_t>(q[q_base + dim]) *
                   static_cast<int32_t>(key[kv_base + dim]);
        }

        float score = static_cast<float>(dot) * q_scale * key_scale *
                      kInvSqrtHeadDim;
        float old_max = running_max[q_head];
        float old_denominator = denominator[q_head];
        float old_weight;
        float new_weight;
        float new_max;

        if (old_denominator == 0.0f) {
            old_weight = 0.0f;
            new_weight = 1.0f;
            new_max = score;
        } else if (score > old_max) {
            old_weight = hls::expf(old_max - score);
            new_weight = 1.0f;
            new_max = score;
        } else {
            old_weight = 1.0f;
            new_weight = hls::expf(score - old_max);
            new_max = old_max;
        }

        denominator[q_head] = old_denominator * old_weight + new_weight;
        running_max[q_head] = new_max;

        for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=16
#pragma HLS DEPENDENCE variable=accumulator inter false
            int output_index = q_base + dim;
            float weighted_value = new_weight *
                                   static_cast<float>(value[kv_base + dim]) *
                                   value_scale;
            accumulator[output_index] =
                accumulator[output_index] * old_weight + weighted_value;
        }
    }
}

} // namespace

extern "C" void bitnet_attention_axis(hls::stream<axis64_t> &S_AXIS,
                                       hls::stream<axis32_t> &M_AXIS) {
#pragma HLS INTERFACE axis port=S_AXIS
#pragma HLS INTERFACE axis port=M_AXIS
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS ALLOCATION instances=quantize_i32 limit=1 function
#pragma HLS ALLOCATION instances=rotate_rope limit=1 function

    int16_t rope_cos[kRopePairs];
    int16_t rope_sin[kRopePairs];
    int32_t q_i32[kHiddenSize];
    int32_t k_i32[kKvSize];
    int32_t v_i32[kKvSize];
    int8_t q_i8[kHiddenSize];
    int8_t k_i8[kKvSize];
    int8_t v_i8[kKvSize];
    int8_t stream_k[kKvSize];
    int8_t stream_v[kKvSize];
    float running_max[kQHeads];
    float denominator[kQHeads];
    float accumulator[kHiddenSize];
    int8_t attention_i8[kHiddenSize];

#pragma HLS ARRAY_PARTITION variable=q_i8 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=k_i8 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=v_i8 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=stream_k cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=stream_v cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=accumulator cyclic factor=16 dim=1

    uint64_t header0 = read_u64(S_AXIS);
    uint32_t magic = static_cast<uint32_t>(header0);
    uint32_t context_len = static_cast<uint32_t>(header0 >> 32);
    uint64_t header1 = read_u64(S_AXIS);
    uint64_t header2 = read_u64(S_AXIS);
    FloatBits q_factor_bits;
    FloatBits k_factor_bits;
    FloatBits v_factor_bits;
    q_factor_bits.u = static_cast<uint32_t>(header1);
    k_factor_bits.u = static_cast<uint32_t>(header1 >> 32);
    v_factor_bits.u = static_cast<uint32_t>(header2);
    uint32_t q15_scale = static_cast<uint32_t>(header2 >> 32);

    if ((magic != kInputMagic) || (context_len == 0u) ||
        (q15_scale != 32767u) ||
        (context_len > static_cast<uint32_t>(kMaxContext))) {
        write_u32(M_AXIS, kOutputMagic, false);
        write_u32(M_AXIS, 1u, true);
        return;
    }

    for (int pair_word = 0; pair_word < kRopePairs / 2; ++pair_word) {
#pragma HLS PIPELINE II=1
        uint64_t packed = read_u64(S_AXIS);
        uint32_t pair0 = static_cast<uint32_t>(packed);
        uint32_t pair1 = static_cast<uint32_t>(packed >> 32);
        rope_cos[pair_word * 2] = static_cast<int16_t>(pair0 & 0xffffu);
        rope_sin[pair_word * 2] = static_cast<int16_t>(pair0 >> 16);
        rope_cos[pair_word * 2 + 1] = static_cast<int16_t>(pair1 & 0xffffu);
        rope_sin[pair_word * 2 + 1] = static_cast<int16_t>(pair1 >> 16);
    }

    for (int i = 0; i < kHiddenSize; i += 2) {
#pragma HLS PIPELINE II=1
        uint64_t packed = read_u64(S_AXIS);
        q_i32[i] = static_cast<int32_t>(packed & 0xffffffffu);
        q_i32[i + 1] = static_cast<int32_t>(packed >> 32);
    }
    for (int i = 0; i < kKvSize; i += 2) {
#pragma HLS PIPELINE II=1
        uint64_t packed = read_u64(S_AXIS);
        k_i32[i] = static_cast<int32_t>(packed & 0xffffffffu);
        k_i32[i + 1] = static_cast<int32_t>(packed >> 32);
    }
    for (int i = 0; i < kKvSize; i += 2) {
#pragma HLS PIPELINE II=1
        uint64_t packed = read_u64(S_AXIS);
        v_i32[i] = static_cast<int32_t>(packed & 0xffffffffu);
        v_i32[i + 1] = static_cast<int32_t>(packed >> 32);
    }

    rotate_rope(q_i32, kQHeads, rope_cos, rope_sin);
    rotate_rope(k_i32, kKvHeads, rope_cos, rope_sin);
    float q_scale = quantize_i32(q_i32, q_i8, kHiddenSize, q_factor_bits.f);
    float k_scale = quantize_i32(k_i32, k_i8, kKvSize, k_factor_bits.f);
    float v_scale = quantize_i32(v_i32, v_i8, kKvSize, v_factor_bits.f);

    for (int head = 0; head < kQHeads; ++head) {
#pragma HLS PIPELINE II=1
        running_max[head] = 0.0f;
        denominator[head] = 0.0f;
    }
    for (int i = 0; i < kHiddenSize; ++i) {
#pragma HLS PIPELINE II=1
        accumulator[i] = 0.0f;
    }

    for (uint32_t position = 0; position + 1u < context_len; ++position) {
        uint64_t scale_word = read_u64(S_AXIS);
        FloatBits past_k_scale;
        FloatBits past_v_scale;
        past_k_scale.u = static_cast<uint32_t>(scale_word);
        past_v_scale.u = static_cast<uint32_t>(scale_word >> 32);

        for (int i = 0; i < kKvSize; i += 8) {
#pragma HLS PIPELINE II=1
            uint64_t packed = read_u64(S_AXIS);
            for (int byte = 0; byte < 8; ++byte) {
#pragma HLS UNROLL
                stream_k[i + byte] = static_cast<int8_t>(packed >> (byte * 8));
            }
        }
        for (int i = 0; i < kKvSize; i += 8) {
#pragma HLS PIPELINE II=1
            uint64_t packed = read_u64(S_AXIS);
            for (int byte = 0; byte < 8; ++byte) {
#pragma HLS UNROLL
                stream_v[i + byte] = static_cast<int8_t>(packed >> (byte * 8));
            }
        }
        update_online_attention(q_i8, stream_k, stream_v, q_scale,
                                past_k_scale.f, past_v_scale.f,
                                running_max, denominator, accumulator);
    }

    update_online_attention(q_i8, k_i8, v_i8, q_scale, k_scale, v_scale,
                            running_max, denominator, accumulator);

    float max_abs = 0.0f;
    for (int i = 0; i < kHiddenSize; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=accumulator inter false
        int head = i / kHeadDim;
        float value = accumulator[i] / denominator[head];
        accumulator[i] = value;
        float magnitude = (value < 0.0f) ? -value : value;
        if (magnitude > max_abs) {
            max_abs = magnitude;
        }
    }

    float attention_scale = (max_abs <= 1.0e-20f) ? 1.0f : max_abs / 127.0f;
    float output_quant_scale = (max_abs <= 1.0e-20f) ? 1.0f : 127.0f / max_abs;
    for (int i = 0; i < kHiddenSize; ++i) {
#pragma HLS PIPELINE II=1
        attention_i8[i] = quantize_value(accumulator[i], output_quant_scale);
    }

    FloatBits attention_scale_bits;
    FloatBits k_scale_bits;
    FloatBits v_scale_bits;
    attention_scale_bits.f = attention_scale;
    k_scale_bits.f = k_scale;
    v_scale_bits.f = v_scale;

    write_u32(M_AXIS, kOutputMagic, false);
    write_u32(M_AXIS, 0u, false);
    write_u32(M_AXIS, attention_scale_bits.u, false);
    write_u32(M_AXIS, k_scale_bits.u, false);
    write_u32(M_AXIS, v_scale_bits.u, false);

    for (int i = 0; i < kHiddenSize; i += 4) {
        uint32_t packed = 0u;
        for (int byte = 0; byte < 4; ++byte) {
            packed |= static_cast<uint32_t>(static_cast<uint8_t>(attention_i8[i + byte]))
                      << (byte * 8);
        }
        write_u32(M_AXIS, packed, false);
    }
    for (int i = 0; i < kKvSize; i += 4) {
        uint32_t packed = 0u;
        for (int byte = 0; byte < 4; ++byte) {
            packed |= static_cast<uint32_t>(static_cast<uint8_t>(k_i8[i + byte]))
                      << (byte * 8);
        }
        write_u32(M_AXIS, packed, false);
    }
    for (int i = 0; i < kKvSize; i += 4) {
        uint32_t packed = 0u;
        for (int byte = 0; byte < 4; ++byte) {
            packed |= static_cast<uint32_t>(static_cast<uint8_t>(v_i8[i + byte]))
                      << (byte * 8);
        }
        write_u32(M_AXIS, packed, (i + 4) == kKvSize);
    }
}
