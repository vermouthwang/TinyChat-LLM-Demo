#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

#ifdef QM_ARM
#include <arm_neon.h>
#endif
#ifdef QM_x86
#include <immintrin.h>
#endif
struct w4a8_thread_args {
    int start_j, end_j;
    const struct matmul_params *params;
};
static void *all_techniques_worker_func(void *args) {
    struct w4a8_thread_args *mat_args = (struct w4a8_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
    const int num_block = k / block_size;  // block_size = 32

    for (int row = 0; row < m; row++) {
        for (int col = mat_args->start_j; col < mat_args->end_j; col++) {

            // order of weights with QM_ARM:
            // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w30,w31)
            // QM_ARM order: (w0,w16),(w1,w17),(w2,w18),(w3,w19),(w4, w20),... (w15,w31)
            //               |--|
            //               4 bits
            //               |------|
            //               8 bits (byte)
            //            low|----------------------------------------------------------|high
            //               0                         128 bit                         127
            float32x4_t sumv0 = vdupq_n_f32(0.0f);
            float32x4_t sumv1 = vdupq_n_f32(0.0f);
            float32x4_t sumv2 = vdupq_n_f32(0.0f);
            float32x4_t sumv3 = vdupq_n_f32(0.0f);
            // pointer of the int4 weights
            const unsigned char *w_start = &params->B.int4_data_ptr[col * k / 2];
            // pointer of the int8 activation
            const signed char *a_start = &params->A.int8_data_ptr[row * k];
            // scale of activation
            float *s_a = &params->A_scales[row * k / 32];
            // scale of weight
            float *s_w = &params->scales[col * k / 32];

            // process four blocks each iteration
            for (int q = 0; q < num_block; q += 4) {
                // load 32x4bit (16 bytes) weight
                const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
                const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
                const uint8x16_t w2 = vld1q_u8(w_start + 32);  // 32 4bit weight
                const uint8x16_t w3 = vld1q_u8(w_start + 48);  // 32 4bit weight
                w_start += 64;

                // TODO: decode each uint8x16_t weight vector into the lower and upper half of the weights as int8x16_t
                // Hint:
                // (1) use `vandq_u8` with the mask_low4bit to get the lower half
                // (2) use `vshrq_n_u8` to right shift 4 bits and get the upper half
                // (3) use `vreinterpretq_s8_u8` to interpret the  vector as int8
                // lowbit mask
                const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
                //
                int8x16_t lower_half_w0 = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                int8x16_t upper_half_w0 = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                int8x16_t lower_half_w1 = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
                int8x16_t upper_half_w1 = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
                int8x16_t lower_half_w2 = vreinterpretq_s8_u8(vandq_u8(w2, mask_low4bit));
                int8x16_t upper_half_w2 = vreinterpretq_s8_u8(vshrq_n_u8(w2, 4));
                int8x16_t lower_half_w3 = vreinterpretq_s8_u8(vandq_u8(w3, mask_low4bit));
                int8x16_t upper_half_w3 = vreinterpretq_s8_u8(vshrq_n_u8(w3, 4));

                // TODO: apply zero_point to weights and convert the range from (0, 15) to (-8, 7)
                // Hint: using `vsubq_s8` to the lower-half and upper-half vectors of weights
                const int8x16_t offsets = vdupq_n_s8(8);
                //
                lower_half_w0 = vsubq_s8(lower_half_w0, offsets);
                upper_half_w0 = vsubq_s8(upper_half_w0, offsets);
                lower_half_w1 = vsubq_s8(lower_half_w1, offsets);
                upper_half_w1 = vsubq_s8(upper_half_w1, offsets);
                lower_half_w2 = vsubq_s8(lower_half_w2, offsets);
                upper_half_w2 = vsubq_s8(upper_half_w2, offsets);
                lower_half_w3 = vsubq_s8(lower_half_w3, offsets);
                upper_half_w3 = vsubq_s8(upper_half_w3, offsets);


                // load 128 8-bit activation
                const int8x16_t a0 = vld1q_s8(a_start);
                const int8x16_t a1 = vld1q_s8(a_start + 16);
                const int8x16_t a2 = vld1q_s8(a_start + 32);
                const int8x16_t a3 = vld1q_s8(a_start + 48);
                const int8x16_t a4 = vld1q_s8(a_start + 64);
                const int8x16_t a5 = vld1q_s8(a_start + 80);
                const int8x16_t a6 = vld1q_s8(a_start + 96);
                const int8x16_t a7 = vld1q_s8(a_start + 112);
                a_start += 128;

                // TODO: perform dot product and store the result into the intermediate sum, int_sum0
                // Hint: use `vdotq_s32` and store the sum for each block in int_sum{0-3}
                // int32x4_t int_sum0, int_sum1, int_sum2, int_sum3;
                //
                int32x4_t int_sum0 = vdupq_n_s32(0);
                int_sum0 = vdotq_s32(int_sum0, lower_half_w0, a0);
                int_sum0 = vdotq_s32(int_sum0, upper_half_w0, a1);

                int32x4_t int_sum1 = vdupq_n_s32(0);
                int_sum1 = vdotq_s32(int_sum1, lower_half_w1, a2);
                int_sum1 = vdotq_s32(int_sum1, upper_half_w1, a3);

                int32x4_t int_sum2 = vdupq_n_s32(0);
                int_sum2 = vdotq_s32(int_sum2, lower_half_w2, a4);
                int_sum2 = vdotq_s32(int_sum2, upper_half_w2, a5);

                int32x4_t int_sum3 = vdupq_n_s32(0);
                int_sum3 = vdotq_s32(int_sum3, lower_half_w3, a6);
                int_sum3 = vdotq_s32(int_sum3, upper_half_w3, a7);
                //
                float s_0 = *s_a++ * *s_w++;
                float s_1 = *s_a++ * *s_w++;
                float s_2 = *s_a++ * *s_w++;
                float s_3 = *s_a++ * *s_w++;

                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum1), s_1);
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum2), s_2);
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum3), s_3);
            }
            params->C.data_ptr[row * n + col] = vaddvq_f32(sumv0);

        }
    }

    return NULL;
}

namespace matmul {
void MatmulOperator::mat_mul_all_techniques(struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    assert(params->block_size % 32 == 0);  // support block size to be multiples of 32
    assert(A->row == C->row);              // support block size to be multiples of 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    const int num_thread = 8;
    pthread_t thread_pool[num_thread];
    struct w4a8_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now

    // TODO: Thread creation
        for (int j = 0; j < num_thread; j++) {
        threads_args[j].start_j = j * (C->column / num_thread);
        threads_args[j].end_j = (j == num_thread - 1) ? C->column : (j + 1) * (C->column / num_thread); // Ensure last thread covers all remaining
        threads_args[j].params = params; // Pass the params pointer directly
        pthread_create(&thread_pool[j], NULL, all_techniques_worker_func, &threads_args[j]);
    }
    // Join threads
    for (int j = 0; j < num_thread; j++)
    {
        pthread_join(thread_pool[j], NULL);
    }

};
}  // namespace matmul
