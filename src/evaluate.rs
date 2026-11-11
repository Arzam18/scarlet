use crate::bitloop;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
pub use crate::board::*;

impl Board {
    pub fn update_eval(&mut self, piece_type: PieceType, from: u64, to: u64) {
        let offset = 384 * (piece_type as usize & 1) + (piece_type as usize >> 1) * 64;
        let acc = self.get_waccum_mut();
        let weightsf = NNUE.feature_weights[offset + (from.trailing_zeros() ^ 7) as usize];
        let weightst = NNUE.feature_weights[offset + (to.trailing_zeros() ^ 7) as usize];

        accumulator_move(&mut acc.vals, &weightsf.vals, &weightst.vals);

        let offset = 384 * (!(piece_type as usize) & 1) + (piece_type as usize >> 1) * 64;
        let acc = self.get_baccum_mut();
        let weightsf = NNUE.feature_weights[offset + (from.trailing_zeros() ^ 63) as usize];
        let weightst = NNUE.feature_weights[offset + (to.trailing_zeros() ^ 63) as usize];

        accumulator_move(&mut acc.vals, &weightsf.vals, &weightst.vals);
    }
    pub fn update_eval_capture(&mut self, piece_type: PieceType, square: u64) {
        let offset = 384 * (piece_type as usize & 1) + (piece_type as usize >> 1) * 64;
        let acc = self.get_waccum_mut();
        let weights = NNUE.feature_weights[offset + (square.trailing_zeros() ^ 7) as usize];

        accumulator_sub(&mut acc.vals, &weights.vals);

        let offset = 384 * (!(piece_type as usize) & 1) + (piece_type as usize >> 1) * 64;
        let acc = self.get_baccum_mut();
        let weights = NNUE.feature_weights[offset + (square.trailing_zeros() ^ 63) as usize];

        accumulator_sub(&mut acc.vals, &weights.vals);
    }
    pub fn update_eval_promotion(&mut self, promotion_type: PieceType, from: u64, to: u64) {
        let offset_promo = 384 * (promotion_type as usize & 1) + (promotion_type as usize >> 1) * 64;
        let offset_pawn = 384 * (promotion_type as usize & 1);
        let acc = self.get_waccum_mut();
        let weightsf = NNUE.feature_weights[offset_pawn + (from.trailing_zeros() ^ 7) as usize];
        let weightst = NNUE.feature_weights[offset_promo + (to.trailing_zeros() ^ 7) as usize];

        accumulator_move(&mut acc.vals, &weightsf.vals, &weightst.vals);

        let offset_promo = 384 * (!(promotion_type as usize) & 1) + (promotion_type as usize >> 1) * 64;
        let offset_pawn = 384 * (!(promotion_type as usize) & 1);
        let acc = self.get_baccum_mut();
        let weightsf = NNUE.feature_weights[offset_pawn + (from.trailing_zeros() ^ 63) as usize];
        let weightst = NNUE.feature_weights[offset_promo + (to.trailing_zeros() ^ 63) as usize];

        accumulator_move(&mut acc.vals, &weightsf.vals, &weightst.vals);

        // self.set_phase(self.phase() + promotion_type as i32 / 2);
        // match self.color() {
        //     Color::White => {
        //         self.set_mg_eval(self.mg_eval() - EVAL_TABLES[PieceType::WhitePawn as usize][from.trailing_zeros() as usize ^ 63] - MG_PV[PieceType::WhitePawn as usize / 2]);
        //         self.set_eg_eval(self.eg_eval() - EVAL_TABLES[PieceType::WhitePawn as usize + 1][from.trailing_zeros() as usize ^ 63] - EG_PV[PieceType::WhitePawn as usize / 2]);

        //         self.set_mg_eval(self.mg_eval() + EVAL_TABLES[promotion_type as usize][to.trailing_zeros() as usize ^ 63] + MG_PV[promotion_type as usize / 2]);
        //         self.set_eg_eval(self.eg_eval() + EVAL_TABLES[promotion_type as usize + 1][to.trailing_zeros() as usize ^  63] + EG_PV[promotion_type as usize / 2]);
        //     }
        //     Color::Black => {
        //         self.set_mg_eval(self.mg_eval() + EVAL_TABLES[PieceType::BlackPawn as usize - 1][from.trailing_zeros() as usize ^ 7] + MG_PV[PieceType::BlackPawn as usize / 2]);
        //         self.set_eg_eval(self.eg_eval() + EVAL_TABLES[PieceType::BlackPawn as usize][from.trailing_zeros() as usize ^ 7] + EG_PV[PieceType::BlackPawn as usize / 2]);
        //         // println!("{}", EVAL_TABLES[PieceType::BlackPawn as usize - 1][from.trailing_zeros() as usize ^ 7]);
        //         // println!("{}", from);

        //         self.set_mg_eval(self.mg_eval() - EVAL_TABLES[promotion_type as usize - 1][to.trailing_zeros() as usize ^ 7] - MG_PV[promotion_type as usize / 2]);
        //         self.set_eg_eval(self.eg_eval() - EVAL_TABLES[promotion_type as usize][to.trailing_zeros() as usize ^ 7] - EG_PV[promotion_type as usize / 2]);
        //     }
        // }
    }
}

const HIDDEN: usize = 512;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

static NNUE: Network = unsafe { std::mem::transmute(*include_bytes!(r"net_witch_512_4.bin")) };

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct Accumulator {
    vals: [i16; HIDDEN],
}

impl Default for Accumulator {
    fn default() -> Self {
        Accumulator { vals: [0; HIDDEN] }
    }
}

#[repr(C, align(64))]
pub struct Network {
    feature_weights: [Accumulator; 768],
    feature_bias: Accumulator,
    output_weights: [i16; 2 * HIDDEN],
    output_bias: i16,
}

fn screlu(x: i16) -> i32 {
    let y = i32::from(x).clamp(0, i32::from(QA));
    y * y
}

// NEON kernels preserve the engine's modulo-2^16 accumulator updates and
// modulo-2^32 output accumulation exactly.
#[inline(always)] fn accumulator_add(acc:&mut [i16;HIDDEN],w:&[i16;HIDDEN]) {
    #[cfg(target_arch="aarch64")] unsafe { let mut i=0; while i<HIDDEN { let a=vld1q_s16(acc.as_ptr().add(i)); let b=vld1q_s16(w.as_ptr().add(i)); vst1q_s16(acc.as_mut_ptr().add(i),vaddq_s16(a,b)); i+=8; } }
    #[cfg(not(target_arch="aarch64"))] for i in 0..HIDDEN { acc[i]=acc[i].wrapping_add(w[i]); }
}
#[inline(always)] fn accumulator_sub(acc:&mut [i16;HIDDEN],w:&[i16;HIDDEN]) {
    #[cfg(target_arch="aarch64")] unsafe { let mut i=0; while i<HIDDEN { let a=vld1q_s16(acc.as_ptr().add(i)); let b=vld1q_s16(w.as_ptr().add(i)); vst1q_s16(acc.as_mut_ptr().add(i),vsubq_s16(a,b)); i+=8; } }
    #[cfg(not(target_arch="aarch64"))] for i in 0..HIDDEN { acc[i]=acc[i].wrapping_sub(w[i]); }
}
#[inline(always)] fn accumulator_move(acc:&mut [i16;HIDDEN],from:&[i16;HIDDEN],to:&[i16;HIDDEN]) {
    #[cfg(target_arch="aarch64")] unsafe { let mut i=0; while i<HIDDEN { let a=vld1q_s16(acc.as_ptr().add(i)); let f=vld1q_s16(from.as_ptr().add(i)); let t=vld1q_s16(to.as_ptr().add(i)); let v=vaddq_s16(vsubq_s16(a,f),t); vst1q_s16(acc.as_mut_ptr().add(i),v); i+=8; } }
    #[cfg(not(target_arch="aarch64"))] for i in 0..HIDDEN { acc[i]=acc[i].wrapping_sub(from[i]).wrapping_add(to[i]); }
}
#[cfg(any(not(target_arch="aarch64"),test))]
#[inline(always)] fn output_sum_scalar(stm:&[i16;HIDDEN],ntm:&[i16;HIDDEN],w:&[i16;2*HIDDEN])->i32 {
    let mut out=0i32; for i in 0..HIDDEN { out=out.wrapping_add(screlu(stm[i]).wrapping_mul(i32::from(w[i]))); out=out.wrapping_add(screlu(ntm[i]).wrapping_mul(i32::from(w[HIDDEN+i]))); } out
}
#[cfg(target_arch="aarch64")]
#[inline]
unsafe fn output_sum_neon(stm:&[i16;HIDDEN],ntm:&[i16;HIDDEN],w:&[i16;2*HIDDEN])->i32 {
    unsafe {
        let zero=vdupq_n_s16(0); let max=vdupq_n_s16(QA);
        let mut lo=vdupq_n_s32(0); let mut hi=vdupq_n_s32(0); let mut i=0;
        while i<HIDDEN {
            let s=vminq_s16(vmaxq_s16(vld1q_s16(stm.as_ptr().add(i)),zero),max);
            let n=vminq_s16(vmaxq_s16(vld1q_s16(ntm.as_ptr().add(i)),zero),max);
            let sl=vmovl_s16(vget_low_s16(s)); let sh=vmovl_s16(vget_high_s16(s));
            let nl=vmovl_s16(vget_low_s16(n)); let nh=vmovl_s16(vget_high_s16(n));
            let sw=vld1q_s16(w.as_ptr().add(i)); let nw=vld1q_s16(w.as_ptr().add(HIDDEN+i));
            let swl=vmovl_s16(vget_low_s16(sw)); let swh=vmovl_s16(vget_high_s16(sw));
            let nwl=vmovl_s16(vget_low_s16(nw)); let nwh=vmovl_s16(vget_high_s16(nw));
            lo=vaddq_s32(lo,vmulq_s32(vmulq_s32(sl,sl),swl));
            hi=vaddq_s32(hi,vmulq_s32(vmulq_s32(sh,sh),swh));
            lo=vaddq_s32(lo,vmulq_s32(vmulq_s32(nl,nl),nwl));
            hi=vaddq_s32(hi,vmulq_s32(vmulq_s32(nh,nh),nwh));
            i+=8;
        }
        vaddvq_s32(lo).wrapping_add(vaddvq_s32(hi))
    }
}

// The bundled network's 1,024 output weights are all in [-127, 127].
// This permits an exact signed-byte SDOT implementation for the output
// layer only. Feature weights remain i16 and are accumulated with NEON.
#[cfg(target_arch = "aarch64")]
struct DotprodState {
    enabled: bool,
    stm_weight_sums: [i32; HIDDEN / 4],
    ntm_weight_sums: [i32; HIDDEN / 4],
}

#[cfg(target_arch = "aarch64")]
fn dotprod_state() -> &'static DotprodState {
    use std::sync::OnceLock;
    static STATE: OnceLock<DotprodState> = OnceLock::new();
    STATE.get_or_init(|| {
        let fits_signed_byte = NNUE.output_weights.iter().all(|&x| {
            x >= i16::from(i8::MIN) && x <= i16::from(i8::MAX)
        });
        let enabled = fits_signed_byte
            && std::arch::is_aarch64_feature_detected!("dotprod");
        let mut stm_weight_sums = [0i32; HIDDEN / 4];
        let mut ntm_weight_sums = [0i32; HIDDEN / 4];
        if fits_signed_byte {
            for group in 0..(HIDDEN / 4) {
                let base = group * 4;
                for lane in 0..4 {
                    stm_weight_sums[group] += i32::from(NNUE.output_weights[base + lane]);
                    ntm_weight_sums[group] += i32::from(NNUE.output_weights[HIDDEN + base + lane]);
                }
            }
        }
        DotprodState { enabled, stm_weight_sums, ntm_weight_sums }
    })
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn squared_bytes_8(values: &[i16; HIDDEN], offset: usize) -> (uint8x8_t, uint8x8_t) {
    unsafe {
        let x = vminq_s16(
            vmaxq_s16(vld1q_s16(values.as_ptr().add(offset)), vdupq_n_s16(0)),
            vdupq_n_s16(QA),
        );
        let low = vmovl_s16(vget_low_s16(x));
        let high = vmovl_s16(vget_high_s16(x));
        let sq_low = vmulq_s32(low, low);
        let sq_high = vmulq_s32(high, high);
        // Each square is <= 65,025, so narrowing to u16 is exact.
        let sq = vcombine_u16(
            vmovn_u32(vreinterpretq_u32_s32(sq_low)),
            vmovn_u32(vreinterpretq_u32_s32(sq_high)),
        );
        (vmovn_u16(sq), vshrn_n_u16(sq, 8))
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn output_sum_dotprod(
    stm: &[i16; HIDDEN],
    ntm: &[i16; HIDDEN],
    w: &[i16; 2 * HIDDEN],
    stm_weight_sums: &[i32; HIDDEN / 4],
    ntm_weight_sums: &[i32; HIDDEN / 4],
) -> i32 {
    unsafe {
        let sign_bit = vdupq_n_u8(0x80);
        let mut total = vdupq_n_s32(0);
        let mut i = 0;
        while i < HIDDEN {
            let (stm_low_0, stm_high_0) = squared_bytes_8(stm, i);
            let (stm_low_1, stm_high_1) = squared_bytes_8(stm, i + 8);
            let (ntm_low_0, ntm_high_0) = squared_bytes_8(ntm, i);
            let (ntm_low_1, ntm_high_1) = squared_bytes_8(ntm, i + 8);
            let stm_low = vcombine_u8(stm_low_0, stm_low_1);
            let stm_high = vcombine_u8(stm_high_0, stm_high_1);
            let ntm_low = vcombine_u8(ntm_low_0, ntm_low_1);
            let ntm_high = vcombine_u8(ntm_high_0, ntm_high_1);

            // Each SDOT call consumes 16 signed-byte weights. Load two
            // 8-lane i16 vectors, then narrow each to 8 lanes and combine.
            // The runtime range check proves these non-saturating narrows exact.
            let stm_w16_0 = vld1q_s16(w.as_ptr().add(i));
            let stm_w16_1 = vld1q_s16(w.as_ptr().add(i + 8));
            let ntm_w16_0 = vld1q_s16(w.as_ptr().add(HIDDEN + i));
            let ntm_w16_1 = vld1q_s16(w.as_ptr().add(HIDDEN + i + 8));
            let stm_w8 = vcombine_s8(vmovn_s16(stm_w16_0), vmovn_s16(stm_w16_1));
            let ntm_w8 = vcombine_s8(vmovn_s16(ntm_w16_0), vmovn_s16(ntm_w16_1));

            // XOR with 0x80 maps unsigned byte x to signed (x - 128).
            // Since x² = lo + 256*hi, the exact offset correction is
            // (128 + 256*128) * sum(weights) = 32,896 * sum(weights).
            let stm_low_signed = vreinterpretq_s8_u8(veorq_u8(stm_low, sign_bit));
            let stm_high_signed = vreinterpretq_s8_u8(veorq_u8(stm_high, sign_bit));
            let ntm_low_signed = vreinterpretq_s8_u8(veorq_u8(ntm_low, sign_bit));
            let ntm_high_signed = vreinterpretq_s8_u8(veorq_u8(ntm_high, sign_bit));

            let mut stm_part = vdotq_s32(vdupq_n_s32(0), stm_low_signed, stm_w8);
            stm_part = vaddq_s32(stm_part, vshlq_n_s32(
                vdotq_s32(vdupq_n_s32(0), stm_high_signed, stm_w8), 8,
            ));
            let mut ntm_part = vdotq_s32(vdupq_n_s32(0), ntm_low_signed, ntm_w8);
            ntm_part = vaddq_s32(ntm_part, vshlq_n_s32(
                vdotq_s32(vdupq_n_s32(0), ntm_high_signed, ntm_w8), 8,
            ));

            let sum_offset = i / 4;
            let stm_sums = vld1q_s32(stm_weight_sums.as_ptr().add(sum_offset));
            let ntm_sums = vld1q_s32(ntm_weight_sums.as_ptr().add(sum_offset));
            let correction_stm = vmulq_n_s32(stm_sums, 32_896);
            let correction_ntm = vmulq_n_s32(ntm_sums, 32_896);
            total = vaddq_s32(
                total,
                vaddq_s32(
                    vaddq_s32(stm_part, correction_stm),
                    vaddq_s32(ntm_part, correction_ntm),
                ),
            );
            i += 16;
        }
        vaddvq_s32(total)
    }
}

impl Board {
    pub fn populate_accumulators(&mut self) {
        let wpieces = [
            self.get_bitboard(PieceType::WhitePawn),
            self.get_bitboard(PieceType::WhiteKnight),
            self.get_bitboard(PieceType::WhiteBishop),
            self.get_bitboard(PieceType::WhiteRook),
            self.get_bitboard(PieceType::WhiteQueen),
            self.get_bitboard(PieceType::WhiteKing),
        ];
        let bpieces = [
            self.get_bitboard(PieceType::BlackPawn),
            self.get_bitboard(PieceType::BlackKnight),
            self.get_bitboard(PieceType::BlackBishop),
            self.get_bitboard(PieceType::BlackRook),
            self.get_bitboard(PieceType::BlackQueen),
            self.get_bitboard(PieceType::BlackKing),
        ];

        // Do white accumulator
        let mut acc = [0; HIDDEN];
        for i in 0..6 {
            let bb = wpieces[i];

            bitloop!(bb {
                let pos = (bb.blsi().trailing_zeros() ^ 7) as usize;
                let weights = NNUE.feature_weights[64 * i + pos];

                accumulator_add(&mut acc, &weights.vals);
            });

            bb = bpieces[i];
            bitloop!(bb {
                let pos = (bb.blsi().trailing_zeros() ^ 7) as usize;
                let weights = NNUE.feature_weights[384 + 64 * i + pos];

                accumulator_add(&mut acc, &weights.vals);
            });
        }

        accumulator_add(&mut acc, &NNUE.feature_bias.vals);

        self.set_waccum(Accumulator { vals: acc });

        // Do black accumulator
        acc = [0; HIDDEN];
        for i in 0..6 {
            let bb = bpieces[i];

            bitloop!(bb {
                let pos = (bb.blsi().trailing_zeros() ^ 63) as usize;
                let weights = NNUE.feature_weights[64 * i + pos];

                accumulator_add(&mut acc, &weights.vals);
            });

            bb = wpieces[i];
            bitloop!(bb {
                let pos = (bb.blsi().trailing_zeros() ^ 63) as usize;
                let weights = NNUE.feature_weights[384 + 64 * i + pos];

                accumulator_add(&mut acc, &weights.vals);
            });
        }

        accumulator_add(&mut acc, &NNUE.feature_bias.vals);

        self.set_baccum(Accumulator { vals: acc });
    }

    pub fn gen_eval(&mut self) -> i32 {
        let stm_accum = if self.color() == Color::White { self.get_waccum() } else { self.get_baccum() };
        let ntm_accum = if self.color() == Color::White { self.get_baccum() } else { self.get_waccum() };

        #[cfg(target_arch = "aarch64")]
        let mut output = {
            let state = dotprod_state();
            if state.enabled {
                // The runtime CPU and network-range checks guard this function.
                unsafe {
                    output_sum_dotprod(
                        &stm_accum.vals,
                        &ntm_accum.vals,
                        &NNUE.output_weights,
                        &state.stm_weight_sums,
                        &state.ntm_weight_sums,
                    )
                }
            } else {
                unsafe { output_sum_neon(&stm_accum.vals, &ntm_accum.vals, &NNUE.output_weights) }
            }
        };
        #[cfg(not(target_arch = "aarch64"))]
        let mut output = output_sum_scalar(&stm_accum.vals, &ntm_accum.vals, &NNUE.output_weights);

        output /= QA as i32;

        output += NNUE.output_bias as i32;

        output *= SCALE;

        output /= QA as i32 * QB as i32;

        self.set_eval(output);

        output
    }
}


#[cfg(all(test, target_arch = "aarch64"))]
mod neon_equivalence_tests {
    use super::*;

    fn next_random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn neon_output_matches_scalar_for_random_lanes() {
        let mut state = 0x6a09_e667_f3bc_c909u64;
        for _ in 0..256 {
            let mut stm = [0i16; HIDDEN];
            let mut ntm = [0i16; HIDDEN];
            let mut weights = [0i16; 2 * HIDDEN];
            for x in &mut stm { *x = next_random(&mut state) as i16; }
            for x in &mut ntm { *x = next_random(&mut state) as i16; }
            for x in &mut weights { *x = next_random(&mut state) as i16; }
            let scalar = output_sum_scalar(&stm, &ntm, &weights);
            let neon = unsafe { output_sum_neon(&stm, &ntm, &weights) };
            assert_eq!(neon, scalar);
        }
    }

    #[test]
    fn neon_accumulator_updates_match_scalar_wrapping_math() {
        let mut state = 0xbb67_ae85_84ca_a73bu64;
        let mut values = [0i16; HIDDEN];
        let mut from = [0i16; HIDDEN];
        let mut to = [0i16; HIDDEN];
        for x in &mut values { *x = next_random(&mut state) as i16; }
        for x in &mut from { *x = next_random(&mut state) as i16; }
        for x in &mut to { *x = next_random(&mut state) as i16; }

        let mut expected = values;
        let mut actual = values;
        accumulator_move(&mut actual, &from, &to);
        for i in 0..HIDDEN {
            expected[i] = expected[i].wrapping_sub(from[i]).wrapping_add(to[i]);
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn dotprod_output_matches_scalar_for_signed_byte_weights() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let mut state = 0x3c6e_f372_fe94_f82bu64;
        for _ in 0..128 {
            let mut stm = [0i16; HIDDEN];
            let mut ntm = [0i16; HIDDEN];
            let mut weights = [0i16; 2 * HIDDEN];
            for x in &mut stm { *x = next_random(&mut state) as i16; }
            for x in &mut ntm { *x = next_random(&mut state) as i16; }
            for x in &mut weights { *x = ((next_random(&mut state) as u8) as i8) as i16; }
            let mut stm_sums = [0i32; HIDDEN / 4];
            let mut ntm_sums = [0i32; HIDDEN / 4];
            for group in 0..(HIDDEN / 4) {
                for lane in 0..4 {
                    stm_sums[group] += i32::from(weights[group * 4 + lane]);
                    ntm_sums[group] += i32::from(weights[HIDDEN + group * 4 + lane]);
                }
            }
            let scalar = output_sum_scalar(&stm, &ntm, &weights);
            let dotprod = unsafe { output_sum_dotprod(&stm, &ntm, &weights, &stm_sums, &ntm_sums) };
            assert_eq!(dotprod, scalar);
        }
    }
}
