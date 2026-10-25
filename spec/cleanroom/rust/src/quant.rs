//! Stored tensor types (IR_SPEC 4.7): the f32 value of every element of each GGUF storage type.

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // subnormal: man * 2^-24, exact in f32
            let v = man as f32 * (1.0f32 / 16777216.0);
            return if sign == 1 { -v } else { v };
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (man << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// (elements per block, bytes per block) of a ggml type id.
pub fn block_size(ty: u32) -> Option<(usize, usize)> {
    Some(match ty {
        0 => (1, 4),
        1 => (1, 2),
        30 => (1, 2),
        8 => (32, 34),
        2 => (32, 18),
        6 => (32, 22),
        7 => (32, 24),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        _ => return None,
    })
}

pub fn type_id(name: &str) -> Option<u32> {
    Some(match name.to_ascii_uppercase().as_str() {
        "F32" => 0,
        "F16" => 1,
        "BF16" => 30,
        "Q8_0" => 8,
        "Q4_0" => 2,
        "Q5_0" => 6,
        "Q5_1" => 7,
        "Q4_K" => 12,
        "Q5_K" => 13,
        "Q6_K" => 14,
        _ => return None,
    })
}

fn h(b: &[u8], o: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[o], b[o + 1]]))
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Dequantises n elements of type `ty` stored in `bytes`.
pub fn dequantize(ty: u32, bytes: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let (be, bb) = block_size(ty).ok_or(format!("storage type {} is not one IR_SPEC 4.7 defines", ty))?;
    if n % be != 0 {
        return Err(format!("{} elements is not a whole number of blocks of {}", n, be));
    }
    let nb = n / be;
    if bytes.len() < nb * bb {
        return Err(format!("{} bytes hold fewer than {} elements", bytes.len(), n));
    }
    let mut out = Vec::with_capacity(n);
    for bi in 0..nb {
        let b = &bytes[bi * bb..(bi + 1) * bb];
        match ty {
            0 => out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            1 => out.push(h(b, 0)),
            30 => out.push(f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)),
            8 => {
                let d = h(b, 0);
                for j in 0..32 {
                    out.push((b[2 + j] as i8) as f32 * d);
                }
            }
            2 => {
                let d = h(b, 0);
                let qs = &b[2..18];
                for j in 0..32 {
                    let q = if j < 16 { (qs[j] & 15) as i32 } else { (qs[j - 16] >> 4) as i32 };
                    out.push((q - 8) as f32 * d);
                }
            }
            6 | 7 => {
                let d = h(b, 0);
                let (m, qo) = if ty == 7 { (h(b, 2), 4) } else { (0.0, 2) };
                let qh = u32::from_le_bytes([b[qo], b[qo + 1], b[qo + 2], b[qo + 3]]);
                let qs = &b[qo + 4..qo + 20];
                for j in 0..32 {
                    let nib = if j < 16 { (qs[j] & 15) as i32 } else { (qs[j - 16] >> 4) as i32 };
                    let q = nib + 16 * ((qh >> j) & 1) as i32;
                    if ty == 6 {
                        out.push((q - 16) as f32 * d);
                    } else {
                        out.push((q as f32 * d) + m);
                    }
                }
            }
            12 | 13 => {
                let d = h(b, 0);
                let dmin = h(b, 2);
                let scales = &b[4..16];
                let (qh, qs) = if ty == 13 { (&b[16..48], &b[48..176]) } else { (&b[0..0], &b[16..144]) };
                for c in 0..4 {
                    for half in 0..2 {
                        let s = 2 * c + half;
                        let (sc, mn) = scale_min_k4(s, scales);
                        let ds = d * sc as f32;
                        let ms = dmin * mn as f32;
                        for l in 0..32 {
                            let byte = qs[32 * c + l];
                            let mut q = if half == 0 { (byte & 15) as i32 } else { (byte >> 4) as i32 };
                            if ty == 13 && (qh[l] >> s) & 1 == 1 {
                                q += 16;
                            }
                            out.push((ds * q as f32) - ms);
                        }
                    }
                }
            }
            14 => {
                let ql_all = &b[0..128];
                let qh_all = &b[128..192];
                let sc_all = &b[192..208];
                let d = h(b, 208);
                let mut vals = [0f32; 256];
                for nh in 0..2 {
                    let ql = &ql_all[64 * nh..];
                    let qh = &qh_all[32 * nh..];
                    let sc = &sc_all[8 * nh..];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 15) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        let base = 128 * nh + l;
                        vals[base] = (d * (sc[is] as i8) as f32) * q1 as f32;
                        vals[base + 32] = (d * (sc[is + 2] as i8) as f32) * q2 as f32;
                        vals[base + 64] = (d * (sc[is + 4] as i8) as f32) * q3 as f32;
                        vals[base + 96] = (d * (sc[is + 6] as i8) as f32) * q4 as f32;
                    }
                }
                out.extend_from_slice(&vals);
            }
            _ => unreachable!(),
        }
    }
    Ok(out)
}
