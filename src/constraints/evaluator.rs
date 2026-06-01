use ndarray::ArrayView1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

// Opcodes for the stack machine.
//
// The Python compiler emits a flat u8 instruction stream.  Each opcode
// is followed by zero or one index into the constant pool or the
// feature-index pool, encoded as a u32 (4 bytes, little-endian) right
// after the opcode byte.
const OP_CONST: u8 = 1;
const OP_FEATURE: u8 = 2;
const OP_ADD: u8 = 3;
const OP_SUB: u8 = 4;
const OP_MUL: u8 = 5;
const OP_DIV: u8 = 6;
const OP_POW: u8 = 7;
const OP_MOD: u8 = 8;
const OP_SAFE_DIV: u8 = 9;
const OP_LOG: u8 = 10;
const OP_LOG_SAFE: u8 = 11;
const OP_LESS_EQUAL: u8 = 20;
const OP_LESS: u8 = 21;
const OP_EQUAL: u8 = 22;
const OP_EQUAL_TOL: u8 = 23;
const OP_AND: u8 = 24; // arg = n_operands
const OP_OR: u8 = 25; // arg = n_operands
const OP_COUNT: u8 = 26; // arg = n_operands
const OP_COUNT_INV: u8 = 27; // arg = n_operands

const LESS_EPS: f64 = 1e-8;

fn read_u32(bytecode: &[u8], ip: usize) -> u32 {
    u32::from_le_bytes([
        bytecode[ip],
        bytecode[ip + 1],
        bytecode[ip + 2],
        bytecode[ip + 3],
    ])
}

fn eval_sample(
    row: ArrayView1<f64>,
    bytecode: &[u8],
    constants: &[f64],
    feature_indices: &[u32],
) -> f64 {
    let mut stack: Vec<f64> = Vec::with_capacity(32);
    let mut ip = 0;
    let len = bytecode.len();

    while ip < len {
        let op = bytecode[ip];
        ip += 1;

        match op {
            OP_CONST => {
                let idx = read_u32(bytecode, ip) as usize;
                ip += 4;
                stack.push(constants[idx]);
            }
            OP_FEATURE => {
                let idx = read_u32(bytecode, ip) as usize;
                ip += 4;
                let feat_idx = feature_indices[idx] as usize;
                stack.push(row[feat_idx]);
            }
            OP_ADD => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a + b);
            }
            OP_SUB => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a - b);
            }
            OP_MUL => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a * b);
            }
            OP_DIV => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a / b);
            }
            OP_POW => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a.powf(b));
            }
            OP_MOD => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a % b);
            }
            OP_SAFE_DIV => {
                let fill = stack.pop().unwrap();
                let den = stack.pop().unwrap();
                let num = stack.pop().unwrap();
                stack.push(if den != 0.0 { num / den } else { fill });
            }
            OP_LOG => {
                let v = stack.pop().unwrap();
                stack.push(v.ln());
            }
            OP_LOG_SAFE => {
                let safe = stack.pop().unwrap();
                let v = stack.pop().unwrap();
                stack.push(if v > 0.0 { v.ln() } else { safe });
            }
            OP_LESS_EQUAL => {
                let r = stack.pop().unwrap();
                let l = stack.pop().unwrap();
                stack.push(f64::max(0.0, l - r));
            }
            OP_LESS => {
                let r = stack.pop().unwrap();
                let l = stack.pop().unwrap();
                stack.push(f64::max(0.0, l - r + LESS_EPS));
            }
            OP_EQUAL => {
                let r = stack.pop().unwrap();
                let l = stack.pop().unwrap();
                stack.push((l - r).abs());
            }
            OP_EQUAL_TOL => {
                let tol = stack.pop().unwrap();
                let r = stack.pop().unwrap();
                let l = stack.pop().unwrap();
                stack.push(f64::max(0.0, (l - r).abs() - tol));
            }
            OP_AND => {
                let n = read_u32(bytecode, ip) as usize;
                ip += 4;
                let start = stack.len() - n;
                let sum: f64 = stack[start..].iter().sum();
                stack.truncate(start);
                stack.push(sum);
            }
            OP_OR => {
                let n = read_u32(bytecode, ip) as usize;
                ip += 4;
                let start = stack.len() - n;
                let min = stack[start..]
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                stack.truncate(start);
                stack.push(min);
            }
            OP_COUNT => {
                let n = read_u32(bytecode, ip) as usize;
                ip += 4;
                let start = stack.len() - n;
                let count = stack[start..].iter().filter(|&&v| v > 0.0).count();
                stack.truncate(start);
                stack.push(count as f64);
            }
            OP_COUNT_INV => {
                let n = read_u32(bytecode, ip) as usize;
                ip += 4;
                let start = stack.len() - n;
                let count = stack[start..].iter().filter(|&&v| v == 0.0).count();
                stack.truncate(start);
                stack.push(count as f64);
            }
            _ => panic!("Unknown opcode: {op}"),
        }
    }

    stack[0]
}

#[pyfunction]
pub fn evaluate_constraints<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    bytecode: PyReadonlyArray1<u8>,
    constants: PyReadonlyArray1<f64>,
    feature_indices: PyReadonlyArray1<u32>,
) -> Bound<'py, PyArray1<f64>> {
    let data_arr = data.as_array();
    let bc = bytecode.as_array().to_vec();
    let consts = constants.as_array().to_vec();
    let feat_idx = feature_indices.as_array().to_vec();
    let n = data_arr.shape()[0];

    let rows: Vec<ArrayView1<f64>> = (0..n).map(|i| data_arr.row(i)).collect();

    let results: Vec<f64> = rows
        .into_par_iter()
        .map(|row| eval_sample(row, &bc, &consts, &feat_idx))
        .collect();

    PyArray1::from_vec(py, results)
}
