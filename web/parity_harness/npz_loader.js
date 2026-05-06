// Minimal NPZ + NPY reader for the parity harness.
//
// .npz is a ZIP of .npy files. We use adm-zip for the ZIP layer + parse
// the .npy header inline. Only float32, int32, and bool dtypes ever
// appear in the fixture — keeps the parser tiny.

import AdmZip from 'adm-zip';
import { readFileSync } from 'node:fs';

const NPY_MAGIC = Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]);

function parseNpyHeader(buf) {
  if (buf.slice(0, 6).compare(NPY_MAGIC) !== 0) {
    throw new Error('bad NPY magic');
  }
  const major = buf[6];
  // header_len encoding depends on version (v1 = u16 LE, v2 = u32 LE)
  let headerLen, headerStart;
  if (major === 1) {
    headerLen = buf.readUInt16LE(8);
    headerStart = 10;
  } else if (major === 2) {
    headerLen = buf.readUInt32LE(8);
    headerStart = 12;
  } else {
    throw new Error(`unsupported NPY version ${major}`);
  }
  const headerStr = buf.slice(headerStart, headerStart + headerLen).toString();

  // The header is a Python literal dict. We don't need a full parser —
  // just regex out descr, fortran_order, shape.
  const descrMatch = headerStr.match(/'descr':\s*'([^']+)'/);
  const fortranMatch = headerStr.match(/'fortran_order':\s*(True|False)/);
  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]*)\)/);
  if (!descrMatch || !fortranMatch || !shapeMatch) {
    throw new Error(`bad NPY header: ${headerStr.slice(0, 80)}`);
  }
  const descr = descrMatch[1];
  if (fortranMatch[1] === 'True') {
    throw new Error('fortran-order arrays not supported in parity harness');
  }
  const shape = shapeMatch[1].trim() === ''
    ? []
    : shapeMatch[1].split(',').map(s => s.trim()).filter(s => s !== '').map(s => parseInt(s, 10));

  return { descr, shape, dataStart: headerStart + headerLen };
}

function decodeNpy(buf) {
  const { descr, shape, dataStart } = parseNpyHeader(buf);
  const data = buf.slice(dataStart);
  let view;
  switch (descr) {
    case '<f4': view = new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4); break;
    case '<f8': view = new Float64Array(data.buffer, data.byteOffset, data.byteLength / 8); break;
    case '<i4': view = new Int32Array(data.buffer, data.byteOffset, data.byteLength / 4); break;
    case '<i8': {
      // BigInt64 is awkward; convert to plain numbers (all our int fields are small).
      const big = new BigInt64Array(data.buffer, data.byteOffset, data.byteLength / 8);
      view = new Int32Array(big.length);
      for (let i = 0; i < big.length; i++) view[i] = Number(big[i]);
      break;
    }
    case '|b1': view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength); break;
    default:
      // Fixed-width UCS-4 string ('<UN'). Decode as ASCII (fixture only stores ASCII).
      if (/^<U\d+$/.test(descr)) {
        const codepoints = new Uint32Array(data.buffer, data.byteOffset, data.byteLength / 4);
        view = String.fromCharCode(...Array.from(codepoints).filter(cp => cp !== 0));
        break;
      }
      throw new Error(`unsupported NPY dtype ${descr}`);
  }
  return { dtype: descr, shape, data: view };
}

export function loadNpz(path) {
  const zip = new AdmZip(path);
  const out = {};
  for (const entry of zip.getEntries()) {
    if (!entry.entryName.endsWith('.npy')) continue;
    const key = entry.entryName.replace(/\.npy$/, '');
    out[key] = decodeNpy(entry.getData());
  }
  return out;
}
