//! Address encoding, used only by `verify` to print something you can paste into a
//! block explorer.
//!
//! Hand-rolled rather than pulling the `bitcoin` crate into the release binary: it
//! would drag in a second copy of libsecp256k1 for a cold path that renders a handful
//! of strings. The tests hold both encoders against `bitcoin`'s own.

use crate::crypto::hash::hash256;

/// Which hash160 to take from a public key.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HashForm {
    /// `hash160(compressed pubkey)`. Serves modern P2PKH (`1...`) *and* native
    /// segwit P2WPKH (`bc1q...`) -- both commit to the same 20 bytes.
    Compressed,
    /// `hash160(uncompressed pubkey)`. Legacy P2PKH from pre-compression wallets.
    Uncompressed,
    /// `hash160(0x0014 || hash160(compressed))`, the redeem script hash behind a
    /// wrapped-segwit `3...` address.
    P2shP2wpkh,
}

impl HashForm {
    pub fn as_str(self) -> &'static str {
        match self {
            HashForm::Compressed => "compressed",
            HashForm::Uncompressed => "uncompressed",
            HashForm::P2shP2wpkh => "p2sh-p2wpkh",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "compressed" => Some(HashForm::Compressed),
            "uncompressed" => Some(HashForm::Uncompressed),
            "p2sh-p2wpkh" => Some(HashForm::P2shP2wpkh),
            _ => None,
        }
    }
}

const BASE58_ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const BECH32_CHARSET: &[u8; 32] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";

/// Mainnet version bytes.
const P2PKH_VERSION: u8 = 0x00;
const P2SH_VERSION: u8 = 0x05;

/// The address a hash160 corresponds to under a given hash form.
pub fn encode(form: HashForm, hash160: &[u8; 20]) -> String {
    match form {
        // A compressed-key hash160 is spendable as either P2PKH or P2WPKH, and the
        // filter cannot tell which the chain actually used, so show both.
        HashForm::Compressed => format!(
            "{} / {}",
            base58check(P2PKH_VERSION, hash160),
            bech32_p2wpkh(hash160)
        ),
        HashForm::Uncompressed => base58check(P2PKH_VERSION, hash160),
        HashForm::P2shP2wpkh => base58check(P2SH_VERSION, hash160),
    }
}

/// Base58Check over `version || payload`, as used by `1...` and `3...` addresses.
pub fn base58check(version: u8, payload: &[u8; 20]) -> String {
    let mut data = Vec::with_capacity(25);
    data.push(version);
    data.extend_from_slice(payload);
    let checksum = hash256(&data);
    data.extend_from_slice(&checksum[..4]);

    // Leading zero bytes become leading '1's rather than being swallowed by the
    // big-integer conversion.
    let leading_zeros = data.iter().take_while(|b| **b == 0).count();

    let mut digits: Vec<u8> = Vec::with_capacity(35);
    for byte in data {
        let mut carry = byte as u32;
        for digit in digits.iter_mut() {
            carry += (*digit as u32) << 8;
            *digit = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }

    let mut out = String::with_capacity(leading_zeros + digits.len());
    out.extend(std::iter::repeat_n('1', leading_zeros));
    out.extend(
        digits
            .iter()
            .rev()
            .map(|d| BASE58_ALPHABET[*d as usize] as char),
    );
    out
}

/// A native segwit v0 `bc1q...` address (BIP173 bech32, not bech32m).
pub fn bech32_p2wpkh(hash160: &[u8; 20]) -> String {
    // Witness version 0, then the program regrouped from 8-bit to 5-bit words.
    let mut data = Vec::with_capacity(33);
    data.push(0u8);
    data.extend(convert_bits(hash160));

    let mut values = hrp_expand("bc");
    values.extend_from_slice(&data);
    values.extend_from_slice(&[0; 6]);
    let polymod = bech32_polymod(&values) ^ 1;

    let mut out = String::with_capacity(42);
    out.push_str("bc1");
    for value in &data {
        out.push(BECH32_CHARSET[*value as usize] as char);
    }
    for i in 0..6 {
        out.push(BECH32_CHARSET[((polymod >> (5 * (5 - i))) & 31) as usize] as char);
    }
    out
}

/// Regroup 8-bit bytes into 5-bit words. The 20-byte witness program is 160 bits,
/// an exact multiple of 5, so no padding case arises.
fn convert_bits(data: &[u8; 20]) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    for byte in data {
        acc = (acc << 8) | *byte as u32;
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            out.push(((acc >> bits) & 31) as u8);
        }
    }
    debug_assert_eq!(bits, 0);
    out
}

fn hrp_expand(hrp: &str) -> Vec<u8> {
    let mut out: Vec<u8> = hrp.bytes().map(|c| c >> 5).collect();
    out.push(0);
    out.extend(hrp.bytes().map(|c| c & 31));
    out
}

fn bech32_polymod(values: &[u8]) -> u32 {
    const GEN: [u32; 5] = [
        0x3b6a_57b2,
        0x2650_8e6d,
        0x1ea1_19fa,
        0x3d42_33dd,
        0x2a14_62b3,
    ];
    let mut chk: u32 = 1;
    for value in values {
        let top = chk >> 25;
        chk = ((chk & 0x1ff_ffff) << 5) ^ (*value as u32);
        for (i, g) in GEN.iter().enumerate() {
            if (top >> i) & 1 == 1 {
                chk ^= g;
            }
        }
    }
    chk
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::hash::hash160;

    /// The genesis coinbase address, the most-checked vector in Bitcoin.
    #[test]
    fn encodes_the_genesis_address() {
        let mut h = [0u8; 20];
        hex::decode_to_slice("62e907b15cbf27d5425399ebf6f0fb50ebb88f18", &mut h).unwrap();
        assert_eq!(
            base58check(P2PKH_VERSION, &h),
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
        );
    }

    /// BIP173's own P2WPKH example vector.
    #[test]
    fn encodes_the_bip173_p2wpkh_vector() {
        let pubkey =
            hex::decode("0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798")
                .unwrap();
        let h = hash160(&pubkey);
        assert_eq!(
            bech32_p2wpkh(&h),
            "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
        );
    }

    /// A hash160 with leading zero bytes must keep its leading '1's -- the classic
    /// base58 bug.
    #[test]
    fn preserves_leading_zeros_in_base58() {
        let h = [0u8; 20];
        let encoded = base58check(P2PKH_VERSION, &h);
        assert!(encoded.starts_with("1111111111111111111"), "{encoded}");
        assert_eq!(encoded, "1111111111111111111114oLvT2");
    }

    /// Both encoders, held against the reference implementation over real public
    /// keys -- so the comparison runs through exactly the address types the scanner
    /// reports, not synthetic hashes.
    #[test]
    fn agrees_with_the_reference_encoders() {
        use bitcoin::hashes::Hash;
        use bitcoin::network::Network;
        use bitcoin::secp256k1::{Secp256k1, SecretKey};
        use bitcoin::{Address, CompressedPublicKey, PublicKey, ScriptHash};

        let secp = Secp256k1::new();
        for i in 1u32..500 {
            let mut key_bytes = [0u8; 32];
            key_bytes[28..].copy_from_slice(&i.wrapping_mul(2_654_435_761).to_be_bytes());
            key_bytes[0] = 1; // keep the scalar comfortably in range
            let sk = SecretKey::from_slice(&key_bytes).unwrap();
            let pk = PublicKey::new(sk.public_key(&secp));
            let compressed = CompressedPublicKey::try_from(pk).unwrap();

            let h_c = hash160(&pk.inner.serialize());
            assert_eq!(
                base58check(P2PKH_VERSION, &h_c),
                Address::p2pkh(pk, Network::Bitcoin).to_string()
            );
            assert_eq!(
                bech32_p2wpkh(&h_c),
                Address::p2wpkh(&compressed, Network::Bitcoin).to_string()
            );

            let h_u = hash160(&pk.inner.serialize_uncompressed());
            let uncompressed = PublicKey {
                compressed: false,
                inner: pk.inner,
            };
            assert_eq!(
                base58check(P2PKH_VERSION, &h_u),
                Address::p2pkh(uncompressed, Network::Bitcoin).to_string()
            );

            let mut script = [0u8; 22];
            script[0] = 0x00;
            script[1] = 0x14;
            script[2..].copy_from_slice(&h_c);
            let h_p = hash160(&script);
            assert_eq!(
                base58check(P2SH_VERSION, &h_p),
                Address::p2sh_from_hash(ScriptHash::from_byte_array(h_p), Network::Bitcoin)
                    .to_string()
            );
            // And that it is the same thing the reference calls a wrapped-segwit
            // address, which is the claim the scanner is actually making.
            assert_eq!(
                base58check(P2SH_VERSION, &h_p),
                Address::p2shwpkh(&compressed, Network::Bitcoin).to_string()
            );
        }
    }
}
