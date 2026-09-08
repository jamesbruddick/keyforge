// Entropy to BIP39 English mnemonic, as the bytes PBKDF2 will stretch.
//
// A mirror of `bip39::write_mnemonic`, with one difference that is not a choice: the CPU
// builds a `String` and the GPU builds a byte array, because the phrase here is never
// shown to anyone. It exists only to be the HMAC key in the next step. Phrases for
// reported hits are re-derived on the host by the CPU implementation, so nothing a user
// ever reads comes out of this function -- see src/gpu/mod.rs.
//
// The wordlist arrives as a device buffer of 2048 fixed-width records, built by
// `gpu::wordlist::encode`: one length byte then eight characters, padded. Fixed width
// keeps the lookup a multiply instead of a prefix-sum over lengths.

#define BIP39_WORD_STRIDE 9
#define BIP39_MAX_PHRASE  216   // 24 words x (8 chars + 1 separator)

// Write the mnemonic for `entropy_len` bytes of entropy. Returns the phrase length.
INLINE u32 bip39_phrase(THREAD const u8* entropy, u32 entropy_len,
                        DEVICE const u8* wordlist, THREAD u8* out) {
    // The checksum is the top `entropy_len * 8 / 32` bits of SHA-256 over the entropy.
    u32 checksum_bits = entropy_len * 8 / 32;
    u8 sha[32];
    sha256_hash(entropy, entropy_len, sha);
    u8 checksum = sha[0] >> (8 - checksum_bits);

    // Walk entropy||checksum eleven bits at a time through a rolling accumulator, exactly
    // as the CPU does. `acc` never holds more than 18 live bits, and the 0x7ff extraction
    // masks off whatever is stale above them.
    u32 acc = 0;
    u32 acc_bits = 0;
    u32 emitted = 0;
    u32 at = 0;

    for (u32 i = 0; i <= entropy_len; i++) {
        u32 width = (i == entropy_len) ? checksum_bits : 8u;
        u32 byte = (i == entropy_len) ? (u32)checksum : (u32)entropy[i];
        acc = (acc << width) | byte;
        acc_bits += width;

        while (acc_bits >= 11) {
            u32 index = (acc >> (acc_bits - 11)) & 0x7ffu;
            acc_bits -= 11;
            if (emitted > 0) out[at++] = ' ';
            DEVICE const u8* word = wordlist + index * BIP39_WORD_STRIDE;
            u32 len = (u32)word[0];
            for (u32 c = 0; c < len; c++) out[at++] = word[1 + c];
            emitted++;
        }
    }
    return at;
}
