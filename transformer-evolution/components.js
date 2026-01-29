// Detailed explanations and code for each component type
const COMPONENT_DETAILS = {
    // ==================== EMBEDDING COMPONENTS ====================
    input_embedding: {
        title: 'Token Embedding',
        overview: 'Token embeddings convert discrete tokens (integers) into dense continuous vectors. Each token in the vocabulary has a learned embedding vector, and the embedding layer acts as a lookup table that maps token IDs to their corresponding vectors.',
        how: 'The input sequence of token IDs is passed through an embedding matrix of shape (vocab_size, d_model). For each token ID, the corresponding row of the embedding matrix is retrieved. This creates a sequence of dense vectors that capture semantic meaning.',
        math: `Embedding lookup:
E ∈ ℝ^(V × d_model)
x = [x₁, x₂, ..., xₙ]  (token IDs)
embeddings = E[x]  (shape: n × d_model)

Where V is vocabulary size and d_model is hidden dimension.`,
        code: `import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        # Output shape: (batch_size, seq_len, d_model)

        # Scale by sqrt(d_model) as in original paper
        return self.embedding(x) * (self.d_model ** 0.5)

# Usage
vocab_size = 50257  # GPT-2 vocabulary
d_model = 768
embedding = TokenEmbedding(vocab_size, d_model)

tokens = torch.tensor([[1, 2, 3, 4]])  # batch_size=1, seq_len=4
embedded = embedding(tokens)  # (1, 4, 768)`,
        tips: [
            'The original Transformer scales embeddings by √d_model to match the scale of positional encodings',
            'Many modern models (GPT-2+) tie the embedding weights with the final output projection (weight tying)',
            'Embedding matrices can be very large - use mixed precision or embedding compression for efficiency',
            'Consider using factorized embeddings for very large vocabularies to reduce parameters'
        ]
    },

    output_embedding: {
        title: 'Output Embedding (Decoder Input)',
        overview: 'In the encoder-decoder architecture, the decoder has its own embedding layer for the target sequence. During training, this receives the shifted target sequence (teacher forcing). During inference, it receives the previously generated tokens.',
        how: 'Works identically to input embedding but for the decoder side. The target sequence is right-shifted by one position and padded with a start token, so position i predicts position i+1.',
        math: `Target sequence: [y₁, y₂, ..., yₘ]
Decoder input: [<sos>, y₁, y₂, ..., yₘ₋₁]
Decoder output predicts: [y₁, y₂, ..., yₘ]`,
        code: `class DecoderEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_idx
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * (self.d_model ** 0.5)

def shift_right(target: torch.Tensor, sos_id: int) -> torch.Tensor:
    """Shift target sequence right for teacher forcing."""
    shifted = torch.zeros_like(target)
    shifted[:, 0] = sos_id
    shifted[:, 1:] = target[:, :-1]
    return shifted`,
        tips: [
            'In encoder-decoder models, encoder and decoder often share embeddings if they use the same vocabulary',
            'Teacher forcing during training means using ground truth as decoder input',
            'During inference, use autoregressive generation with previously generated tokens'
        ]
    },

    segment_embedding: {
        title: 'Segment Embedding',
        overview: 'Segment embeddings (introduced in BERT) distinguish between different segments or sentences in the input. This is crucial for tasks involving sentence pairs like question answering or natural language inference.',
        how: 'Each token is assigned a segment ID (typically 0 or 1) indicating which segment it belongs to. These IDs are embedded and added to the token embeddings, allowing the model to differentiate between segments.',
        math: `Segment embedding: S ∈ ℝ^(2 × d_model)
segment_ids = [0, 0, 0, 1, 1, 1, 1]
segment_emb = S[segment_ids]

Final embedding = token_emb + position_emb + segment_emb`,
        code: `class BERTEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 512,
        n_segments: int = 2
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor
    ) -> torch.Tensor:
        seq_len = token_ids.size(1)
        positions = torch.arange(seq_len, device=token_ids.device)

        x = (
            self.token_embedding(token_ids) +
            self.position_embedding(positions) +
            self.segment_embedding(segment_ids)
        )
        return self.dropout(self.norm(x))

# Usage for sentence pair
tokens = torch.tensor([[101, 2054, 2003, 102, 2009, 2003, 3835, 102]])
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
# [CLS] What is [SEP] It is great [SEP]`,
        tips: [
            'BERT uses 2 segment types, but this can be extended for multiple documents',
            'The [SEP] token usually belongs to the segment before it',
            'Some models (RoBERTa) remove segment embeddings without performance loss',
            'Segment embeddings are learned during pre-training'
        ]
    },

    // ==================== POSITIONAL COMPONENTS ====================
    pos_encoding_enc: {
        title: 'Positional Encoding (Sinusoidal)',
        overview: 'Since transformers process all positions in parallel (no recurrence), they need explicit position information. The original Transformer uses fixed sinusoidal positional encodings that can theoretically generalize to longer sequences.',
        how: 'For each position and each dimension, a sinusoidal function (sin for even dimensions, cos for odd) with a specific frequency is computed. Lower dimensions have higher frequencies, encoding fine position differences, while higher dimensions encode coarser patterns.',
        math: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where pos is position, i is dimension index.
The wavelengths form a geometric progression from 2π to 10000·2π.`,
        code: `import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Compute division term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)`,
        tips: [
            'Sinusoidal encodings can theoretically extrapolate to longer sequences than seen in training',
            'The encoding allows the model to learn relative positions via linear projections',
            'PE(pos+k) can be represented as a linear function of PE(pos)',
            'Modern models often prefer learned positional embeddings or RoPE instead'
        ]
    },

    pos_encoding_dec: {
        title: 'Positional Encoding (Decoder)',
        overview: 'The decoder uses the same sinusoidal positional encoding as the encoder. Positions are encoded independently of the encoder positions - the decoder learns to use cross-attention to align with encoder positions.',
        how: 'Identical to encoder positional encoding. Added to decoder token embeddings before the first decoder layer.',
        math: `Same as encoder:
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`,
        code: `# Same implementation as encoder - typically shared
class Transformer(nn.Module):
    def __init__(self, d_model: int, ...):
        super().__init__()
        # Shared positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)

        self.encoder_embedding = nn.Embedding(src_vocab, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab, d_model)

    def encode(self, src):
        x = self.encoder_embedding(src)
        x = self.pos_encoding(x)
        return self.encoder(x)

    def decode(self, tgt, memory):
        x = self.decoder_embedding(tgt)
        x = self.pos_encoding(x)  # Same encoding
        return self.decoder(x, memory)`,
        tips: [
            'Encoder and decoder typically share the same positional encoding module',
            'Decoder positions are independent of encoder - cross-attention handles alignment',
            'The decoder can have different sequence lengths than the encoder'
        ]
    },

    pos_encoding: {
        title: 'Learned Positional Embedding',
        overview: 'GPT and most modern models use learned positional embeddings instead of fixed sinusoidal encodings. Each position has its own learnable embedding vector that is trained along with the rest of the model.',
        how: 'A position embedding matrix is created with shape (max_seq_len, d_model). Position indices are used to look up the corresponding embeddings, which are then added to the token embeddings.',
        math: `Position embedding: P ∈ ℝ^(max_len × d_model)
positions = [0, 1, 2, ..., n-1]
pos_emb = P[positions]

Final: x = token_emb + pos_emb`,
        code: `class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)

# Combined token + position embedding (GPT-style)
class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)

        tok = self.token_emb(x)
        pos = self.pos_emb(positions)
        return self.dropout(tok + pos)`,
        tips: [
            'Learned embeddings often perform slightly better than sinusoidal on tasks within training length',
            'Cannot extrapolate to longer sequences than max_len',
            'Initialize with small values (normal distribution with small std)',
            'Modern models prefer RoPE for better length generalization'
        ]
    },

    // ==================== ATTENTION COMPONENTS ====================
    encoder_mha: {
        title: 'Multi-Head Self-Attention (Encoder)',
        overview: 'Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. Each head learns different attention patterns, and their outputs are concatenated and projected.',
        how: 'The input is projected into Q, K, V for each head. Attention scores are computed as scaled dot-product of Q and K, normalized with softmax, then used to weight V. All heads are concatenated and linearly projected.',
        math: `MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)Wᴼ

where headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)

Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V

Parameters: Wᵢᵠ, Wᵢᴷ, Wᵢⱽ ∈ ℝ^(d_model × dₖ)
           Wᴼ ∈ ℝ^(h·dₖ × d_model)`,
        code: `class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # (B, L, D)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head: (B, L, H, D_k) -> (B, H, L, D_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # (B, H, L, D_k)

        # Concatenate heads: (B, H, L, D_k) -> (B, L, D)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(out)`,
        tips: [
            'Use 8-16 heads for models with d_model=512-1024',
            'Each head has d_k = d_model / n_heads dimensions',
            'The scale factor √d_k prevents softmax saturation with large d_k',
            'Consider FlashAttention for memory-efficient implementation'
        ]
    },

    masked_mha: {
        title: 'Masked Multi-Head Self-Attention',
        overview: 'Masked self-attention prevents positions from attending to subsequent positions. This is crucial for autoregressive language modeling where each position should only see previous tokens (causal masking).',
        how: 'Same as regular MHA, but with a causal mask applied before softmax. The mask sets attention scores to -inf for positions j > i, ensuring token i cannot attend to any token after it.',
        math: `Causal Mask: M[i,j] = 0 if j ≤ i else -∞

Attention(Q, K, V) = softmax((QKᵀ + M) / √dₖ)V

The mask ensures:
- Position 0 attends only to position 0
- Position 1 attends to positions 0, 1
- Position n attends to positions 0, 1, ..., n`,
        code: `class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

        # Causal mask (lower triangular)
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('mask', mask.view(1, 1, max_len, max_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Combined QKV projection (more efficient)
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Attention with causal mask
        scores = (Q @ K.transpose(-2, -1)) * (self.d_k ** -0.5)
        scores = scores.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)`,
        tips: [
            'The causal mask is typically pre-computed and registered as a buffer',
            'For inference, use KV-caching to avoid recomputing attention for previous tokens',
            'Combined QKV projection is more efficient than separate Q, K, V projections',
            'Use torch.tril() to create the lower-triangular causal mask'
        ]
    },

    cross_attention: {
        title: 'Cross-Attention (Encoder-Decoder)',
        overview: 'Cross-attention allows the decoder to attend to encoder outputs. Queries come from the decoder, while keys and values come from the encoder, enabling the decoder to extract relevant information from the input sequence.',
        how: 'The decoder hidden state is projected to queries, while encoder outputs are projected to keys and values. This allows each decoder position to attend to all encoder positions.',
        math: `CrossAttention:
Q = decoder_hidden @ Wᵠ
K = encoder_output @ Wᴷ
V = encoder_output @ Wⱽ

Output = softmax(QKᵀ / √dₖ)V

No mask needed - decoder can attend to all encoder positions.`,
        code: `class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_kv = nn.Linear(d_model, 2 * d_model)  # Combined K, V
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: torch.Tensor = None  # Padding mask for encoder
    ) -> torch.Tensor:
        B, L_dec, D = decoder_hidden.shape
        L_enc = encoder_output.size(1)

        # Queries from decoder
        Q = self.W_q(decoder_hidden)
        Q = Q.view(B, L_dec, self.n_heads, self.d_k).transpose(1, 2)

        # Keys and Values from encoder
        KV = self.W_kv(encoder_output).reshape(B, L_enc, 2, self.n_heads, self.d_k)
        KV = KV.permute(2, 0, 3, 1, 4)
        K, V = KV[0], KV[1]

        # Cross-attention
        scores = (Q @ K.transpose(-2, -1)) * (self.d_k ** -0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, L_dec, D)
        return self.W_o(out)`,
        tips: [
            'Cross-attention is the key mechanism for encoder-decoder information flow',
            'Apply padding mask to prevent attending to encoder padding tokens',
            'The encoder output is computed once and reused for all decoder layers',
            'During inference, encoder output can be cached since it doesn\'t change'
        ]
    },

    mha: {
        title: 'Multi-Head Attention (Bidirectional)',
        overview: 'BERT uses bidirectional self-attention where each token can attend to all tokens in the sequence, both before and after. This enables richer contextual understanding compared to causal models.',
        how: 'Standard multi-head attention without causal masking. Only padding masks are applied to prevent attention to padding tokens.',
        math: `Bidirectional Attention (no causal mask):
Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V

Token i can attend to all tokens j ∈ [1, n]
Only masked positions are padding tokens.`,
        code: `class BidirectionalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            padding_mask: (batch, seq_len) - True for valid, False for padding
        """
        if padding_mask is not None:
            # Expand mask for attention: (B, 1, 1, L)
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = None

        return self.mha(x, mask=attn_mask)

def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Create mask where True = valid token, False = padding."""
    return seq != pad_idx`,
        tips: [
            'Bidirectional attention is ideal for understanding tasks (classification, NER)',
            'Not suitable for generation - use causal attention for that',
            'BERT masks 15% of tokens during pre-training to create a prediction task',
            'The [CLS] token can attend to all other tokens, making it useful for classification'
        ]
    },

    rope_mha: {
        title: 'Multi-Head Attention with RoPE',
        overview: 'Rotary Position Embedding (RoPE) encodes position information by rotating the query and key vectors. This allows for better length generalization and naturally decaying attention with distance.',
        how: 'Instead of adding position embeddings, RoPE rotates Q and K vectors by angles proportional to their positions. The dot product QKᵀ then naturally incorporates relative position information.',
        math: `RoPE applies rotation matrix R to Q and K:
Rₘ = [cos(mθ₁)  -sin(mθ₁)  0  ...
      sin(mθ₁)   cos(mθ₁)  0  ...
      0          0         cos(mθ₂) ...]

θᵢ = 10000^(-2i/d)

Attention = softmax((RₘQ)(RₙK)ᵀ / √d)V
         = softmax(Q·Rₘ₋ₙ·Kᵀ / √d)V

The rotation encodes relative position m-n.`,
        code: `class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192, base: int = 10000):
        super().__init__()
        self.d_model = d_model

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple:
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings to Q and K."""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionalEmbedding(self.d_k)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(x, L)
        q, k = apply_rotary_emb(q, k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)`,
        tips: [
            'RoPE enables better length generalization than learned positions',
            'The rotation naturally encodes relative positions in the attention scores',
            'RoPE can be combined with position interpolation for even longer contexts',
            'Apply RoPE to Q and K only, not V (values don\'t need position encoding)'
        ]
    },

    gqa: {
        title: 'Grouped Query Attention (GQA)',
        overview: 'GQA groups query heads to share key-value heads, reducing KV-cache memory during inference while maintaining model quality. It\'s a middle ground between Multi-Head Attention (MHA) and Multi-Query Attention (MQA).',
        how: 'Instead of n_heads separate K and V projections, GQA uses n_kv_heads (typically n_heads/8). Each group of query heads shares one set of K and V, significantly reducing memory during autoregressive inference.',
        math: `MHA: n_heads Q, K, V projections each
MQA: n_heads Q projections, 1 K, 1 V projection
GQA: n_heads Q projections, n_kv_heads K, V projections

Group size = n_heads / n_kv_heads
KV-cache reduction = n_heads / n_kv_heads (e.g., 8x)`,
        code: `class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,  # Number of KV heads (< n_heads)
        max_len: int = 8192
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # How many Q heads per KV head
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_k, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, L, _ = x.shape

        # Project Q, K, V
        q = self.W_q(x).view(B, L, self.n_heads, self.d_k)
        k = self.W_k(x).view(B, L, self.n_kv_heads, self.d_k)
        v = self.W_v(x).view(B, L, self.n_kv_heads, self.d_k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(x, L)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Repeat K, V to match Q heads
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Standard attention
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.W_o(out)

# LLaMA 2 70B uses n_heads=64, n_kv_heads=8 (8x reduction)`,
        tips: [
            'GQA reduces KV-cache memory by n_heads/n_kv_heads factor',
            'Minimal quality loss compared to full MHA',
            'Enables longer context lengths with same memory budget',
            'repeat_interleave is used during attention computation to expand K, V'
        ]
    },

    // ==================== FFN COMPONENTS ====================
    encoder_ffn: {
        title: 'Feed-Forward Network (FFN)',
        overview: 'The FFN applies two linear transformations with a non-linearity in between. It processes each position independently, allowing the model to transform representations after attention aggregation.',
        how: 'The FFN expands the hidden dimension (typically 4x), applies an activation function (ReLU in original), then projects back down. This creates a bottleneck that forces the model to learn compressed representations.',
        math: `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

W₁ ∈ ℝ^(d_model × d_ff)
W₂ ∈ ℝ^(d_ff × d_model)

Typically d_ff = 4 × d_model`,
        code: `class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Original Transformer FFN
class TransformerFFN(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))`,
        tips: [
            'The expansion ratio of 4x is empirically found to work well',
            'FFN processes each position independently (no cross-position interaction)',
            'Modern models use GELU or SwiGLU instead of ReLU',
            'Can be viewed as a two-layer MLP applied to each token'
        ]
    },

    decoder_ffn: {
        title: 'Feed-Forward Network (Decoder)',
        overview: 'Identical to encoder FFN. Applied after cross-attention in the decoder to transform the combined encoder-decoder representations.',
        how: 'Same architecture as encoder FFN. In the decoder, it processes representations that have been enriched by both self-attention and cross-attention.',
        math: `Same as encoder FFN:
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂`,
        code: `# Decoder FFN is identical to encoder FFN
# The difference is in the surrounding architecture

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, n_heads)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        # Self-attention with residual
        x = x + self.self_attn(self.norm1(x), tgt_mask)

        # Cross-attention with residual
        x = x + self.cross_attn(self.norm2(x), encoder_out, src_mask)

        # FFN with residual
        x = x + self.ffn(self.norm3(x))

        return x`,
        tips: [
            'Decoder FFN processes representations enriched by both attention types',
            'Same hyperparameters as encoder FFN (d_ff = 4 * d_model)',
            'The FFN is where most parameters live in the Transformer'
        ]
    },

    ffn: {
        title: 'Feed-Forward Network (GPT)',
        overview: 'GPT models use GELU activation instead of ReLU. GELU (Gaussian Error Linear Unit) provides smoother gradients and has become the standard for language models.',
        how: 'Same structure as original FFN but with GELU activation, which applies a smooth gating based on the input value\'s position in the Gaussian CDF.',
        math: `FFN(x) = GELU(xW₁ + b₁)W₂ + b₂

GELU(x) = x · Φ(x)
        ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

Where Φ(x) is the Gaussian CDF.`,
        code: `class GPTMLP(nn.Module):
    """GPT-style MLP with GELU activation."""
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# Alternative: new GELU approximation (faster)
class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))`,
        tips: [
            'GELU provides smoother gradients than ReLU',
            'PyTorch\'s nn.GELU() uses the exact formula, not approximation',
            'The approximation is sometimes used for speed',
            'GELU is now the standard for most language models'
        ]
    },

    swiglu_ffn: {
        title: 'SwiGLU Feed-Forward Network',
        overview: 'SwiGLU combines Swish activation with Gated Linear Units. It uses three linear projections instead of two, with one projection serving as a gate. This has become standard in LLaMA and most modern LLMs.',
        how: 'The input is projected to two separate vectors (gate and up). The gate projection is passed through Swish (SiLU), then multiplied element-wise with the up projection, and finally projected down.',
        math: `SwiGLU(x) = (Swish(xW_gate) ⊙ xW_up)W_down

Swish(x) = x · σ(x) = x · sigmoid(x)

W_gate, W_up ∈ ℝ^(d_model × d_ff)
W_down ∈ ℝ^(d_ff × d_model)

Note: d_ff is often 2/3 × 4 × d_model to maintain param count`,
        code: `class SwiGLU(nn.Module):
    """SwiGLU activation with gated linear unit."""
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        bias: bool = False  # LLaMA removes bias
    ):
        super().__init__()
        # d_ff is typically 2/3 of what it would be with standard FFN
        # to maintain similar parameter count (3 matrices vs 2)
        d_ff = d_ff or int(2 * (4 * d_model) / 3)
        # Round to multiple of 256 for efficiency
        d_ff = 256 * ((d_ff + 255) // 256)

        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(gate) * up, then project down
        gate = F.silu(self.w_gate(x))  # Swish = SiLU
        up = self.w_up(x)
        return self.w_down(gate * up)

# Full LLaMA-style FFN
class LLaMAMLP(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))`,
        tips: [
            'SwiGLU consistently outperforms GELU in language modeling',
            'Uses 3 weight matrices instead of 2, so d_ff is reduced to maintain param count',
            'LLaMA removes all bias terms for efficiency',
            'F.silu() in PyTorch implements Swish activation'
        ]
    },

    experts: {
        title: 'Mixture of Experts (MoE) FFN',
        overview: 'MoE replaces the single FFN with multiple expert FFN networks. A router selects which experts to use for each token, providing more parameters while keeping compute constant (sparse activation).',
        how: 'For each token, the router computes scores for all experts. The top-k experts (usually k=2) are selected, their outputs are computed, and the results are weighted by the router scores.',
        math: `Router scores: g = softmax(xW_router)
Top-k selection: indices, weights = topk(g, k)

Output = Σᵢ weights[i] · Expert_i(x)

With 8 experts and top-2: 8x more params, ~2x compute`,
        code: `class MoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # Router
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Expert FFNs (using SwiGLU)
        self.experts = nn.ModuleList([
            SwiGLU(d_model, d_ff) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_flat = x.view(-1, D)  # (B*L, D)

        # Router scores
        router_logits = self.router(x_flat)  # (B*L, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = expert(expert_input)

                # Weight by router probability
                idx = (topk_indices[mask] == i).float()
                weights = (topk_weights[mask] * idx).sum(dim=-1, keepdim=True)
                output[mask] += weights * expert_output

        return output.view(B, L, D)`,
        tips: [
            'Load balancing loss ensures experts are used evenly',
            'Top-2 routing is most common (good trade-off)',
            'Experts can be distributed across GPUs for efficient parallelism',
            'Capacity factor limits how many tokens each expert processes'
        ]
    },

    router: {
        title: 'Expert Router',
        overview: 'The router is a learned linear layer that decides which experts should process each token. It outputs probability scores for all experts, from which top-k are selected.',
        how: 'A simple linear projection from token representations to expert scores, followed by softmax and top-k selection. Auxiliary losses ensure load balancing.',
        math: `Router scores: g = softmax(xW + noise)
Selection: indices = topk(g, k)

Load balancing loss:
L_aux = n_experts · Σᵢ fᵢ · Pᵢ
where fᵢ = fraction of tokens routed to expert i
      Pᵢ = average router probability for expert i`,
        code: `class Router(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,  # For load balancing
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.noise_std = noise_std

        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, d_model)
        logits = self.gate(x)  # (batch, seq_len, n_experts)

        # Add noise during training for load balancing
        if self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)

        # Top-k selection
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        # Normalize selected probabilities
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Compute load balancing loss
        aux_loss = self.load_balancing_loss(probs, topk_indices)

        return topk_probs, topk_indices, aux_loss

    def load_balancing_loss(self, probs, indices):
        # Fraction of tokens per expert
        n_tokens = probs.shape[0] * probs.shape[1]
        expert_mask = F.one_hot(indices, self.n_experts).sum(dim=2)
        tokens_per_expert = expert_mask.sum(dim=[0, 1]) / n_tokens

        # Average probability per expert
        avg_prob = probs.mean(dim=[0, 1])

        # Auxiliary loss
        return self.n_experts * (tokens_per_expert * avg_prob).sum()`,
        tips: [
            'Noise injection during training helps with load balancing',
            'The auxiliary loss coefficient is typically 0.01-0.1',
            'Expert capacity limits prevent overloading popular experts',
            'Some implementations use different routing strategies (hash, learned)'
        ]
    },

    // ==================== NORMALIZATION COMPONENTS ====================
    encoder_add_norm1: {
        title: 'Add & Layer Normalization (Post-LN)',
        overview: 'The original Transformer uses Post-LayerNorm: the residual connection adds the sub-layer output to the input, then LayerNorm normalizes the sum. This is applied after both attention and FFN.',
        how: 'LayerNorm normalizes across the feature dimension for each token independently. Combined with residual connections, it stabilizes training by maintaining reasonable activation scales.',
        math: `Post-LN Residual:
output = LayerNorm(x + SubLayer(x))

LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β

Where μ, σ² are computed over the feature dimension:
μ = (1/d) Σᵢ xᵢ
σ² = (1/d) Σᵢ (xᵢ - μ)²`,
        code: `class PostLNResidual(nn.Module):
    """Post-LayerNorm residual block (original Transformer)."""
    def __init__(self, d_model: int, sublayer: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Post-LN: normalize after adding residual
        return self.norm(x + self.dropout(self.sublayer(x, *args, **kwargs)))

# Usage in encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = PostLNResidual(
            d_model,
            MultiHeadAttention(d_model, n_heads)
        )
        self.ffn = PostLNResidual(
            d_model,
            FeedForward(d_model, d_ff)
        )

    def forward(self, x, mask=None):
        x = self.self_attn(x, mask=mask)
        x = self.ffn(x)
        return x`,
        tips: [
            'Post-LN was the original design but can be harder to train at depth',
            'The residual connection enables gradient flow through deep networks',
            'LayerNorm has learnable scale (γ) and shift (β) parameters',
            'ε (epsilon) prevents division by zero (typically 1e-5 or 1e-6)'
        ]
    },

    encoder_add_norm2: {
        title: 'Add & Layer Normalization (After FFN)',
        overview: 'Second Add & Norm in encoder, applied after the FFN. Same operation as after attention: residual addition followed by layer normalization.',
        how: 'Identical to the first Add & Norm. Ensures the output of each sub-layer has normalized features and maintains the residual pathway.',
        math: `output = LayerNorm(x + FFN(x))`,
        code: `# Same as encoder_add_norm1, applied after FFN
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention + Add & Norm
        attn_out = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x`,
        tips: [
            'Both norms use the same hyperparameters but separate learnable parameters',
            'The second norm\'s output is what\'s passed to the next encoder layer',
            'Each encoder layer has its own pair of LayerNorm modules'
        ]
    },

    add_norm1: {
        title: 'Add & LayerNorm (GPT Post-LN)',
        overview: 'GPT-1 used Post-LN like the original Transformer. The sum of input and attention output is normalized before being passed to the FFN.',
        how: 'Standard residual + LayerNorm. GPT-1 followed the original Transformer design closely.',
        math: `x = LayerNorm(x + Attention(x))`,
        code: `class GPT1Block(nn.Module):
    """GPT-1 style block with Post-LayerNorm."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ffn = GPTMLP(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Post-LN: norm after residual
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x`,
        tips: [
            'GPT-1 used Post-LN (same as original Transformer)',
            'GPT-2 switched to Pre-LN for better training stability',
            'Post-LN can lead to gradient issues in very deep models'
        ]
    },

    add_norm2: {
        title: 'Add & LayerNorm (After FFN)',
        overview: 'Second normalization layer in GPT-1/original style blocks. Normalizes the sum of FFN output and its input.',
        how: 'Same Post-LN pattern applied after FFN.',
        math: `output = LayerNorm(x + FFN(x))`,
        code: `# See add_norm1 for full block implementation
# This is the second norm in the sequence`,
        tips: [
            'Final output of each block before passing to next layer',
            'Post-LN at depth can suffer from gradient issues',
            'This is why GPT-2+ moved to Pre-LN'
        ]
    },

    decoder_add_norm1: {
        title: 'Add & Norm (After Masked Attention)',
        overview: 'First normalization in decoder, applied after masked self-attention. Maintains normalized activations while preserving the residual connection.',
        how: 'Same Post-LN as encoder. In decoder, this normalizes after causal self-attention.',
        math: `x = LayerNorm(x + MaskedAttention(x))`,
        code: `class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, n_heads)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # Masked self-attention + Norm
        x = self.norm1(x + self.dropout(self.self_attn(x, tgt_mask)))

        # Cross-attention + Norm
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, src_mask)))

        # FFN + Norm
        x = self.norm3(x + self.dropout(self.ffn(x)))

        return x`,
        tips: [
            'Decoder has 3 norms vs encoder\'s 2 (extra for cross-attention)',
            'Self-attention sees only previous positions due to causal mask',
            'The masked attention output captures decoder context'
        ]
    },

    decoder_add_norm2: {
        title: 'Add & Norm (After Cross-Attention)',
        overview: 'Second normalization in decoder, applied after cross-attention with encoder outputs. This is unique to encoder-decoder models.',
        how: 'Normalizes the combination of decoder self-attention output and cross-attention output.',
        math: `x = LayerNorm(x + CrossAttention(x, encoder_output))`,
        code: `# See decoder_add_norm1 for full implementation
# This is the second norm, after cross-attention`,
        tips: [
            'This norm processes representations enriched by encoder information',
            'The cross-attention allows decoder to focus on relevant encoder positions',
            'Decoder-only models (GPT) don\'t have this layer'
        ]
    },

    decoder_add_norm3: {
        title: 'Add & Norm (After Decoder FFN)',
        overview: 'Third and final normalization in decoder, applied after the FFN. Output goes to next decoder layer or to final projection.',
        how: 'Final Post-LN before the decoder layer output.',
        math: `output = LayerNorm(x + FFN(x))`,
        code: `# See decoder_add_norm1 for full implementation
# This is the final norm in each decoder layer`,
        tips: [
            'Decoder layers have 3 sub-layers vs encoder\'s 2',
            'This output either goes to next decoder layer or to output projection',
            'In the final decoder layer, this feeds into the linear + softmax'
        ]
    },

    pre_norm1: {
        title: 'Pre-LayerNorm (Before Attention)',
        overview: 'GPT-2 introduced Pre-LN where LayerNorm is applied before each sub-layer instead of after. This improves training stability and removes the need for careful learning rate warmup.',
        how: 'The input is normalized first, then passed through attention. The original input is added to the attention output (residual). This keeps gradients well-behaved in deep networks.',
        math: `Pre-LN Residual:
output = x + Attention(LayerNorm(x))

vs Post-LN:
output = LayerNorm(x + Attention(x))`,
        code: `class PreLNBlock(nn.Module):
    """GPT-2 style block with Pre-LayerNorm."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ffn = GPTMLP(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: norm before sublayer, residual after
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# Comparison
class PostLN(nn.Module):
    def forward(self, x):
        return self.norm(x + self.sublayer(x))  # norm AFTER

class PreLN(nn.Module):
    def forward(self, x):
        return x + self.sublayer(self.norm(x))  # norm BEFORE`,
        tips: [
            'Pre-LN enables training without learning rate warmup',
            'Gradients flow better through the residual path',
            'Pre-LN is now standard in most language models',
            'Requires a final LayerNorm after all blocks (see final_norm)'
        ]
    },

    pre_norm2: {
        title: 'Pre-LayerNorm (Before FFN)',
        overview: 'Second Pre-LN in GPT-2 style blocks, applied before the FFN. Same pattern: normalize, transform, add residual.',
        how: 'Input is normalized, passed through FFN, then added to the original input.',
        math: `output = x + FFN(LayerNorm(x))`,
        code: `# Part of PreLNBlock above
def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.ffn(self.ln2(x))  # Pre-LN before FFN
    return x`,
        tips: [
            'Same Pre-LN pattern as before attention',
            'Each block has two Pre-LN layers (before attn, before FFN)',
            'The residual stream flows through unchanged except for additions'
        ]
    },

    rms_norm1: {
        title: 'RMSNorm (Pre-Norm)',
        overview: 'RMSNorm (Root Mean Square Normalization) simplifies LayerNorm by removing the mean-centering step. It only uses the RMS for scaling, making it faster with similar performance.',
        how: 'Instead of computing mean and variance, RMSNorm only computes the RMS (root mean square) and scales by it. This removes one reduction operation and the learnable bias.',
        math: `RMSNorm(x) = γ · x / RMS(x)

RMS(x) = √((1/d) Σᵢ xᵢ²)

vs LayerNorm:
LayerNorm(x) = γ · (x - μ) / σ + β`,
        code: `class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA)."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * (x / rms)

# Optimized version
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)`,
        tips: [
            'RMSNorm is ~7-10% faster than LayerNorm',
            'No learnable bias parameter (only scale)',
            'Performance is similar to LayerNorm in practice',
            'Standard in LLaMA and most modern LLMs'
        ]
    },

    rms_norm2: {
        title: 'RMSNorm (Before FFN)',
        overview: 'Second RMSNorm in LLaMA-style blocks, applied before the SwiGLU FFN.',
        how: 'Same as first RMSNorm, separate learnable weights.',
        math: `output = x + SwiGLU(RMSNorm(x))`,
        code: `class LLaMABlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = RoPEAttention(d_model, n_heads)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x`,
        tips: [
            'Each block has separate RMSNorm instances',
            'LLaMA also removes all bias terms throughout the model',
            'The combination of RMSNorm + SwiGLU + no bias is efficient'
        ]
    },

    final_norm: {
        title: 'Final Layer Normalization',
        overview: 'Pre-LN architectures require a final normalization after all transformer blocks. Without this, the output would be unnormalized since Pre-LN normalizes before each sublayer, not after.',
        how: 'A single LayerNorm (or RMSNorm) applied to the output of the final transformer block, before the language model head.',
        math: `final_output = LayerNorm(transformer_output)
logits = Linear(final_output)`,
        code: `class GPT2Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff):
        super().__init__()
        self.embedding = GPTEmbedding(vocab_size, d_model)

        # Transformer blocks with Pre-LN
        self.blocks = nn.ModuleList([
            PreLNBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        # Final LayerNorm (required for Pre-LN)
        self.ln_f = nn.LayerNorm(d_model)

        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.ln_f(x)

        return self.lm_head(x)`,
        tips: [
            'Essential for Pre-LN architectures',
            'Post-LN doesn\'t need this (output is already normalized)',
            'LLaMA uses RMSNorm for the final norm as well',
            'Sometimes weight-tied with embedding normalization'
        ]
    },

    // ==================== OUTPUT COMPONENTS ====================
    linear: {
        title: 'Output Linear Projection (LM Head)',
        overview: 'The language model head projects from hidden dimension back to vocabulary size. Each position\'s representation is mapped to a score for each vocabulary token.',
        how: 'A simple linear projection without activation. The output logits represent unnormalized log probabilities for each token in the vocabulary.',
        math: `logits = xW + b

W ∈ ℝ^(d_model × vocab_size)
logits ∈ ℝ^(seq_len × vocab_size)

Often tied with input embedding (W = Embedding.weight.T)`,
        code: `class LanguageModelHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, tie_weights: nn.Embedding = None):
        super().__init__()

        if tie_weights is not None:
            # Weight tying with embedding
            self.weight = tie_weights.weight
            self.bias = None
        else:
            self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'weight'):
            # Tied weights
            return F.linear(x, self.weight)
        return self.proj(x)

# Full model with weight tying
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, ...):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.transformer = ...

        # Tie weights: lm_head uses embedding weights
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying

    def forward(self, x):
        x = self.tok_emb(x)
        x = self.transformer(x)
        return self.lm_head(x)`,
        tips: [
            'Weight tying reduces parameters by vocab_size × d_model',
            'Most modern LLMs use weight tying',
            'The bias is usually omitted (bias=False)',
            'Output is logits, not probabilities (apply softmax for probabilities)'
        ]
    },

    softmax: {
        title: 'Softmax Output',
        overview: 'Softmax converts logits to a probability distribution over the vocabulary. During training, cross-entropy loss is computed. During inference, various sampling strategies select the next token.',
        how: 'Softmax normalizes logits to sum to 1, creating a valid probability distribution. Temperature can be applied to control randomness.',
        math: `Softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)

With temperature T:
Softmax(zᵢ, T) = exp(zᵢ/T) / Σⱼ exp(zⱼ/T)

T < 1: sharper (more confident)
T > 1: softer (more random)`,
        code: `def generate_next_token(
    logits: torch.Tensor,  # (batch, vocab_size)
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample next token with various strategies."""

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative prob > top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

    # Sample from distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token

# Greedy decoding (no sampling)
def greedy_decode(logits):
    return logits.argmax(dim=-1)`,
        tips: [
            'Training uses cross-entropy loss directly on logits (more numerically stable)',
            'Temperature=0 is equivalent to greedy (argmax) decoding',
            'Top-k and top-p can be combined for better generation quality',
            'Beam search maintains multiple hypotheses for more coherent generation'
        ]
    },

    cls_output: {
        title: '[CLS] Pooler Output',
        overview: 'BERT uses the [CLS] token\'s final representation for classification tasks. A pooler layer transforms this representation for downstream tasks.',
        how: 'The hidden state at position 0 (the [CLS] token) is extracted and passed through a dense layer with tanh activation. This creates a fixed-size representation of the entire sequence.',
        math: `pooled = tanh(Linear(hidden_states[0]))

The [CLS] representation aggregates information
from all tokens through bidirectional attention.`,
        code: `class BertPooler(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Take the hidden state of [CLS] token (first token)
        cls_token = hidden_states[:, 0]
        pooled = self.dense(cls_token)
        pooled = self.activation(pooled)
        return pooled

# Usage for classification
class BertForClassification(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model
        self.pooler = BertPooler(bert_model.config.d_model)
        self.classifier = nn.Linear(bert_model.config.d_model, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled = self.pooler(outputs.last_hidden_state)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits`,
        tips: [
            'The [CLS] token learns to aggregate sequence information',
            'For some tasks, mean pooling of all tokens works better',
            'The pooler is task-specific and often fine-tuned',
            'tanh squashes values to [-1, 1] for stable downstream training'
        ]
    },

    mlm_head: {
        title: 'Masked Language Model Head',
        overview: 'The MLM head predicts masked tokens during BERT pre-training. It projects hidden states back to vocabulary space with additional transformations.',
        how: 'A two-layer head: first a dense layer with GELU activation, then layer norm, and finally projection to vocabulary. This is more expressive than a simple linear projection.',
        math: `MLM prediction:
h = GELU(Linear(hidden_states))
h = LayerNorm(h)
logits = Linear(h)  # to vocab_size`,
        code: `class BertMLMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, d_model)
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)  # (batch, seq_len, vocab_size)
        return logits

class BertForMLM(nn.Module):
    def __init__(self, bert_model, vocab_size):
        super().__init__()
        self.bert = bert_model
        self.mlm_head = BertMLMHead(bert_model.config.d_model, vocab_size)

        # Tie weights with embedding
        self.mlm_head.decoder.weight = bert_model.embeddings.word_embeddings.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        logits = self.mlm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss`,
        tips: [
            'MLM randomly masks 15% of tokens: 80% [MASK], 10% random, 10% unchanged',
            'Loss is only computed on masked positions (labels=-100 elsewhere)',
            'Weight tying with embeddings is common',
            'The extra dense + norm layers help with the MLM objective'
        ]
    },

    // Container blocks (for visual representation)
    encoder_block: {
        title: 'Encoder Block',
        overview: 'The encoder block contains multi-head self-attention followed by a feed-forward network, each with residual connections and layer normalization. Multiple encoder blocks are stacked (Nx).',
        how: 'Each encoder block processes the entire sequence in parallel. Information flows through self-attention (mixing positions) then FFN (transforming each position). The stack of N blocks creates deep representations.',
        math: `EncoderBlock(x):
  h = LayerNorm(x + MultiHeadAttention(x, x, x))
  output = LayerNorm(h + FeedForward(h))

Full Encoder: x → EncoderBlock₁ → EncoderBlock₂ → ... → EncoderBlockₙ`,
        code: `class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  # Final norm (optional)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Original Transformer uses N=6 encoder layers`,
        tips: [
            'Original Transformer uses N=6 encoder layers',
            'BERT-base uses 12 layers, BERT-large uses 24',
            'Each layer has its own learnable parameters (not shared)',
            'The encoder output is passed to all decoder cross-attention layers'
        ]
    },

    decoder_block: {
        title: 'Decoder Block',
        overview: 'The decoder block has three sub-layers: masked self-attention, cross-attention (encoder-decoder), and FFN. Each has residual connections and normalization. This enables autoregressive generation while attending to encoder outputs.',
        how: 'Masked self-attention sees only previous positions (causal). Cross-attention attends to all encoder positions. This allows the decoder to generate one token at a time while using full input context.',
        math: `DecoderBlock(x, encoder_output):
  h1 = LayerNorm(x + MaskedSelfAttention(x))
  h2 = LayerNorm(h1 + CrossAttention(h1, encoder_output))
  output = LayerNorm(h2 + FeedForward(h2))`,
        code: `class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

# Autoregressive generation
def generate(model, encoder_output, max_len, sos_id, eos_id):
    generated = [sos_id]

    for _ in range(max_len):
        tgt = torch.tensor([generated]).unsqueeze(0)
        out = model.decode(tgt, encoder_output)
        next_token = out[:, -1].argmax(dim=-1).item()

        if next_token == eos_id:
            break
        generated.append(next_token)

    return generated`,
        tips: [
            'Decoder-only models (GPT) remove cross-attention',
            'KV-caching speeds up autoregressive generation',
            'The causal mask prevents attending to future tokens',
            'Cross-attention allows flexible alignment with encoder'
        ]
    },

    // ==================== RLHF / PPO COMPONENTS ====================
    reward_head: {
        title: 'Reward Model Head',
        overview: 'The reward model outputs a scalar score indicating how good a response is according to human preferences. It is trained on pairs of responses where humans indicated which one was better.',
        how: 'A linear layer projects the final hidden state (usually of the last token or a special token) to a single scalar value. The model is trained with a ranking loss on preference pairs.',
        math: `Reward Model Training:
r(x, y) = Linear(transformer_output[:, -1, :])

Loss = -log(σ(r(x, y_w) - r(x, y_l)))

Where y_w is the preferred (winning) response
and y_l is the rejected (losing) response.`,
        code: `class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.transformer = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Use last token's hidden state
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)

def reward_loss(rewards_chosen, rewards_rejected):
    """Bradley-Terry preference loss."""
    return -torch.log(torch.sigmoid(rewards_chosen - rewards_rejected)).mean()

# Training loop
for batch in dataloader:
    r_chosen = reward_model(batch['chosen_ids'])
    r_rejected = reward_model(batch['rejected_ids'])
    loss = reward_loss(r_chosen, r_rejected)
    loss.backward()`,
        tips: [
            'Reward models are typically smaller than the policy model',
            'Use the same tokenizer as the policy model',
            'Normalize rewards during PPO training for stability',
            'Consider reward model ensembles for robustness'
        ]
    },

    value_head: {
        title: 'Value Head (Critic)',
        overview: 'In PPO, the value head estimates the expected cumulative reward from a state. It serves as the critic in actor-critic RL, helping compute advantages for policy gradient updates.',
        how: 'A separate linear head on top of the transformer outputs a scalar value estimate for each token position. This is used to compute GAE (Generalized Advantage Estimation) for variance reduction.',
        math: `Value estimation:
V(s) = Linear(hidden_state)

Advantage (GAE):
Â_t = Σₖ (γλ)^k δ_{t+k}
δ_t = r_t + γV(s_{t+1}) - V(s_t)

PPO uses advantages for policy updates.`,
        code: `class ActorCriticLM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.transformer = base_model
        hidden_size = base_model.config.hidden_size

        # Actor (policy) head - uses LM head
        self.lm_head = base_model.lm_head

        # Critic (value) head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden = outputs.hidden_states[-1]

        # Policy logits
        logits = self.lm_head(hidden)

        # Value estimates
        values = self.value_head(hidden).squeeze(-1)

        return logits, values

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages)`,
        tips: [
            'Value head is trained alongside policy to minimize MSE on returns',
            'Separate optimizers or learning rates often help stability',
            'Value function can share transformer weights or be separate',
            'GAE with λ=0.95 is a common choice for variance-bias tradeoff'
        ]
    },

    kl_penalty: {
        title: 'KL Divergence Penalty',
        overview: 'The KL penalty prevents the policy from diverging too far from the reference model during RLHF training. This maintains generation quality and prevents reward hacking.',
        how: 'For each token, compute the KL divergence between the policy and reference model distributions. This is subtracted from the reward (or added as a penalty to the loss).',
        math: `KL Penalty:
KL(π_θ || π_ref) = Σ_t π_θ(a_t|s_t) log(π_θ(a_t|s_t) / π_ref(a_t|s_t))

Per-token approximation:
KL_t ≈ log π_θ(a_t|s_t) - log π_ref(a_t|s_t)

Modified reward:
r' = r - β * KL`,
        code: `def compute_kl_penalty(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    kl_coef: float = 0.1
) -> torch.Tensor:
    """Compute KL penalty between policy and reference."""
    # Per-token KL divergence (approximation)
    kl = policy_logprobs - ref_logprobs
    return kl_coef * kl

def get_logprobs(logits, labels):
    """Get log probabilities of chosen actions."""
    logprobs = F.log_softmax(logits, dim=-1)
    return torch.gather(logprobs, -1, labels.unsqueeze(-1)).squeeze(-1)

# During PPO training
with torch.no_grad():
    ref_logits = ref_model(input_ids).logits
    ref_logprobs = get_logprobs(ref_logits, response_ids)

policy_logits = policy_model(input_ids).logits
policy_logprobs = get_logprobs(policy_logits, response_ids)

kl_penalty = compute_kl_penalty(policy_logprobs, ref_logprobs, kl_coef=0.1)
modified_rewards = rewards - kl_penalty.sum(dim=-1)`,
        tips: [
            'β (kl_coef) typically ranges from 0.01 to 0.2',
            'Adaptive KL coefficient can help maintain target KL',
            'KL penalty is crucial for preventing reward hacking',
            'Monitor KL divergence during training to detect collapse'
        ]
    },

    ppo_loss: {
        title: 'PPO Clipped Loss',
        overview: 'PPO uses a clipped surrogate objective that prevents too large policy updates. This makes training more stable compared to vanilla policy gradient methods.',
        how: 'The loss clips the probability ratio between new and old policies, preventing updates that would change the policy too drastically in a single step.',
        math: `PPO Clipped Objective:
L^CLIP = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

Where:
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
ε is the clip range (typically 0.2)
Â_t is the advantage estimate`,
        code: `def ppo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2
) -> torch.Tensor:
    """Compute PPO clipped policy loss."""
    # Probability ratio
    ratio = torch.exp(logprobs - old_logprobs)

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)

    # PPO loss (negative because we maximize)
    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()

    return policy_loss

def ppo_step(policy, ref_model, reward_model, batch, optimizer):
    """Single PPO training step."""
    # Get old logprobs
    with torch.no_grad():
        old_logits = policy(batch['input_ids']).logits
        old_logprobs = get_logprobs(old_logits, batch['response_ids'])

    # Get rewards
    with torch.no_grad():
        rewards = reward_model(batch['full_ids'])

    # Compute advantages
    _, values = policy(batch['input_ids'])
    advantages = compute_gae(rewards, values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO update
    logits, new_values = policy(batch['input_ids'])
    logprobs = get_logprobs(logits, batch['response_ids'])

    policy_loss = ppo_loss(logprobs, old_logprobs, advantages)
    value_loss = F.mse_loss(new_values, rewards)

    loss = policy_loss + 0.5 * value_loss
    loss.backward()
    optimizer.step()`,
        tips: [
            'Clip range of 0.2 is standard but can be tuned',
            'Multiple PPO epochs per batch can improve sample efficiency',
            'Normalize advantages for more stable training',
            'Use gradient clipping alongside PPO clipping'
        ]
    },

    // ==================== DPO COMPONENTS ====================
    implicit_reward: {
        title: 'Implicit Reward (DPO)',
        overview: 'DPO shows that the optimal policy under a reward function can be expressed in closed form. This means we can bypass reward model training and directly optimize the policy using preference data.',
        how: 'The implicit reward is the log ratio of policy to reference probabilities, scaled by the temperature β. This closed-form solution means we can train with a simple classification loss.',
        math: `Implicit Reward:
r(x, y) = β log(π_θ(y|x) / π_ref(y|x)) + β log Z(x)

The partition function Z(x) cancels out in the loss,
giving us the DPO objective.

Optimal policy (Bradley-Terry):
π*(y|x) = π_ref(y|x) exp(r(x,y)/β) / Z(x)`,
        code: `def compute_implicit_reward(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """Compute implicit reward under DPO formulation."""
    # Sum log probs over sequence
    policy_sum = policy_logprobs.sum(dim=-1)
    ref_sum = ref_logprobs.sum(dim=-1)

    # Implicit reward (up to constant)
    return beta * (policy_sum - ref_sum)

# The key insight: we don't need explicit rewards!
# The ratio of probabilities IS the reward.`,
        tips: [
            'β controls how much the policy can deviate from reference',
            'Higher β means more conservative updates',
            'The implicit reward naturally prevents reward hacking',
            'No need to train or maintain a separate reward model'
        ]
    },

    dpo_loss: {
        title: 'DPO Loss',
        overview: 'DPO loss is a simple binary cross-entropy on preference pairs. It directly increases the probability of chosen responses relative to rejected ones, weighted by the implicit reward difference.',
        how: 'For each preference pair, compute log probability ratios for both chosen and rejected responses. The loss pushes the chosen response to have higher implicit reward.',
        math: `DPO Loss:
L_DPO = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x))
                  - β log(π_θ(y_l|x)/π_ref(y_l|x)))]

Simplified:
L_DPO = -E[log σ(β(r_θ(y_w) - r_θ(y_l)))]

Where r_θ(y) = log(π_θ(y|x)/π_ref(y|x))`,
        code: `def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """Direct Preference Optimization loss."""
    # Compute log ratios
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps

    # DPO loss
    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -F.logsigmoid(logits).mean()

    return loss

class DPOTrainer:
    def __init__(self, policy, ref_policy, beta=0.1):
        self.policy = policy
        self.ref_policy = ref_policy
        self.beta = beta

    def train_step(self, batch):
        # Get policy log probs
        chosen_logps = self.get_logprobs(self.policy, batch['chosen'])
        rejected_logps = self.get_logprobs(self.policy, batch['rejected'])

        # Get reference log probs (frozen)
        with torch.no_grad():
            ref_chosen_logps = self.get_logprobs(self.ref_policy, batch['chosen'])
            ref_rejected_logps = self.get_logprobs(self.ref_policy, batch['rejected'])

        loss = dpo_loss(
            chosen_logps, rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            self.beta
        )
        return loss`,
        tips: [
            'DPO is much simpler to implement than PPO',
            'No reward model, no value function, no RL tricks needed',
            'Reference model must stay frozen throughout training',
            'β=0.1 is a common starting point'
        ]
    },

    // ==================== DEEPSEEK COMPONENTS ====================
    mla_compress: {
        title: 'KV Compression (MLA)',
        overview: 'Multi-head Latent Attention compresses the KV cache by projecting keys and values into a low-dimensional latent space. This dramatically reduces memory requirements during inference.',
        how: 'Instead of storing full-dimensional K and V for each head, MLA projects them to a shared low-rank latent representation. During attention, they are projected back up.',
        math: `Standard MHA KV cache: O(batch × seq × heads × d_head)

MLA compression:
c = x W_c  (compress to latent dim d_l << d)
K = c W_K^up
V = c W_V^up

KV cache: O(batch × seq × d_l)
Reduction: d_l / (heads × d_head) ≈ 93%`,
        code: `class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_latent):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_latent = d_latent

        # Compression
        self.kv_compress = nn.Linear(d_model, d_latent, bias=False)

        # Decompression
        self.k_decompress = nn.Linear(d_latent, d_model, bias=False)
        self.v_decompress = nn.Linear(d_latent, d_model, bias=False)

        # Query projection (not compressed)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        B, L, D = x.shape

        # Query (full dimension)
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head)

        # Compress KV to latent space
        kv_latent = self.kv_compress(x)  # (B, L, d_latent)

        # Decompress for attention
        K = self.k_decompress(kv_latent).view(B, L, self.n_heads, self.d_head)
        V = self.v_decompress(kv_latent).view(B, L, self.n_heads, self.d_head)

        # Standard attention computation
        # ... (same as regular MHA)

        # During inference, only cache kv_latent (much smaller!)
        return output, kv_latent`,
        tips: [
            'MLA achieves 93%+ KV cache reduction in DeepSeek V2',
            'd_latent is typically 512-1024 vs 8192+ for full KV',
            'The compression is learned end-to-end',
            'Query projection is NOT compressed (maintains quality)'
        ]
    },

    shared_experts: {
        title: 'Shared Experts (DeepSeekMoE)',
        overview: 'Shared experts are always activated for every token, regardless of routing. They capture common knowledge that all tokens need, while routed experts handle specialized patterns.',
        how: 'In DeepSeekMoE, 2 experts are "shared" and always contribute to the output. The router only selects from the remaining routed experts. This ensures baseline quality.',
        math: `DeepSeekMoE output:
y = Σ (shared expert outputs) + Σ (top-k routed expert outputs)

Shared experts: Always activated (2 experts)
Routed experts: Top-6 selected from 160 experts

Total active: 2 + 6 = 8 experts per token`,
        code: `class DeepSeekMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_shared: int = 2,
        n_routed: int = 160,
        top_k: int = 6
    ):
        super().__init__()
        self.n_shared = n_shared
        self.top_k = top_k

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            SwiGLU(d_model, d_ff) for _ in range(n_shared)
        ])

        # Routed experts
        self.routed_experts = nn.ModuleList([
            SwiGLU(d_model, d_ff) for _ in range(n_routed)
        ])

        # Router for routed experts only
        self.router = nn.Linear(d_model, n_routed, bias=False)

    def forward(self, x):
        B, L, D = x.shape

        # Shared expert contributions (always computed)
        shared_out = sum(expert(x) for expert in self.shared_experts)

        # Route to top-k routed experts
        router_logits = self.router(x)
        topk_weights, topk_indices = torch.topk(
            F.softmax(router_logits, dim=-1),
            self.top_k, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Compute routed expert outputs
        routed_out = self.compute_routed(x, topk_weights, topk_indices)

        return shared_out + routed_out`,
        tips: [
            'Shared experts provide a quality floor',
            'Fine-grained experts (160) allow better specialization',
            'Shared + routed split improves training stability',
            'Device-limited routing ensures efficient parallelism'
        ]
    },

    // ==================== GRPO / DeepSeek R1 COMPONENTS ====================
    grpo_loss: {
        title: 'GRPO Loss (Group Relative Policy Optimization)',
        overview: 'GRPO simplifies PPO by using group statistics instead of a learned value function. For each prompt, multiple responses are sampled, and their rewards are used to estimate the baseline.',
        how: 'Instead of training a critic, GRPO samples G responses per prompt and uses the mean reward as the baseline. This eliminates the value function while maintaining variance reduction.',
        math: `GRPO Objective:
L_GRPO = -E[Σᵢ (rᵢ - mean(r)) log π_θ(yᵢ|x)]

Where:
- G responses sampled per prompt
- rᵢ is reward for response i
- mean(r) is the group baseline

No value function needed!`,
        code: `class GRPOTrainer:
    def __init__(self, policy, ref_policy, reward_fn, group_size=8):
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_fn = reward_fn
        self.group_size = group_size

    def train_step(self, prompts):
        all_losses = []

        for prompt in prompts:
            # Sample G responses
            responses = [
                self.policy.generate(prompt)
                for _ in range(self.group_size)
            ]

            # Get rewards
            rewards = torch.tensor([
                self.reward_fn(prompt, resp) for resp in responses
            ])

            # Group baseline (no critic needed!)
            baseline = rewards.mean()

            # Compute advantages
            advantages = rewards - baseline

            # Policy gradient with group-relative advantages
            for resp, adv in zip(responses, advantages):
                logprobs = self.get_logprobs(self.policy, prompt, resp)
                loss = -adv * logprobs.sum()
                all_losses.append(loss)

        return torch.stack(all_losses).mean()

# Key insight: The group mean IS the baseline
# No need to train a separate value network`,
        tips: [
            'GRPO eliminates the need for value function training',
            'Group size of 4-16 typically works well',
            'Computationally cheaper than PPO (no critic)',
            'Naturally handles sparse rewards'
        ]
    },

    reasoning_head: {
        title: 'Reasoning Head (<think> tokens)',
        overview: 'DeepSeek R1 uses special tokens to separate reasoning from final answers. The model learns to emit chain-of-thought in <think> blocks before producing the answer, without explicit CoT supervision.',
        how: 'During RL training, the model discovers that breaking down problems step-by-step leads to higher rewards. The <think> tokens mark this internal reasoning process.',
        math: `Response format:
<think>
[Step-by-step reasoning...]
</think>
[Final answer]

The RL reward encourages:
1. Correct final answer (accuracy reward)
2. Proper format (format reward)
3. Concise reasoning (length penalty)`,
        code: `class ReasoningLM(nn.Module):
    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer

        # Add special tokens
        special_tokens = {
            'additional_special_tokens': ['<think>', '</think>']
        }
        tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(tokenizer))

        self.think_start = tokenizer.convert_tokens_to_ids('<think>')
        self.think_end = tokenizer.convert_tokens_to_ids('</think>')

    def generate_with_reasoning(self, prompt, max_length=2048):
        """Generate response with explicit reasoning phase."""
        # Model naturally learns to use <think> tokens
        # through RL training (no explicit supervision!)
        return self.model.generate(
            prompt,
            max_length=max_length,
            do_sample=True,
            temperature=0.7
        )

def extract_reasoning_and_answer(response, tokenizer):
    """Parse reasoning and answer from response."""
    text = tokenizer.decode(response)

    if '<think>' in text and '</think>' in text:
        reasoning = text.split('<think>')[1].split('</think>')[0]
        answer = text.split('</think>')[1].strip()
    else:
        reasoning = ""
        answer = text

    return reasoning, answer`,
        tips: [
            'Reasoning emerges naturally from RL without CoT supervision',
            'The model learns to self-verify in the thinking phase',
            'Longer reasoning often correlates with harder problems',
            'Format rewards ensure proper <think> tag usage'
        ]
    }
};

// Additional component details for RLHF/DPO architectures
const TRAINING_COMPONENT_DETAILS = {
    ref_model: {
        title: 'Reference Model',
        overview: 'The reference model is a frozen copy of the initial supervised fine-tuned (SFT) model. It serves as an anchor to prevent the policy from drifting too far during training.',
        how: 'During training, outputs from the reference model are compared with the policy model to compute KL divergence penalties or probability ratios. It remains unchanged throughout training.',
        math: `KL Divergence: D_KL(π_θ || π_ref)

Reference log prob: log π_ref(y|x)
Used in PPO: r_modified = r - β * KL(π_θ || π_ref)
Used in DPO: log(π_ref(y_w|x)) and log(π_ref(y_l|x))`,
        code: `class ReferenceModel:
    def __init__(self, base_model):
        self.model = copy.deepcopy(base_model)
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_log_probs(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs`,
        tips: [
            'Always freeze reference model to prevent gradient updates',
            'Use torch.no_grad() for memory efficiency during inference',
            'The reference is typically the SFT model before RLHF',
            'KL penalty strength β is a key hyperparameter (0.01-0.2)'
        ]
    },
    ref_embedding: {
        title: 'Reference Embedding Layer',
        overview: 'The embedding layer in the frozen reference model that converts tokens to dense vectors. Identical to the policy embedding but not updated during training.',
        how: 'Maps input token IDs to learned vector representations. Since the reference model is frozen, these embeddings remain unchanged and provide a stable baseline.',
        math: `E_ref: V → R^d

x_embed = E_ref[token_id]
where E_ref ∈ R^{|V| × d}`,
        code: `# Reference embedding (frozen)
ref_embeddings = ref_model.embeddings(input_ids)

# These are compared with policy embeddings
# to measure how much the model has changed`,
        tips: [
            'Reference embeddings stay frozen throughout training',
            'Helps maintain stability by anchoring to original model',
            'No gradients flow through reference model'
        ]
    },
    ref_transformer: {
        title: 'Reference Transformer Blocks',
        overview: 'The transformer layers in the frozen reference model. They process embeddings through attention and FFN layers to produce contextual representations.',
        how: 'Identical architecture to the policy transformer, but all weights are frozen. Used to compute reference log probabilities for KL divergence or DPO loss.',
        math: `h_ref = TransformerBlocks_ref(x_embed)

Same computation as policy, but:
∂L/∂θ_ref = 0 (frozen)`,
        code: `# Forward pass through frozen reference
with torch.no_grad():
    ref_hidden = ref_model.transformer(
        input_embeds=embeddings,
        attention_mask=mask
    ).last_hidden_state`,
        tips: [
            'Reference transformer architecture matches policy exactly',
            'Frozen weights prevent mode collapse',
            'Provides stable log-probability baseline'
        ]
    },
    ref_lm_head: {
        title: 'Reference LM Head',
        overview: 'The language model head in the reference model that converts hidden states to vocabulary logits. Used to compute reference probabilities.',
        how: 'Projects the final hidden state to vocabulary size, then applies softmax to get token probabilities. These probabilities form the reference distribution.',
        math: `logits_ref = h_ref × W_lm + b
P_ref(token|context) = softmax(logits_ref)`,
        code: `# Get reference log probabilities
ref_logits = ref_model.lm_head(ref_hidden)
ref_log_probs = F.log_softmax(ref_logits, dim=-1)

# Select log probs for actual tokens
ref_token_log_probs = torch.gather(
    ref_log_probs, -1, tokens.unsqueeze(-1)
).squeeze(-1)`,
        tips: [
            'Reference probabilities anchor the policy',
            'Used in KL computation: log(π_θ) - log(π_ref)',
            'Critical for preventing reward hacking'
        ]
    },
    policy_model: {
        title: 'Policy Model',
        overview: 'The trainable language model being optimized through RLHF or DPO. It learns to generate responses that maximize reward while staying close to the reference model.',
        how: 'Generates responses given prompts, and its parameters are updated based on the training signal (PPO loss or DPO loss). The KL penalty prevents it from deviating too far from the reference.',
        math: `Objective: max_θ E[r(x,y)] - β * D_KL(π_θ || π_ref)

Policy: π_θ(y|x) - parameterized by θ
Update: θ ← θ + α∇_θ L(θ)`,
        code: `class PolicyModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model  # Trainable

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def generate(self, prompt, **kwargs):
        return self.model.generate(prompt, **kwargs)`,
        tips: [
            'Policy model parameters are updated during training',
            'Initialized from SFT model (same as reference)',
            'Balance reward maximization with staying near reference'
        ]
    },
    policy_embedding: {
        title: 'Policy Embedding Layer',
        overview: 'The trainable embedding layer in the policy model. Unlike the reference, these embeddings can be updated during training.',
        how: 'Converts input tokens to dense vectors. During RLHF, gradients flow through this layer to potentially adjust how tokens are represented.',
        math: `E_π: V → R^d

x_embed = E_π[token_id]
∂L/∂E_π ≠ 0 (trainable)`,
        code: `# Policy embeddings (trainable)
policy_embeddings = policy_model.embeddings(input_ids)

# Gradients will update these embeddings
# based on the RLHF/DPO loss signal`,
        tips: [
            'Embedding updates are usually small in RLHF',
            'Main learning happens in attention layers',
            'Shared embedding matrices with LM head common'
        ]
    },
    policy_transformer: {
        title: 'Policy Transformer Blocks',
        overview: 'The trainable transformer layers that learn to produce better responses. These layers are where most of the behavioral changes occur during RLHF.',
        how: 'Processes embeddings through self-attention and FFN layers. Gradients from the reward signal flow through these layers, adjusting attention patterns and representations.',
        math: `h_policy = TransformerBlocks_θ(x_embed)

θ includes attention weights and FFN parameters
∂L/∂θ = ∇_θ[r(x,y) - β*KL]`,
        code: `# Forward pass through trainable policy
policy_hidden = policy_model.transformer(
    input_embeds=embeddings,
    attention_mask=mask
).last_hidden_state

# Gradients flow back to update transformer weights`,
        tips: [
            'Attention patterns shift to prefer rewarded behaviors',
            'FFN layers learn reward-relevant features',
            'Layer-wise learning rates can help stability'
        ]
    },
    policy_lm_head: {
        title: 'Policy LM Head',
        overview: 'The trainable language model head that outputs token probabilities. This layer directly determines the policy\'s token distribution.',
        how: 'Projects hidden states to vocabulary logits. During RLHF, this layer learns to assign higher probabilities to tokens that lead to higher rewards.',
        math: `logits_π = h_policy × W_lm + b
π_θ(token|context) = softmax(logits_π)

Optimized to maximize: log π_θ(y_good) - log π_θ(y_bad)`,
        code: `# Get policy log probabilities
policy_logits = policy_model.lm_head(policy_hidden)
policy_log_probs = F.log_softmax(policy_logits, dim=-1)

# Used in PPO: log π_θ(a|s) for policy gradient
# Used in DPO: log π_θ(y_w|x) - log π_θ(y_l|x)`,
        tips: [
            'Log probabilities used for both sampling and loss',
            'Often tied to input embeddings (weight sharing)',
            'Key layer for controlling output distribution'
        ]
    },
    reward_model: {
        title: 'Reward Model',
        overview: 'A separate model trained to predict human preferences. Given a prompt and response, it outputs a scalar reward indicating quality.',
        how: 'Trained on human preference data (response A > response B). During RLHF, it provides reward signals to guide policy optimization without needing human feedback at every step.',
        math: `r(x, y) = RewardModel(x, y) ∈ R

Training objective (Bradley-Terry):
L = -log(σ(r(y_w) - r(y_l)))

where y_w is preferred over y_l`,
        code: `class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask)
        # Pool to single vector (e.g., last token)
        pooled = hidden[:, -1, :]
        reward = self.reward_head(pooled)
        return reward.squeeze(-1)`,
        tips: [
            'Train on diverse preference pairs',
            'Reward model quality limits RLHF performance',
            'DPO eliminates need for explicit reward model',
            'Reward hacking is a common failure mode'
        ]
    },
    reward_embedding: {
        title: 'Reward Model Embedding',
        overview: 'The embedding layer in the reward model that converts tokens to vectors for reward prediction.',
        how: 'Same architecture as the language model embeddings. Often initialized from a pretrained model and fine-tuned on preference data.',
        math: `E_r: V → R^d

Input to reward model transformer layers`,
        code: `# Reward model uses same embedding architecture
reward_embeds = reward_model.embeddings(
    torch.cat([prompt_ids, response_ids], dim=1)
)`,
        tips: [
            'Often initialized from same base model as policy',
            'Shared vocabulary with the language model',
            'Processes both prompt and response together'
        ]
    },
    reward_transformer: {
        title: 'Reward Model Transformer',
        overview: 'Transformer layers in the reward model that process the prompt-response pair to extract features for reward prediction.',
        how: 'Processes the concatenation of prompt and response through self-attention layers. The final hidden states are used by the reward head to predict a scalar score.',
        math: `h_r = TransformerBlocks_r(concat(x, y))

Learns to identify features that correlate with human preferences`,
        code: `# Process full sequence through reward model
reward_hidden = reward_model.transformer(
    input_embeds=embeddings,
    attention_mask=attention_mask
).last_hidden_state`,
        tips: [
            'Can use bidirectional attention (not causal)',
            'Often smaller than the policy model',
            'Focus on response quality features'
        ]
    },
    advantage: {
        title: 'Advantage Estimation',
        overview: 'The advantage function measures how much better an action is compared to the expected value. It reduces variance in policy gradient estimation.',
        how: 'Computed as A(s,a) = Q(s,a) - V(s), or estimated using Generalized Advantage Estimation (GAE). Higher advantage means the action was better than expected.',
        math: `Advantage: A_t = Q(s_t, a_t) - V(s_t)

GAE: Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
where δ_t = r_t + γV(s_{t+1}) - V(s_t)`,
        code: `def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages)`,
        tips: [
            'GAE with λ=0.95 is common for RLHF',
            'Normalizing advantages improves stability',
            'Higher λ = more bias, lower variance'
        ]
    },
    prompt: {
        title: 'Input Prompt',
        overview: 'The initial text provided to the model that sets up the context for generation. In RLHF, prompts are sampled from a dataset to train the policy.',
        how: 'Prompts are tokenized and fed to both the policy and reference models. The policy generates a response, which is then evaluated by the reward model.',
        math: `x ~ D_prompts (prompt distribution)

Policy generates: y ~ π_θ(·|x)
Reward: r = R(x, y)`,
        code: `# Sample prompts from dataset
prompts = prompt_dataset.sample(batch_size)

# Tokenize
prompt_tokens = tokenizer(
    prompts,
    padding=True,
    return_tensors='pt'
)`,
        tips: [
            'Diverse prompts improve generalization',
            'Prompt quality affects learned behaviors',
            'Balance instruction types in training data'
        ]
    },
    response: {
        title: 'Policy Gradient Update',
        overview: 'The mechanism by which the policy is updated based on rewards. Uses the policy gradient theorem to improve expected reward.',
        how: 'Computes gradients of expected reward with respect to policy parameters, then applies gradient ascent. PPO uses clipping to ensure stable updates.',
        math: `∇_θ J(θ) = E[∇_θ log π_θ(a|s) * A(s,a)]

PPO Clipped:
L_CLIP = min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)`,
        code: `# Policy gradient with PPO clipping
ratio = torch.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
policy_loss = -torch.min(surr1, surr2).mean()

# Update policy
optimizer.zero_grad()
policy_loss.backward()
optimizer.step()`,
        tips: [
            'Clip ratio typically ε=0.2',
            'Multiple gradient steps per batch (PPO epochs)',
            'Monitor KL divergence for stability'
        ]
    },
    chosen: {
        title: 'Chosen Response (y_w)',
        overview: 'The preferred response in a preference pair. Human annotators or AI judges select this as the better response to the prompt.',
        how: 'In DPO, the model learns to assign higher probability to chosen responses compared to rejected ones. The implicit reward for chosen responses is higher.',
        math: `y_w (winner) - preferred response

DPO learns: log(π_θ(y_w|x)/π_ref(y_w|x)) should be HIGH`,
        code: `# Chosen response processing
chosen_tokens = tokenizer(chosen_response)
chosen_log_probs = get_log_probs(policy, chosen_tokens)
ref_chosen_log_probs = get_log_probs(ref_model, chosen_tokens)

chosen_ratio = chosen_log_probs - ref_chosen_log_probs`,
        tips: [
            'Quality of preference data is crucial',
            'Chosen doesn\'t mean perfect, just better',
            'Human and AI preferences can differ'
        ]
    },
    rejected: {
        title: 'Rejected Response (y_l)',
        overview: 'The less preferred response in a preference pair. The model learns to decrease the probability of generating similar responses.',
        how: 'In DPO, the difference between chosen and rejected log-probability ratios drives learning. Rejected responses receive lower implicit rewards.',
        math: `y_l (loser) - rejected response

DPO learns: log(π_θ(y_l|x)/π_ref(y_l|x)) should be LOW`,
        code: `# Rejected response processing
rejected_tokens = tokenizer(rejected_response)
rejected_log_probs = get_log_probs(policy, rejected_tokens)
ref_rejected_log_probs = get_log_probs(ref_model, rejected_tokens)

rejected_ratio = rejected_log_probs - ref_rejected_log_probs`,
        tips: [
            'Rejected responses help define boundaries',
            'Hard negatives improve discrimination',
            'Avoid obviously bad rejected examples'
        ]
    },
    ref_logprob: {
        title: 'Reference Log Probability',
        overview: 'The log probability of a response under the frozen reference model. Used as a baseline to measure how much the policy has changed.',
        how: 'Computed by running the response through the reference model and extracting log probabilities for each token. Summed to get total sequence log probability.',
        math: `log π_ref(y|x) = Σ_t log π_ref(y_t|y_{<t}, x)`,
        code: `with torch.no_grad():
    ref_logits = ref_model(input_ids).logits
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    ref_logprob = gather_log_probs(ref_log_probs, labels)`,
        tips: [
            'Always computed with torch.no_grad()',
            'Sum over sequence for total log prob',
            'Used in both PPO (KL) and DPO (ratios)'
        ]
    },
    policy_logprob: {
        title: 'Policy Log Probability',
        overview: 'The log probability of a response under the current policy model. The key quantity being optimized during training.',
        how: 'Computed by forward pass through the policy, then extracting and summing log probabilities for the actual tokens generated.',
        math: `log π_θ(y|x) = Σ_t log π_θ(y_t|y_{<t}, x)`,
        code: `policy_logits = policy_model(input_ids).logits
policy_log_probs = F.log_softmax(policy_logits, dim=-1)
policy_logprob = gather_log_probs(policy_log_probs, labels)`,
        tips: [
            'This is where gradients flow through',
            'Compare with reference for ratio calculation',
            'Higher for chosen, lower for rejected (DPO goal)'
        ]
    },
    gradient: {
        title: 'Gradient Update',
        overview: 'The gradient descent step that updates model parameters based on the computed loss. Uses Adam or similar optimizers.',
        how: 'Backpropagation computes gradients of the loss with respect to parameters, then the optimizer updates weights in the direction that minimizes loss.',
        math: `θ ← θ - α∇_θL(θ)

With Adam:
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²
θ_t = θ_{t-1} - α·m̂_t/(√v̂_t + ε)`,
        code: `# Compute loss (DPO or PPO)
loss = compute_loss(...)

# Backpropagation
optimizer.zero_grad()
loss.backward()

# Gradient clipping (optional)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Update step
optimizer.step()`,
        tips: [
            'Gradient clipping helps stability',
            'Learning rate warmup is common',
            'Monitor gradient norms for debugging'
        ]
    },
    mla_attention: {
        title: 'Multi-head Latent Attention',
        overview: 'The attention computation in MLA after decompressing keys and values from the latent space. Performs standard scaled dot-product attention.',
        how: 'Takes compressed KV latent, decompresses to full dimension, then computes attention scores and weighted values.',
        math: `K = Decompress(c_KV)
V = Decompress(c_KV)
Attn = softmax(QK^T/√d)V`,
        code: `# Decompress from latent
K = self.k_up_proj(kv_latent)
V = self.v_up_proj(kv_latent)

# Standard attention
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)
attn_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, V)`,
        tips: [
            'Only the latent c_KV is cached',
            'Decompression happens at attention time',
            'Enables massive KV cache reduction'
        ]
    },
    routed_experts: {
        title: 'Routed Expert FFNs',
        overview: 'Sparse experts activated by the router based on input tokens. Only a subset (e.g., top-6) are activated per token, enabling massive scale with constant compute.',
        how: 'Router selects top-k experts per token. Each selected expert processes the token, outputs are weighted by router scores and combined.',
        math: `Output = Σᵢ gᵢ · Expertᵢ(x)
where gᵢ = router_weight if i ∈ top-k else 0`,
        code: `# Select top-k experts per token
router_logits = self.router(x)
top_k_logits, top_k_indices = torch.topk(router_logits, k=6)
top_k_weights = F.softmax(top_k_logits, dim=-1)

# Process through selected experts
expert_outputs = []
for i, idx in enumerate(top_k_indices):
    expert_out = self.experts[idx](x)
    expert_outputs.append(top_k_weights[i] * expert_out)

output = sum(expert_outputs)`,
        tips: [
            'DeepSeek V2 uses 160 routed + 2 shared experts',
            'Top-6 routing balances quality vs compute',
            'Auxiliary loss balances expert utilization'
        ]
    },
    gqa_bias: {
        title: 'GQA with QKV Bias',
        overview: 'Grouped Query Attention with bias terms in Q, K, V projections. Qwen 2.5 adds biases back for improved modeling capacity.',
        how: 'Uses fewer KV heads than query heads (like GQA) but includes bias terms in all projections. Also uses YaRN RoPE for extended context.',
        math: `Q = xW_Q + b_Q
K = xW_K + b_K
V = xW_V + b_V

with grouped heads: K,V have n_kv_heads < n_heads`,
        code: `class GQAWithBias(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model // (n_heads // n_kv_heads), bias=True)
        self.v_proj = nn.Linear(d_model, d_model // (n_heads // n_kv_heads), bias=True)`,
        tips: [
            'Qwen adds biases unlike LLaMA',
            'YaRN RoPE enables 128K+ context',
            'Biases can improve model expressiveness'
        ]
    },
    swiglu_expanded: {
        title: 'Expanded SwiGLU FFN',
        overview: 'SwiGLU FFN with larger intermediate dimension. Qwen 2.5 uses an expanded FFN ratio for increased model capacity.',
        how: 'Same SwiGLU mechanism but with larger hidden dimension (e.g., 8/3 × d_model instead of 4 × d_model).',
        math: `FFN(x) = (swish(xW₁) ⊙ xW₃)W₂
where hidden_dim = 8/3 × d_model (expanded)`,
        code: `class ExpandedSwiGLU(nn.Module):
    def __init__(self, d_model, expansion=8/3):
        hidden = int(d_model * expansion)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))`,
        tips: [
            'Larger FFN increases model capacity',
            '8/3 ratio common in modern LLMs',
            'Trade-off between capacity and compute'
        ]
    },
    gqa_128k: {
        title: 'GQA with 128K Context',
        overview: 'Grouped Query Attention optimized for 128K token context windows. LLaMA 3.1 uses improved RoPE scaling for extreme sequence lengths.',
        how: 'Uses GQA for KV efficiency, combined with modified RoPE that scales better to very long sequences through adjusted frequency bases.',
        math: `RoPE_{long} uses adjusted θ_i = 10000^{-2i/d} × scaling_factor

Attention: softmax(QK^T/√d)V with 128K positions`,
        code: `# RoPE with extended context
def compute_rope_embeddings(positions, dim, base=10000, scaling=8.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
    # Apply scaling for long context
    inv_freq = inv_freq / scaling
    freqs = torch.outer(positions, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)`,
        tips: [
            '128K context enables book-length inputs',
            'RoPE scaling is key to length generalization',
            'Memory-efficient attention variants help'
        ]
    },
    base_model: {
        title: 'DeepSeek V3 Base Model',
        overview: 'The foundation 671B MoE model that DeepSeek R1 builds upon. Features MLA for efficient attention and DeepSeekMoE for sparse computation.',
        how: 'Combines all DeepSeek innovations: Multi-head Latent Attention for KV compression, auxiliary-loss-free load balancing, and shared+routed expert design.',
        math: `Total params: 671B
Active params per token: ~37B
Architecture: MLA + DeepSeekMoE`,
        code: `# DeepSeek V3 combines innovations
model = DeepSeekV3(
    num_layers=60,
    hidden_size=7168,
    num_attention_heads=128,
    mla_latent_dim=512,  # KV compression
    num_experts=160,
    num_shared_experts=2,
    top_k=6
)`,
        tips: [
            '671B total, ~37B active per token',
            'MLA reduces KV cache by 93%',
            'Competitive with GPT-4 at lower cost'
        ]
    },
    mla: {
        title: 'MLA + MoE Transformer',
        overview: 'The transformer block combining Multi-head Latent Attention with Mixture of Experts FFN. Core building block of DeepSeek models.',
        how: 'Each block: RMSNorm → MLA (compressed KV) → Residual → RMSNorm → DeepSeekMoE (shared + routed) → Residual',
        math: `h = x + MLA(RMSNorm(x))
output = h + MoE(RMSNorm(h))`,
        code: `class DeepSeekBlock(nn.Module):
    def forward(self, x, kv_cache=None):
        # MLA with compressed KV
        h = x + self.mla(self.norm1(x), kv_cache)
        # MoE FFN
        output = h + self.moe(self.norm2(h))
        return output`,
        tips: [
            'MLA provides attention efficiency',
            'MoE provides compute scaling',
            'Combination enables trillion-scale models'
        ]
    },
    answer_head: {
        title: 'Answer Generation Head',
        overview: 'The output head for generating final answers after the reasoning phase. Produces tokens outside the <think> tags.',
        how: 'After reasoning completes, the model switches to answer generation mode. The same LM head is used but generates answer tokens instead of reasoning tokens.',
        math: `P(answer_t | reasoning, answer_{<t}) = LMHead(h_t)`,
        code: `# After reasoning, generate answer
def generate_answer(model, reasoning_hidden):
    answer_tokens = []
    for _ in range(max_answer_length):
        logits = model.lm_head(hidden)
        next_token = sample(logits)
        if next_token == END_TOKEN:
            break
        answer_tokens.append(next_token)
    return answer_tokens`,
        tips: [
            'Same model head, different generation phase',
            'Reasoning informs the answer',
            'Format reward ensures proper separation'
        ]
    },
    prompt_group: {
        title: 'GRPO Prompt Input',
        overview: 'The input prompt in Group Relative Policy Optimization. Same prompt is used to generate multiple response samples.',
        how: 'A single prompt is sampled, then the policy generates G different responses. These are evaluated together with the group mean as baseline.',
        math: `x ~ D_prompts
Generate G responses: {y₁, y₂, ..., y_G} ~ π_θ(·|x)`,
        code: `# Sample prompt and generate group
prompt = sample_prompt(dataset)
responses = []
for _ in range(G):
    response = policy.generate(prompt, do_sample=True)
    responses.append(response)`,
        tips: [
            'Group size G typically 4-16',
            'Same prompt for all samples in group',
            'Reduces variance in reward estimation'
        ]
    },
    group_samples: {
        title: 'Group Response Sampling',
        overview: 'Multiple responses generated from the same prompt for GRPO. Each response receives a reward, and the group statistics form the baseline.',
        how: 'For each prompt, generate G responses using temperature sampling. All responses are evaluated by the reward function, then relative advantages are computed.',
        math: `{y₁, ..., y_G} ~ π_θ(·|x)
{r₁, ..., r_G} = rewards

Advantages: A_i = r_i - mean({r_j})`,
        code: `# Generate group of samples
group_responses = []
for _ in range(group_size):
    response = model.generate(
        prompt,
        do_sample=True,
        temperature=0.8
    )
    group_responses.append(response)

# Compute rewards for all
rewards = [reward_fn(prompt, r) for r in group_responses]`,
        tips: [
            'Temperature sampling for diversity',
            'Larger groups = more stable baseline',
            'All samples contribute to gradient'
        ]
    },
    accuracy_reward: {
        title: 'Accuracy Reward',
        overview: 'Reward signal based on whether the model\'s answer is correct. For math/code problems, this is verified against ground truth.',
        how: 'The final answer is extracted and compared against the correct answer. Binary reward (correct/incorrect) or partial credit for close answers.',
        math: `r_accuracy = 1 if answer == ground_truth else 0

For math: check numerical equivalence
For code: run test cases`,
        code: `def accuracy_reward(response, ground_truth):
    # Extract answer from response
    answer = extract_answer(response)

    # Compare with ground truth
    if is_math_problem:
        return 1.0 if math_equal(answer, ground_truth) else 0.0
    elif is_code_problem:
        return run_tests(answer, test_cases)`,
        tips: [
            'Primary signal for reasoning ability',
            'Binary rewards work well with RL',
            'Verifiable tasks enable scalable training'
        ]
    },
    format_reward: {
        title: 'Format Reward',
        overview: 'Reward for proper use of <think> and </think> tags. Ensures the model separates reasoning from final answer correctly.',
        how: 'Checks that reasoning is enclosed in proper tags, appears before the answer, and follows the expected format.',
        math: `r_format = 1 if valid_format(response) else penalty

Valid: <think>reasoning</think>answer`,
        code: `def format_reward(response):
    # Check for proper think tags
    has_think_open = '<think>' in response
    has_think_close = '</think>' in response

    if has_think_open and has_think_close:
        think_start = response.index('<think>')
        think_end = response.index('</think>')
        if think_start < think_end:
            return 1.0
    return -0.5  # Penalty for bad format`,
        tips: [
            'Enforces structured reasoning output',
            'Small negative penalty for violations',
            'Enables easy separation of CoT from answer'
        ]
    },
    length_penalty: {
        title: 'Length Penalty',
        overview: 'Penalty for excessively long reasoning chains. Prevents the model from generating unnecessarily verbose thinking.',
        how: 'Applies a small negative reward proportional to reasoning length beyond a threshold. Encourages concise but complete reasoning.',
        math: `r_length = -λ * max(0, len(reasoning) - threshold)

λ typically small (0.001)`,
        code: `def length_penalty(response, threshold=1000, lambda_=0.001):
    reasoning = extract_reasoning(response)
    excess_length = max(0, len(reasoning) - threshold)
    return -lambda_ * excess_length`,
        tips: [
            'Prevents reward hacking via verbose output',
            'Threshold tuned per task type',
            'Small weight to not dominate accuracy'
        ]
    },
    group_baseline: {
        title: 'Group Baseline',
        overview: 'The mean reward across all responses in a GRPO group. Used to compute relative advantages without a learned value function.',
        how: 'Simply averages the rewards of all G samples from the same prompt. Each sample\'s advantage is its reward minus this mean.',
        math: `baseline = (1/G) Σᵢ rᵢ

Advantage_i = r_i - baseline`,
        code: `def compute_group_baseline(rewards):
    baseline = sum(rewards) / len(rewards)
    advantages = [r - baseline for r in rewards]
    return advantages`,
        tips: [
            'Eliminates need for value network',
            'Simple but effective variance reduction',
            'Group size affects baseline quality'
        ]
    },
    gradient_update: {
        title: 'GRPO Policy Update',
        overview: 'The policy gradient update in GRPO using group-relative advantages. Updates the policy to increase probability of better-than-average responses.',
        how: 'Computes policy gradient weighted by advantages, then applies gradient ascent with KL constraint.',
        math: `L = Σᵢ (rᵢ - baseline) log π_θ(yᵢ|x) - β KL(π_θ || π_ref)`,
        code: `# GRPO update
for prompt, responses, rewards in batches:
    baseline = rewards.mean()
    advantages = rewards - baseline

    log_probs = compute_log_probs(policy, responses)
    ref_log_probs = compute_log_probs(ref_model, responses)

    loss = -(advantages * log_probs).mean()
    loss += beta * (log_probs - ref_log_probs).mean()  # KL

    loss.backward()
    optimizer.step()`,
        tips: [
            'No critic network needed',
            'KL penalty prevents drift',
            'Simpler than PPO, competitive results'
        ]
    }
};

// Merge training components into main details
Object.assign(COMPONENT_DETAILS, TRAINING_COMPONENT_DETAILS);

// Helper to get details for any component, with fallbacks
function getComponentDetails(componentId) {
    // Direct match
    if (COMPONENT_DETAILS[componentId]) {
        return COMPONENT_DETAILS[componentId];
    }

    // Try base name (e.g., 'encoder_mha' for 'encoder_mha')
    const baseName = componentId.replace(/_enc$|_dec$/, '');
    if (COMPONENT_DETAILS[baseName]) {
        return COMPONENT_DETAILS[baseName];
    }

    // Try partial matches for common prefixes
    const prefixMappings = {
        'ref_': 'ref_model',
        'policy_': 'policy_model',
        'reward_': 'reward_model'
    };

    for (const [prefix, fallbackId] of Object.entries(prefixMappings)) {
        if (componentId.startsWith(prefix) && COMPONENT_DETAILS[fallbackId]) {
            // Check for more specific match first
            const specificKey = componentId;
            if (COMPONENT_DETAILS[specificKey]) {
                return COMPONENT_DETAILS[specificKey];
            }
        }
    }

    // Generic fallbacks based on type
    const genericFallbacks = {
        embedding: COMPONENT_DETAILS.input_embedding,
        positional: COMPONENT_DETAILS.pos_encoding_enc,
        attention: COMPONENT_DETAILS.encoder_mha,
        ffn: COMPONENT_DETAILS.encoder_ffn,
        norm: COMPONENT_DETAILS.encoder_add_norm1,
        output: COMPONENT_DETAILS.linear,
        softmax: COMPONENT_DETAILS.softmax,
        block: COMPONENT_DETAILS.encoder_block,
    };

    // Find component type from architecture - with safety check
    if (typeof ARCHITECTURES !== 'undefined') {
        for (const arch of Object.values(ARCHITECTURES)) {
            const comp = arch.components.find(c => c.id === componentId);
            if (comp && genericFallbacks[comp.type]) {
                return genericFallbacks[comp.type];
            }
        }
    }

    // Last resort: try to infer from component ID
    const typeHints = {
        'embedding': COMPONENT_DETAILS.input_embedding,
        'transformer': COMPONENT_DETAILS.encoder_mha,
        'lm_head': COMPONENT_DETAILS.linear,
        'head': COMPONENT_DETAILS.linear,
        'norm': COMPONENT_DETAILS.encoder_add_norm1,
        'ffn': COMPONENT_DETAILS.encoder_ffn,
        'attn': COMPONENT_DETAILS.encoder_mha,
        'attention': COMPONENT_DETAILS.encoder_mha,
    };

    for (const [hint, details] of Object.entries(typeHints)) {
        if (componentId.toLowerCase().includes(hint)) {
            return details;
        }
    }

    return null;
}
