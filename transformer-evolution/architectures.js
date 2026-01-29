// Architecture definitions for different transformer variants
const ARCHITECTURES = {
    original: {
        id: 'original',
        name: 'Original Transformer',
        year: 2017,
        paper: 'Attention Is All You Need',
        paperUrl: 'https://arxiv.org/abs/1706.03762',
        authors: 'Vaswani et al.',
        description: 'The original Transformer architecture introduced self-attention as the primary mechanism for sequence modeling, completely replacing recurrence. It uses an encoder-decoder structure designed for sequence-to-sequence tasks like machine translation.',
        keyInnovation: 'Self-Attention Mechanism',
        category: 'architecture',
        changes: [],
        components: [
            // Encoder side
            { id: 'input_embedding', type: 'embedding', x: 80, y: 520, width: 120, height: 40, label: 'Input Embedding', side: 'encoder' },
            { id: 'pos_encoding_enc', type: 'positional', x: 80, y: 460, width: 120, height: 35, label: 'Positional Encoding', side: 'encoder' },
            { id: 'encoder_mha', type: 'attention', x: 60, y: 340, width: 160, height: 50, label: 'Multi-Head Attention', side: 'encoder' },
            { id: 'encoder_add_norm1', type: 'norm', x: 80, y: 280, width: 120, height: 35, label: 'Add & Norm', side: 'encoder' },
            { id: 'encoder_ffn', type: 'ffn', x: 60, y: 210, width: 160, height: 50, label: 'Feed Forward', side: 'encoder' },
            { id: 'encoder_add_norm2', type: 'norm', x: 80, y: 150, width: 120, height: 35, label: 'Add & Norm', side: 'encoder' },
            { id: 'encoder_block', type: 'block', x: 40, y: 130, width: 200, height: 300, label: 'Encoder Block', sublabel: 'Nx', side: 'encoder', isContainer: true },

            // Decoder side
            { id: 'output_embedding', type: 'embedding', x: 320, y: 520, width: 120, height: 40, label: 'Output Embedding', side: 'decoder' },
            { id: 'pos_encoding_dec', type: 'positional', x: 320, y: 460, width: 120, height: 35, label: 'Positional Encoding', side: 'decoder' },
            { id: 'masked_mha', type: 'attention', x: 300, y: 380, width: 160, height: 50, label: 'Masked Multi-Head', sublabel: 'Attention', side: 'decoder' },
            { id: 'decoder_add_norm1', type: 'norm', x: 320, y: 320, width: 120, height: 35, label: 'Add & Norm', side: 'decoder' },
            { id: 'cross_attention', type: 'attention', x: 300, y: 250, width: 160, height: 50, label: 'Multi-Head Attention', sublabel: 'Cross-Attention', side: 'decoder' },
            { id: 'decoder_add_norm2', type: 'norm', x: 320, y: 190, width: 120, height: 35, label: 'Add & Norm', side: 'decoder' },
            { id: 'decoder_ffn', type: 'ffn', x: 300, y: 120, width: 160, height: 50, label: 'Feed Forward', side: 'decoder' },
            { id: 'decoder_add_norm3', type: 'norm', x: 320, y: 60, width: 120, height: 35, label: 'Add & Norm', side: 'decoder' },
            { id: 'decoder_block', type: 'block', x: 280, y: 40, width: 200, height: 430, label: 'Decoder Block', sublabel: 'Nx', side: 'decoder', isContainer: true },

            // Output
            { id: 'linear', type: 'output', x: 320, y: 10, width: 120, height: 35, label: 'Linear', side: 'output' },
            { id: 'softmax', type: 'softmax', x: 320, y: -35, width: 120, height: 30, label: 'Softmax', side: 'output' },
        ],
        connections: [
            { from: 'input_embedding', to: 'pos_encoding_enc' },
            { from: 'pos_encoding_enc', to: 'encoder_mha' },
            { from: 'encoder_mha', to: 'encoder_add_norm1', type: 'residual' },
            { from: 'encoder_add_norm1', to: 'encoder_ffn' },
            { from: 'encoder_ffn', to: 'encoder_add_norm2', type: 'residual' },
            { from: 'encoder_add_norm2', to: 'cross_attention', type: 'cross' },
            { from: 'output_embedding', to: 'pos_encoding_dec' },
            { from: 'pos_encoding_dec', to: 'masked_mha' },
            { from: 'masked_mha', to: 'decoder_add_norm1', type: 'residual' },
            { from: 'decoder_add_norm1', to: 'cross_attention' },
            { from: 'cross_attention', to: 'decoder_add_norm2', type: 'residual' },
            { from: 'decoder_add_norm2', to: 'decoder_ffn' },
            { from: 'decoder_ffn', to: 'decoder_add_norm3', type: 'residual' },
            { from: 'decoder_add_norm3', to: 'linear' },
            { from: 'linear', to: 'softmax' },
        ]
    },

    gpt1: {
        id: 'gpt1',
        name: 'GPT-1',
        year: 2018,
        paper: 'Improving Language Understanding by Generative Pre-Training',
        paperUrl: 'https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf',
        authors: 'Radford et al. (OpenAI)',
        description: 'GPT-1 introduced the decoder-only architecture for language modeling. It removed the encoder entirely and used only masked self-attention, proving that large-scale unsupervised pre-training followed by supervised fine-tuning could achieve state-of-the-art results.',
        keyInnovation: 'Decoder-Only Pre-training',
        category: 'architecture',
        changes: [
            { type: 'removed', title: 'Encoder Removed', description: 'Eliminated the entire encoder stack, using only the decoder for autoregressive language modeling.' },
            { type: 'removed', title: 'Cross-Attention Removed', description: 'Without an encoder, cross-attention layers were no longer needed.' },
            { type: 'modified', title: 'Pre-training Paradigm', description: 'Introduced unsupervised pre-training on large text corpus followed by task-specific fine-tuning.' },
        ],
        components: [
            { id: 'input_embedding', type: 'embedding', x: 180, y: 520, width: 140, height: 45, label: 'Token Embedding', side: 'decoder' },
            { id: 'pos_encoding', type: 'positional', x: 180, y: 455, width: 140, height: 40, label: 'Positional Embedding', sublabel: 'Learned', side: 'decoder' },
            { id: 'masked_mha', type: 'attention', x: 160, y: 340, width: 180, height: 55, label: 'Masked Multi-Head', sublabel: 'Self-Attention', side: 'decoder' },
            { id: 'add_norm1', type: 'norm', x: 180, y: 270, width: 140, height: 40, label: 'Add & LayerNorm', side: 'decoder' },
            { id: 'ffn', type: 'ffn', x: 160, y: 195, width: 180, height: 55, label: 'Feed Forward', sublabel: 'GELU Activation', side: 'decoder' },
            { id: 'add_norm2', type: 'norm', x: 180, y: 130, width: 140, height: 40, label: 'Add & LayerNorm', side: 'decoder' },
            { id: 'decoder_block', type: 'block', x: 140, y: 110, width: 220, height: 310, label: 'Transformer Block', sublabel: '12x', side: 'decoder', isContainer: true },
            { id: 'linear', type: 'output', x: 180, y: 50, width: 140, height: 40, label: 'Linear', side: 'output' },
            { id: 'softmax', type: 'softmax', x: 180, y: -5, width: 140, height: 35, label: 'Softmax', side: 'output' },
        ],
        connections: [
            { from: 'input_embedding', to: 'pos_encoding' },
            { from: 'pos_encoding', to: 'masked_mha' },
            { from: 'masked_mha', to: 'add_norm1', type: 'residual' },
            { from: 'add_norm1', to: 'ffn' },
            { from: 'ffn', to: 'add_norm2', type: 'residual' },
            { from: 'add_norm2', to: 'linear' },
            { from: 'linear', to: 'softmax' },
        ]
    },

    bert: {
        id: 'bert',
        name: 'BERT',
        year: 2018,
        paper: 'BERT: Pre-training of Deep Bidirectional Transformers',
        paperUrl: 'https://arxiv.org/abs/1810.04805',
        authors: 'Devlin et al. (Google)',
        description: 'BERT introduced bidirectional pre-training using masked language modeling (MLM). Unlike GPT which only looks at previous tokens, BERT can attend to both left and right context, making it better suited for understanding tasks like classification and question answering.',
        keyInnovation: 'Bidirectional Masked Language Modeling',
        category: 'architecture',
        changes: [
            { type: 'modified', title: 'Bidirectional Attention', description: 'Removed causal masking to allow tokens to attend to both past and future context.' },
            { type: 'added', title: 'Masked Language Modeling', description: 'Pre-training objective that randomly masks 15% of tokens and predicts them.' },
            { type: 'added', title: 'Segment Embeddings', description: 'Added segment embeddings to distinguish between sentence pairs.' },
            { type: 'added', title: '[CLS] Token', description: 'Special classification token used for downstream tasks.' },
        ],
        components: [
            { id: 'input_embedding', type: 'embedding', x: 180, y: 530, width: 140, height: 45, label: 'Token Embedding', side: 'encoder' },
            { id: 'segment_embedding', type: 'embedding', x: 180, y: 475, width: 140, height: 35, label: 'Segment Embedding', side: 'encoder' },
            { id: 'pos_encoding', type: 'positional', x: 180, y: 420, width: 140, height: 35, label: 'Position Embedding', sublabel: 'Learned', side: 'encoder' },
            { id: 'mha', type: 'attention', x: 160, y: 310, width: 180, height: 55, label: 'Multi-Head Attention', sublabel: 'Bidirectional', side: 'encoder' },
            { id: 'add_norm1', type: 'norm', x: 180, y: 245, width: 140, height: 40, label: 'Add & LayerNorm', side: 'encoder' },
            { id: 'ffn', type: 'ffn', x: 160, y: 175, width: 180, height: 50, label: 'Feed Forward', sublabel: 'GELU Activation', side: 'encoder' },
            { id: 'add_norm2', type: 'norm', x: 180, y: 115, width: 140, height: 40, label: 'Add & LayerNorm', side: 'encoder' },
            { id: 'encoder_block', type: 'block', x: 140, y: 95, width: 220, height: 300, label: 'Encoder Block', sublabel: '12x / 24x', side: 'encoder', isContainer: true },
            { id: 'cls_output', type: 'output', x: 180, y: 35, width: 140, height: 40, label: '[CLS] Pooler', side: 'output' },
            { id: 'mlm_head', type: 'output', x: 180, y: -20, width: 140, height: 35, label: 'MLM Head', side: 'output' },
        ],
        connections: [
            { from: 'input_embedding', to: 'segment_embedding' },
            { from: 'segment_embedding', to: 'pos_encoding' },
            { from: 'pos_encoding', to: 'mha' },
            { from: 'mha', to: 'add_norm1', type: 'residual' },
            { from: 'add_norm1', to: 'ffn' },
            { from: 'ffn', to: 'add_norm2', type: 'residual' },
            { from: 'add_norm2', to: 'cls_output' },
            { from: 'cls_output', to: 'mlm_head' },
        ]
    },

    gpt2: {
        id: 'gpt2',
        name: 'GPT-2',
        year: 2019,
        paper: 'Language Models are Unsupervised Multitask Learners',
        paperUrl: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
        authors: 'Radford et al. (OpenAI)',
        description: 'GPT-2 scaled up GPT-1 significantly and introduced Pre-Norm (moving LayerNorm before attention/FFN instead of after). It demonstrated emergent zero-shot capabilities without any fine-tuning, showing that scale alone can unlock new abilities.',
        keyInnovation: 'Pre-LayerNorm & Zero-Shot Learning',
        category: 'architecture',
        changes: [
            { type: 'modified', title: 'Pre-LayerNorm', description: 'Moved LayerNorm to before each sub-layer (Pre-LN) instead of after (Post-LN), improving training stability.' },
            { type: 'modified', title: 'Scale Increase', description: 'Scaled to 1.5B parameters (vs 117M in GPT-1) with 48 layers and 1600 hidden dimensions.' },
            { type: 'added', title: 'Final LayerNorm', description: 'Added a final LayerNorm after the last transformer block.' },
        ],
        components: [
            { id: 'input_embedding', type: 'embedding', x: 180, y: 520, width: 140, height: 45, label: 'Token Embedding', side: 'decoder' },
            { id: 'pos_encoding', type: 'positional', x: 180, y: 455, width: 140, height: 40, label: 'Position Embedding', sublabel: 'Learned', side: 'decoder' },
            { id: 'pre_norm1', type: 'norm', x: 180, y: 390, width: 140, height: 35, label: 'LayerNorm', sublabel: 'Pre-Norm', side: 'decoder' },
            { id: 'masked_mha', type: 'attention', x: 160, y: 320, width: 180, height: 50, label: 'Masked Multi-Head', sublabel: 'Self-Attention', side: 'decoder' },
            { id: 'pre_norm2', type: 'norm', x: 180, y: 255, width: 140, height: 35, label: 'LayerNorm', sublabel: 'Pre-Norm', side: 'decoder' },
            { id: 'ffn', type: 'ffn', x: 160, y: 185, width: 180, height: 50, label: 'Feed Forward', sublabel: 'GELU', side: 'decoder' },
            { id: 'decoder_block', type: 'block', x: 140, y: 165, width: 220, height: 280, label: 'Transformer Block', sublabel: '48x', side: 'decoder', isContainer: true },
            { id: 'final_norm', type: 'norm', x: 180, y: 105, width: 140, height: 40, label: 'Final LayerNorm', side: 'decoder' },
            { id: 'linear', type: 'output', x: 180, y: 45, width: 140, height: 40, label: 'LM Head', side: 'output' },
            { id: 'softmax', type: 'softmax', x: 180, y: -10, width: 140, height: 35, label: 'Softmax', side: 'output' },
        ],
        connections: [
            { from: 'input_embedding', to: 'pos_encoding' },
            { from: 'pos_encoding', to: 'pre_norm1' },
            { from: 'pre_norm1', to: 'masked_mha' },
            { from: 'masked_mha', to: 'pre_norm2', type: 'residual' },
            { from: 'pre_norm2', to: 'ffn' },
            { from: 'ffn', to: 'final_norm', type: 'residual' },
            { from: 'final_norm', to: 'linear' },
            { from: 'linear', to: 'softmax' },
        ]
    },

    llama: {
        id: 'llama',
        name: 'LLaMA',
        year: 2023,
        paper: 'LLaMA: Open and Efficient Foundation Language Models',
        paperUrl: 'https://arxiv.org/abs/2302.13971',
        authors: 'Touvron et al. (Meta)',
        description: 'LLaMA introduced several architectural improvements that became standard in modern LLMs: RMSNorm for efficiency, Rotary Position Embeddings (RoPE) for better length generalization, SwiGLU activation for improved performance, and removed bias terms throughout.',
        keyInnovation: 'RMSNorm + RoPE + SwiGLU',
        category: 'architecture',
        changes: [
            { type: 'modified', title: 'RMSNorm', description: 'Replaced LayerNorm with RMSNorm, which is faster and doesn\'t require centering (no mean computation).' },
            { type: 'modified', title: 'RoPE Embeddings', description: 'Replaced learned position embeddings with Rotary Position Embeddings applied at each attention layer.' },
            { type: 'modified', title: 'SwiGLU Activation', description: 'Replaced GELU with SwiGLU (Swish + Gated Linear Unit) in the FFN for better performance.' },
            { type: 'removed', title: 'Bias Terms', description: 'Removed all bias terms in linear layers throughout the model.' },
        ],
        components: [
            { id: 'input_embedding', type: 'embedding', x: 180, y: 520, width: 140, height: 45, label: 'Token Embedding', sublabel: 'No Position Emb', side: 'decoder' },
            { id: 'rms_norm1', type: 'norm', x: 180, y: 420, width: 140, height: 40, label: 'RMSNorm', sublabel: 'Pre-Norm', side: 'decoder' },
            { id: 'rope_mha', type: 'attention', x: 160, y: 340, width: 180, height: 55, label: 'Multi-Head Attention', sublabel: 'with RoPE', side: 'decoder' },
            { id: 'rms_norm2', type: 'norm', x: 180, y: 265, width: 140, height: 40, label: 'RMSNorm', sublabel: 'Pre-Norm', side: 'decoder' },
            { id: 'swiglu_ffn', type: 'ffn', x: 160, y: 185, width: 180, height: 55, label: 'SwiGLU FFN', sublabel: 'No Bias', side: 'decoder' },
            { id: 'decoder_block', type: 'block', x: 140, y: 165, width: 220, height: 320, label: 'Transformer Block', sublabel: '32x / 80x', side: 'decoder', isContainer: true },
            { id: 'final_norm', type: 'norm', x: 180, y: 100, width: 140, height: 40, label: 'Final RMSNorm', side: 'decoder' },
            { id: 'linear', type: 'output', x: 180, y: 40, width: 140, height: 40, label: 'LM Head', side: 'output' },
            { id: 'softmax', type: 'softmax', x: 180, y: -15, width: 140, height: 35, label: 'Softmax', side: 'output' },
        ],
        connections: [
            { from: 'input_embedding', to: 'rms_norm1' },
            { from: 'rms_norm1', to: 'rope_mha' },
            { from: 'rope_mha', to: 'rms_norm2', type: 'residual' },
            { from: 'rms_norm2', to: 'swiglu_ffn' },
            { from: 'swiglu_ffn', to: 'final_norm', type: 'residual' },
            { from: 'final_norm', to: 'linear' },
            { from: 'linear', to: 'softmax' },
        ]
    },

    llama2: {
        id: 'llama2',
        name: 'LLaMA 2',
        year: 2023,
        paper: 'Llama 2: Open Foundation and Fine-Tuned Chat Models',
        paperUrl: 'https://arxiv.org/abs/2307.09288',
        authors: 'Touvron et al. (Meta)',
        description: 'LLaMA 2 introduced Grouped Query Attention (GQA) which provides a balance between Multi-Head Attention and Multi-Query Attention, significantly reducing KV-cache memory during inference while maintaining model quality.',
        keyInnovation: 'Grouped Query Attention (GQA)',
        category: 'architecture',
        changes: [
            { type: 'modified', title: 'Grouped Query Attention', description: 'Groups query heads to share key-value heads, reducing KV-cache memory by 8x while maintaining quality.' },
            { type: 'modified', title: 'Context Length', description: 'Extended context length from 2K to 4K tokens through continued pre-training.' },
            { type: 'added', title: 'Ghost Attention', description: 'Instruction fine-tuning technique for multi-turn conversations (in chat models).' },
        ],
        components: [
            { id: 'input_embedding', type: 'embedding', x: 180, y: 520, width: 140, height: 45, label: 'Token Embedding', side: 'decoder' },
            { id: 'rms_norm1', type: 'norm', x: 180, y: 420, width: 140, height: 40, label: 'RMSNorm', sublabel: 'Pre-Norm', side: 'decoder' },
            { id: 'gqa', type: 'attention', x: 160, y: 335, width: 180, height: 60, label: 'Grouped Query Attn', sublabel: 'with RoPE', side: 'decoder' },
            { id: 'rms_norm2', type: 'norm', x: 180, y: 260, width: 140, height: 40, label: 'RMSNorm', sublabel: 'Pre-Norm', side: 'decoder' },
            { id: 'swiglu_ffn', type: 'ffn', x: 160, y: 180, width: 180, height: 55, label: 'SwiGLU FFN', side: 'decoder' },
            { id: 'decoder_block', type: 'block', x: 140, y: 160, width: 220, height: 325, label: 'Transformer Block', sublabel: '32x / 80x', side: 'decoder', isContainer: true },
            { id: 'final_norm', type: 'norm', x: 180, y: 95, width: 140, height: 40, label: 'Final RMSNorm', side: 'decoder' },
            { id: 'linear', type: 'output', x: 180, y: 35, width: 140, height: 40, label: 'LM Head', side: 'output' },
            { id: 'softmax', type: 'softmax', x: 180, y: -20, width: 140, height: 35, label: 'Softmax', side: 'output' },
        ],
        connections: [
            { from: 'input_embedding', to: 'rms_norm1' },
            { from: 'rms_norm1', to: 'gqa' },
            { from: 'gqa', to: 'rms_norm2', type: 'residual' },
            { from: 'rms_norm2', to: 'swiglu_ffn' },
            { from: 'swiglu_ffn', to: 'final_norm', type: 'residual' },
            { from: 'final_norm', to: 'linear' },
            { from: 'linear', to: 'softmax' },
        ]
    },

    rlhf: {
        id: 'rlhf',
        name: 'RLHF / PPO',
        year: 2022,
        paper: 'Training language models to follow instructions with human feedback',
        paperUrl: 'https://arxiv.org/abs/2203.02155',
        authors: 'Ouyang et al. (OpenAI)',
        description: 'Reinforcement Learning from Human Feedback (RLHF) with PPO is a training paradigm that aligns language models with human preferences. It involves training a reward model on human comparisons, then using PPO to optimize the LLM to maximize the reward while staying close to the original model.',
        keyInnovation: 'Human Preference Alignment via RL',
        category: 'training',
        changes: [
            { type: 'added', title: 'Reward Model', description: 'A separate model trained on human preference comparisons to score LLM outputs.' },
            { type: 'added', title: 'PPO Training', description: 'Proximal Policy Optimization updates the LLM to maximize reward while constraining divergence.' },
            { type: 'added', title: 'KL Penalty', description: 'Regularization term to prevent the model from diverging too far from the reference policy.' },
            { type: 'added', title: 'Value Head', description: 'Additional head for estimating expected future rewards (critic in actor-critic).' },
        ],
        components: [
            // Reference/SFT Model (frozen)
            { id: 'ref_model', type: 'block', x: 40, y: 380, width: 140, height: 180, label: 'Reference Model', sublabel: 'Frozen', side: 'reference', isContainer: true },
            { id: 'ref_embedding', type: 'embedding', x: 55, y: 510, width: 110, height: 35, label: 'Embedding', side: 'reference' },
            { id: 'ref_transformer', type: 'attention', x: 55, y: 450, width: 110, height: 45, label: 'Transformer', side: 'reference' },
            { id: 'ref_lm_head', type: 'output', x: 55, y: 395, width: 110, height: 35, label: 'LM Head', side: 'reference' },

            // Policy Model (trainable)
            { id: 'policy_model', type: 'block', x: 180, y: 320, width: 140, height: 240, label: 'Policy Model', sublabel: 'Trainable', side: 'policy', isContainer: true },
            { id: 'policy_embedding', type: 'embedding', x: 195, y: 510, width: 110, height: 35, label: 'Embedding', side: 'policy' },
            { id: 'policy_transformer', type: 'attention', x: 195, y: 440, width: 110, height: 50, label: 'Transformer', side: 'policy' },
            { id: 'policy_lm_head', type: 'output', x: 195, y: 380, width: 110, height: 35, label: 'LM Head', side: 'policy' },
            { id: 'value_head', type: 'output', x: 195, y: 335, width: 110, height: 30, label: 'Value Head', side: 'policy' },

            // Reward Model (frozen)
            { id: 'reward_model', type: 'block', x: 320, y: 380, width: 140, height: 180, label: 'Reward Model', sublabel: 'Frozen', side: 'reward', isContainer: true },
            { id: 'reward_embedding', type: 'embedding', x: 335, y: 510, width: 110, height: 35, label: 'Embedding', side: 'reward' },
            { id: 'reward_transformer', type: 'attention', x: 335, y: 450, width: 110, height: 45, label: 'Transformer', side: 'reward' },
            { id: 'reward_head', type: 'softmax', x: 335, y: 395, width: 110, height: 35, label: 'Reward Head', sublabel: 'Scalar', side: 'reward' },

            // PPO Components
            { id: 'kl_penalty', type: 'norm', x: 110, y: 280, width: 130, height: 35, label: 'KL Divergence', sublabel: 'Penalty', side: 'loss' },
            { id: 'advantage', type: 'ffn', x: 180, y: 220, width: 140, height: 40, label: 'Advantage', sublabel: 'Estimation', side: 'loss' },
            { id: 'ppo_loss', type: 'output', x: 180, y: 160, width: 140, height: 40, label: 'PPO Loss', sublabel: 'Clipped', side: 'loss' },

            // Generation flow
            { id: 'prompt', type: 'embedding', x: 180, y: 600, width: 140, height: 35, label: 'Prompt', side: 'input' },
            { id: 'response', type: 'ffn', x: 180, y: 100, width: 140, height: 35, label: 'Policy Gradient', sublabel: 'Update', side: 'output' },
        ],
        connections: [
            { from: 'prompt', to: 'ref_embedding' },
            { from: 'prompt', to: 'policy_embedding' },
            { from: 'ref_embedding', to: 'ref_transformer' },
            { from: 'ref_transformer', to: 'ref_lm_head' },
            { from: 'policy_embedding', to: 'policy_transformer' },
            { from: 'policy_transformer', to: 'policy_lm_head' },
            { from: 'policy_transformer', to: 'value_head' },
            { from: 'policy_lm_head', to: 'reward_embedding' },
            { from: 'reward_embedding', to: 'reward_transformer' },
            { from: 'reward_transformer', to: 'reward_head' },
            { from: 'ref_lm_head', to: 'kl_penalty' },
            { from: 'policy_lm_head', to: 'kl_penalty' },
            { from: 'reward_head', to: 'advantage' },
            { from: 'value_head', to: 'advantage' },
            { from: 'kl_penalty', to: 'ppo_loss' },
            { from: 'advantage', to: 'ppo_loss' },
            { from: 'ppo_loss', to: 'response' },
        ]
    },

    dpo: {
        id: 'dpo',
        name: 'DPO',
        year: 2023,
        paper: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model',
        paperUrl: 'https://arxiv.org/abs/2305.18290',
        authors: 'Rafailov et al. (Stanford)',
        description: 'Direct Preference Optimization (DPO) simplifies RLHF by eliminating the need for a separate reward model and RL training. It directly optimizes the policy using a classification loss on preference pairs, treating the language model itself as an implicit reward model.',
        keyInnovation: 'RL-Free Preference Learning',
        category: 'training',
        changes: [
            { type: 'removed', title: 'Reward Model', description: 'No separate reward model needed - the policy implicitly defines the reward.' },
            { type: 'removed', title: 'PPO/RL Training', description: 'Replaced complex RL with simple supervised learning on preference pairs.' },
            { type: 'modified', title: 'Training Objective', description: 'Binary cross-entropy loss on chosen vs rejected responses, weighted by implicit reward difference.' },
            { type: 'added', title: 'Reference Model', description: 'Frozen copy of initial model used to compute log probability ratios.' },
        ],
        components: [
            // Reference Model (frozen)
            { id: 'ref_model', type: 'block', x: 60, y: 320, width: 150, height: 220, label: 'Reference Model', sublabel: 'Frozen πref', side: 'reference', isContainer: true },
            { id: 'ref_embedding', type: 'embedding', x: 75, y: 490, width: 120, height: 35, label: 'Embedding', side: 'reference' },
            { id: 'ref_transformer', type: 'attention', x: 75, y: 420, width: 120, height: 50, label: 'Transformer', side: 'reference' },
            { id: 'ref_lm_head', type: 'output', x: 75, y: 355, width: 120, height: 35, label: 'LM Head', side: 'reference' },

            // Policy Model (trainable)
            { id: 'policy_model', type: 'block', x: 290, y: 320, width: 150, height: 220, label: 'Policy Model', sublabel: 'Trainable πθ', side: 'policy', isContainer: true },
            { id: 'policy_embedding', type: 'embedding', x: 305, y: 490, width: 120, height: 35, label: 'Embedding', side: 'policy' },
            { id: 'policy_transformer', type: 'attention', x: 305, y: 420, width: 120, height: 50, label: 'Transformer', side: 'policy' },
            { id: 'policy_lm_head', type: 'output', x: 305, y: 355, width: 120, height: 35, label: 'LM Head', side: 'policy' },

            // Input data
            { id: 'prompt', type: 'embedding', x: 175, y: 600, width: 150, height: 35, label: 'Prompt x', side: 'input' },
            { id: 'chosen', type: 'ffn', x: 100, y: 550, width: 120, height: 30, label: 'Chosen yw', side: 'input' },
            { id: 'rejected', type: 'ffn', x: 280, y: 550, width: 120, height: 30, label: 'Rejected yl', side: 'input' },

            // Log probability computation
            { id: 'ref_logprob', type: 'norm', x: 75, y: 280, width: 120, height: 30, label: 'log πref(y|x)', side: 'compute' },
            { id: 'policy_logprob', type: 'norm', x: 305, y: 280, width: 120, height: 30, label: 'log πθ(y|x)', side: 'compute' },

            // Implicit reward
            { id: 'implicit_reward', type: 'attention', x: 175, y: 200, width: 150, height: 50, label: 'Implicit Reward', sublabel: 'β log(πθ/πref)', side: 'compute' },

            // DPO Loss
            { id: 'dpo_loss', type: 'output', x: 175, y: 120, width: 150, height: 50, label: 'DPO Loss', sublabel: '-log σ(rw - rl)', side: 'loss' },

            // Gradient update
            { id: 'gradient', type: 'softmax', x: 175, y: 50, width: 150, height: 35, label: 'Gradient Update', side: 'output' },
        ],
        connections: [
            { from: 'prompt', to: 'chosen' },
            { from: 'prompt', to: 'rejected' },
            { from: 'chosen', to: 'ref_embedding' },
            { from: 'rejected', to: 'ref_embedding' },
            { from: 'chosen', to: 'policy_embedding' },
            { from: 'rejected', to: 'policy_embedding' },
            { from: 'ref_embedding', to: 'ref_transformer' },
            { from: 'ref_transformer', to: 'ref_lm_head' },
            { from: 'ref_lm_head', to: 'ref_logprob' },
            { from: 'policy_embedding', to: 'policy_transformer' },
            { from: 'policy_transformer', to: 'policy_lm_head' },
            { from: 'policy_lm_head', to: 'policy_logprob' },
            { from: 'ref_logprob', to: 'implicit_reward' },
            { from: 'policy_logprob', to: 'implicit_reward' },
            { from: 'implicit_reward', to: 'dpo_loss' },
            { from: 'dpo_loss', to: 'gradient' },
        ]
    },

    mixtral: {
        id: 'mixtral',
        name: 'Mixtral (MoE)',
        year: 2024,
        paper: 'Mixtral of Experts',
        paperUrl: 'https://arxiv.org/abs/2401.04088',
        authors: 'Mistral AI',
        description: 'Mixtral introduced Mixture of Experts (MoE) to the LLaMA-style architecture. Each FFN layer is replaced with 8 expert FFN networks, and a router selects the top-2 experts for each token, providing 8x more parameters while only using 2x compute.',
        keyInnovation: 'Sparse Mixture of Experts',
        category: 'architecture',
        changes: [
            { type: 'modified', title: 'Expert FFN Layers', description: 'Replaced single FFN with 8 expert FFNs. Router selects top-2 experts per token.' },
            { type: 'added', title: 'Router Network', description: 'Learned routing mechanism that decides which experts process each token.' },
            { type: 'added', title: 'Load Balancing Loss', description: 'Auxiliary loss to ensure experts are used evenly across tokens.' },
        ],
        components: [
            { id: 'input_embedding', type: 'embedding', x: 180, y: 540, width: 140, height: 45, label: 'Token Embedding', side: 'decoder' },
            { id: 'rms_norm1', type: 'norm', x: 180, y: 450, width: 140, height: 40, label: 'RMSNorm', side: 'decoder' },
            { id: 'gqa', type: 'attention', x: 160, y: 370, width: 180, height: 55, label: 'Grouped Query Attn', sublabel: 'Sliding Window', side: 'decoder' },
            { id: 'rms_norm2', type: 'norm', x: 180, y: 300, width: 140, height: 40, label: 'RMSNorm', side: 'decoder' },
            { id: 'router', type: 'output', x: 180, y: 240, width: 140, height: 35, label: 'Router', sublabel: 'Top-2', side: 'decoder' },
            { id: 'experts', type: 'ffn', x: 140, y: 150, width: 220, height: 70, label: 'Expert FFNs (8x)', sublabel: 'SwiGLU', side: 'decoder' },
            { id: 'decoder_block', type: 'block', x: 120, y: 130, width: 260, height: 390, label: 'MoE Block', sublabel: '32x', side: 'decoder', isContainer: true },
            { id: 'final_norm', type: 'norm', x: 180, y: 65, width: 140, height: 40, label: 'Final RMSNorm', side: 'decoder' },
            { id: 'linear', type: 'output', x: 180, y: 5, width: 140, height: 40, label: 'LM Head', side: 'output' },
            { id: 'softmax', type: 'softmax', x: 180, y: -50, width: 140, height: 35, label: 'Softmax', side: 'output' },
        ],
        connections: [
            { from: 'input_embedding', to: 'rms_norm1' },
            { from: 'rms_norm1', to: 'gqa' },
            { from: 'gqa', to: 'rms_norm2', type: 'residual' },
            { from: 'rms_norm2', to: 'router' },
            { from: 'router', to: 'experts' },
            { from: 'experts', to: 'final_norm', type: 'residual' },
            { from: 'final_norm', to: 'linear' },
            { from: 'linear', to: 'softmax' },
        ]
    },

    deepseek: {
        id: 'deepseek',
        name: 'DeepSeek V2',
        year: 2024,
        paper: 'DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model',
        paperUrl: 'https://arxiv.org/abs/2405.04434',
        authors: 'DeepSeek AI',
        description: 'DeepSeek V2 introduced Multi-head Latent Attention (MLA) which compresses KV cache by projecting keys and values into a lower-dimensional latent space. Combined with DeepSeekMoE architecture using fine-grained experts and shared experts, it achieves superior efficiency.',
        keyInnovation: 'Multi-head Latent Attention (MLA)',
        category: 'architecture',
        changes: [
            { type: 'added', title: 'Multi-head Latent Attention', description: 'Projects KV into low-rank latent space, reducing KV cache by 93.3% while maintaining quality.' },
            { type: 'modified', title: 'Fine-grained MoE', description: 'Uses 160 fine-grained experts (vs 8 in Mixtral) with top-6 routing for better specialization.' },
            { type: 'added', title: 'Shared Experts', description: '2 shared experts that are always activated, capturing common knowledge.' },
            { type: 'added', title: 'Device-Limited Routing', description: 'Ensures each token is only routed to experts on a limited number of devices.' },
        ],
        components: [
            { id: 'input_embedding', type: 'embedding', x: 180, y: 580, width: 140, height: 45, label: 'Token Embedding', side: 'decoder' },
            { id: 'rms_norm1', type: 'norm', x: 180, y: 505, width: 140, height: 35, label: 'RMSNorm', side: 'decoder' },

            // MLA components
            { id: 'mla_compress', type: 'attention', x: 140, y: 430, width: 100, height: 50, label: 'KV Compress', sublabel: 'd→dₗ', side: 'decoder' },
            { id: 'mla_attention', type: 'attention', x: 260, y: 430, width: 100, height: 50, label: 'MLA', sublabel: 'Attention', side: 'decoder' },
            { id: 'mla_block', type: 'block', x: 125, y: 415, width: 250, height: 80, label: 'Multi-head Latent Attention', sublabel: '', side: 'decoder', isContainer: true },

            { id: 'rms_norm2', type: 'norm', x: 180, y: 350, width: 140, height: 35, label: 'RMSNorm', side: 'decoder' },

            // DeepSeekMoE
            { id: 'router', type: 'output', x: 180, y: 295, width: 140, height: 30, label: 'Router', sublabel: 'Top-6', side: 'decoder' },
            { id: 'shared_experts', type: 'ffn', x: 100, y: 225, width: 100, height: 50, label: 'Shared (2x)', sublabel: 'Always On', side: 'decoder' },
            { id: 'routed_experts', type: 'ffn', x: 220, y: 225, width: 120, height: 50, label: 'Routed (160x)', sublabel: 'Sparse', side: 'decoder' },
            { id: 'moe_block', type: 'block', x: 85, y: 205, width: 270, height: 135, label: 'DeepSeekMoE', sublabel: '', side: 'decoder', isContainer: true },

            { id: 'decoder_block', type: 'block', x: 70, y: 185, width: 300, height: 345, label: 'DeepSeek Block', sublabel: '60x', side: 'decoder', isContainer: true },
            { id: 'final_norm', type: 'norm', x: 180, y: 115, width: 140, height: 35, label: 'Final RMSNorm', side: 'decoder' },
            { id: 'linear', type: 'output', x: 180, y: 55, width: 140, height: 40, label: 'LM Head', side: 'output' },
            { id: 'softmax', type: 'softmax', x: 180, y: 0, width: 140, height: 35, label: 'Softmax', side: 'output' },
        ],
        connections: [
            { from: 'input_embedding', to: 'rms_norm1' },
            { from: 'rms_norm1', to: 'mla_compress' },
            { from: 'mla_compress', to: 'mla_attention' },
            { from: 'mla_attention', to: 'rms_norm2', type: 'residual' },
            { from: 'rms_norm2', to: 'router' },
            { from: 'router', to: 'shared_experts' },
            { from: 'router', to: 'routed_experts' },
            { from: 'shared_experts', to: 'final_norm', type: 'residual' },
            { from: 'routed_experts', to: 'final_norm', type: 'residual' },
            { from: 'final_norm', to: 'linear' },
            { from: 'linear', to: 'softmax' },
        ]
    },

    qwen2: {
        id: 'qwen2',
        name: 'Qwen 2.5',
        year: 2024,
        paper: 'Qwen2.5 Technical Report',
        paperUrl: 'https://arxiv.org/abs/2412.15115',
        authors: 'Qwen Team (Alibaba)',
        description: 'Qwen 2.5 builds on modern transformer practices with several refinements: QKV bias for improved training stability, dual chunk attention for long contexts, and YaRN-based RoPE scaling. It also features SwiGLU with expanded intermediate size.',
        keyInnovation: 'QKV Bias + Dual Chunk Attention',
        category: 'architecture',
        changes: [
            { type: 'added', title: 'QKV Bias', description: 'Adds learnable bias to Q, K, V projections for better training dynamics.' },
            { type: 'added', title: 'Dual Chunk Attention', description: 'Processes long sequences in chunks with local and global attention patterns.' },
            { type: 'modified', title: 'YaRN RoPE', description: 'Uses Yet another RoPE extension for better long-context generalization.' },
            { type: 'modified', title: 'Expanded FFN', description: 'Uses larger intermediate size (13824 for 7B) for increased capacity.' },
        ],
        components: [
            { id: 'input_embedding', type: 'embedding', x: 180, y: 540, width: 140, height: 45, label: 'Token Embedding', sublabel: '151k vocab', side: 'decoder' },
            { id: 'rms_norm1', type: 'norm', x: 180, y: 460, width: 140, height: 35, label: 'RMSNorm', sublabel: 'Pre-Norm', side: 'decoder' },
            { id: 'gqa_bias', type: 'attention', x: 160, y: 375, width: 180, height: 60, label: 'GQA + Bias', sublabel: 'YaRN RoPE', side: 'decoder' },
            { id: 'rms_norm2', type: 'norm', x: 180, y: 300, width: 140, height: 35, label: 'RMSNorm', sublabel: 'Pre-Norm', side: 'decoder' },
            { id: 'swiglu_expanded', type: 'ffn', x: 160, y: 215, width: 180, height: 60, label: 'SwiGLU FFN', sublabel: 'Expanded', side: 'decoder' },
            { id: 'decoder_block', type: 'block', x: 140, y: 195, width: 220, height: 325, label: 'Transformer Block', sublabel: '28x-80x', side: 'decoder', isContainer: true },
            { id: 'final_norm', type: 'norm', x: 180, y: 130, width: 140, height: 35, label: 'Final RMSNorm', side: 'decoder' },
            { id: 'linear', type: 'output', x: 180, y: 65, width: 140, height: 40, label: 'LM Head', sublabel: 'Tied Weights', side: 'output' },
            { id: 'softmax', type: 'softmax', x: 180, y: 10, width: 140, height: 35, label: 'Softmax', side: 'output' },
        ],
        connections: [
            { from: 'input_embedding', to: 'rms_norm1' },
            { from: 'rms_norm1', to: 'gqa_bias' },
            { from: 'gqa_bias', to: 'rms_norm2', type: 'residual' },
            { from: 'rms_norm2', to: 'swiglu_expanded' },
            { from: 'swiglu_expanded', to: 'final_norm', type: 'residual' },
            { from: 'final_norm', to: 'linear' },
            { from: 'linear', to: 'softmax' },
        ]
    },

    llama3: {
        id: 'llama3',
        name: 'LLaMA 3.1',
        year: 2024,
        paper: 'The Llama 3 Herd of Models',
        paperUrl: 'https://arxiv.org/abs/2407.21783',
        authors: 'Meta AI',
        description: 'LLaMA 3.1 scales to 405B parameters with 128K context length. It uses standard dense attention throughout (no MoE), with improved data quality and longer training. The 405B model achieves GPT-4 level performance on most benchmarks.',
        keyInnovation: '128K Context + 405B Dense Model',
        category: 'architecture',
        changes: [
            { type: 'modified', title: '128K Context', description: 'Extended context to 128K through continued pre-training with progressively longer sequences.' },
            { type: 'modified', title: 'Larger Vocabulary', description: 'Expanded vocabulary to 128K tokens with improved tokenizer efficiency.' },
            { type: 'modified', title: 'Scale', description: '405B parameters with 126 layers - largest dense open model.' },
            { type: 'added', title: 'Tool Use Training', description: 'Trained on tool use, including code execution and web browsing.' },
        ],
        components: [
            { id: 'input_embedding', type: 'embedding', x: 180, y: 520, width: 140, height: 45, label: 'Token Embedding', sublabel: '128K vocab', side: 'decoder' },
            { id: 'rms_norm1', type: 'norm', x: 180, y: 440, width: 140, height: 35, label: 'RMSNorm', sublabel: 'Pre-Norm', side: 'decoder' },
            { id: 'gqa_128k', type: 'attention', x: 160, y: 360, width: 180, height: 55, label: 'GQA', sublabel: '128K Context RoPE', side: 'decoder' },
            { id: 'rms_norm2', type: 'norm', x: 180, y: 285, width: 140, height: 35, label: 'RMSNorm', sublabel: 'Pre-Norm', side: 'decoder' },
            { id: 'swiglu_ffn', type: 'ffn', x: 160, y: 205, width: 180, height: 55, label: 'SwiGLU FFN', side: 'decoder' },
            { id: 'decoder_block', type: 'block', x: 140, y: 185, width: 220, height: 295, label: 'Transformer Block', sublabel: '126x', side: 'decoder', isContainer: true },
            { id: 'final_norm', type: 'norm', x: 180, y: 120, width: 140, height: 35, label: 'Final RMSNorm', side: 'decoder' },
            { id: 'linear', type: 'output', x: 180, y: 55, width: 140, height: 40, label: 'LM Head', side: 'output' },
            { id: 'softmax', type: 'softmax', x: 180, y: 0, width: 140, height: 35, label: 'Softmax', side: 'output' },
        ],
        connections: [
            { from: 'input_embedding', to: 'rms_norm1' },
            { from: 'rms_norm1', to: 'gqa_128k' },
            { from: 'gqa_128k', to: 'rms_norm2', type: 'residual' },
            { from: 'rms_norm2', to: 'swiglu_ffn' },
            { from: 'swiglu_ffn', to: 'final_norm', type: 'residual' },
            { from: 'final_norm', to: 'linear' },
            { from: 'linear', to: 'softmax' },
        ]
    },

    deepseekr1: {
        id: 'deepseekr1',
        name: 'DeepSeek R1',
        year: 2025,
        paper: 'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning',
        paperUrl: 'https://arxiv.org/abs/2501.12948',
        authors: 'DeepSeek AI',
        description: 'DeepSeek R1 demonstrates that reasoning capabilities can emerge through pure RL without supervised fine-tuning on reasoning data. Using GRPO (Group Relative Policy Optimization), the model develops chain-of-thought reasoning and self-verification behaviors.',
        keyInnovation: 'Emergent Reasoning via Pure RL',
        category: 'training',
        changes: [
            { type: 'added', title: 'GRPO Algorithm', description: 'Group Relative Policy Optimization - estimates baselines from group samples instead of critic.' },
            { type: 'added', title: 'Cold Start Strategy', description: 'Initial RL without SFT, letting reasoning patterns emerge naturally.' },
            { type: 'added', title: 'Multi-stage Training', description: 'Pure RL → Rejection sampling → RL refinement pipeline.' },
            { type: 'modified', title: 'Reward Design', description: 'Combines accuracy reward, format reward, and length penalty.' },
        ],
        components: [
            // Base DeepSeek V3 Model
            { id: 'base_model', type: 'block', x: 130, y: 350, width: 240, height: 220, label: 'DeepSeek V3 Base', sublabel: '671B MoE', side: 'model', isContainer: true },
            { id: 'mla', type: 'attention', x: 160, y: 510, width: 180, height: 45, label: 'MLA + MoE', sublabel: 'Transformer', side: 'model' },
            { id: 'reasoning_head', type: 'output', x: 180, y: 440, width: 140, height: 40, label: 'Reasoning Head', sublabel: '<think> tokens', side: 'model' },
            { id: 'answer_head', type: 'output', x: 180, y: 375, width: 140, height: 40, label: 'Answer Head', side: 'model' },

            // GRPO Training
            { id: 'prompt_group', type: 'embedding', x: 180, y: 620, width: 140, height: 35, label: 'Prompt', side: 'input' },
            { id: 'group_samples', type: 'ffn', x: 130, y: 280, width: 240, height: 50, label: 'Group Samples (G)', sublabel: 'Multiple Responses', side: 'sampling' },

            // Reward components
            { id: 'accuracy_reward', type: 'softmax', x: 80, y: 195, width: 110, height: 40, label: 'Accuracy', sublabel: 'Reward', side: 'reward' },
            { id: 'format_reward', type: 'softmax', x: 195, y: 195, width: 110, height: 40, label: 'Format', sublabel: 'Reward', side: 'reward' },
            { id: 'length_penalty', type: 'softmax', x: 310, y: 195, width: 110, height: 40, label: 'Length', sublabel: 'Penalty', side: 'reward' },

            // GRPO Loss
            { id: 'group_baseline', type: 'norm', x: 130, y: 125, width: 120, height: 35, label: 'Group Baseline', sublabel: 'mean(rewards)', side: 'loss' },
            { id: 'grpo_loss', type: 'output', x: 260, y: 125, width: 120, height: 35, label: 'GRPO Loss', side: 'loss' },

            { id: 'gradient_update', type: 'attention', x: 180, y: 50, width: 140, height: 40, label: 'Policy Update', side: 'output' },
        ],
        connections: [
            { from: 'prompt_group', to: 'mla' },
            { from: 'mla', to: 'reasoning_head' },
            { from: 'reasoning_head', to: 'answer_head' },
            { from: 'answer_head', to: 'group_samples' },
            { from: 'group_samples', to: 'accuracy_reward' },
            { from: 'group_samples', to: 'format_reward' },
            { from: 'group_samples', to: 'length_penalty' },
            { from: 'accuracy_reward', to: 'group_baseline' },
            { from: 'format_reward', to: 'group_baseline' },
            { from: 'length_penalty', to: 'group_baseline' },
            { from: 'group_baseline', to: 'grpo_loss' },
            { from: 'grpo_loss', to: 'gradient_update' },
        ]
    },
};

// Order for timeline - separate by category for better organization
const ARCHITECTURE_ORDER = [
    'original', 'gpt1', 'bert', 'gpt2',
    'rlhf',
    'llama', 'llama2', 'dpo',
    'mixtral', 'deepseek', 'qwen2', 'llama3',
    'deepseekr1'
];

// Component colors
const COMPONENT_COLORS = {
    embedding: { fill: '#3b82f6', stroke: '#2563eb' },
    positional: { fill: '#06b6d4', stroke: '#0891b2' },
    attention: { fill: '#8b5cf6', stroke: '#7c3aed' },
    ffn: { fill: '#22c55e', stroke: '#16a34a' },
    norm: { fill: '#f59e0b', stroke: '#d97706' },
    output: { fill: '#ef4444', stroke: '#dc2626' },
    softmax: { fill: '#f97316', stroke: '#ea580c' },
    block: { fill: 'transparent', stroke: '#6366f1' },
};

// Category labels for timeline grouping
const CATEGORY_LABELS = {
    architecture: 'Architecture',
    training: 'Training',
    moe: 'Mixture of Experts'
};
