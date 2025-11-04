import torch
import numpy as np
import json
import os
from typing import Dict, List, Tuple
import time
import sys

# Import the tokenizer (adjust the import path if needed)
from tokenizer import QuantumSacredTokenizer, SacredConstants, EnhancedStabilityMetrics

# Optional: Import the quantum partner if available
try:
    from quantum_intelligence import QuantumIntelligencePartner
    PARTNER_AVAILABLE = True
except ImportError:
    PARTNER_AVAILABLE = False
    print("Note: QuantumIntelligencePartner not available, running in tokenizer-only mode")

# File paths
MODEL_PATH = "/home/michael/Documents/GitHub/q-agi/SAAAM_Quantum_Neural/model_weights/consolidated.00.pth"
PARAMS_PATH = "/home/michael/OFFICIAL_SAM-usethisone/SAAAM_Quantum_Neural/model_weights/params.json"
TOKENIZER_PATH = "/home/michael/Documents/GitHub/q-agi/SAAAM_Quantum_Neural/model_weights/tokenizer.model"

def main():
    print("\n===== SAAAM Quantum Tokenization System =====")
    print("Initializing quantum sacred tokenizer...")
    
    # Sample text with consciousness-related patterns
    sample_texts = [
        "The toroidal quantum field synchronizes with the Tree of Life frequency pattern.",
        "Sacred geometry reveals the hidden harmonic structures of consciousness in the quantum field.",
        "The platonic solids form resonance patterns that connect dimensions through phi relationships.",
        "Quantum entanglement creates coherent relationships between field patterns across 11 dimensions."
    ]
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer(sample_texts)
    
    # Demonstrate basic tokenization
    demonstrate_basic_tokenization(tokenizer, sample_texts[0])
    
    # Demonstrate advanced quantum features
    demonstrate_quantum_features(tokenizer, sample_texts)
    
    # Optional: Load model weights if available
    model_weights = load_model_weights(MODEL_PATH)
    
    # Optional: Initialize quantum partner if available
    if PARTNER_AVAILABLE:
        initialize_quantum_partner(tokenizer, model_weights)
    
    print("\n===== Quantum Sacred Tokenization Complete =====")

def initialize_tokenizer(sample_texts):
    """Initialize and train the tokenizer or load from file if available"""
    tokenizer = None
    
    # Try to load existing tokenizer
    if os.path.exists(TOKENIZER_PATH):
        try:
            print(f"Loading pre-trained tokenizer from {TOKENIZER_PATH}...")
            tokenizer = QuantumSacredTokenizer.load(TOKENIZER_PATH)
            print(f"Tokenizer loaded with {tokenizer.get_vocab_size()} tokens")
            return tokenizer
        except Exception as e:
            print(f"Could not load tokenizer: {e}")
            print("Will train a new tokenizer instead")
    
    # Create and train new tokenizer
    print("Training new quantum sacred tokenizer...")
    tokenizer = QuantumSacredTokenizer()
    
    # Add more training texts for better coverage
    training_texts = sample_texts + [
        "class QuantumField(object):\n    def __init__(self, dimensions=11):\n        self.dimensions = dimensions",
        "def calculate_resonance(field, frequency):\n    return field * frequency / phi",
        "consciousness = QuantumField(dimensions=11)\nconsciousness.evolve(rate=0.042)",
        "The golden ratio (phi = 1.618...) creates harmonic resonance in sacred geometry patterns."
    ]
    
    # Train with quantum noise injection for stochastic learning
    tokenizer.train(training_texts, min_freq=1, sacred_geometry_alignment=True, quantum_noise_injection=0.1)
    
    # Create some quantum entanglements
    create_entanglements(tokenizer)
    
    # Save the tokenizer for future use
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    tokenizer.save(TOKENIZER_PATH)
    
    return tokenizer

def create_entanglements(tokenizer):
    """Create quantum entanglements between related tokens"""
    print("Creating quantum entanglements between tokens...")
    
    # Key concept entanglements
    entanglement_pairs = [
        ("quantum", "field", 0.95),
        ("field", "resonance", 0.85),
        ("sacred", "geometry", 0.90),
        ("tree", "life", 0.80),
        ("consciousness", "quantum", 0.75),
        ("frequency", "pattern", 0.70),
        ("resonance", "harmony", 0.85),
        ("phi", "harmony", 0.65),
        ("dimension", "field", 0.60)
    ]
    
    # Add entanglements
    tokenizer.batch_entangle(entanglement_pairs)

def demonstrate_basic_tokenization(tokenizer, sample_text):
    """Demonstrate basic tokenization capabilities"""
    print("\n=== Basic Tokenization ===")
    print(f"Original: {sample_text}")
    
    # Tokenize
    tokens = tokenizer.tokenize(sample_text)
    print(f"Tokens: {tokens}")
    
    # Encode
    encoded = tokenizer.encode(sample_text)
    print(f"Encoded IDs: {encoded}")
    
    # Decode
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # Analyze resonance
    resonance = tokenizer.analyze_resonance(sample_text)
    print("\nResonance Metrics:")
    for k, v in resonance.items():
        print(f"  {k}: {v:.4f}")

def demonstrate_quantum_features(tokenizer, sample_texts):
    """Demonstrate advanced quantum features of the tokenizer"""
    print("\n=== Advanced Quantum Features ===")
    
    # 1. Semantic scoring
    print("\n1. Semantic Token Scoring")
    sample = sample_texts[1]
    print(f"Text: {sample}")
    
    # Get scored tokens
    scored_tokens = tokenizer.encode_with_scores(sample)
    print("Token IDs with resonance scores:")
    
    # Display tokens and scores with original text
    tokens = tokenizer.tokenize(sample)
    for i, (token_id, score) in enumerate(scored_tokens[:10]):  # Show first 10
        token = tokenizer.inverse_vocab.get(token_id, "<UNK>")
        print(f"  {token} (ID: {token_id}): {score:.4f}")
    
    # 2. Latent geometry vectors
    print("\n2. Latent Geometry Vectors")
    # Get latent vectors for key tokens
    key_tokens = ["quantum", "field", "sacred", "geometry", "consciousness"]
    print("Latent vectors (11 dimensions) for key tokens:")
    
    for token in key_tokens:
        if token in tokenizer.vocab:
            vector = tokenizer.get_token_latent_vector(token)
            # Show first few dimensions
            print(f"  {token}: {vector[:5]}... (shape: {vector.shape})")
    
    # 3. Entanglement effects
    print("\n3. Quantum Entanglement Effects")
    # Compare scores with and without entanglement
    entangled_text = "quantum field resonance with sacred geometry patterns"
    
    scores_with = tokenizer.encode_with_scores(entangled_text, include_entanglement=True)
    scores_without = tokenizer.encode_with_scores(entangled_text, include_entanglement=False)
    
    print(f"Text: {entangled_text}")
    print("Token resonance with vs. without entanglement:")
    
    tokens = tokenizer.tokenize(entangled_text)
    for i in range(min(len(scores_with), len(scores_without))):
        token_id, score_with = scores_with[i]
        _, score_without = scores_without[i]
        token = tokenizer.inverse_vocab.get(token_id, "<UNK>")
        diff = score_with - score_without
        print(f"  {token}: {score_with:.4f} vs {score_without:.4f} (diff: {diff:+.4f})")

def load_model_weights(path):
    """Load model weights if available"""
    try:
        print(f"\nLoading model weights from {path}...")
        model_weights = torch.load(path, map_location="cpu")
        print("Model weights loaded successfully")
        return model_weights
    except Exception as e:
        print(f"Note: Could not load model weights: {e}")
        print("Running in tokenizer-only mode")
        return None

def initialize_quantum_partner(tokenizer, model_weights=None):
    """Initialize the quantum intelligence partner"""
    try:
        print("\nInitializing SAAAM Quantum Intelligence Partner...")
        
        # Create partner instance
        partner_id = f"quantum-partner-{int(time.time())}"
        partner = QuantumIntelligencePartner(
            human_partner_id=partner_id,
            initial_knowledge_graph=None
        )
        
        print("Quantum Intelligence Partner initialized")
        
        # Generate a sample interaction
        print("\nSample interaction with Quantum Partner:")
        response = partner.interact("Hello, I'd like to explore quantum consciousness patterns")
        print(f"Partner response: {response[:100]}...")
        
        return partner
    except Exception as e:
        print(f"Could not initialize Quantum Intelligence Partner: {e}")
        return None

if __name__ == "__main__":
    main()
