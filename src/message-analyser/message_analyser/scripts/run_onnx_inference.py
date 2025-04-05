#!/usr/bin/env python3
"""
ONNX Runtime inference implementation for Gemma-2B model to extract criminal activity from conversations.

This optimized back-end implementation uses an exported ONNX model for efficient inference
with proper resource management and error handling.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gemma-onnx-inference')

class GemmaOnnxInference:
    """
    Optimized ONNX inference engine for Gemma-2B model with focus on text generation
    for criminal activity extraction from conversations.
    """
    
    def __init__(
        self,
        model_path: str = "models/gemma-2b.onnx",
        tokenizer_name: str = "google/gemma-2b",
        max_length: int = 1024,
        num_threads: int = 4
    ):
        """
        Initialize the ONNX inference engine.
        
        Args:
            model_path: Path to the ONNX model file
            tokenizer_name: Hugging Face tokenizer name or path
            max_length: Maximum sequence length for generation
            num_threads: Number of threads for inference
        """
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.num_threads = num_threads
        
        # Initialize components
        self._validate_model()
        self.tokenizer = self._load_tokenizer()
        self.session = self._create_session()
        
        logger.info(f"Gemma ONNX inference engine initialized with model: {model_path}")
    
    def _validate_model(self) -> None:
        """Validate that the ONNX model file exists."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found at: {self.model_path}")
        
        # Check for external data files
        model_dir = os.path.dirname(self.model_path)
        data_file = os.path.join(model_dir, "gemma-2b_data.bin")
        weights_file = os.path.join(model_dir, "gemma-2b-weights.bin")
        
        if not (os.path.exists(data_file) or os.path.exists(weights_file)):
            logger.warning(
                "External data files not found. Model may not load correctly. "
                "Ensure external data files are present in the model directory."
            )
    
    def _load_tokenizer(self):
        """Load the Hugging Face tokenizer for the model."""
        try:
            logger.info(f"Loading tokenizer: {self.tokenizer_name}")
            return AutoTokenizer.from_pretrained(self.tokenizer_name)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise RuntimeError(f"Tokenizer initialization failed: {str(e)}")
    
    def _create_session(self):
        """Create an optimized ONNX Runtime session."""
        try:
            # Configure session options for optimal performance
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = self.num_threads
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Enable memory optimizations
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
            
            # Select appropriate execution providers
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
                logger.info("Using CUDA acceleration for inference")
            
            # Create and return the session
            logger.info(f"Creating ONNX Runtime session with providers: {providers}")
            return ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {str(e)}")
            raise RuntimeError(f"ONNX session initialization failed: {str(e)}")
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the ONNX model for a single prompt.
        
        Args:
            prompt: The input prompt for text generation
            
        Returns:
            Generated text response
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Convert to numpy arrays for ONNX Runtime
            onnx_inputs = {
                name: tensor.numpy() for name, tensor in inputs.items()
            }
            
            # Run inference
            start_time = time.time()
            outputs = self.session.run(None, onnx_inputs)
            inference_time = time.time() - start_time
            
            # Process outputs - get logits from the model output
            logits = outputs[0]
            
            # Get the most likely tokens (simple greedy decoding)
            predictions = np.argmax(logits, axis=-1)
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
            
            logger.info(f"Text generated in {inference_time:.2f}s")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return f"Error generating text: {str(e)}"
    
    def extract_criminal_activity(self, conversation: str) -> str:
        """
        Format the prompt and extract criminal activity from a conversation.
        
        Args:
            conversation: The conversation text to analyze
            
        Returns:
            Generated analysis of criminal elements in the conversation
        """
        # Format the prompt with the criminal activity extraction template
        prompt_template = """You are an AI crime analyst. Given the following conversation, identify potential crime elements within each message and categorize them based on the following categories:
            1. Actus Reus (Conduct) - A direct criminal act or an unlawful omission.
            2. Mens Rea (Intent) - A guilty mindset, including general intent, specific intent, motive, or knowledge.
            3. Concurrence - A combination of Actus Reus and Mens Rea.
            4. Causation (Harm) - A direct or indirect link between an act and its consequences.

        For each message, output in the format:
        Message from [Speaker]: "[Message]" | Crime element: [Category]

        If a message does not contain a crime element, output:
        Message from [Speaker]: "[Message]" | Crime element: None

        Conversation:
        {0}""".format(conversation)
        
        # Generate text with the formatted prompt
        return self.generate_text(prompt_template)
    
    def process_conversations_file(self, csv_path: str, output_path: Optional[str] = None) -> List[str]:
        """
        Process a CSV file containing conversations and generate criminal activity extractions.
        
        Args:
            csv_path: Path to the CSV file with conversations
            output_path: Optional path to save results to a CSV file
            
        Returns:
            List of generated extractions
        """
        try:
            # Load the CSV file
            logger.info(f"Loading conversations from {csv_path}")
            df = pd.read_csv(csv_path)
            
            if "conversation" not in df.columns:
                raise ValueError("CSV file must contain a 'conversation' column")
            
            # Extract conversations
            conversations = df["conversation"].tolist()
            
            # Process each conversation
            generated_results = []
            for i, conversation in enumerate(conversations):
                #if(i==2): break
                logger.info(f"Processing conversation {i+1}/{len(conversations)}")
                
                # Generate extraction for this conversation
                result = self.extract_criminal_activity(conversation)
                generated_results.append(result)
                
                # Log progress
                logger.info(f"Completed conversation {i+1}")
                print(f"\nGenerated extraction for conversation {i+1}:\n{result}\n")
            
            # Save results if output path is provided
            if output_path:
                output_df = df.copy()
                output_df["generated_by_gemma_onnx"] = generated_results
                output_df.to_csv(output_path, index=False)
                logger.info(f"Results saved to {output_path}")
            
            return generated_results
            
        except Exception as e:
            logger.error(f"Error processing conversations file: {str(e)}")
            raise

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gemma ONNX Inference for Criminal Activity Extraction")
    parser.add_argument("--model", type=str, default="models/gemma-2b.onnx", help="Path to ONNX model")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with conversations")
    parser.add_argument("--output", type=str, help="Path to save output CSV")
    parser.add_argument("--threads", type=int, default=4, help="Number of inference threads")
    args = parser.parse_args()
    
    # Initialize the inference engine
    engine = GemmaOnnxInference(
        model_path=args.model,
        num_threads=args.threads
    )
    
    # Process the conversations
    engine.process_conversations_file(args.csv, args.output)

if __name__ == "__main__":
    main()