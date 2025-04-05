from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import onnx
from pathlib import Path
from huggingface_hub import login

def export_to_onnx():
    # Hugging Face token

    login()#Add token ="hf_your_token" if needed


    # Model configuration
    model_name = "google/gemma-2b"
    model_dir = Path("models")
    onnx_model_path = model_dir / "gemma-2b.onnx"
    external_data_path = model_dir / "gemma-2b_data.bin"
    
    try:
        # Ensure output directory exists
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load model and tokenizer with memory optimization
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.eval()

        # Create proper dummy inputs
        dummy_inputs = tokenizer("Sample input", return_tensors="pt")
        
        # Create a wrapper class to handle complex outputs
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, input_ids, attention_mask):
                with torch.no_grad():  # Prevent gradient computation
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.logits
                
        # Create wrapped model
        wrapped_model = ModelWrapper(model)
        
        # Single-step export with proper external data configuration
        torch.onnx.export(
            wrapped_model,
            args=tuple(dummy_inputs.values()),
            f=str(onnx_model_path),
            input_names=list(dummy_inputs.keys()),
            output_names=["logits"],
            dynamic_axes={
                **{k: {0: "batch_size", 1: "sequence_length"} for k in dummy_inputs.keys()},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            use_external_data_format=True,
            verbose=False,  # Set to True only for debugging
            keep_initializers_as_inputs=False  # Better for inference optimization
        )
        
        # Clean up memory before loading ONNX model
        del model, wrapped_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc
        gc.collect()
        
        # Consolidate external data (if needed)
        onnx_model = onnx.load(str(onnx_model_path), load_external_data=False)
        from onnx.external_data_helper import convert_model_to_external_data
        
        # Use absolute path for reliable loading
        convert_model_to_external_data(
            onnx_model,
            all_tensors_to_one_file=True,
            location=str(external_data_path.name),  # Use filename only, not path
            size_threshold=1024
        )
        onnx.save(onnx_model, str(onnx_model_path), save_as_external_data=True)
        
        # Verify model integrity
        verify_onnx_model(onnx_model_path)
        
        print(f"Model successfully exported to {onnx_model_path}")
        return True
    
    except Exception as e:
        print(f"Error exporting model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def verify_onnx_model(onnx_model_path="models/gemma-2b.onnx"):
    onnx.checker.check_model(onnx_model_path)
    print("ONNX model verification successful")

def export_with_consolidated_tensors():
    """Export ONNX model with all tensors consolidated into a single file"""
    import onnx
    from pathlib import Path
    import shutil
    import os
    
    # Define paths
    model_dir = Path("models")
    onnx_model_path = model_dir / "gemma-2b.onnx"
    external_data_path = model_dir / "gemma-2b-weights.bin"
    
    # Step 1: Create a temporary directory for clean export
    temp_dir = model_dir / "temp_export"
    temp_dir.mkdir(exist_ok=True)
    temp_model_path = temp_dir / "model.onnx"
    
    try:
        # Step 2: Load existing model if available, otherwise export new one
        if onnx_model_path.exists():
            # Load existing model with all external data
            model = onnx.load(str(onnx_model_path), load_external_data=True)
        else:
            # Export new model (your existing export code)
            # ...existing export code...
            model = onnx.load(str(onnx_model_path))
        
        # Step 3: Force consolidation to a single external data file
        from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
        
        # Ensure all tensors are loaded into the model
        if onnx_model_path.exists():
            load_external_data_for_model(model, str(model_dir))
        
        # Save with consolidated external data
        convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location=external_data_path.name,
            size_threshold=0  # Force all tensors to external storage
        )
        
        # Step 4: Save to temporary location first
        onnx.save(
            model, 
            str(temp_model_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path.name
        )
        
        # Step 5: Verify the temporary model has correct structure
        if not (temp_dir / external_data_path.name).exists():
            raise ValueError("Failed to consolidate tensors to a single file")
            
        # Step 6: Replace original files with optimized version
        if onnx_model_path.exists():
            onnx_model_path.unlink()
        shutil.copy2(str(temp_model_path), str(onnx_model_path))
        
        if external_data_path.exists():
            external_data_path.unlink()
        shutil.copy2(str(temp_dir / external_data_path.name), str(external_data_path))
        
        # Step 7: Clean up individual tensor files
        for file in model_dir.glob("onnx__MatMul_*"):
            file.unlink()
            
        print(f"Model successfully exported with consolidated tensors")
        print(f"- Model file: {onnx_model_path}")
        print(f"- Weights file: {external_data_path}")
        
        return True
    
    except Exception as e:
        print(f"Error consolidating tensors: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir))

def onnx_inference(prompt):
    import onnxruntime as ort
    import numpy as np
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    
    # Preprocess input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].numpy()
    attention_mask = inputs["attention_mask"].numpy()
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession("models/gemma-2b.onnx")
    
    # Run inference
    outputs = ort_session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    )
    
    # Process outputs
    logits = outputs[0]
    predictions = np.argmax(logits, axis=-1)
    decoded = tokenizer.batch_decode(predictions)
    
    print("\nGenerated text:")
    print(decoded[0])

if __name__ == "__main__":
    export_to_onnx()

    export_with_consolidated_tensors()
    
    verify_onnx_model()
    
    # Example usage
    test_prompt = "Extract messages from the given conversation that contain any indications of possible criminal activity"
    onnx_inference(test_prompt)