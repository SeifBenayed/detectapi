from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import numpy as np
import os
import glob
import dataclasses
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@dataclasses.dataclass
class DetectorArgs:
    reference_model_name: str = "falcon-7b"  # Use smaller model by default
    scoring_model_name: str = "falcon-7b-instruct"
    device: str = "cpu"
    cache_dir: str = "../cache"
    batch_size: int = 4  # Reduced batch size for better stability
    max_length: int = 512  # Maximum input length to process

class FastDetectGPT:
    def __init__(self, args):
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        
        logger.info(f"Loading tokenizers and models...")
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        
        if args.reference_model_name != args.scoring_model_name:
            self.reference_tokenizer = load_tokenizer(args.reference_model_name, args.cache_dir)
            self.reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
            self.reference_model.eval()
        else:
            self.reference_tokenizer = self.scoring_tokenizer
            self.reference_model = self.scoring_model
        
        # pre-calculated parameters by fitting a LogisticRegression on detection results
        linear_params = {
            'gpt-j-6B_gpt-neo-2.7B': (1.87, -2.19),
            'gpt-neo-2.7B_gpt-neo-2.7B': (1.97, -1.47),
            'falcon-7b_falcon-7b-instruct': (2.42, -2.83),
        }
        key = f'{args.reference_model_name}_{args.scoring_model_name}'
        self.linear_k, self.linear_b = linear_params.get(key, (2.0, -2.0))  # Default if not found
        logger.info(f"Detector initialized with {args.reference_model_name} and {args.scoring_model_name}")

    def compute_crit(self, text):
        """Compute criterion for a single text with improved error handling."""
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty or invalid text")
            
            # Ensure text length is reasonable
            text = text[:self.args.max_length * 4]  # Rough character limit
            
            # Process with scoring model - FIXED: use dictionary access for tokenizer output
            scoring_tokenized = self.scoring_tokenizer(
                text, 
                truncation=True, 
                max_length=self.args.max_length,
                return_tensors="pt", 
                padding=True, 
                return_token_type_ids=False
            )
            
            if scoring_tokenized is None or 'input_ids' not in scoring_tokenized:
                raise ValueError("Scoring tokenizer failed to process the text")
                
            # Move to device - dictionary style
            scoring_tokenized = {k: v.to(self.args.device) for k, v in scoring_tokenized.items()}
            scoring_labels = scoring_tokenized['input_ids'][:, 1:]  # FIXED: use dictionary access
            
            with torch.no_grad():
                scoring_outputs = self.scoring_model(**scoring_tokenized)
                if scoring_outputs is None or not hasattr(scoring_outputs, 'logits'):
                    raise ValueError("Scoring model failed to process the tokenized text")
                    
                scoring_logits = scoring_outputs.logits[:, :-1]
                
                # If using the same model for reference and scoring
                if self.args.reference_model_name == self.args.scoring_model_name:
                    reference_logits = scoring_logits
                    reference_labels = scoring_labels
                else:
                    # Process with reference model
                    reference_tokenized = self.reference_tokenizer(
                        text, 
                        truncation=True, 
                        max_length=self.args.max_length,
                        return_tensors="pt", 
                        padding=True, 
                        return_token_type_ids=False
                    )
                    
                    if reference_tokenized is None or 'input_ids' not in reference_tokenized:
                        raise ValueError("Reference tokenizer failed to process the text")
                        
                    # Move to device - dictionary style
                    reference_tokenized = {k: v.to(self.args.device) for k, v in reference_tokenized.items()}
                    reference_labels = reference_tokenized['input_ids'][:, 1:]  # FIXED: use dictionary access
                    
                    # Check if token lengths match, pad if needed
                    if reference_labels.size(1) != scoring_labels.size(1):
                        logger.warning(f"Token length mismatch: {reference_labels.size(1)} vs {scoring_labels.size(1)}")
                        
                        # Use the shorter length for both
                        min_length = min(reference_labels.size(1), scoring_labels.size(1))
                        reference_labels = reference_labels[:, :min_length]
                        scoring_labels = scoring_labels[:, :min_length]
                    
                    reference_outputs = self.reference_model(**reference_tokenized)
                    if reference_outputs is None or not hasattr(reference_outputs, 'logits'):
                        raise ValueError("Reference model failed to process the tokenized text")
                        
                    reference_logits = reference_outputs.logits[:, :-1]
                    
                    # Match dimensions if needed
                    if reference_logits.size(1) != scoring_logits.size(1):
                        min_length = min(reference_logits.size(1), scoring_logits.size(1))
                        reference_logits = reference_logits[:, :min_length]
                        scoring_logits = scoring_logits[:, :min_length]
                        reference_labels = reference_labels[:, :min_length] if reference_labels.size(1) > min_length else reference_labels
                        scoring_labels = scoring_labels[:, :min_length] if scoring_labels.size(1) > min_length else scoring_labels
                
                # Compute criterion
                crit = self.criterion_fn(reference_logits, scoring_logits, scoring_labels)
                
            return crit, scoring_labels.size(1)
            
        except Exception as e:
            logger.error(f"Error in compute_crit: {str(e)}")
            raise

    def compute_prob(self, text):
        """Compute probability for a single text."""
        try:
            crit, ntoken = self.compute_crit(text)
            prob = sigmoid(self.linear_k * crit + self.linear_b)
            return prob, crit, ntoken
        except Exception as e:
            logger.error(f"Error in compute_prob: {str(e)}")
            raise

    def compute_batch_probs(self, texts):
        """Process a batch of texts efficiently with robust error handling."""
        results = []
        
        # Process in smaller batches to avoid memory issues
        for i in range(0, len(texts), self.args.batch_size):
            batch_texts = texts[i:i + self.args.batch_size]
            logger.info(f"Processing batch {i//self.args.batch_size + 1}/{(len(texts)-1)//self.args.batch_size + 1}")
            
            batch_results = []
            for text in batch_texts:
                try:
                    if not isinstance(text, str) or not text.strip():
                        batch_results.append({
                            "criterion": 0.0,
                            "probability": 0.0,
                            "num_tokens": 0,
                            "is_machine_generated": False,
                            "error": "Empty text"
                        })
                        continue
                    
                    # Text preprocessing
                    cleaned_text = text.strip()
                    
                    prob, crit, ntokens = self.compute_prob(cleaned_text)
                    batch_results.append({
                        "criterion": float(crit),
                        "probability": float(prob),
                        "num_tokens": int(ntokens),
                        "is_machine_generated": prob > 0.5,
                        "error": None
                    })
                except Exception as e:
                    logger.error(f"Error processing text: {str(e)}")
                    batch_results.append({
                        "criterion": 0.0,
                        "probability": 0.0,
                        "num_tokens": 0,
                        "is_machine_generated": False,
                        "error": str(e)
                    })
            
            results.extend(batch_results)
            
            # Clear CUDA cache if using GPU
            if self.args.device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return results
        
    def compute_multi_threaded(self, texts, max_workers=2):
        """Process texts using multiple threads with better error handling."""
        results = [None] * len(texts)
        logger.info(f"Processing {len(texts)} texts with {max_workers} threads")
        
        def process_text(idx, text):
            try:
                if not isinstance(text, str) or not text.strip():
                    return idx, {
                        "criterion": 0.0,
                        "probability": 0.0,
                        "num_tokens": 0,
                        "is_machine_generated": False,
                        "error": "Empty text"
                    }
                
                # Text preprocessing
                cleaned_text = text.strip()
                
                prob, crit, ntokens = self.compute_prob(cleaned_text)
                return idx, {
                    "criterion": float(crit),
                    "probability": float(prob),
                    "num_tokens": int(ntokens),
                    "is_machine_generated": prob > 0.5,
                    "error": None
                }
            except Exception as e:
                logger.error(f"Thread error processing text {idx}: {str(e)}")
                return idx, {
                    "criterion": 0.0,
                    "probability": 0.0,
                    "num_tokens": 0,
                    "is_machine_generated": False,
                    "error": str(e)
                }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_text, i, text): i for i, text in enumerate(texts)}
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    logger.error(f"Error getting future result: {str(e)}")
        
        # Make sure all results are populated
        for i in range(len(results)):
            if results[i] is None:
                results[i] = {
                    "criterion": 0.0,
                    "probability": 0.0,
                    "num_tokens": 0,
                    "is_machine_generated": False,
                    "error": "Processing failed"
                }
                
        return results

# Initialize FastAPI app
app = FastAPI(
    title="Fast-DetectGPT API",
    description="API for detecting machine-generated text using Fast-DetectGPT",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize detector with default arguments
detector = None

@app.on_event("startup")
async def startup_event():
    global detector
    args = DetectorArgs()
    detector = FastDetectGPT(args)

# Request/Response Models
class TextRequest(BaseModel):
    text: str = Field(..., max_length=8192)

class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    multi_threaded: bool = Field(False, description="Whether to use multi-threading")

class DetectionResponse(BaseModel):
    criterion: float
    probability: float
    num_tokens: int
    is_machine_generated: bool
    error: Optional[str] = None

class BatchDetectionResponse(BaseModel):
    results: List[DetectionResponse]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    detector_loaded: bool
    device: str
    reference_model: str
    scoring_model: str

@app.post("/detect", response_model=DetectionResponse)
async def detect_machine_generated_text(request: TextRequest):
    """Detect if the provided text is machine-generated"""
    text = request.text.strip()
    if not curl -X POST http://20.106.35.65:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a sample text to test if the API is working properly."}':
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        prob, crit, ntokens = detector.compute_prob(text)
        return DetectionResponse(
            criterion=float(crit),
            probability=float(prob),
            num_tokens=int(ntokens),
            is_machine_generated=prob > 0.5,
            error=None
        )
    except Exception as e:
        logger.error(f"Error in /detect endpoint: {str(e)}")
        return DetectionResponse(
            criterion=0.0,
            probability=0.0,
            num_tokens=0,
            is_machine_generated=False,
            error=str(e)
        )

@app.post("/detect_batch", response_model=BatchDetectionResponse)
async def detect_batch(request: BatchTextRequest):
    """Detect if multiple texts are machine-generated (batch processing)"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    try:
        import time
        start_time = time.time()
        
        # Default to batch processing as it's more stable
        if request.multi_threaded:
            # Use multi-threading for better CPU utilization
            results = detector.compute_multi_threaded(request.texts, max_workers=2)
        else:
            # Use batch processing
            results = detector.compute_batch_probs(request.texts)
            
        processing_time = time.time() - start_time
        
        return BatchDetectionResponse(
            results=results,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error in /detect_batch endpoint: {str(e)}")
        # Return error results for all texts
        error_results = [
            DetectionResponse(
                criterion=0.0,
                probability=0.0,
                num_tokens=0,
                is_machine_generated=False,
                error=f"Batch processing error: {str(e)}"
            ) for _ in request.texts
        ]
        return BatchDetectionResponse(
            results=error_results,
            processing_time=0.0
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is healthy and detector is loaded"""
    return HealthResponse(
        status="healthy",
        detector_loaded=detector is not None,
        device=detector.args.device if detector else "unknown",
        reference_model=detector.args.reference_model_name if detector else "unknown",
        scoring_model=detector.args.scoring_model_name if detector else "unknown"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hemden:app", host="0.0.0.0", port=8000)
