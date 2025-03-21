import torch
import argparse
import os
import cag.dataset as cagds
import cag.similarity as cagsim
from time import time
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
)
from transformers.cache_utils import DynamicCache
import logging
import faiss   # Add FAISS
import numpy as np   # Add numpy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found")

global model_name, model, tokenizer
global rand_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

"""KV Cache test"""
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])

# Add embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# load embedding model
embedding_tokenizer, embedding_model = AutoTokenizer.from_pretrained(EMBEDDING_MODEL), AutoModel.from_pretrained(EMBEDDING_MODEL).to(device)

def load_embedding_model():
    return embedding_tokenizer, embedding_model

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def get_batch_embeddings(text_list, tokenizer, model):
    inputs = tokenizer(
        text_list, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def store_embeddings_faiss(text_list, faiss_index_path, nlist=10):
    tokenizer, model = load_embedding_model()
    embeddings = get_batch_embeddings(text_list, tokenizer, model)  # 배치 단위 임베딩 처리

    d = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    
    index.train(embeddings)
    index.add(embeddings)
    
    index.nprobe = min(5, nlist)  

    faiss.write_index(index, faiss_index_path)
    
    return index

def load_faiss_index(faiss_index_path):
    return faiss.read_index(faiss_index_path)

def prepare_index(text_list, faiss_index_path):
    if os.path.exists(faiss_index_path):
        return load_faiss_index(faiss_index_path)
    return store_embeddings_faiss(text_list, faiss_index_path)

def find_similar_documents(
    query, faiss_index, text_list, top_k=3
    ):
    """
    주어진 쿼리와 가장 유사한 문서들을 검색합니다.

    Args:
        query: 검색할 쿼리 텍스트
        faiss_index: FAISS 인덱스
        clusters: 문서 클러스터 정보
        text_list: 원본 텍스트 리스트
        top_k: 반환할 가장 유사한 문서 개수

    Returns:
        가장 유사한 문서들의 리스트
    """
    # 쿼리 임베딩 생성성
    tokenizer, model = load_embedding_model()
    query_embedding = get_embedding(query, tokenizer, model)

    # FAISS로 가장 가까운 이웃 검색
    D, I = faiss_index.search(query_embedding, top_k)

    # 검색된 클러스터에서 추가 문서 검색
    similar_docs = [text_list[i] for i in I[0]]

    return similar_docs[:top_k]

def save_results(output_path: str, results: dict):
    """
    실험 결과를 파일에 저장합니다.

    Args:
        output_path: 결과를 저장할 파일 경로
        results: 저장할 결과 딕셔너리
    """
    with open(output_path, "a") as f:
        f.write("\n=== Results Update ===\n")
        f.write(f"Number of samples processed: {len(results['prompts'])}\n")
        if results["similarity"]:
            avg_similarity = sum(results["similarity"]) / len(results["similarity"])
            f.write(f"Average Similarity: {avg_similarity:.4f}\n")
        avg_cache_time = sum(results["cache_time"]) / len(results["cache_time"])
        avg_generate_time = sum(results["generate_time"]) / len(results["generate_time"])
        f.write(f"Average Cache Time: {avg_cache_time:.2f}s\n")
        f.write(f"Average Generate Time: {avg_generate_time:.2f}s\n")
        f.write("=====================\n")

def kvcache_test(args: argparse.Namespace):
    answer_instruction = "Answer the question with a super short answer."
    text_list, dataset = cagds.get(
        args.dataset,
        max_knowledge=args.maxKnowledge,
        max_paragraph=args.maxParagraph,
        max_questions=args.maxQuestion,
    )

    faiss_index_path = "./data_cache/faiss.index"
    faiss_index = prepare_index(text_list, faiss_index_path)

    kvcache_path = "./data_cache/cache_knowledges.pt"
    dataset = list(dataset)
    max_questions = (
        min(len(dataset), args.maxQuestion)
        if args.maxQuestion is not None
        else len(dataset)
    )

    results = {
        "cache_time": [],
        "generate_time": [],
        "similarity": [],
        "prompts": [],
        "responses": [],
    }

    for id, (question, ground_truth) in enumerate(dataset[:max_questions]):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # 질문과 관련된 문서 검색
        relevant_docs = find_similar_documents(
            question, faiss_index, text_list
        )
        knowledges = "\n\n\n".join(relevant_docs)

        # KV Cache 준비
        cache_t1 = time()
        knowledge_cache, prepare_time = prepare_kvcache(
            knowledges, filepath=kvcache_path, answer_instruction=answer_instruction
        )
        cache_t2 = time()

        kv_len = knowledge_cache.key_cache[0].shape[-2]
        print(f"KVcache prepared in {prepare_time} seconds")

        if args.usePrompt:
            prompt = f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an assistant for giving short answers based on given context.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Context information is bellow.
------------------------------------------------
{knowledges}
------------------------------------------------
{answer_instruction}
Question:
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            generate_t1 = time()
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            output = generate(model, input_ids, DynamicCache())
            generated_text = tokenizer.decode(
                output[0], skip_special_tokens=True, temperature=None
            )
            generate_t2 = time()
        else:
            prompt = f"""
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            generate_t1 = time()
            clean_up(knowledge_cache, kv_len)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            output = generate(model, input_ids, knowledge_cache)
            generated_text = tokenizer.decode(
                output[0], skip_special_tokens=True, temperature=None
            )
            generate_t2 = time()

        print("Q: ", question)
        print("A: ", generated_text)

        # Evaluate bert-score similarity
        similarity = cagsim.bert(generated_text, ground_truth)

        print(
            f"[{id}]: Semantic Similarity: {round(similarity, 5)},",
            f"cache time: {cache_t2 - cache_t1},",
            f"generate time: {generate_t2 - generate_t1}",
        )

        results["prompts"].append(question)
        results["responses"].append(generated_text)
        results["cache_time"].append(cache_t2 - cache_t1)
        results["generate_time"].append(generate_t2 - generate_t1)
        results["similarity"].append(similarity)

        # 10개 질문마다 중간 결과 저장
        if (id + 1) % 10 == 0:
            save_results(args.output, results)

    # 최종 결과 저장
    save_results(args.output, results)

    # 평균 계산
    avg_similarity = sum(results["similarity"]) / len(results["similarity"])
    avg_cache_time = sum(results["cache_time"]) / len(results["cache_time"])
    avg_generate_time = sum(results["generate_time"]) / len(results["generate_time"])

    print(f"\nFinal Results:")
    print(f"Average Semantic Similarity: {avg_similarity:.4f}")
    print(f"Average Cache Time: {avg_cache_time:.2f}s")
    print(f"Average Generate Time: {avg_generate_time:.2f}s")


# ------------------------------------------------


def generate(
    model, input_ids: torch.Tensor, past_key_values, max_new_tokens: int = 300
) -> torch.Tensor:
    """
    Generate text with greedy decoding.

    Args:
        model: HuggingFace model with automatic device mapping
        input_ids: Input token ids
        past_key_values: KV Cache for knowledge
        max_new_tokens: Maximum new tokens to generate
    """

    origin_ids = input_ids
    input_ids = input_ids.to(device)

    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_token, past_key_values=past_key_values, use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            next_token = next_token.to(device)

            past_key_values = outputs.past_key_values

            output_ids = torch.cat([output_ids, next_token], dim=1)

            if next_token.item() in model.config.eos_token_id:
                break
    return output_ids[:, origin_ids.shape[-1] :]


def preprocess_knowledge(
    model,
    tokenizer,
    prompt: str,
) -> DynamicCache:
    """
    Prepare knowledge kv cache for CAG.
    Args:
        model: HuggingFace model with automatic device mapping
        tokenizer: HuggingFace tokenizer
        prompt: The knowledge to preprocess, which is basically a prompt

    Returns:
        DynamicCache: KV Cache
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    past_key_values = DynamicCache()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
    return outputs.past_key_values


def write_kv_cache(kv: DynamicCache, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    """
    Write the KV Cache to a file.
    """
    torch.save(kv, path)


def clean_up(kv: DynamicCache, origin_len: int):
    """
    Truncate the KV Cache to the original length.
    """
    for i in range(len(kv.key_cache)):
        kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
        kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]


# cluster들별로 kvcache read하도록 수정
def read_kv_cache(path: str) -> DynamicCache | None:
    """
    Read the KV Cache from a file. If the cache file is invalid or empty, return None.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        kv = torch.load(path, weights_only=True)
        return kv
    else:
        # Regenerate cache if it doesn't exist or is too small
        return None


def prepare_kvcache(
    documents,
    filepath: str = "./data_cache/cache_knowledges.pt",
    answer_instruction: str | None = None,
):
    # Prepare the knowledges kvcache

    if answer_instruction is None:
        answer_instruction = "Answer the question with a super short answer."
    knowledges = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an assistant for giving short answers based on given context.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is bellow.
    ------------------------------------------------
    {documents}
    ------------------------------------------------
    {answer_instruction}
    Question:
    """
    # Get the knowledge cache
    t1 = time()
    kv = preprocess_knowledge(model, tokenizer, knowledges)
    print("kvlen: ", kv.key_cache[0].shape[-2])
    write_kv_cache(kv, filepath)
    t2 = time()
    logger.info(f"KV cache prepared in {t2 - t1:.2f} seconds.")
    return kv, t2 - t1


# Define quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4",  # Normalize float 4 quantization
    bnb_4bit_compute_dtype=torch.float16,  # Compute dtype for 4-bit base matrices
    bnb_4bit_use_double_quant=True,  # Use nested quantization
)


def load_quantized_model(model_name, hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically choose best device
        trust_remote_code=True,  # Required for some models
        token=hf_token,
    )

    return tokenizer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAG test with specified parameters."
    )
    # parser.add_argument('--method', choices=['rag', 'kvcache'], required=True, help='Method to use (rag or kvcache)')
    # parser.add_argument('--kvcache', choices=['file', 'variable'], required=True, help='Method to use (from_file or from_var)')
    parser.add_argument(
        "--modelname",
        required=False,
        default="meta-llama/Llama-3.2-1B-Instruct",
        type=str,
        help="Model name to use",
    )
    parser.add_argument(
        "--quantized", required=False, default=False, type=bool, help="Quantized model"
    )
    parser.add_argument(
        "--faiss_index", required=True, help="Path to save FAISS index"
    )  # Add FAISS index path
    parser.add_argument(
        "--kvcache",
        choices=["file"],
        required=True,
        help="Method to use (from_file or from_var)",
    )
    parser.add_argument(
        "--similarity",
        choices=["bertscore"],
        required=True,
        help="Similarity metric to use (bertscore)",
    )
    parser.add_argument(
        "--output", required=True, type=str, help="Output file to save the results"
    )
    parser.add_argument(
        "--maxQuestion",
        required=False,
        default=None,
        type=int,
        help="Maximum number of questions to test",
    )
    parser.add_argument(
        "--maxKnowledge",
        required=False,
        default=None,
        type=int,
        help="Maximum number of knowledge items to use",
    )
    parser.add_argument(
        "--maxParagraph",
        required=False,
        default=None,
        type=int,
        help="Maximum number of paragraph to use",
    )
    parser.add_argument(
        "--usePrompt", default=False, action="store_true", help="Do not use cache"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset to use (kis, kis_sample, squad-dev, squad-train)",
        choices=[
            "kis",
            "kis_sample",
            "squad-dev",
            "squad-train",
            "hotpotqa-dev",
            "hotpotqa-train",
            "hotpotqa-test",
        ],
    )
    parser.add_argument(
        "--randomSeed",
        required=False,
        default=None,
        type=int,
        help="Random seed to use",
    )
    # 48 Articles, each article average 40~50 paragraph, each average 5~10 questions

    args = parser.parse_args()

    print(
        "maxKnowledge",
        args.maxKnowledge,
        "maxParagraph",
        args.maxParagraph,
        "maxQuestion",
        args.maxQuestion,
        "randomeSeed",
        args.randomSeed,
    )

    model_name = args.modelname
    rand_seed = args.randomSeed if args.randomSeed is not None else None

    if args.quantized:
        tokenizer, model = load_quantized_model(
            model_name=model_name, hf_token=HF_TOKEN
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
        )

    def unique_path(path, i=0):
        if os.path.exists(path):
            # path = path.split("_")[:-1] if i != 0 else path
            return unique_path(path + "_" + str(i), i + 1)
        return path

    if os.path.exists(args.output):
        args.output = unique_path(args.output)

    kvcache_test(args)
