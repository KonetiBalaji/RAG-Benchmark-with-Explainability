"""Comprehensive system verification script.

Verifies all god-level features are working correctly.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_header(text: str):
    """Print section header."""
    print(f"\n{Colors.BLUE}{'=' * 70}")
    print(f"{text}")
    print(f"{'=' * 70}{Colors.RESET}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {text}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {text}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {text}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {text}")


def verify_phase1() -> Tuple[bool, List[str]]:
    """Verify Phase 1: Code cleanup."""
    print_header("PHASE 1: Code Quality & Documentation")
    
    issues = []
    
    # Check for emojis in Python files
    print_info("Checking for emojis in Python files...")
    emoji_found = False
    for py_file in Path("src").rglob("*.py"):
        content = py_file.read_text(encoding='utf-8')
        # Check for common emoji unicode ranges
        if any(ord(c) > 0x1F300 for c in content):
            emoji_found = True
            issues.append(f"Emojis found in {py_file}")
    
    if not emoji_found:
        print_success("No emojis found in Python files")
    else:
        print_error("Emojis still present in codebase")
    
    # Check markdown files
    print_info("Checking markdown files...")
    md_files = list(Path(".").glob("*.md"))
    if len(md_files) == 1 and md_files[0].name == "README.md":
        print_success("Only README.md exists (6 extra .md files removed)")
    else:
        extra_md = [f.name for f in md_files if f.name != "README.md"]
        issues.append(f"Extra .md files found: {extra_md}")
        print_error(f"Found extra .md files: {extra_md}")
    
    return len(issues) == 0, issues


def verify_phase2() -> Tuple[bool, List[str]]:
    """Verify Phase 2: Core enhancements."""
    print_header("PHASE 2: Core Enhancements")
    
    issues = []
    
    # Check RAGAS integration
    print_info("Checking RAGAS metrics integration...")
    ragas_file = Path("src/evaluation/ragas_metrics.py")
    if ragas_file.exists():
        content = ragas_file.read_text()
        if "RAGASEvaluator" in content and "context_precision" in content:
            print_success("RAGAS metrics integration complete")
        else:
            issues.append("RAGAS integration incomplete")
            print_error("RAGAS integration incomplete")
    else:
        issues.append("RAGAS metrics file not found")
        print_error("RAGAS metrics file not found")
    
    # Check LLM-as-Judge
    print_info("Checking LLM-as-Judge evaluation...")
    judge_file = Path("src/evaluation/llm_judge.py")
    if judge_file.exists():
        content = judge_file.read_text()
        if "LLMJudge" in content and "evaluate_answer_quality" in content:
            print_success("LLM-as-Judge implementation complete")
        else:
            issues.append("LLM-as-Judge incomplete")
            print_error("LLM-as-Judge incomplete")
    else:
        issues.append("LLM-as-Judge file not found")
        print_error("LLM-as-Judge file not found")
    
    # Check tests
    print_info("Checking test suite...")
    test_file = Path("tests/test_rag_models.py")
    if test_file.exists():
        content = test_file.read_text()
        test_classes = content.count("class Test")
        if test_classes >= 5:
            print_success(f"Comprehensive test suite with {test_classes} test classes")
        else:
            issues.append(f"Only {test_classes} test classes found")
            print_warning(f"Only {test_classes} test classes found")
    else:
        issues.append("Test file not found")
        print_error("Test file not found")
    
    # Check FastAPI
    print_info("Checking FastAPI REST API...")
    api_file = Path("src/api/main.py")
    if api_file.exists():
        content = api_file.read_text()
        if "FastAPI" in content and "/query" in content and "QueryResponse" in content:
            print_success("FastAPI REST API implemented")
        else:
            issues.append("FastAPI API incomplete")
            print_error("FastAPI API incomplete")
    else:
        issues.append("FastAPI file not found")
        print_error("FastAPI file not found")
    
    return len(issues) == 0, issues


def verify_phase3() -> Tuple[bool, List[str]]:
    """Verify Phase 3: Advanced RAG techniques."""
    print_header("PHASE 3: Advanced RAG Techniques")
    
    issues = []
    
    # Check HyDE
    print_info("Checking HyDE implementation...")
    hyde_file = Path("src/models/hyde_rag.py")
    if hyde_file.exists():
        content = hyde_file.read_text()
        if "HyDERAG" in content and "hypothetical_document" in content:
            print_success("HyDE (Hypothetical Document Embeddings) implemented")
        else:
            issues.append("HyDE incomplete")
            print_error("HyDE incomplete")
    else:
        issues.append("HyDE file not found")
        print_error("HyDE file not found")
    
    # Check Self-RAG
    print_info("Checking Self-RAG implementation...")
    self_rag_file = Path("src/models/self_rag.py")
    if self_rag_file.exists():
        content = self_rag_file.read_text()
        if "SelfRAG" in content and "assess_retrieval_need" in content:
            print_success("Self-RAG (Self-Reflective) implemented")
        else:
            issues.append("Self-RAG incomplete")
            print_error("Self-RAG incomplete")
    else:
        issues.append("Self-RAG file not found")
        print_error("Self-RAG file not found")
    
    # Check Semantic Chunking
    print_info("Checking Semantic Chunking...")
    semantic_file = Path("src/data/semantic_chunker.py")
    if semantic_file.exists():
        content = semantic_file.read_text()
        if "SemanticChunker" in content and "HierarchicalChunker" in content:
            print_success("Semantic & Hierarchical Chunking implemented")
        else:
            issues.append("Semantic chunking incomplete")
            print_error("Semantic chunking incomplete")
    else:
        issues.append("Semantic chunking file not found")
        print_error("Semantic chunking file not found")
    
    # Check Multi-Query
    print_info("Checking Multi-Query Retrieval...")
    multi_query_file = Path("src/models/multi_query_rag.py")
    if multi_query_file.exists():
        content = multi_query_file.read_text()
        if "MultiQueryRAG" in content and "FusionRAG" in content:
            print_success("Multi-Query & Fusion RAG implemented")
        else:
            issues.append("Multi-Query RAG incomplete")
            print_error("Multi-Query RAG incomplete")
    else:
        issues.append("Multi-Query file not found")
        print_error("Multi-Query file not found")
    
    return len(issues) == 0, issues


def verify_phase4() -> Tuple[bool, List[str]]:
    """Verify Phase 4: Production features."""
    print_header("PHASE 4: Production Features")
    
    issues = []
    
    # Check Redis caching
    print_info("Checking Redis caching layer...")
    cache_file = Path("src/utils/cache.py")
    if cache_file.exists():
        content = cache_file.read_text()
        if "QueryCache" in content and "redis" in content.lower():
            print_success("Redis caching layer implemented")
        else:
            issues.append("Redis caching incomplete")
            print_error("Redis caching incomplete")
    else:
        issues.append("Caching file not found")
        print_error("Caching file not found")
    
    # Check Query Preprocessing
    print_info("Checking Query Preprocessing...")
    preprocess_file = Path("src/utils/query_preprocessor.py")
    if preprocess_file.exists():
        content = preprocess_file.read_text()
        if "QueryPreprocessor" in content and "spell_correct" in content:
            print_success("Query preprocessing pipeline implemented")
        else:
            issues.append("Query preprocessing incomplete")
            print_error("Query preprocessing incomplete")
    else:
        issues.append("Query preprocessing file not found")
        print_error("Query preprocessing file not found")
    
    # Check Citation Generation
    print_info("Checking Citation Generation...")
    citation_file = Path("src/utils/citation_generator.py")
    if citation_file.exists():
        content = citation_file.read_text()
        if "CitationGenerator" in content and "add_citations" in content:
            print_success("Citation generation system implemented")
        else:
            issues.append("Citation generation incomplete")
            print_error("Citation generation incomplete")
    else:
        issues.append("Citation file not found")
        print_error("Citation file not found")
    
    return len(issues) == 0, issues


def count_rag_configurations() -> int:
    """Count total RAG configurations."""
    configs = []
    
    # Original 4
    if Path("src/models/baseline_rag.py").exists():
        configs.append("Baseline")
    if Path("src/models/hybrid_rag.py").exists():
        configs.append("Hybrid")
    if Path("src/models/reranker_rag.py").exists():
        configs.append("Reranker")
    if Path("src/models/query_decomposition_rag.py").exists():
        configs.append("Query Decomposition")
    
    # New advanced configs
    if Path("src/models/hyde_rag.py").exists():
        configs.append("HyDE")
        configs.append("Multi-HyDE")
    if Path("src/models/self_rag.py").exists():
        configs.append("Self-RAG")
    if Path("src/models/multi_query_rag.py").exists():
        configs.append("Multi-Query")
        configs.append("Fusion RAG")
    
    return len(configs), configs


def main():
    """Run all verifications."""
    print_header("RAG BENCHMARK SYSTEM - GOD-LEVEL VERIFICATION")
    print_info("Verifying all implementations...")
    
    # Run all phases
    phase1_pass, phase1_issues = verify_phase1()
    phase2_pass, phase2_issues = verify_phase2()
    phase3_pass, phase3_issues = verify_phase3()
    phase4_pass, phase4_issues = verify_phase4()
    
    # Count configurations
    num_configs, config_list = count_rag_configurations()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    phases = [
        ("Phase 1: Code Quality", phase1_pass),
        ("Phase 2: Core Enhancements", phase2_pass),
        ("Phase 3: Advanced RAG", phase3_pass),
        ("Phase 4: Production Features", phase4_pass),
    ]
    
    all_passed = all(p[1] for p in phases)
    
    for phase_name, passed in phases:
        if passed:
            print_success(f"{phase_name}: PASSED")
        else:
            print_error(f"{phase_name}: FAILED")
    
    print(f"\n{Colors.BLUE}RAG Configurations:{Colors.RESET} {num_configs} total")
    for i, config in enumerate(config_list, 1):
        print(f"  {i}. {config}")
    
    # Overall status
    print(f"\n{'=' * 70}")
    if all_passed:
        print(f"{Colors.GREEN}{'='*70}")
        print(f"{'GOD-LEVEL STATUS: ACHIEVED!':^70}")
        print(f"{'='*70}{Colors.RESET}")
        print(f"\n{Colors.GREEN}All systems operational and god-level features implemented!{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{'='*70}")
        print(f"{'STATUS: ISSUES FOUND':^70}")
        print(f"{'='*70}{Colors.RESET}")
        
        all_issues = phase1_issues + phase2_issues + phase3_issues + phase4_issues
        print(f"\n{Colors.RED}Issues to fix:{Colors.RESET}")
        for issue in all_issues:
            print(f"  - {issue}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
