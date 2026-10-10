import sys
import os
import argparse


def run_diagnostics():
    print("=" * 60)
    print(" SentinelCFR Air-Gapped System Diagnostics & Health Check")
    print("=" * 60)

    # 1. RAG & Vector Store Module
    print("\n[1/2] Checking RAG & Vector Store Module...")
    try:
        from app import rag
        chunk_count = rag.COL.count()
        print("  [OK] MiniLM embedding model loaded locally (local_files_only=True).")
        print(f"  [OK] Chroma collection ('safety_docs') chunk count: {chunk_count}")

        # Test Ollama LLM connectivity
        base_url = str(rag.llm.base_url)
        print(f"  [..] Testing Ollama connectivity at {base_url}...")
        try:
            models_response = rag.llm.models.list()
            available_models = [m.id for m in models_response.data]
            model_found = any(rag.MODEL in m for m in available_models)
            print("  [OK] Ollama is reachable.")
            if model_found:
                print(f"  [OK] Target model '{rag.MODEL}' is registered and ready.")
            else:
                print(f"  [WARN] Target model '{rag.MODEL}' not found in Ollama list: {available_models}")
        except Exception as err:
            print(f"  [WARN] Could not connect to Ollama endpoint ({base_url}): {err}")
    except Exception as e:
        print(f"  [FAIL] RAG module check failed: {e}")
        sys.exit(1)

    # 2. Agent Module
    print("\n[2/2] Checking Compliance Agent Module...")
    try:
        from app import agent
        print("  [OK] Agent module loaded successfully.")
        print(f"  [OK] Target agent model: {agent.MODEL}")
        print(f"  [OK] Registered agent tools: {[t['function']['name'] for t in agent.TOOLS]}")
    except Exception as e:
        print(f"  [FAIL] Agent module check failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" SentinelCFR Diagnostics completed successfully.")
    print("=" * 60)
    sys.exit(0)


def run_ui(host="0.0.0.0", port=7860):
    print(f"Starting SentinelCFR Gradio UI on http://{host}:{port} ...")
    from app.rules import demo
    demo.launch(server_name=host, server_port=port)


def run_api(host="0.0.0.0", port=8000):
    print(f"Starting SentinelCFR FastAPI server on http://{host}:{port} ...")
    import uvicorn
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(
        description="SentinelCFR - Air-Gapped Industrial Safety Regulatory Engine Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --check       Run SentinelCFR system diagnostics and exit
  python demo.py --ui          Launch the SentinelCFR Gradio web interface
  python demo.py --api         Launch the SentinelCFR FastAPI production REST API
        """
    )
    parser.add_argument("--check", action="store_true", help="Run non-blocking diagnostics and system health check")
    parser.add_argument("--ui", action="store_true", help="Start the SentinelCFR Gradio web interface (default port 7860)")
    parser.add_argument("--api", action="store_true", help="Start the SentinelCFR FastAPI REST server via uvicorn (default port 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to (default: 7860 for UI, 8000 for API)")

    args = parser.parse_args()

    if args.check:
        run_diagnostics()
    elif args.ui:
        port = args.port or 7860
        run_ui(host=args.host, port=port)
    elif args.api:
        port = args.port or 8000
        run_api(host=args.host, port=port)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()