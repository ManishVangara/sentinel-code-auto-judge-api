# SentinelCode AutoJudge API: Technical Architecture & Fundamentals Report

## 1. Executive Summary
The **SentinelCode AutoJudge API** is a robust, multi-language code execution engine designed for competitive programming platforms, online coding assessments, and educational tools. It provides a secure, sandboxed environment to compile, execute, and validate code submissions against test cases. The system leverages modern AI models (OpenAI, Anthropic, Hugging Face) for automated test case generation and employs advanced heuristic analysis for language detection and security validation.

---

## 2. Core Software Architecture

### 2.1 Backend Framework: FastAPI (ASGI)
The application is built on **FastAPI**, a modern, high-performance web framework for building APIs with Python 3.8+.
- **Asynchronous Capabilities**: Utilizes Python's `async/await` syntax for potentially handling concurrent requests efficiently (though the core `subprocess` calls are CPU-bound/blocking in this implementation).
- **Data Validation**: Uses **Pydantic** models (`CodeRequest`, `TestCase`) for strict type checking and request validation, ensuring data integrity before processing.
- **RESTful Design**: Exposes standard endpoints (e.g., `POST /run`, `GET /languages`) following REST principles.

### 2.2 Execution Engine: Process Isolation
The core responsibility of the application is executing untrusted code securely.
- **Subprocess Management**: The application spawns separate child processes for compilation and execution using Python's `subprocess` module. This provides a basic level of isolation from the main server process.
- **Inter-Process Communication (IPC)**: Communication with the child process occurs via standard I/O streams—**stdin** (input), **stdout** (output), and **stderr** (error logging). The `communicate()` method is used to send input and capture output securely.

### 2.3 Modular AI Integration (Open/Closed Principle)
The system integrates with multiple AI providers for test case generation.
- **Factory Pattern**: The `AIProviderFactory` class encapsulates the creation logic for different AI clients (OpenAI, Anthropic, Hugging Face). This allows new providers to be added without modifying the core consumption logic.
- **Polymorphism**: The system treats different AI providers uniformly through a common interface assumption (client instantiation and API calling).

---

## 3. Key Software Fundamentals & Concepts

### 3.1 Operating System Concepts
*   **Process Management & Lifecycles**:
    *   **Fork/Exec**: The system effectively forks the current process and executes a new program (compiler or interpreter).
    *   **Exit Codes**: It relies on standard POSIX exit codes (0 for success, non-zero for error) to determine compilation or runtime status.
    *   **Resource Limits (ulimit)**: The `resource` module (Unix only) is used to set hard limits on CPU time (`RLIMIT_CPU`) and file size (`RLIMIT_FSIZE`) to prevent denial-of-service (DoS) attacks via infinite loops or disk filling.

*   **File System Operations**:
    *   **Temporary Directories**: Usage of `tempfile.TemporaryDirectory` ensures that code files are created in an isolated, ephemeral location and automatically cleaned up after execution, preventing file system clutter and race conditions.
    *   **File I/O**: Direct manipulation of source files (`.py`, `.c`, `.java`) and executables.

### 3.2 Compiler & Interpreter Theory
*   **Compilation Pipeline**: For C/C++ and Go, the system manages the build pipeline: `Source Code -> Compiler -> Linker -> Executable Binary`.
*   **Interpretation**: For Python, JavaScript (Node.js), and Perl, the system invokes the runtime environment directly on the source file.
*   **Static Analysis (Lightweight)**:
    *   **Syntax Validation**: Uses Python's `ast` (Abstract Syntax Tree) module to validate Python syntax without execution.
    *   **Heuristic Analysis**: Uses regex-based weighted pattern matching to guess the programming language (e.g., detecting `public class` for Java vs `def` for Python), implementing a form of statistical classification.

### 3.3 Security Engineering
*   **Input Sanitization**:
    *   **Regex Filtering**: A "blacklist" approach is used via `FORBIDDEN_PATTERNS` to detect dangerous keywords (e.g., `import os`, `system()`, `exec()`) before execution.
    *   **Regex**: Use of regular expressions for both security scanning and output cleaning.
*   **Sandboxing**:
    *   **Timeouts**: Strict timeouts (`MAX_EXECUTION_TIME`) prevent infinite loops.
    *   **Memory Monitoring**: Uses `psutil` to track memory consumption of the child process tree (RSS - Resident Set Size).

### 3.4 Design Patterns Used
*   **Factory Pattern**: `AIProviderFactory` creates objects without specifying the exact class of object that will be created.
*   **Strategy Pattern**: Although implemented procedurally, the `compile_*` and `get_execute_command` functions act as strategies for handling different languages (Python strategy, Java strategy, etc.).

---

## 4. Technical Workflows

### 4.1 The "Run Code" Pipeline
1.  **Request Receipt**: JSON payload with code and language.
2.  **Validation Layer**:
    *   Check for supported language.
    *   **Sanitization**: Remove invisible Unicode characters.
    *   **Security Scan**: Grep for forbidden patterns.
    *   **Language Mismatch**: Detect if declared language matches code signatures.
3.  **Preparation**:
    *   Create unique temp dir.
    *   Write source code to file (e.g., `<uuid>.py`).
4.  **Compilation (if applicable)**:
    *   Run `gcc`/`javac`/`go build`.
    *   Capture stderr for compilation errors.
5.  **Test Case Generation (Optional, AI-driven)**:
    *   If no input needed -> Skip.
    *   If input needed -> Prompt LLM (GPT-4/Claude) to generate input/output pairs.
6.  **Execution Loop**:
    *   For each test case:
        *   Spawn process.
        *   Feed input via `stdin`.
        *   Capture `stdout`/`stderr`.
        *   Enforce timeout/memory limits.
        *   Compare actual vs. expected output.
7.  **Response**: Aggregate results into JSON report.

---

## 5. Conclusion
The codebase demonstrates a solid application of **systems programming** (process/resource management) wrapped in a modern **web API** architecture. It effectively balances the need for executing arbitrary code with security constraints through multilayered validation (static analysis, timeouts, resource limits).
