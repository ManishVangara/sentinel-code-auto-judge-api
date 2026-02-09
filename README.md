# SentinelCode AutoJudge API

**A Secure, AI-Powered Multi-Language Code Execution & Evaluation Service**

SentinelCode AutoJudge API is a FastAPI-based, production-ready code execution engine that securely runs and evaluates code across multiple programming languages. It features strong sandboxing, smart language detection, resource monitoring, and optional AI-powered test case generation.

This project is designed for real-world use cases such as online judges, coding platforms, automated graders, and secure code execution services.

---

## 🚀 Features

### Core Features
- **Multi-Language Support**  
  Execute code written in:
  - Python
  - C
  - C++
  - Java
  - JavaScript (Node.js)
  - Go
  - Perl  
  *(Requires respective compilers/interpreters to be installed)*

- **Secure Sandboxing**
  - Execution time limits
  - Memory usage caps
  - Restricted system access
  - Forbidden pattern detection

- **Smart Language Detection**
  - Multi-stage, weighted pattern matching
  - Syntax validation per language
  - Prevents declared vs detected language mismatch

- **AI-Supported Test Case Generation**
  - Automatic test case creation using:
    - OpenAI
    - Anthropic (Claude)
    - Hugging Face models  
  *(API keys required)*

- **Resource Monitoring**
  - Tracks execution time
  - Tracks memory usage using `psutil`

- **Flexible Input Handling**
  - Supports string-based inputs
  - Supports array-based test cases
  - Suitable for real-world judging scenarios

---

## 🔐 Security Features & Considerations

- Maximum code size: **10 KB**
- Default execution timeout: **5 seconds**
- Maximum memory usage: **150 MB**
- Detection of forbidden and dangerous operations
- Directory traversal protection
- System file access prevention
- Process isolation and cleanup after execution

---

## 📦 Prerequisites

### Required
- **Python 3.8+**

### Language Compilers / Interpreters (as needed)

| Language | Requirement |
|--------|-------------|
| C | `gcc` |
| C++ | `g++` |
| Java | `javac`, `java` |
| JavaScript | `node` |
| Go | `go` |
| Perl | `perl` |

---

## 🧰 Python Dependencies

Install required Python libraries:

```bash
pip install -r requirements.txt
